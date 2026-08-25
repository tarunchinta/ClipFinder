"""Instagram reel ingestion: yt-dlp download, Azure Blob storage, indexing enqueue."""

import asyncio
import functools
import logging
import os
import tempfile
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.indexed_file import IndexedFile, IndexingStatus
from app.models.user import User
from app.services.service_bus_publisher import publish_video_indexing_jobs

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = ("instagram.com", "instagr.am")


class InstagramIngestError(Exception):
    """Raised when a reel cannot be ingested; message is safe to show the caller."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _validate_reel_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InstagramIngestError(
            "URL must start with http:// or https://", status_code=400
        )
    host = (parsed.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _ALLOWED_HOSTS):
        raise InstagramIngestError(
            "URL must be an instagram.com link", status_code=400
        )


def _truncate(value: str | None, limit: int) -> str | None:
    """Clamp a metadata string to its column width; empty values become NULL."""
    return value[:limit] if value else None


def _timestamp_to_datetime(timestamp: int | float | None) -> datetime | None:
    """Convert a yt-dlp unix timestamp to a naive UTC datetime, matching the model."""
    if timestamp is None:
        return None
    try:
        return datetime.utcfromtimestamp(timestamp)
    except (OverflowError, OSError, ValueError):
        logger.warning("Reel reported an unusable timestamp: %r", timestamp)
        return None


def _download_reel_sync(url: str, out_dir: str) -> dict:
    """
    Download a reel with yt-dlp. Returns the video id, the metadata Instagram
    reports about the post (title, description, uploader, uploader_id,
    published_at, duration) and the downloaded file path.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "mp4/best",
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
    except DownloadError as e:
        raise InstagramIngestError(f"Could not download reel: {e}") from e

    if not info.get("id") or not os.path.isfile(file_path):
        raise InstagramIngestError("Download produced no video file")
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "published_at": _timestamp_to_datetime(info.get("timestamp")),
        "duration": info.get("duration"),
        "file_path": file_path,
    }


def _upload_video_to_azure_blob_sync(blob_path: str, file_path: str) -> str | None:
    """Upload the reel video to the videos container; return the blob URL or None."""
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        logger.error("Azure Blob not configured, cannot store reel video")
        return None
    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings

        client = BlobServiceClient.from_connection_string(
            settings.azure_blob_connection_string,
        )
        container_client = client.get_container_client(
            settings.azure_blob_videos_container_name
        )
        try:
            container_client.create_container()
        except Exception:
            pass  # already exists
        blob_client = container_client.get_blob_client(blob_path)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type="video/mp4"),
            )
        return blob_client.url
    except Exception as e:
        logger.error("Azure Blob video upload failed: %s", e)
        return None


async def ingest_instagram_reel(
    url: str,
    user: User,
    session: AsyncSession,
) -> dict:
    """
    Download an Instagram reel, store it in Azure Blob, upsert an IndexedFile row,
    and enqueue full frame + transcript indexing.

    Raises InstagramIngestError with a caller-safe message on failure.
    """
    settings = get_settings()
    _validate_reel_url(url)

    loop = asyncio.get_running_loop()
    with tempfile.TemporaryDirectory(prefix="reel_") as tmpdir:
        info = await loop.run_in_executor(
            None, functools.partial(_download_reel_sync, url, tmpdir)
        )

        duration = info.get("duration")
        if duration and duration > settings.mcp_max_reel_seconds:
            raise InstagramIngestError(
                f"Reel is {duration:.0f}s long; the limit is "
                f"{settings.mcp_max_reel_seconds}s (MCP_MAX_REEL_SECONDS)",
                status_code=400,
            )

        drive_file_id = f"instagram:{info['id']}"
        existing_id = (
            await session.execute(
                select(IndexedFile.id).where(
                    IndexedFile.user_id == user.id,
                    IndexedFile.drive_file_id == drive_file_id,
                )
            )
        ).scalar_one_or_none()

        filename = f"{(info.get('title') or info['id'])[:490]}.mp4"
        size_bytes = os.path.getsize(info["file_path"])
        title = _truncate(info.get("title"), 500)
        uploader = _truncate(info.get("uploader"), 255)
        uploader_id = _truncate(info.get("uploader_id"), 255)
        now = datetime.utcnow()
        stmt = insert(IndexedFile).values(
            user_id=user.id,
            google_account_id="instagram",
            folder_id="instagram",
            drive_file_id=drive_file_id,
            filename=filename,
            mime_type="video/mp4",
            file_type="video",
            size_bytes=size_bytes,
            duration_seconds=duration,
            source_type="instagram",
            source_url=url,
            title=title,
            description=info.get("description"),
            uploader=uploader,
            uploader_id=uploader_id,
            published_at=info.get("published_at"),
            indexing_status=IndexingStatus.PENDING.value,
            transcript_status=IndexingStatus.PENDING.value,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_drive_file",
            set_={
                "filename": stmt.excluded.filename,
                "size_bytes": stmt.excluded.size_bytes,
                "duration_seconds": stmt.excluded.duration_seconds,
                "source_url": stmt.excluded.source_url,
                "title": stmt.excluded.title,
                "description": stmt.excluded.description,
                "uploader": stmt.excluded.uploader,
                "uploader_id": stmt.excluded.uploader_id,
                "published_at": stmt.excluded.published_at,
                "indexing_status": IndexingStatus.PENDING.value,
                "transcript_status": IndexingStatus.PENDING.value,
                "error_message": None,
                "frames_total": None,
                "frames_completed": 0,
                "frames_failed": 0,
                "updated_at": now,
                "indexed_at": None,
            },
        ).returning(IndexedFile)
        indexed_file = (await session.execute(stmt)).scalar_one()
        await session.commit()

        blob_path = f"{user.id}/{indexed_file.id}/source.mp4"
        blob_video_url = await loop.run_in_executor(
            None,
            functools.partial(
                _upload_video_to_azure_blob_sync, blob_path, info["file_path"]
            ),
        )
        if not blob_video_url:
            indexed_file.indexing_status = IndexingStatus.FAILED.value
            indexed_file.error_message = "Failed to store reel video in Azure Blob"
            indexed_file.updated_at = datetime.utcnow()
            await session.commit()
            raise InstagramIngestError(indexed_file.error_message)

        indexed_file.blob_video_url = blob_video_url
        indexed_file.updated_at = datetime.utcnow()
        await session.commit()

    try:
        publish_video_indexing_jobs([indexed_file], [{}])
    except Exception:
        logger.exception(
            "Service Bus publish failed for reel %s, falling back to in-process indexing",
            indexed_file.id,
        )
        from app.tasks import _run_frame_indexing_async

        task = asyncio.create_task(_run_frame_indexing_async(str(indexed_file.id)))
        task.add_done_callback(_log_indexing_task_result)

    return {
        "video_id": str(indexed_file.id),
        "id": info["id"],
        "title": indexed_file.title,
        "description": indexed_file.description,
        "uploader": indexed_file.uploader,
        "uploader_id": indexed_file.uploader_id,
        "published_at": (
            indexed_file.published_at.isoformat()
            if indexed_file.published_at
            else None
        ),
        "duration_seconds": duration,
        "source_url": url,
        "file_path": blob_path,
        "blob_url": indexed_file.blob_video_url,
        "status": indexed_file.indexing_status,
        "deduplicated": existing_id is not None,
    }


def _log_indexing_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("In-process reel indexing task failed")
