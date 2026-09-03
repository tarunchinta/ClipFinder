"""Video frame extraction, embedding, and storage for per-frame video search."""

import asyncio
import functools
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

import httpx
from sqlalchemy import literal_column, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_maker
from app.models.indexed_file import IndexedFile, IndexingStatus
from app.models.video_frame_embedding import VideoFrameEmbedding
from app.observability import (
    emit_video_frame_trace,
    flush_langfuse,
    trace_index_file,
    trace_video_frame,
)
from app.services.color_signature import (
    apply_color_signature,
    merge_signatures,
    signature_from_image_bytes,
    signature_from_jpeg_path,
)
from app.services.transcription import transcribe_video_with_own_session
from app.services.vision_embedding import get_vision_embedding_service

logger = logging.getLogger(__name__)


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str) -> None:
    # #region agent log
    import os
    payload = {"sessionId": "b87a53", "location": location, "message": message, "data": data, "hypothesisId": hypothesis_id, "timestamp": __import__("time").time() * 1000}
    logger.info("[DEBUG b87a53] %s", json.dumps(payload))
    for base in (Path(__file__).resolve().parent.parent.parent, Path(os.getcwd()), Path(__import__("tempfile").gettempdir())):
        try:
            log_path = base / "debug-b87a53.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            break
        except Exception:
            continue
    # #endregion


# Default SAS expiry for thumbnail URLs returned to the frontend (1 hour)
BLOB_SAS_EXPIRY_HOURS = 100

# Extract every 5th frame (frame indices 0, 5, 10, ...)
FRAME_INTERVAL = 5

# Downsample frames during ffmpeg extract (~CLIP input size; saves blob/API payload)
FRAME_MAX_DIMENSION = 336
FRAME_JPEG_QUALITY = 5


@dataclass(frozen=True)
class ExtractedFrame:
    """One extracted frame ready for CLIP indexing."""

    frame_index: int
    time_seconds: float
    blob_path: str
    frame_image_url: str | None
    local_jpeg_path: str


async def _download_video_from_drive(
    drive_file_id: str,
    google_access_token: str,
) -> bytes | None:
    """Download video file content from Google Drive using OAuth token."""
    url = f"https://www.googleapis.com/drive/v3/files/{drive_file_id}?alt=media"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {google_access_token}"},
            )
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        logger.error(f"Drive download failed {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Drive download error: {e}")
        return None


def _download_video_from_azure_blob_sync(blob_video_url: str) -> bytes | None:
    """Download a stored source video from Azure Blob Storage by its blob URL."""
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        logger.error("Azure Blob not configured, cannot download source video")
        return None
    try:
        parsed = urlparse(blob_video_url)
        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) != 2:
            logger.error("Unexpected blob video URL format: %s", blob_video_url[:120])
            return None
        container_name, blob_name = path_parts[0], path_parts[1]
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(
            settings.azure_blob_connection_string,
        )
        blob_client = client.get_container_client(container_name).get_blob_client(blob_name)
        return blob_client.download_blob().readall()
    except Exception as e:
        logger.error("Azure Blob video download failed for %s: %s", blob_video_url[:120], e)
        return None


async def _download_video_bytes(
    row: IndexedFile,
    google_access_token: str | None,
) -> bytes | None:
    """Fetch the source video bytes for a row, dispatching on its ingestion source."""
    if getattr(row, "source_type", "drive") == "instagram" and row.blob_video_url:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(_download_video_from_azure_blob_sync, row.blob_video_url),
        )
    if not google_access_token:
        logger.error("No Google access token for Drive video %s", row.id)
        return None
    return await _download_video_from_drive(row.drive_file_id, google_access_token)


def _get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe. Returns 30.0 as fallback."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return 30.0
        rate = out.stdout.strip().rstrip(",").strip()
        if not rate:
            return 30.0
        if "/" in rate:
            parts = rate.split("/")
            num, den = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "1")
            if not den or float(den) == 0:
                return 30.0
            return float(num) / float(den)
        try:
            return float(rate)
        except ValueError:
            return 30.0
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
        logger.warning(f"ffprobe failed, using 30 fps: {e}")
        return 30.0


def _extract_frames_ffmpeg(video_path: str, out_dir: str, every_n: int = FRAME_INTERVAL) -> list[tuple[int, float, str]]:
    """
    Extract every Nth frame from video, downsampled to FRAME_MAX_DIMENSION max side.
    Returns list of (frame_index, time_seconds, path_to_jpeg).
    """
    vf = (
        f"select=not(mod(n\\,{every_n})),"
        f"scale={FRAME_MAX_DIMENSION}:{FRAME_MAX_DIMENSION}:force_original_aspect_ratio=decrease"
    )
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", vf,
                "-vsync", "vfr",
                "-frame_pts", "1",
                "-q:v", str(FRAME_JPEG_QUALITY),
                pattern,
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"ffmpeg extract failed: {e}")
        return []

    fps = _get_video_fps(video_path)
    results = []
    for i, p in enumerate(sorted(Path(out_dir).glob("frame_*.jpg"))):
        frame_index = i * every_n
        time_seconds = frame_index / fps if fps else 0.0
        results.append((frame_index, time_seconds, str(p)))
    return results


def _extract_poster_ffmpeg(video_path: str, out_path: str) -> bool:
    """Extract a single poster JPEG at t=0, downsampled like frame extracts."""
    vf = (
        f"scale={FRAME_MAX_DIMENSION}:{FRAME_MAX_DIMENSION}"
        ":force_original_aspect_ratio=decrease"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", "0", "-i", video_path,
                "-vframes", "1",
                "-vf", vf,
                "-q:v", str(FRAME_JPEG_QUALITY),
                out_path,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error("ffmpeg poster extract failed: %s", e)
        return False


def thumbnail_blob_path(user_id: UUID, file_id: UUID) -> str:
    """Azure Blob path for a file's poster JPEG in the frames container."""
    return f"{user_id}/{file_id}/thumbnail.jpg"


def upload_thumbnail_jpeg_sync(user_id: UUID, file_id: UUID, image_bytes: bytes) -> str | None:
    """Upload poster JPEG bytes to Azure Blob; return blob URL or None."""
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        logger.warning("Azure Blob not configured, cannot store thumbnail")
        return None
    return _upload_frame_to_azure_blob(
        settings.azure_blob_container_name,
        thumbnail_blob_path(user_id, file_id),
        image_bytes,
    )


def _upload_frame_to_azure_blob(
    container_name: str,
    blob_path: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
) -> str | None:
    """Upload frame image to Azure Blob Storage; return blob URL or None."""
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        logger.warning("Azure Blob not configured, skipping frame upload to Blob")
        return None
    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings

        client = BlobServiceClient.from_connection_string(
            settings.azure_blob_connection_string,
        )
        container_client = client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return blob_client.url
    except Exception as e:
        logger.error(f"Azure Blob upload failed: {e}")
        return None


def _download_frame_from_azure_blob(container_name: str, blob_path: str) -> bytes | None:
    """Download frame bytes from Azure Blob Storage."""
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        return None
    try:
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(
            settings.azure_blob_connection_string,
        )
        blob_client = client.get_container_client(container_name).get_blob_client(blob_path)
        return blob_client.download_blob().readall()
    except Exception as e:
        logger.error(f"Azure Blob download failed for {blob_path}: {e}")
        return None


def download_stored_image_bytes_sync(image_url: str | None) -> bytes | None:
    """Download JPEG bytes from an Azure Blob URL (or any http URL)."""
    if not image_url:
        return None
    if "blob.core.windows.net" in image_url:
        parsed = urlparse(image_url)
        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) == 2:
            data = _download_frame_from_azure_blob(path_parts[0], path_parts[1])
            if data:
                return data
    try:
        import urllib.request

        with urllib.request.urlopen(image_url, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        logger.warning("Failed to download image %s: %s", image_url[:120], e)
        return None


def _parse_blob_connection_string(connection_string: str) -> tuple[str, str]:
    """Extract account name and account key from Azure Storage connection string."""
    account_name, account_key = "", ""
    for part in connection_string.split(";"):
        part = part.strip()
        if part.startswith("AccountName="):
            account_name = part.split("=", 1)[1].strip()
        elif part.startswith("AccountKey="):
            account_key = part.split("=", 1)[1].strip()
    return account_name, account_key


def get_blob_url_with_sas(blob_url: str | None) -> str | None:
    """
    If blob_url is an Azure Blob URL and Azure Blob is configured, return the URL with
    a short-lived read-only SAS token so the frontend can load the image without
    anonymous container access. Otherwise return blob_url unchanged.
    """
    if not blob_url or "blob.core.windows.net" not in blob_url:
        return blob_url
    settings = get_settings()
    if not settings.azure_blob_connection_string:
        return blob_url
    try:
        parsed = urlparse(blob_url)
        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) != 2:
            return blob_url
        container_name, blob_name = path_parts[0], path_parts[1]
        account_name, account_key = _parse_blob_connection_string(
            settings.azure_blob_connection_string
        )
        if not account_name or not account_key:
            return blob_url
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=BLOB_SAS_EXPIRY_HOURS),
        )
        return blob_url + ("&" if "?" in blob_url else "?") + sas_token
    except Exception as e:
        logger.warning("Failed to generate SAS for blob URL %s: %s", blob_url[:80], e)
        return blob_url


def _upload_frame_to_supabase(
    bucket: str,
    path_in_bucket: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
) -> str | None:
    """Upload frame image to Supabase Storage; return public URL or None."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase not configured, skipping frame upload")
        return None
    try:
        from supabase import create_client
        client = create_client(settings.supabase_url, settings.supabase_key)
        client.storage.from_(bucket).upload(
            path_in_bucket,
            image_bytes,
            file_options={"content-type": content_type},
        )
        return client.storage.from_(bucket).get_public_url(path_in_bucket)
    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        return None


async def _load_frame_image_bytes(
    *,
    image_bytes: bytes | None = None,
    local_jpeg_path: str | None = None,
    blob_path: str | None = None,
) -> bytes | None:
    """Resolve frame image bytes from in-memory, local path, or blob storage."""
    if image_bytes is not None:
        return image_bytes
    if local_jpeg_path:
        try:
            with open(local_jpeg_path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.error("Failed to read local frame %s: %s", local_jpeg_path, e)
            return None
    if blob_path:
        settings = get_settings()
        loop = asyncio.get_running_loop()
        if settings.azure_blob_connection_string:
            return await loop.run_in_executor(
                None,
                functools.partial(
                    _download_frame_from_azure_blob,
                    settings.azure_blob_container_name,
                    blob_path,
                ),
            )
        logger.warning("Blob path provided but Azure Blob is not configured")
    return None


async def _frame_embedding_exists(
    session: AsyncSession,
    video_id: UUID,
    frame_index: int,
) -> bool:
    row = (
        await session.execute(
            select(VideoFrameEmbedding.id).where(
                VideoFrameEmbedding.video_id == video_id,
                VideoFrameEmbedding.frame_index == frame_index,
            )
        )
    ).scalar_one_or_none()
    return row is not None


async def _upsert_frame_embedding(
    session: AsyncSession,
    *,
    video_id: UUID,
    frame_index: int,
    time_seconds: float,
    embedding: list[float],
    frame_image_url: str | None,
) -> bool:
    """
    Upsert a frame embedding. Returns True if a new row was inserted (not an update).
    """
    stmt = insert(VideoFrameEmbedding).values(
        id=uuid.uuid4(),
        video_id=video_id,
        frame_index=frame_index,
        time_seconds=time_seconds,
        embedding=embedding,
        frame_image_url=frame_image_url,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_video_frame_video_id_frame_index",
        set_={
            "time_seconds": stmt.excluded.time_seconds,
            "embedding": stmt.excluded.embedding,
            "frame_image_url": stmt.excluded.frame_image_url,
        },
    ).returning(literal_column("(xmax = 0)").label("inserted"))
    result = await session.execute(stmt)
    return bool(result.scalar_one())


async def _record_frame_result(
    session: AsyncSession,
    video_id: UUID,
    *,
    success: bool,
) -> None:
    if success:
        values = {
            "frames_completed": IndexedFile.frames_completed + 1,
            "updated_at": datetime.utcnow(),
        }
    else:
        values = {
            "frames_failed": IndexedFile.frames_failed + 1,
            "updated_at": datetime.utcnow(),
        }
    await session.execute(
        update(IndexedFile).where(IndexedFile.id == video_id).values(**values)
    )


async def _maybe_mark_video_indexing_complete(
    session: AsyncSession,
    video_id: UUID,
) -> None:
    row = (
        await session.execute(select(IndexedFile).where(IndexedFile.id == video_id))
    ).scalar_one_or_none()
    if not row or row.frames_total is None:
        return
    if row.frames_completed + row.frames_failed < row.frames_total:
        return

    now = datetime.utcnow()
    if row.frames_failed >= row.frames_total:
        row.indexing_status = IndexingStatus.FAILED.value
        row.error_message = "All frames failed to index"
    else:
        row.indexing_status = IndexingStatus.COMPLETED.value
        row.error_message = None
        row.indexed_at = now
    row.updated_at = now


async def extract_and_upload_frames(
    video_id: UUID,
    google_access_token: str | None,
    session: AsyncSession,
    work_dir: str,
) -> tuple[list[ExtractedFrame], IndexedFile | None, str | None]:
    """
    Download video from its source (Drive or blob-stored Instagram reel), extract
    frames with ffmpeg, upload JPEGs to storage.

    Sets indexing_status=processing and frames_total on the video row.
    Returns (frames, video_row, error_message).
    """
    stmt = select(IndexedFile).where(
        IndexedFile.id == video_id,
        IndexedFile.file_type == "video",
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if not row:
        return [], None, "Video not found or not a video"

    row.indexing_status = IndexingStatus.PROCESSING.value
    row.frames_total = None
    row.frames_completed = 0
    row.frames_failed = 0
    row.error_message = None
    row.updated_at = datetime.utcnow()
    await session.commit()

    video_bytes = await _download_video_bytes(row, google_access_token)
    if not video_bytes:
        row.indexing_status = IndexingStatus.FAILED.value
        row.error_message = "Failed to download source video"
        row.updated_at = datetime.utcnow()
        await session.commit()
        return [], row, row.error_message

    video_path = os.path.join(work_dir, "video")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    loop = asyncio.get_running_loop()
    poster_path = os.path.join(work_dir, "thumbnail.jpg")
    raw_frames, poster_ok = await asyncio.gather(
        loop.run_in_executor(
            None,
            functools.partial(_extract_frames_ffmpeg, video_path, work_dir, FRAME_INTERVAL),
        ),
        loop.run_in_executor(
            None,
            functools.partial(_extract_poster_ffmpeg, video_path, poster_path),
        ),
    )
    if not raw_frames:
        row.indexing_status = IndexingStatus.FAILED.value
        row.error_message = "No frames extracted (ffmpeg failed or no frames)"
        row.updated_at = datetime.utcnow()
        await session.commit()
        return [], row, row.error_message

    settings = get_settings()
    extracted: list[ExtractedFrame] = []
    for frame_index, time_seconds, jpeg_path in raw_frames:
        with open(jpeg_path, "rb") as f:
            image_bytes = f.read()

        blob_path = f"{row.user_id}/{video_id}/frame_{frame_index:05d}.jpg"
        frame_image_url: str | None = None
        if settings.azure_blob_connection_string:
            frame_image_url = await loop.run_in_executor(
                None,
                functools.partial(
                    _upload_frame_to_azure_blob,
                    settings.azure_blob_container_name,
                    blob_path,
                    image_bytes,
                ),
            )
        elif settings.supabase_url and settings.supabase_key:
            frame_image_url = await loop.run_in_executor(
                None,
                functools.partial(
                    _upload_frame_to_supabase,
                    settings.supabase_storage_bucket,
                    blob_path,
                    image_bytes,
                ),
            )

        extracted.append(
            ExtractedFrame(
                frame_index=frame_index,
                time_seconds=time_seconds,
                blob_path=blob_path,
                frame_image_url=frame_image_url,
                local_jpeg_path=jpeg_path,
            )
        )

    if poster_ok:
        with open(poster_path, "rb") as f:
            poster_bytes = f.read()
        poster_url: str | None = None
        if settings.azure_blob_connection_string:
            poster_url = await loop.run_in_executor(
                None,
                functools.partial(
                    _upload_frame_to_azure_blob,
                    settings.azure_blob_container_name,
                    thumbnail_blob_path(row.user_id, video_id),
                    poster_bytes,
                ),
            )
        if poster_url:
            row.blob_thumbnail_url = poster_url

    row.frames_total = len(extracted)
    row.updated_at = datetime.utcnow()
    await session.commit()

    logger.info(
        "Extracted and uploaded %d frames for video_id=%s",
        len(extracted),
        video_id,
    )
    return extracted, row, None


async def _finalize_frame_index_result(
    session: AsyncSession,
    video_id: UUID,
    *,
    success: bool,
    update_completion: bool,
    frame_index: int | None = None,
) -> None:
    if not update_completion:
        return
    if not success and frame_index is not None:
        if await _frame_embedding_exists(session, video_id, frame_index):
            return
    await _record_frame_result(session, video_id, success=success)
    await _maybe_mark_video_indexing_complete(session, video_id)
    await session.commit()


async def index_single_frame(
    video_id: UUID,
    frame_index: int,
    time_seconds: float,
    session: AsyncSession,
    *,
    image_bytes: bytes | None = None,
    local_jpeg_path: str | None = None,
    blob_path: str | None = None,
    frame_image_url: str | None = None,
    filename: str | None = None,
    trace_ctx: dict | None = None,
    update_completion: bool = True,
) -> bool:
    """
    CLIP-embed one frame and upsert into video_frame_embeddings.

    Returns True on success. When update_completion is True, atomically increments
    frames_completed or frames_failed and may mark the video indexing_status complete.
    """
    frame_meta = {
        "video_id": str(video_id),
        "filename": filename or "",
        "frame_index": frame_index,
        "time_seconds": time_seconds,
    }
    frame_start = time.perf_counter()
    success = False

    with trace_video_frame(metadata=frame_meta, trace_context=trace_ctx):
        try:
            resolved_bytes = await _load_frame_image_bytes(
                image_bytes=image_bytes,
                local_jpeg_path=local_jpeg_path,
                blob_path=blob_path,
            )
            if not resolved_bytes:
                await session.rollback()
                await _finalize_frame_index_result(
                    session,
                    video_id,
                    success=False,
                    update_completion=update_completion,
                    frame_index=frame_index,
                )
                return False

            vision = get_vision_embedding_service()
            embedding = await vision.generate_embedding_from_image_bytes(resolved_bytes)
            if not embedding:
                await session.rollback()
                await _finalize_frame_index_result(
                    session,
                    video_id,
                    success=False,
                    update_completion=update_completion,
                    frame_index=frame_index,
                )
                return False

            inserted = await _upsert_frame_embedding(
                session,
                video_id=video_id,
                frame_index=frame_index,
                time_seconds=time_seconds,
                embedding=embedding,
                frame_image_url=frame_image_url,
            )

            if update_completion and inserted:
                await _record_frame_result(session, video_id, success=True)
                await _maybe_mark_video_indexing_complete(session, video_id)

            await session.commit()
            success = True
            logger.info(
                "Indexed frame video_id=%s frame_index=%s embedding_len=%s",
                video_id,
                frame_index,
                len(embedding),
            )
        except Exception as e:
            logger.exception(
                "Frame indexing failed for video_id=%s frame_index=%s: %s",
                video_id,
                frame_index,
                e,
            )
            await session.rollback()
            try:
                await _finalize_frame_index_result(
                    session,
                    video_id,
                    success=False,
                    update_completion=update_completion,
                    frame_index=frame_index,
                )
            except Exception:
                await session.rollback()
            return False
        finally:
            frame_duration_ms = (time.perf_counter() - frame_start) * 1000
            emit_video_frame_trace(metadata=frame_meta, duration_ms=frame_duration_ms)

    return success


async def index_image_thumbnail(
    file_id: UUID,
    google_access_token: str,
    session: AsyncSession,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict:
    """
    Download thumbnail, generate CLIP embedding, and update indexed_files for an image.

    Intended for image-index queue workers; callable directly for in-process use.
    """
    result = {"success": False, "error": None}
    stmt = select(IndexedFile).where(
        IndexedFile.id == file_id,
        IndexedFile.file_type == "image",
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if not row:
        result["error"] = "Image not found or not an image"
        return result

    file_meta = {
        "file_id": str(file_id),
        "filename": row.filename,
    }
    if trace_id:
        file_meta["trace_id"] = trace_id
    if parent_span_id:
        file_meta["parent_span_id"] = parent_span_id

    with trace_index_file("image", metadata=file_meta):
        if not row.thumbnail_url:
            result["error"] = "No thumbnail URL on file record"
            row.vision_indexing_status = IndexingStatus.FAILED.value
            row.error_message = result["error"]
            row.updated_at = datetime.utcnow()
            await session.commit()
            return result

        vision = get_vision_embedding_service()

        row.vision_indexing_status = IndexingStatus.PROCESSING.value
        row.updated_at = datetime.utcnow()
        await session.commit()

        actual_url = row.thumbnail_url
        if row.drive_file_id and google_access_token:
            fresh_url = await vision._get_fresh_thumbnail_url(
                row.drive_file_id, google_access_token
            )
            if fresh_url:
                actual_url = fresh_url

        image_bytes = await vision._download_image(actual_url, google_access_token)
        if not image_bytes:
            row.vision_indexing_status = IndexingStatus.FAILED.value
            row.error_message = "Failed to download thumbnail"
            row.updated_at = datetime.utcnow()
            await session.commit()
            result["error"] = row.error_message
            return result

        sig = signature_from_image_bytes(image_bytes)
        if sig:
            apply_color_signature(row, sig)

        if not vision.is_configured:
            now = datetime.utcnow()
            row.vision_indexing_status = None
            row.updated_at = now
            await session.commit()
            result["success"] = True
            return result

        embedding = await vision.generate_embedding_from_image_bytes(image_bytes)
        if not embedding:
            row.vision_indexing_status = IndexingStatus.FAILED.value
            row.error_message = "Failed to generate vision embedding"
            row.updated_at = datetime.utcnow()
            await session.commit()
            result["error"] = row.error_message
            return result

        now = datetime.utcnow()
        row.vision_embedding = embedding
        row.vision_indexing_status = IndexingStatus.COMPLETED.value
        row.vision_indexed_at = now
        row.error_message = None
        row.updated_at = now
        await session.commit()
        result["success"] = True
        return result


async def _embed_thumbnail_with_own_session(
    video_id: UUID,
    local_jpeg_path: str,
) -> bool:
    """Generate thumbnail_embedding from a local poster JPEG in a dedicated session."""
    if not os.path.isfile(local_jpeg_path):
        return False
    vision = get_vision_embedding_service()
    if not vision.is_configured:
        return False
    try:
        with open(local_jpeg_path, "rb") as f:
            image_bytes = f.read()
    except OSError as e:
        logger.error("Failed to read poster %s: %s", local_jpeg_path, e)
        return False
    embedding = await vision.generate_embedding_from_image_bytes(image_bytes)
    if not embedding:
        logger.warning("Thumbnail embedding failed for video_id=%s", video_id)
        return False
    async with async_session_maker() as session:
        row = (
            await session.execute(select(IndexedFile).where(IndexedFile.id == video_id))
        ).scalar_one_or_none()
        if not row:
            return False
        row.thumbnail_embedding = embedding
        row.updated_at = datetime.utcnow()
        await session.commit()
    return True


def _signatures_from_jpeg_paths(jpeg_paths: list[str]):
    sigs = []
    for path in jpeg_paths:
        if not path or not os.path.isfile(path):
            continue
        sig = signature_from_jpeg_path(path)
        if sig:
            sigs.append(sig)
    return merge_signatures(sigs)


async def _write_color_signature_with_own_session(
    video_id: UUID,
    jpeg_paths: list[str],
) -> bool:
    """Aggregate Lab signatures from local frame JPEGs onto indexed_files."""
    loop = asyncio.get_running_loop()
    merged = await loop.run_in_executor(
        None,
        functools.partial(_signatures_from_jpeg_paths, jpeg_paths),
    )
    if not merged:
        logger.warning("Color signature extract produced nothing for video_id=%s", video_id)
        return False
    async with async_session_maker() as session:
        row = (
            await session.execute(select(IndexedFile).where(IndexedFile.id == video_id))
        ).scalar_one_or_none()
        if not row:
            return False
        apply_color_signature(row, merged)
        row.updated_at = datetime.utcnow()
        await session.commit()
    return True


async def _index_frame_with_own_session(
    video_id: UUID,
    frame: ExtractedFrame,
    *,
    filename: str,
    trace_ctx: dict | None,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        async with async_session_maker() as frame_session:
            return await index_single_frame(
                video_id,
                frame.frame_index,
                frame.time_seconds,
                frame_session,
                local_jpeg_path=frame.local_jpeg_path,
                blob_path=frame.blob_path,
                frame_image_url=frame.frame_image_url,
                filename=filename,
                trace_ctx=trace_ctx,
            )


async def run_frame_indexing_for_video(
    video_id: UUID,
    google_access_token: str | None,
    session: AsyncSession,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict:
    """
    For one video: extract frames, then CLIP+DB each frame in parallel (bounded).

    Also transcribes the video's audio (Whisper) while frames are being embedded.

    Returns stats: { "frames_processed": int, "frames_failed": int,
    "transcript_segments": int, "transcript_error": str | None, "error": str | None }.
    """
    result = {
        "frames_processed": 0,
        "frames_failed": 0,
        "transcript_segments": 0,
        "transcript_error": None,
        "error": None,
    }
    logger.info("Video frame indexing starting for video_id=%s", video_id)
    _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "Entry", {"video_id": str(video_id)}, "H5")

    trace_ctx = None
    if trace_id and parent_span_id:
        trace_ctx = {"trace_id": trace_id, "parent_span_id": parent_span_id}

    settings = get_settings()
    parallelism = max(1, settings.frame_index_parallelism)
    semaphore = asyncio.Semaphore(parallelism)

    with tempfile.TemporaryDirectory() as tmpdir:
        frames, row, extract_error = await extract_and_upload_frames(
            video_id=video_id,
            google_access_token=google_access_token,
            session=session,
            work_dir=tmpdir,
        )
        if extract_error or not row:
            result["error"] = extract_error or "Video extraction failed"
            _debug_log(
                "video_frame_indexing.py:run_frame_indexing_for_video",
                "Extract failed",
                {"video_id": str(video_id), "error": result["error"]},
                "H5",
            )
            return result

        logger.info(
            "Indexing %d frames for video_id=%s with parallelism=%d",
            len(frames),
            video_id,
            parallelism,
        )

        # Transcribe audio concurrently with frame embedding (video file lives in tmpdir)
        transcription_task = asyncio.create_task(
            transcribe_video_with_own_session(
                video_id,
                os.path.join(tmpdir, "video"),
                filename=row.filename,
            )
        )
        thumbnail_task = asyncio.create_task(
            _embed_thumbnail_with_own_session(
                video_id,
                os.path.join(tmpdir, "thumbnail.jpg"),
            )
        )
        jpeg_paths = [frame.local_jpeg_path for frame in frames]
        poster_path = os.path.join(tmpdir, "thumbnail.jpg")
        if os.path.isfile(poster_path):
            jpeg_paths.append(poster_path)
        color_task = asyncio.create_task(
            _write_color_signature_with_own_session(video_id, jpeg_paths)
        )

        outcomes = await asyncio.gather(
            *[
                _index_frame_with_own_session(
                    video_id,
                    frame,
                    filename=row.filename,
                    trace_ctx=trace_ctx,
                    semaphore=semaphore,
                )
                for frame in frames
            ],
            return_exceptions=True,
        )

        try:
            transcript_result = await transcription_task
            result["transcript_segments"] = transcript_result.get("segments", 0)
            result["transcript_error"] = transcript_result.get("error")
        except Exception as e:
            logger.exception("Transcription task failed for video_id=%s: %s", video_id, e)
            result["transcript_error"] = str(e)

        try:
            await thumbnail_task
        except Exception as e:
            logger.exception("Thumbnail embedding failed for video_id=%s: %s", video_id, e)

        try:
            await color_task
        except Exception as e:
            logger.exception("Color signature failed for video_id=%s: %s", video_id, e)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                result["frames_failed"] += 1
                logger.exception("Unexpected frame task error: %s", outcome)
            elif outcome:
                result["frames_processed"] += 1
            else:
                result["frames_failed"] += 1

    flush_langfuse()
    _debug_log(
        "video_frame_indexing.py:run_frame_indexing_for_video",
        "Completed",
        {
            "video_id": str(video_id),
            "frames_processed": result["frames_processed"],
            "error": result["error"],
        },
        "H5",
    )
    return result
