"""Video frame extraction, embedding, and storage for per-frame video search."""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.indexed_file import IndexedFile
from app.models.video_frame_embedding import VideoFrameEmbedding
from app.observability import emit_video_frame_trace, flush_langfuse, trace_video_frame
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
# Commit DB every N frames so the connection isn't idle too long (avoids connection closed by server)
COMMIT_BATCH_SIZE = 10


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
        # r_frame_rate is like "30/1", "30000/1001", or sometimes "1," (trailing comma)
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
    Extract every Nth frame from video. Returns list of (frame_index, time_seconds, path_to_jpeg).
    """
    # select=not(mod(n\,5)) -> frames 0, 5, 10, ...
    vf = f"select=not(mod(n\\,{every_n}))"
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", vf,
                "-vsync", "vfr",
                "-frame_pts", "1",
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


async def run_frame_indexing_for_video(
    video_id: UUID,
    google_access_token: str,
    session: AsyncSession,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict:
    """
    For one video: download from Drive, extract every 5th frame, upload to Azure Blob
    (or Supabase if Blob not configured), generate CLIP embeddings, insert into
    video_frame_embeddings. Returns stats: { "frames_processed": int, "frames_failed": int,
    "error": str | None }.
    trace_id, parent_span_id: optional Langfuse trace context to attach frame spans to the batch trace.
    """
    result = {"frames_processed": 0, "frames_failed": 0, "error": None}
    logger.info("Video frame indexing starting for video_id=%s", video_id)
    # #region agent log
    _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "Entry", {"video_id": str(video_id)}, "H5")
    # #endregion

    # Load video row
    stmt = select(IndexedFile).where(
        IndexedFile.id == video_id,
        IndexedFile.file_type == "video",
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if not row:
        result["error"] = "Video not found or not a video"
        # #region agent log
        _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "Video row not found", {"video_id": str(video_id)}, "H5")
        # #endregion
        logger.warning("Video not found or not a video: video_id=%s", video_id)
        return result

    drive_file_id = row.drive_file_id
    video_bytes = await _download_video_from_drive(drive_file_id, google_access_token)
    if not video_bytes:
        result["error"] = "Failed to download video from Drive"
        # #region agent log
        _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "Drive download failed", {"video_id": str(video_id)}, "H5")
        # #endregion
        logger.warning("Failed to download video from Drive: video_id=%s", video_id)
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        frames = _extract_frames_ffmpeg(video_path, tmpdir, FRAME_INTERVAL)
        if not frames:
            result["error"] = "No frames extracted (ffmpeg failed or no frames)"
            # #region agent log
            _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "No frames extracted", {"video_id": str(video_id)}, "H5")
            # #endregion
            logger.warning("No frames extracted for video_id=%s", video_id)
            return result

        logger.info(
            "Extracted %d frames for video_id=%s, preparing to insert into video_frame_embeddings",
            len(frames),
            video_id,
        )

        settings = get_settings()
        vision = get_vision_embedding_service()

        trace_ctx = None
        if trace_id and parent_span_id:
            trace_ctx = {"trace_id": trace_id, "parent_span_id": parent_span_id}

        for frame_index, time_seconds, jpeg_path in frames:
            frame_meta = {
                "video_id": str(video_id),
                "filename": row.filename,
                "frame_index": frame_index,
                "time_seconds": time_seconds,
            }
            frame_start = time.perf_counter()
            with trace_video_frame(metadata=frame_meta, trace_context=trace_ctx):
                with open(jpeg_path, "rb") as f:
                    image_bytes = f.read()

                path_in_bucket = f"{row.user_id}/{video_id}/frame_{frame_index:05d}.jpg"
                if settings.azure_blob_connection_string:
                    frame_image_url = _upload_frame_to_azure_blob(
                        settings.azure_blob_container_name,
                        path_in_bucket,
                        image_bytes,
                    )
                elif settings.supabase_url and settings.supabase_key:
                    frame_image_url = _upload_frame_to_supabase(
                        settings.supabase_storage_bucket,
                        path_in_bucket,
                        image_bytes,
                    )
                else:
                    frame_image_url = None

                embedding = await vision.generate_embedding_from_image_bytes(image_bytes)
                if not embedding:
                    result["frames_failed"] += 1
                    continue

                rec = VideoFrameEmbedding(
                    video_id=video_id,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                    embedding=embedding,
                    frame_image_url=frame_image_url,
                )
                session.add(rec)
                result["frames_processed"] += 1
                logger.info(
                    "video_frame_embeddings row to insert: video_id=%s frame_index=%s time_seconds=%s embedding_len=%s frame_image_url=%s",
                    video_id,
                    frame_index,
                    time_seconds,
                    len(embedding) if embedding else 0,
                    "set" if frame_image_url else "null",
                )

                if result["frames_processed"] % COMMIT_BATCH_SIZE == 0:
                    try:
                        await session.commit()
                        logger.info(
                            "Committed batch of %d frames for video_id=%s",
                            COMMIT_BATCH_SIZE,
                            video_id,
                        )
                    except Exception as e:
                        logger.exception(
                            "Batch commit failed for video_id=%s: %s",
                            video_id,
                            e,
                        )
                        result["error"] = str(e)
                        raise

            frame_duration_ms = (time.perf_counter() - frame_start) * 1000
            emit_video_frame_trace(metadata=frame_meta, duration_ms=frame_duration_ms)

    # Commit any remaining rows (last partial batch)
    if result["frames_processed"] > 0 and result["frames_processed"] % COMMIT_BATCH_SIZE != 0:
        try:
            await session.commit()
            logger.info(
                "Committed video_frame_embeddings for video_id=%s (%d rows total)",
                video_id,
                result["frames_processed"],
            )
        except Exception as e:
            logger.exception(
                "Final commit failed for video_id=%s: %s. Pending rows: %d",
                video_id,
                e,
                result["frames_processed"],
            )
            result["error"] = str(e)
            raise
    flush_langfuse()
    # #region agent log
    _debug_log("video_frame_indexing.py:run_frame_indexing_for_video", "Completed", {"video_id": str(video_id), "frames_processed": result["frames_processed"], "error": result["error"]}, "H5")
    # #endregion
    return result
