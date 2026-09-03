"""Celery tasks for background processing."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.celery_app import celery_app
from app.services.google_auth import get_valid_access_token_via_postgrest
from app.services.indexing_store import get_file, get_user
from app.services.postgrest import postgrest_session
from app.services.video_frame_indexing import (
    index_image_thumbnail,
    run_frame_indexing_for_video,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(value: str | None) -> datetime | None:
    """PostgREST returns timestamps as ISO strings; token expiry compares datetimes."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not parse timestamp %r", value)
        return None
    # google_token_expires_at is a naive UTC column; normalise so comparisons work.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


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


async def _run_frame_indexing_async(
    video_id: str,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict:
    """Load video and user, get token, run frame indexing. Runs in async context."""
    # #region agent log
    _debug_log("tasks.py:_run_frame_indexing_async", "Task started", {"video_id": video_id}, "H4")
    # #endregion
    video_uuid = UUID(video_id)
    # One pooled PostgREST connection for the whole job: the frame, thumbnail,
    # colour and transcript coroutines below all share this client.
    async with postgrest_session() as db:
        indexed_file = await get_file(db, video_uuid, file_type="video")
        if not indexed_file:
            # #region agent log
            _debug_log("tasks.py:_run_frame_indexing_async", "Video not found or not video", {"video_id": video_id, "found": False, "file_type": None}, "H5")
            # #endregion
            return {"error": "Video not found or not a video"}

        user = await get_user(db, UUID(str(indexed_file["user_id"])))
        if not user:
            return {"error": "User not found"}

        # Instagram-sourced videos are downloaded from Azure Blob and need no Google token
        access_token = None
        if (indexed_file.get("source_type") or "drive") == "drive":
            access_token = await get_valid_access_token_via_postgrest(
                db,
                user_id=UUID(str(user["id"])),
                access_token=user.get("google_access_token"),
                refresh_token=user.get("google_refresh_token"),
                expires_at=_parse_timestamp(user.get("google_token_expires_at")),
            )
            if not access_token:
                # #region agent log
                _debug_log("tasks.py:_run_frame_indexing_async", "No access token", {"video_id": video_id}, "H5")
                # #endregion
                return {"error": "Could not get valid Google access token"}

        out = await run_frame_indexing_for_video(
            video_id=video_uuid,
            google_access_token=access_token,
            db=db,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        # #region agent log
        _debug_log("tasks.py:_run_frame_indexing_async", "Task completed", {"video_id": video_id, "result_keys": list(out.keys()) if isinstance(out, dict) else None, "error": out.get("error") if isinstance(out, dict) else None, "frames_processed": out.get("frames_processed") if isinstance(out, dict) else None}, "H5")
        # #endregion
        return out


async def _run_image_indexing_async(
    file_id: str,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict:
    """Load image and user, get token, run thumbnail vision indexing."""
    file_uuid = UUID(file_id)
    async with postgrest_session() as db:
        indexed_file = await get_file(db, file_uuid, file_type="image")
        if not indexed_file:
            return {"error": "Image not found or not an image"}

        user = await get_user(db, UUID(str(indexed_file["user_id"])))
        if not user:
            return {"error": "User not found"}

        access_token = await get_valid_access_token_via_postgrest(
            db,
            user_id=UUID(str(user["id"])),
            access_token=user.get("google_access_token"),
            refresh_token=user.get("google_refresh_token"),
            expires_at=_parse_timestamp(user.get("google_token_expires_at")),
        )
        if not access_token:
            return {"error": "Could not get valid Google access token"}

        return await index_image_thumbnail(
            file_id=file_uuid,
            google_access_token=access_token,
            db=db,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )


@celery_app.task(bind=True, name="app.tasks.index_video_frames")
def index_video_frames_task(self, video_id: str) -> dict:
    """
    Celery task: run video frame extraction, upload to Supabase, and store embeddings.
    video_id: UUID string of the indexed_files row (must be file_type=video).
    """
    try:
        return asyncio.run(_run_frame_indexing_async(video_id))
    except Exception as e:
        logger.exception("Video frame indexing task failed: %s", e)
        return {"error": str(e), "frames_processed": 0, "frames_failed": 0}
