"""Celery tasks for background processing."""

import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.celery_app import celery_app
from app.database import async_session_maker
from app.models.indexed_file import IndexedFile
from app.models.user import User
from app.services.google_auth import get_valid_access_token
from app.services.video_frame_indexing import run_frame_indexing_for_video

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
    async with async_session_maker() as session:
        stmt = select(IndexedFile).where(IndexedFile.id == video_uuid)
        indexed_file = (await session.execute(stmt)).scalar_one_or_none()
        if not indexed_file or indexed_file.file_type != "video":
            # #region agent log
            _debug_log("tasks.py:_run_frame_indexing_async", "Video not found or not video", {"video_id": video_id, "found": indexed_file is not None, "file_type": getattr(indexed_file, "file_type", None)}, "H5")
            # #endregion
            return {"error": "Video not found or not a video"}

        user_stmt = select(User).where(User.id == indexed_file.user_id)
        user = (await session.execute(user_stmt)).unique().scalar_one_or_none()
        if not user:
            return {"error": "User not found"}

        access_token = await get_valid_access_token(
            user_id=user.id,
            access_token=user.google_access_token,
            refresh_token=user.google_refresh_token,
            expires_at=user.google_token_expires_at,
            session=session,
        )
        if not access_token:
            # #region agent log
            _debug_log("tasks.py:_run_frame_indexing_async", "No access token", {"video_id": video_id}, "H5")
            # #endregion
            return {"error": "Could not get valid Google access token"}

        out = await run_frame_indexing_for_video(
            video_id=video_uuid,
            google_access_token=access_token,
            session=session,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        # #region agent log
        _debug_log("tasks.py:_run_frame_indexing_async", "Task completed", {"video_id": video_id, "result_keys": list(out.keys()) if isinstance(out, dict) else None, "error": out.get("error") if isinstance(out, dict) else None, "frames_processed": out.get("frames_processed") if isinstance(out, dict) else None}, "H5")
        # #endregion
        return out


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
