"""Indexing reads and writes expressed as PostgREST calls.

The worker path (Service Bus triggers and the Celery task) used the ORM; it now
goes through here. Rows come back as plain dicts rather than `IndexedFile` /
`User` instances, so callers index them by column name.

Two rules shape the split between PATCH and RPC:

* Anything touching a `vector` column (`embedding`, `thumbnail_embedding`,
  `vision_embedding`, `color_histogram`, `text_embedding`) goes through an RPC
  that takes the value as text and casts it, because PostgREST would otherwise
  send a JSON array to a type it cannot infer.
* Anything that must be atomic goes through an RPC. `frames_completed + 1`
  followed by a "are we done yet" check is a read-modify-write, and frames are
  indexed concurrently, so it has to happen inside one Postgres statement.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from app.services.postgrest import PostgrestClient, to_vector

logger = logging.getLogger(__name__)

INDEXED_FILES = "indexed_files"
VIDEO_FRAME_EMBEDDINGS = "video_frame_embeddings"
VIDEO_TRANSCRIPT_SEGMENTS = "video_transcript_segments"
# fastapi-users' SQLAlchemyBaseUserTableUUID names the table "user".
USERS = "user"

# Columns the worker actually reads off indexed_files. Selecting explicitly keeps
# the 768-dim embedding columns out of every response.
FILE_COLUMNS = (
    "id,user_id,filename,file_type,source_type,source_url,drive_file_id,"
    "blob_video_url,blob_thumbnail_url,thumbnail_url,drive_url,"
    "indexing_status,vision_indexing_status,transcript_status,error_message,"
    "frames_total,frames_completed,frames_failed,indexed_at,updated_at"
)

USER_COLUMNS = "id,email,google_access_token,google_refresh_token,google_token_expires_at"


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


async def get_file(
    db: PostgrestClient,
    file_id: UUID,
    *,
    file_type: str | None = None,
) -> dict[str, Any] | None:
    """Load one indexed_files row, optionally asserting its file_type."""
    match: dict[str, Any] = {"id": file_id}
    if file_type:
        match["file_type"] = file_type
    return await db.select_one(INDEXED_FILES, match=match, columns=FILE_COLUMNS)


async def get_user(db: PostgrestClient, user_id: UUID) -> dict[str, Any] | None:
    return await db.select_one(USERS, match={"id": user_id}, columns=USER_COLUMNS)


async def update_file(
    db: PostgrestClient,
    file_id: UUID,
    **values: Any,
) -> None:
    """PATCH plain (non-vector) columns on indexed_files, stamping updated_at."""
    values.setdefault("updated_at", _utcnow())
    await db.update(INDEXED_FILES, match={"id": file_id}, values=values)


async def update_user_google_token(
    db: PostgrestClient,
    user_id: UUID,
    *,
    access_token: str,
    expires_at: datetime,
) -> None:
    await db.update(
        USERS,
        match={"id": user_id},
        values={
            "google_access_token": access_token,
            "google_token_expires_at": expires_at.isoformat(),
        },
    )


async def upsert_frame_embedding(
    db: PostgrestClient,
    *,
    video_id: UUID,
    frame_index: int,
    time_seconds: float,
    embedding: Sequence[float],
    frame_image_url: str | None,
    count_completion: bool,
) -> bool:
    """Upsert one frame and, in the same transaction, count it toward completion.

    Returns True when the row was newly inserted (a re-index that overwrites an
    existing frame must not double-count the progress counters).
    """
    payload = await db.rpc(
        "distill_upsert_frame_embedding",
        {
            "p_video_id": str(video_id),
            "p_frame_index": frame_index,
            "p_time_seconds": float(time_seconds),
            "p_embedding": to_vector(embedding),
            "p_frame_image_url": frame_image_url,
            "p_count_completion": count_completion,
        },
    )
    if isinstance(payload, dict):
        return bool(payload.get("inserted"))
    return False


async def record_frame_failure(
    db: PostgrestClient,
    *,
    video_id: UUID,
    frame_index: int | None,
) -> None:
    """Count a failed frame, unless that frame already has an embedding stored."""
    await db.rpc(
        "distill_record_frame_failure",
        {
            "p_video_id": str(video_id),
            "p_frame_index": frame_index,
        },
    )


async def set_thumbnail_embedding(
    db: PostgrestClient,
    *,
    video_id: UUID,
    embedding: Sequence[float],
) -> None:
    await db.rpc(
        "distill_set_thumbnail_embedding",
        {
            "p_file_id": str(video_id),
            "p_embedding": to_vector(embedding),
        },
    )


async def set_color_signature(
    db: PostgrestClient,
    *,
    file_id: UUID,
    histogram: Sequence[float],
    palette: Any,
    mean_l: float,
    std_l: float,
    mean_a: float,
    mean_b: float,
) -> None:
    await db.rpc(
        "distill_set_color_signature",
        {
            "p_file_id": str(file_id),
            "p_histogram": to_vector([float(x) for x in histogram]),
            "p_palette": palette,
            "p_mean_l": float(mean_l),
            "p_std_l": float(std_l),
            "p_mean_a": float(mean_a),
            "p_mean_b": float(mean_b),
        },
    )


async def set_vision_embedding(
    db: PostgrestClient,
    *,
    file_id: UUID,
    embedding: Sequence[float],
    status: str | None,
    indexed_at: datetime,
) -> None:
    await db.rpc(
        "distill_set_vision_embedding",
        {
            "p_file_id": str(file_id),
            "p_embedding": to_vector(embedding),
            "p_status": status,
            "p_indexed_at": indexed_at.isoformat(),
        },
    )


async def replace_transcript_segments(
    db: PostgrestClient,
    *,
    video_id: UUID,
    segments: list[dict[str, Any]],
) -> None:
    """Swap a video's transcript segments atomically.

    Delete-then-insert has to be one transaction; a re-index that failed halfway
    through the insert would otherwise leave the video with no transcript at all.
    """
    await db.rpc(
        "distill_replace_transcript_segments",
        {
            "p_video_id": str(video_id),
            "p_segments": segments,
        },
    )


def transcript_segment_payload(
    *,
    segment_index: int,
    start_seconds: float,
    end_seconds: float,
    text: str,
    words: Any,
    text_embedding: Sequence[float] | None,
) -> dict[str, Any]:
    """Shape one segment for distill_replace_transcript_segments."""
    return {
        "segment_index": segment_index,
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "text": text,
        "words": words,
        "text_embedding": to_vector(text_embedding),
    }
