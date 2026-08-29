"""
Backfill poster thumbnails and caption embeddings for Instagram reels.

Selects source_type='instagram' rows that already have a stored source video
but are missing blob_thumbnail_url, thumbnail_embedding, and/or
description_embedding. Downloads the video from Azure Blob, extracts a poster
JPEG with ffmpeg, uploads it, and generates Gemini embeddings.

Usage:
    cd backend
    python backfill_thumbnails.py
    python backfill_thumbnails.py --limit 20

Requires DATABASE_URL, AZURE_BLOB_CONNECTION_STRING, and GEMINI_API_KEY
(embeddings are skipped when Gemini is not configured).
"""

import argparse
import asyncio
import functools
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import or_, select

from app.database import async_session_maker
from app.models.indexed_file import IndexedFile
from app.services.video_frame_indexing import (
    _download_video_from_azure_blob_sync,
    _extract_poster_ffmpeg,
    upload_thumbnail_jpeg_sync,
)
from app.services.vision_embedding import get_vision_embedding_service

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def _backfill_row(row: IndexedFile, vision) -> dict:
    stats = {"thumbnail": False, "thumbnail_embedding": False, "caption_embedding": False}
    loop = asyncio.get_running_loop()

    needs_poster = not row.blob_thumbnail_url or row.thumbnail_embedding is None
    if needs_poster and row.blob_video_url:
        video_bytes = await loop.run_in_executor(
            None,
            functools.partial(_download_video_from_azure_blob_sync, row.blob_video_url),
        )
        if not video_bytes:
            logger.warning("Could not download video for %s", row.id)
        else:
            with tempfile.TemporaryDirectory(prefix="thumb_bf_") as tmpdir:
                video_path = os.path.join(tmpdir, "video")
                poster_path = os.path.join(tmpdir, "thumbnail.jpg")
                with open(video_path, "wb") as f:
                    f.write(video_bytes)
                poster_ok = await loop.run_in_executor(
                    None,
                    functools.partial(_extract_poster_ffmpeg, video_path, poster_path),
                )
                if poster_ok:
                    with open(poster_path, "rb") as f:
                        poster_bytes = f.read()
                    if not row.blob_thumbnail_url:
                        url = await loop.run_in_executor(
                            None,
                            functools.partial(
                                upload_thumbnail_jpeg_sync,
                                row.user_id,
                                row.id,
                                poster_bytes,
                            ),
                        )
                        if url:
                            row.blob_thumbnail_url = url
                            stats["thumbnail"] = True
                    if row.thumbnail_embedding is None and vision.is_configured:
                        embedding = await vision.generate_embedding_from_image_bytes(
                            poster_bytes
                        )
                        if embedding:
                            row.thumbnail_embedding = embedding
                            stats["thumbnail_embedding"] = True

    if (
        row.description
        and row.description_embedding is None
        and vision.is_configured
    ):
        desc_embedding = await vision.generate_document_text_embedding(row.description)
        if desc_embedding:
            row.description_embedding = desc_embedding
            stats["caption_embedding"] = True

    if any(stats.values()):
        row.updated_at = datetime.utcnow()
    return stats


async def run(limit: int | None) -> int:
    vision = get_vision_embedding_service()
    async with async_session_maker() as session:
        stmt = (
            select(IndexedFile)
            .where(
                IndexedFile.source_type == "instagram",
                IndexedFile.blob_video_url.isnot(None),
                or_(
                    IndexedFile.blob_thumbnail_url.is_(None),
                    IndexedFile.thumbnail_embedding.is_(None),
                    IndexedFile.description_embedding.is_(None),
                ),
            )
            .order_by(IndexedFile.created_at)
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = list((await session.execute(stmt)).scalars().all())
        logger.info("Backfilling %d Instagram row(s)", len(rows))

        totals = {"thumbnail": 0, "thumbnail_embedding": 0, "caption_embedding": 0}
        for row in rows:
            stats = await _backfill_row(row, vision)
            for key, ok in stats.items():
                if ok:
                    totals[key] += 1
            await session.commit()
            logger.info(
                "  %s thumbnail=%s thumb_emb=%s caption_emb=%s",
                row.id,
                stats["thumbnail"],
                stats["thumbnail_embedding"],
                stats["caption_embedding"],
            )

        print(
            f"Done. uploaded={totals['thumbnail']} "
            f"thumbnail_embeddings={totals['thumbnail_embedding']} "
            f"caption_embeddings={totals['caption_embedding']} "
            f"of {len(rows)} candidates"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    args = parser.parse_args()
    return asyncio.run(run(args.limit))


if __name__ == "__main__":
    sys.exit(main())
