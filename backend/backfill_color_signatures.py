"""
Backfill Lab color signatures for indexed files.

Selects rows missing color_histogram. Videos reuse already-stored frame JPEGs
in blob storage (no Gemini re-embed). Images/posters fall back to
blob_thumbnail_url or thumbnail_url.

Usage:
    cd backend
    python backfill_color_signatures.py
    python backfill_color_signatures.py --limit 20

Requires DATABASE_URL. Azure Blob (or reachable thumbnail URLs) is needed to
read the stored JPEGs.
"""

import argparse
import asyncio
import functools
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select

from app.database import async_session_maker
from app.models.indexed_file import IndexedFile
from app.models.video_frame_embedding import VideoFrameEmbedding
from app.services.color_signature import (
    apply_color_signature,
    merge_signatures,
    signature_from_image_bytes,
)
from app.services.video_frame_indexing import download_stored_image_bytes_sync

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MAX_FRAMES = 32


def _signature_from_urls(urls: list[str]):
    sigs = []
    for url in urls:
        image_bytes = download_stored_image_bytes_sync(url)
        if not image_bytes:
            continue
        sig = signature_from_image_bytes(image_bytes)
        if sig:
            sigs.append(sig)
    return merge_signatures(sigs)


async def _source_urls(session, row: IndexedFile) -> list[str]:
    urls: list[str] = []
    if row.file_type == "video":
        frames = (
            await session.execute(
                select(VideoFrameEmbedding.frame_image_url)
                .where(
                    VideoFrameEmbedding.video_id == row.id,
                    VideoFrameEmbedding.frame_image_url.isnot(None),
                )
                .order_by(VideoFrameEmbedding.frame_index)
                .limit(MAX_FRAMES)
            )
        ).scalars().all()
        urls.extend(u for u in frames if u)
    if row.blob_thumbnail_url:
        urls.append(row.blob_thumbnail_url)
    if row.thumbnail_url and row.thumbnail_url not in urls:
        urls.append(row.thumbnail_url)
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


async def run(limit: int | None) -> int:
    async with async_session_maker() as session:
        stmt = (
            select(IndexedFile)
            .where(IndexedFile.color_histogram.is_(None))
            .order_by(IndexedFile.created_at)
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = list((await session.execute(stmt)).scalars().all())
        logger.info("Backfilling color signatures for %d row(s)", len(rows))

        written = 0
        skipped = 0
        loop = asyncio.get_running_loop()
        for row in rows:
            urls = await _source_urls(session, row)
            if not urls:
                skipped += 1
                logger.info("  %s skipped (no stored JPEG URLs)", row.id)
                continue
            sig = await loop.run_in_executor(
                None, functools.partial(_signature_from_urls, urls)
            )
            if not sig:
                skipped += 1
                logger.info("  %s skipped (extract failed)", row.id)
                continue
            apply_color_signature(row, sig)
            row.updated_at = datetime.utcnow()
            await session.commit()
            written += 1
            logger.info("  %s wrote signature (%d source JPEG(s))", row.id, len(urls))

        print(f"Done. wrote={written} skipped={skipped} of {len(rows)} candidates")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    args = parser.parse_args()
    return asyncio.run(run(args.limit))


if __name__ == "__main__":
    sys.exit(main())
