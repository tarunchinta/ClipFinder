"""FastMCP server with ClipFinder tools: reel ingest, hybrid search, transcripts."""

import logging
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from app.database import async_session_maker
from app.mcp_server.bound_user import get_bound_user
from app.services.indexing import IndexingService
from app.services.instagram_ingest import (
    InstagramIngestError,
    ingest_instagram_reel as _ingest_reel,
)
from app.services.video_frame_indexing import get_blob_url_with_sas

logger = logging.getLogger(__name__)

# streamable_http_path="/" because main.py mounts this app at /mcp; the SDK's
# default internal path of /mcp would otherwise make the endpoint /mcp/mcp.
mcp = FastMCP(
    "clipfinder",
    instructions=(
        "Index Instagram reels into ClipFinder and search them. Indexing is "
        "asynchronous: after ingest_instagram_reel returns, poll get_transcript "
        "(transcript_status) or search_clips (indexing_status) for completion."
    ),
    stateless_http=True,
    streamable_http_path="/",
)


@mcp.tool()
async def ingest_instagram_reel(url: str) -> dict:
    """
    Download an Instagram reel (video + metadata) and index it for search.

    Indexing (frame embeddings + speech transcript) runs in the background;
    this returns immediately with the new video_id and status "pending".
    Poll get_transcript(video_id) until transcript_status is "completed", or
    check indexing_status on search_clips results. Re-ingesting the same reel
    returns the existing video_id with deduplicated=true and re-indexes it.

    Args:
        url: A public Instagram reel/post URL, e.g. https://www.instagram.com/reel/XXXX/
    """
    async with async_session_maker() as session:
        user = await get_bound_user(session)
        try:
            result = await _ingest_reel(url, user, session)
            if result.get("blob_thumbnail_url"):
                result["blob_thumbnail_url"] = get_blob_url_with_sas(
                    result["blob_thumbnail_url"]
                )
            return result
        except InstagramIngestError as e:
            raise ValueError(str(e)) from e


@mcp.tool()
async def search_clips(query: str, limit: int = 10) -> dict:
    """
    Hybrid search over all indexed clips: filenames, poster thumbnails, visual
    frame content, Instagram captions, and spoken transcript (lexical + semantic,
    fused with reciprocal rank fusion).

    Each result includes links: instagram_url (original reel), video_url
    (temporary playable link to the stored video, valid ~100 hours),
    thumbnail_url (poster JPEG), and the best-matching frame/transcript moment
    with timestamps in seconds.

    Args:
        query: Natural-language description of what is shown or said in the clip.
        limit: Max results (1-50, default 10).
    """
    async with async_session_maker() as session:
        user = await get_bound_user(session)
        service = IndexingService(session)
        results = await service.hybrid_search_rrf(
            user_id=user.id,
            query=query,
            limit=max(1, min(limit, 50)),
        )
        return {
            "query": query,
            "results": [
                {
                    "video_id": str(r["file"].id),
                    "filename": r["file"].filename,
                    "file_type": r["file"].file_type,
                    "source_type": r["file"].source_type,
                    "indexing_status": r["file"].indexing_status,
                    "transcript_status": r["file"].transcript_status,
                    "duration_seconds": r["file"].duration_seconds,
                    "instagram_url": r["file"].source_url,
                    "drive_url": r["file"].drive_url,
                    "video_url": get_blob_url_with_sas(r["file"].blob_video_url),
                    "thumbnail_url": get_blob_url_with_sas(r["file"].blob_thumbnail_url),
                    "hybrid_score": r["hybrid_score"],
                    "text_score": r["text_score"],
                    "thumbnail_score": r["thumbnail_score"],
                    "frame_score": r["frame_score"],
                    "caption_score": r["caption_score"],
                    "transcript_score": r["transcript_score"],
                    "matched_thumbnail": (
                        {
                            "thumbnail_image_url": r["matched_thumbnail"].get(
                                "thumbnailImageUrl"
                            ),
                        }
                        if r["matched_thumbnail"]
                        else None
                    ),
                    "matched_frame": (
                        {
                            "frame_image_url": r["matched_frame"].get("frameImageUrl"),
                            "time_seconds": r["matched_frame"].get("timeSeconds"),
                        }
                        if r["matched_frame"]
                        else None
                    ),
                    "matched_caption": (
                        {"text": r["matched_caption"].get("text")}
                        if r["matched_caption"]
                        else None
                    ),
                    "matched_transcript": (
                        {
                            "text": r["matched_transcript"].get("text"),
                            "start_seconds": r["matched_transcript"].get("startSeconds"),
                            "end_seconds": r["matched_transcript"].get("endSeconds"),
                        }
                        if r["matched_transcript"]
                        else None
                    ),
                }
                for r in results
            ],
        }


@mcp.tool()
async def get_transcript(video_id: str) -> dict:
    """
    Get the full speech transcript of an indexed video, as ordered segments
    with start/end timestamps in seconds plus the joined full text.

    If transcript_status is "pending" or "processing", indexing is still
    running: the segments transcribed so far are returned — poll again later.

    Args:
        video_id: The video's id, as returned by ingest_instagram_reel or search_clips.
    """
    try:
        video_uuid = UUID(video_id)
    except ValueError:
        raise ValueError(f"'{video_id}' is not a valid video id (expected a UUID)")

    async with async_session_maker() as session:
        user = await get_bound_user(session)
        service = IndexingService(session)
        data = await service.get_full_transcript(video_uuid, user.id)
        if data is None:
            raise ValueError(f"Video {video_id} not found")

        file_row = data["file"]
        segments = data["segments"]
        return {
            "video_id": video_id,
            "filename": file_row.filename,
            "source_url": file_row.source_url,
            "indexing_status": file_row.indexing_status,
            "transcript_status": file_row.transcript_status,
            "duration_seconds": file_row.duration_seconds,
            "segments": [
                {
                    "index": s.segment_index,
                    "start_seconds": s.start_seconds,
                    "end_seconds": s.end_seconds,
                    "text": s.text,
                }
                for s in segments
            ],
            "full_text": " ".join(s.text.strip() for s in segments),
        }
