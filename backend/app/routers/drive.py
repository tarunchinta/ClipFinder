"""Google Drive API routes for folder and file operations."""

import asyncio
import logging
from typing import Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import get_settings
from app.database import get_async_session
from app.models.indexed_file import IndexingStatus
from app.models.user import User
from app.observability import flush_langfuse, trace_batch_upload, trace_retrieval
from app.services.google_auth import get_valid_access_token
from app.services.google_drive import GoogleDriveService, MAX_CLIPS_PER_FOLDER
from app.services.indexing import IndexingService
from app.services.service_bus_publisher import (
    publish_image_indexing_jobs,
    publish_video_indexing_jobs,
)
from app.services.vision_embedding import normalize_drive_thumbnail_url

logger = logging.getLogger(__name__)

try:
    from app.tasks import _run_frame_indexing_async, _run_image_indexing_async
except Exception as e:
    logger.warning("Indexing task runners unavailable (tasks module failed to load): %s", e)
    _run_frame_indexing_async = None
    _run_image_indexing_async = None

router = APIRouter(prefix="/api/drive", tags=["drive"])


class FolderInfo(BaseModel):
    """Folder information response."""
    id: str
    name: str


class FileInfo(BaseModel):
    """File information with validation status."""
    id: str
    name: str
    mimeType: str
    mediaType: str  # "video" or "image"
    size: int
    sizeMB: float
    durationSeconds: Optional[float] = None
    modifiedTime: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    driveUrl: Optional[str] = None
    downloadUrl: Optional[str] = None
    isValid: bool
    errors: list[str]
    width: Optional[int] = None
    height: Optional[int] = None


class FolderFilesResponse(BaseModel):
    """Response for folder files listing."""
    folder: FolderInfo
    files: list[FileInfo]
    totalFiles: int
    validFiles: int
    invalidFiles: int
    maxClips: int = MAX_CLIPS_PER_FOLDER


def get_drive_service(user: User) -> GoogleDriveService:
    """
    Get Google Drive service for a user.
    
    Raises HTTPException if user doesn't have valid Google tokens.
    """
    if not user.google_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Drive not connected. Please re-authenticate with Google.",
        )
    
    return GoogleDriveService(
        access_token=user.google_access_token,
        refresh_token=user.google_refresh_token,
    )


@router.get("/thumbnail/{file_id}")
async def get_thumbnail(
    file_id: str,
    user: User = Depends(current_active_user),
):
    """
    Proxy endpoint for Google Drive thumbnails.
    
    Uses the Google Drive API to get a fresh thumbnail link, then fetches
    and proxies the image content.
    """
    if not user.google_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Drive not connected. Please re-authenticate.",
        )
    
    try:
        drive_service = get_drive_service(user)
        
        # Get file metadata including thumbnail link
        file_metadata = drive_service.service.files().get(
            fileId=file_id,
            fields="thumbnailLink"
        ).execute()
        
        thumbnail_url = file_metadata.get("thumbnailLink")
        
        if not thumbnail_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No thumbnail available for this file",
            )
        
        # Request a larger thumbnail (default is small); capped at 336px for embedding alignment
        thumbnail_url = normalize_drive_thumbnail_url(thumbnail_url)
        
        # Fetch the thumbnail - Google's lh3.googleusercontent.com URLs 
        # don't need auth once we have the link
        async with httpx.AsyncClient() as client:
            response = await client.get(
                thumbnail_url,
                follow_redirects=True,
                timeout=10.0,
            )
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "image/jpeg")
                return Response(
                    content=response.content,
                    media_type=content_type,
                    headers={
                        "Cache-Control": "public, max-age=3600",
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Thumbnail not available",
                )
                
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout fetching thumbnail",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching thumbnail: {str(e)}",
        )


@router.get("/folders/{folder_id}/files", response_model=FolderFilesResponse)
async def list_folder_files(
    folder_id: str,
    user: User = Depends(current_active_user),
):
    """
    List media files (videos and images) in a Google Drive folder with validation.
    
    Returns files with validation status indicating if they meet
    the MVP constraints (videos ≤30s duration, all files ≤100MB size).
    """
    drive_service = get_drive_service(user)
    
    # Get folder info
    folder = drive_service.get_folder_info(folder_id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found or not accessible",
        )
    
    # List and validate files
    files = drive_service.list_media_files(folder_id)
    
    # Count valid/invalid
    valid_count = sum(1 for f in files if f["isValid"])
    invalid_count = len(files) - valid_count
    
    return FolderFilesResponse(
        folder=FolderInfo(id=folder["id"], name=folder["name"]),
        files=[FileInfo(**f) for f in files],
        totalFiles=len(files),
        validFiles=valid_count,
        invalidFiles=invalid_count,
    )


@router.get("/folders/{folder_id}")
async def get_folder_info(
    folder_id: str,
    user: User = Depends(current_active_user),
):
    """Get information about a Google Drive folder."""
    drive_service = get_drive_service(user)
    
    folder = drive_service.get_folder_info(folder_id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found or not accessible",
        )
    
    return FolderInfo(id=folder["id"], name=folder["name"])


@router.get("/user/token-status")
async def get_token_status(user: User = Depends(current_active_user)):
    """
    Check if user has valid Google Drive tokens.
    
    Used by frontend to determine if picker can be shown.
    """
    has_token = bool(user.google_access_token)
    
    return {
        "hasGoogleToken": has_token,
        "tokenExpiresAt": user.google_token_expires_at.isoformat() if user.google_token_expires_at else None,
    }


# --- Indexing Endpoints ---

class IndexedFileInfo(BaseModel):
    """Indexed file information."""
    id: UUID
    driveFileId: str
    filename: str
    mimeType: str
    fileType: str
    sizeBytes: int
    indexingStatus: str

    class Config:
        from_attributes = True


class IndexFilesResponse(BaseModel):
    """Response for indexing files in a folder."""
    folder: FolderInfo
    indexedCount: int
    skippedCount: int
    files: list[IndexedFileInfo]


class IndexingStatsResponse(BaseModel):
    """Response for indexing statistics."""
    total: int
    pending: int
    processing: int
    completed: int
    failed: int


class SearchFileInfo(BaseModel):
    """File information for search results."""
    id: UUID
    driveFileId: str
    filename: str
    mimeType: str
    fileType: str
    sizeBytes: int
    durationSeconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thumbnailUrl: Optional[str] = None
    driveUrl: Optional[str] = None
    indexingStatus: str

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Response for file search."""
    files: list[SearchFileInfo]
    totalCount: int
    query: str
    fuzzy: bool = False


def get_google_account_id(user: User) -> str:
    """
    Get the Google account ID from user's OAuth accounts.
    
    Raises HTTPException if no Google account is linked.
    """
    google_account = next(
        (acc for acc in user.oauth_accounts if acc.oauth_name == "google"),
        None
    )
    
    if not google_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Google account linked to this user.",
        )
    
    return google_account.account_id


@router.post("/folders/{folder_id}/index", response_model=IndexFilesResponse)
async def index_folder_files(
    folder_id: str,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Index valid media files in a Google Drive folder.
    
    This endpoint:
    1. Fetches files from the specified folder
    2. Filters to valid files only (≤30s duration for videos, ≤100MB size)
    3. Generates vision embeddings for thumbnails
    4. Saves file metadata to the database with status='pending'
    
    For re-indexing, existing files are reset to 'pending' status.
    """
    logger.info("[index] POST /folders/%s/index called", folder_id)
    drive_service = get_drive_service(user)

    # Get folder info
    folder = drive_service.get_folder_info(folder_id)
    if not folder:
        logger.warning("[index] Folder not found: %s", folder_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found or not accessible",
        )
    logger.info("[index] Folder resolved: id=%s name=%s", folder["id"], folder.get("name", ""))

    # Get Google account ID from OAuth
    google_account_id = get_google_account_id(user)
    logger.info("[index] Listing media files in folder_id=%s", folder_id)

    # List and validate files
    all_files = drive_service.list_media_files(folder_id)
    logger.info("[index] list_media_files returned %d file(s)", len(all_files))

    # Filter to valid files only
    valid_files = [f for f in all_files if f["isValid"]]
    skipped_count = len(all_files) - len(valid_files)
    if skipped_count:
        logger.info("[index] Skipped %d invalid file(s), %d valid", skipped_count, len(valid_files))

    if not valid_files:
        logger.info("[index] No valid files to index, returning empty")
        return IndexFilesResponse(
            folder=FolderInfo(id=folder["id"], name=folder["name"]),
            indexedCount=0,
            skippedCount=skipped_count,
            files=[],
        )

    logger.info("[index] Saving %d file(s) for indexing (metadata only)", len(valid_files))

    batch_metadata = {
        "folder_id": folder_id,
        "folder_name": folder.get("name", ""),
        "user_id": str(user.id),
        "file_count": len(valid_files),
    }
    with trace_batch_upload("folder_index", metadata=batch_metadata):
        indexing_service = IndexingService(session)
        indexed_files, video_trace_contexts, image_trace_contexts = (
            await indexing_service.save_files_for_indexing(
                user_id=user.id,
                google_account_id=google_account_id,
                folder_id=folder_id,
                files=valid_files,
            )
        )
    logger.info("[index] save_files_for_indexing done: %d indexed_files", len(indexed_files))

    settings = get_settings()
    if not settings.service_bus_connection_string:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Bus not configured; indexing unavailable",
        )

    images = [
        f for f in indexed_files
        if f.file_type == "image" and f.vision_indexing_status == IndexingStatus.PENDING.value
    ]
    videos = [f for f in indexed_files if f.file_type == "video"]
    logger.info(
        "[index] Jobs to enqueue: %d image(s), %d video(s)",
        len(images),
        len(videos),
    )

    try:
        if images:
            publish_image_indexing_jobs(images, image_trace_contexts)
        if videos:
            publish_video_indexing_jobs(videos, video_trace_contexts)
    except Exception:
        logger.exception(
            "[index] Service Bus publish failed, falling back to in-process indexing"
        )
        if images:
            _enqueue_in_process_image_indexing(images, image_trace_contexts)
        if videos:
            _enqueue_in_process_video_indexing(videos, video_trace_contexts)

    logger.info("[index] Returning response: indexedCount=%d", len(indexed_files))

    response_files = [
        IndexedFileInfo(
            id=f.id,
            driveFileId=f.drive_file_id,
            filename=f.filename,
            mimeType=f.mime_type,
            fileType=f.file_type,
            sizeBytes=f.size_bytes,
            indexingStatus=f.indexing_status,
        )
        for f in indexed_files
    ]
    flush_langfuse()
    return IndexFilesResponse(
        folder=FolderInfo(id=folder["id"], name=folder["name"]),
        indexedCount=len(indexed_files),
        skippedCount=skipped_count,
        files=response_files,
    )


@router.get("/folders/{folder_id}/index/stats", response_model=IndexingStatsResponse)
async def get_folder_indexing_stats(
    folder_id: str,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get indexing statistics for a folder.
    
    Returns counts of files by indexing status.
    """
    indexing_service = IndexingService(session)
    stats = await indexing_service.get_indexing_stats(
        user_id=user.id,
        folder_id=folder_id,
    )
    
    return IndexingStatsResponse(**stats)


# --- Search Endpoints ---

@router.get("/search", response_model=SearchResponse)
async def search_files(
    q: str = "",
    file_type: Optional[str] = None,
    fuzzy: bool = False,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Search indexed files by filename with multi-term matching.
    
    Args:
        q: Search query (supports multiple space-separated terms, all must match)
        file_type: Optional filter by file type ("video" or "image")
        fuzzy: If True, enable typo-tolerant fuzzy matching using trigram similarity.
               If False (default), require exact substring matches for each term.
    
    Returns:
        List of matching files with metadata, ordered by relevance
    """
    indexing_service = IndexingService(session)
    files = await indexing_service.search_files(
        user_id=user.id,
        query=q,
        file_type=file_type,
        fuzzy=fuzzy,
    )
    
    # Convert to response format
    response_files = [
        SearchFileInfo(
            id=f.id,
            driveFileId=f.drive_file_id,
            filename=f.filename,
            mimeType=f.mime_type,
            fileType=f.file_type,
            sizeBytes=f.size_bytes,
            durationSeconds=f.duration_seconds,
            width=f.width,
            height=f.height,
            thumbnailUrl=f.thumbnail_url,
            driveUrl=f.drive_url,
            indexingStatus=f.indexing_status,
        )
        for f in files
    ]
    
    return SearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
        fuzzy=fuzzy,
    )


def _enqueue_in_process_image_indexing(
    images: list,
    image_trace_contexts: list[dict] | None = None,
) -> None:
    """
    Fallback path: enqueue in-process image vision indexing via _run_image_indexing_async.
    """
    if _run_image_indexing_async is None:
        logger.warning(
            "[index] Image vision indexing skipped: task runner not available (%d image(s) not indexed)",
            len(images),
        )
        return

    logger.info(
        "[index] Enqueueing image vision indexing for %d image(s) (in-process)",
        len(images),
    )

    contexts = image_trace_contexts if image_trace_contexts is not None else []
    if len(contexts) != len(images):
        contexts = [{} for _ in images]

    def _log_task_result(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception:
            logger.exception("[index] Image vision indexing task failed")

    for f, ctx in zip(images, contexts):
        logger.info(
            "[index] Enqueueing image vision indexing: file_id=%s filename=%s",
            f.id,
            f.filename,
        )
        t = asyncio.create_task(
            _run_image_indexing_async(
                str(f.id),
                trace_id=ctx.get("trace_id") if ctx else None,
                parent_span_id=ctx.get("parent_span_id") if ctx else None,
            )
        )
        t.add_done_callback(_log_task_result)

    logger.info(
        "[index] Image vision indexing enqueued for %d image(s), running in background",
        len(images),
    )


def _enqueue_in_process_video_indexing(
    videos: list,
    video_trace_contexts: list[dict] | None = None,
) -> None:
    """
    Fallback path: enqueue in-process video frame indexing tasks using _run_frame_indexing_async.
    video_trace_contexts: optional list of {"trace_id", "parent_span_id"} per video (same order as videos).
    """
    if _run_frame_indexing_async is None:
        logger.warning(
            "[index] Video frame indexing skipped: task runner not available (%d video(s) not indexed)",
            len(videos),
        )
        return

    logger.info(
        "[index] Enqueueing video frame indexing for %d video(s) (in-process)",
        len(videos),
    )

    contexts = video_trace_contexts if video_trace_contexts is not None else []
    if len(contexts) != len(videos):
        contexts = [{} for _ in videos]

    def _log_task_result(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception:
            logger.exception("[index] Video frame indexing task failed")

    for f, ctx in zip(videos, contexts):
        logger.info(
            "[index] Enqueueing video frame indexing: video_id=%s filename=%s",
            f.id,
            f.filename,
        )
        t = asyncio.create_task(
            _run_frame_indexing_async(
                str(f.id),
                trace_id=ctx.get("trace_id") if ctx else None,
                parent_span_id=ctx.get("parent_span_id") if ctx else None,
            )
        )
        t.add_done_callback(_log_task_result)

    logger.info(
        "[index] Video frame indexing enqueued for %d video(s), running in background",
        len(videos),
    )


# --- Vision Embedding & Search Endpoints ---

class GenerateVisionEmbeddingsResponse(BaseModel):
    """Response for batch vision embedding generation."""
    success: bool
    processed: int
    failed: int
    totalFound: int
    error: Optional[str] = None


class VisionSearchFileInfo(BaseModel):
    """File information for vision search results with similarity score."""
    id: UUID
    driveFileId: str
    filename: str
    mimeType: str
    fileType: str
    sizeBytes: int
    durationSeconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thumbnailUrl: Optional[str] = None
    driveUrl: Optional[str] = None
    indexingStatus: str
    visionSimilarityScore: float

    class Config:
        from_attributes = True


class VisionSearchResponse(BaseModel):
    """Response for vision semantic file search."""
    files: list[VisionSearchFileInfo]
    totalCount: int
    query: str


class MatchedFrameInfo(BaseModel):
    """When a search hit is a video frame, this describes the matched frame."""
    frameImageUrl: Optional[str] = None
    timeSeconds: float
    frameIndex: int


class MatchedTranscriptInfo(BaseModel):
    """When a search hit matched transcript text, this describes the matched segment.

    startSeconds is word-accurate when a query token matched a WhisperX-aligned
    word (matchedWord is set); otherwise it is the segment start.
    """
    text: str
    startSeconds: float
    endSeconds: float
    segmentIndex: int
    matchedWord: Optional[str] = None


class VisionHybridSearchFileInfo(BaseModel):
    """File information for hybrid search results with per-leg RRF score components."""
    id: UUID
    driveFileId: str
    filename: str
    mimeType: str
    fileType: str
    sizeBytes: int
    durationSeconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thumbnailUrl: Optional[str] = None
    driveUrl: Optional[str] = None
    indexingStatus: str
    textScore: float
    visionScore: float
    transcriptScore: float = 0.0
    colorScore: float = 0.0
    hybridScore: float
    matchedFrame: Optional[MatchedFrameInfo] = None
    matchedTranscript: Optional[MatchedTranscriptInfo] = None

    class Config:
        from_attributes = True


class VisionHybridSearchResponse(BaseModel):
    """Response for hybrid file search (reciprocal rank fusion)."""
    files: list[VisionHybridSearchFileInfo]
    totalCount: int
    query: str


class VisionIndexingStatsResponse(BaseModel):
    """Response for vision indexing statistics."""
    total: int
    withThumbnail: int
    pending: int
    processing: int
    completed: int
    failed: int
    notStarted: int


@router.post("/vision-embeddings/generate", response_model=GenerateVisionEmbeddingsResponse)
async def generate_missing_vision_embeddings(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate vision embeddings for files that don't have them yet.
    
    This endpoint processes files with thumbnails in batches and generates
    Gemini Embedding 2 vectors via Google AI Studio.
    
    Call this endpoint to backfill vision embeddings for files that were
    indexed before vision embedding generation was enabled.
    
    Uses the current user's Google Drive token to fetch fresh thumbnail URLs.
    """
    with trace_retrieval("generate_vision_embeddings", metadata={"user_id": str(user.id)}):
        indexing_service = IndexingService(session)
        stats = await indexing_service.generate_missing_vision_embeddings(
            user_id=user.id,
            google_access_token=user.google_access_token,  # Pass token for fresh thumbnails
            batch_size=50,  # Smaller batch size due to image download overhead
        )
    flush_langfuse()
    return GenerateVisionEmbeddingsResponse(
        success=stats.get("success", False),
        processed=stats.get("processed", 0),
        failed=stats.get("failed", 0),
        totalFound=stats.get("total_found", 0),
        error=stats.get("error"),
    )


@router.get("/vision-embeddings/stats", response_model=VisionIndexingStatsResponse)
async def get_vision_indexing_stats(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get vision indexing statistics for the current user.
    
    Returns counts of files by vision indexing status.
    """
    indexing_service = IndexingService(session)
    stats = await indexing_service.get_vision_indexing_stats(user_id=user.id)
    
    return VisionIndexingStatsResponse(
        total=stats.get("total", 0),
        withThumbnail=stats.get("with_thumbnail", 0),
        pending=stats.get("pending", 0),
        processing=stats.get("processing", 0),
        completed=stats.get("completed", 0),
        failed=stats.get("failed", 0),
        notStarted=stats.get("not_started", 0),
    )


@router.get("/search/vision", response_model=VisionSearchResponse)
async def vision_search_files(
    q: str = "",
    file_type: Optional[str] = None,
    limit: int = 20,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Search indexed files using vision semantic similarity.
    
    Uses Gemini Embedding 2 to find files with visually similar thumbnails
    to the search query concept. This enables finding files based on
    visual content rather than filename.
    
    Args:
        q: Search query (natural language description of what you're looking for)
        file_type: Optional filter by file type ("video" or "image")
        limit: Maximum number of results to return (default 20, max 100)
    
    Returns:
        List of matching files with vision similarity scores, ordered by relevance
    """
    if not q or not q.strip():
        return VisionSearchResponse(
            files=[],
            totalCount=0,
            query=q,
        )
    
    # Cap limit at 100
    limit = min(limit, 100)

    with trace_retrieval("vision_semantic_search", metadata={"query": q[:100], "limit": limit}):
        indexing_service = IndexingService(session)
        results = await indexing_service.vision_semantic_search(
            user_id=user.id,
            query=q,
            file_type=file_type,
            limit=limit,
        )

    # Convert to response format
    response_files = [
        VisionSearchFileInfo(
            id=f.id,
            driveFileId=f.drive_file_id,
            filename=f.filename,
            mimeType=f.mime_type,
            fileType=f.file_type,
            sizeBytes=f.size_bytes,
            durationSeconds=f.duration_seconds,
            width=f.width,
            height=f.height,
            thumbnailUrl=f.thumbnail_url,
            driveUrl=f.drive_url,
            indexingStatus=f.indexing_status,
            visionSimilarityScore=score,
        )
        for f, score in results
    ]
    flush_langfuse()
    return VisionSearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
    )


@router.get("/search/vision-hybrid", response_model=VisionHybridSearchResponse)
async def vision_hybrid_search_files(
    q: str = "",
    file_type: Optional[str] = None,
    limit: int = 50,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Hybrid search fusing lexical, semantic, and color-grade legs with Reciprocal
    Rank Fusion (RRF).

    Legs: filename trigram similarity, Gemini vision embeddings (thumbnails +
    video frames), transcript full-text search, transcript segment embeddings,
    and a Lab color signature (only when the query names a look/grade).
    Each leg contributes 1/(k + rank) per file; a clip whose spoken words match
    the query ranks high and carries the matched segment's timestamp.

    Args:
        q: Search query (required)
        file_type: Optional filter by file type ("video" or "image")
        limit: Maximum number of results to return (default 50, max 100)

    Returns:
        List of matching files with per-leg and fused RRF scores, ordered by
        hybrid score, including matched frame and/or transcript segment info.
    """
    if not q or not q.strip():
        return VisionHybridSearchResponse(
            files=[],
            totalCount=0,
            query=q,
        )

    # Cap limit at 100
    limit = min(limit, 100)

    with trace_retrieval(
        "hybrid_search_rrf",
        metadata={
            "query": q[:100],
            "limit": limit,
            "file_type": file_type,
        },
    ) as span:
        indexing_service = IndexingService(session)
        results = await indexing_service.hybrid_search_rrf(
            user_id=user.id,
            query=q,
            file_type=file_type,
            limit=limit,
        )
        if span:
            span.update(
                metadata={
                    "result_count": len(results),
                    "top_results": [
                        {
                            "file_id": str(r["file"].id),
                            "filename": r["file"].filename[:80],
                            "text_score": round(r["text_score"], 4),
                            "vision_score": round(r["vision_score"], 4),
                            "transcript_score": round(r["transcript_score"], 4),
                            "color_score": round(r["color_score"], 4),
                            "hybrid_score": round(r["hybrid_score"], 4),
                            "matched_frame": r["matched_frame"] is not None,
                            "matched_transcript": r["matched_transcript"] is not None,
                        }
                        for r in results[:5]
                    ],
                }
            )

    # Convert to response format (include matched frame/transcript for timestamp deep links)
    response_files = [
        VisionHybridSearchFileInfo(
            id=r["file"].id,
            driveFileId=r["file"].drive_file_id,
            filename=r["file"].filename,
            mimeType=r["file"].mime_type,
            fileType=r["file"].file_type,
            sizeBytes=r["file"].size_bytes,
            durationSeconds=r["file"].duration_seconds,
            width=r["file"].width,
            height=r["file"].height,
            thumbnailUrl=r["file"].thumbnail_url,
            driveUrl=r["file"].drive_url,
            indexingStatus=r["file"].indexing_status,
            textScore=r["text_score"],
            visionScore=r["vision_score"],
            transcriptScore=r["transcript_score"],
            colorScore=r["color_score"],
            hybridScore=r["hybrid_score"],
            matchedFrame=MatchedFrameInfo(**r["matched_frame"]) if r["matched_frame"] else None,
            matchedTranscript=(
                MatchedTranscriptInfo(**r["matched_transcript"])
                if r["matched_transcript"]
                else None
            ),
        )
        for r in results
    ]
    flush_langfuse()
    return VisionHybridSearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
    )