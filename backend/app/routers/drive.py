"""Google Drive API routes for folder and file operations."""

import asyncio
import json
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
from app.models.user import User
from app.observability import flush_langfuse, trace_batch_upload, trace_retrieval
from app.services.google_auth import get_valid_access_token
from app.services.google_drive import GoogleDriveService, MAX_CLIPS_PER_FOLDER
from app.services.indexing import IndexingService

logger = logging.getLogger(__name__)

try:
    from app.tasks import _run_frame_indexing_async
except Exception as e:
    logger.warning("Video frame indexing unavailable (tasks module failed to load): %s", e)
    _run_frame_indexing_async = None

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
        
        # Increase thumbnail size by modifying the URL (default is small)
        # Google thumbnail URLs have =s220 or similar, we can change to =s400
        if "=s" in thumbnail_url:
            thumbnail_url = thumbnail_url.rsplit("=s", 1)[0] + "=s400"
        
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
    3. Generates text embeddings for filenames
    4. Generates vision embeddings for thumbnails
    5. Saves file metadata to the database with status='pending'
    
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

    logger.info("[index] Getting access token for vision embeddings")
    access_token = await get_valid_access_token(
        user_id=user.id,
        access_token=user.google_access_token,
        refresh_token=user.google_refresh_token,
        expires_at=user.google_token_expires_at,
        session=session
    )
    logger.info("[index] Saving %d file(s) for indexing (text + vision embeddings)", len(valid_files))

    batch_metadata = {
        "folder_id": folder_id,
        "folder_name": folder.get("name", ""),
        "user_id": str(user.id),
        "file_count": len(valid_files),
    }
    with trace_batch_upload("folder_index", metadata=batch_metadata):
        indexing_service = IndexingService(session)
        indexed_files, video_trace_contexts = await indexing_service.save_files_for_indexing(
            user_id=user.id,
            google_account_id=google_account_id,
            folder_id=folder_id,
            files=valid_files,
            google_access_token=access_token,
        )
    logger.info("[index] save_files_for_indexing done: %d indexed_files", len(indexed_files))

    videos = [f for f in indexed_files if f.file_type == "video"]
    logger.info("[index] Videos to frame-index: %d (file_types in batch: %s)", len(videos), [f.file_type for f in indexed_files])
    if videos:
        settings = get_settings()
        if settings.service_bus_connection_string:
            logger.info(
                "[index] Service Bus configured, sending %d message(s) to queue '%s'",
                len(videos),
                settings.video_indexing_queue_name,
            )
            try:
                from azure.servicebus import ServiceBusClient, ServiceBusMessage

                with ServiceBusClient.from_connection_string(
                    conn_str=settings.service_bus_connection_string,
                    logging_enable=False,
                ) as sb_client:
                    sender = sb_client.get_queue_sender(
                        queue_name=settings.video_indexing_queue_name
                    )
                    with sender:
                        for i, f in enumerate(videos):
                            ctx = video_trace_contexts[i] if i < len(video_trace_contexts) else {}
                            body = json.dumps({
                                "video_id": str(f.id),
                                "trace_id": ctx.get("trace_id"),
                                "parent_span_id": ctx.get("parent_span_id"),
                            })
                            logger.info(
                                "[index] Sending Service Bus message for video_id=%s filename=%s",
                                f.id,
                                f.filename,
                            )
                            sender.send_messages(ServiceBusMessage(body))
                logger.info(
                    "[index] Successfully sent %d Service Bus message(s) for video frame indexing",
                    len(videos),
                )
            except Exception:
                logger.exception(
                    "[index] Failed to send messages to Service Bus, falling back to in-process video frame indexing"
                )
                _enqueue_in_process_video_indexing(videos, video_trace_contexts)
        else:
            logger.info(
                "[index] Service Bus connection string not set; using in-process video frame indexing"
            )
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


# --- Embedding & Search Endpoints ---

class GenerateEmbeddingsResponse(BaseModel):
    """Response for batch embedding generation."""
    success: bool
    processed: int
    failed: int
    totalFound: int
    error: Optional[str] = None


class SemanticSearchFileInfo(BaseModel):
    """File information for semantic search results with similarity score."""
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
    similarityScore: float

    class Config:
        from_attributes = True


class SemanticSearchResponse(BaseModel):
    """Response for semantic file search."""
    files: list[SemanticSearchFileInfo]
    totalCount: int
    query: str


@router.post("/embeddings/generate", response_model=GenerateEmbeddingsResponse)
async def generate_missing_embeddings(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate embeddings for files that don't have them yet.
    
    This endpoint processes files in batches and generates
    filename embeddings using Azure OpenAI's text-embedding-3-small model.
    
    Call this endpoint to backfill embeddings for files that were
    indexed before embedding generation was enabled.
    """
    with trace_retrieval("generate_text_embeddings", metadata={"user_id": str(user.id)}):
        indexing_service = IndexingService(session)
        stats = await indexing_service.generate_missing_embeddings(
            user_id=user.id,
            batch_size=100,
        )
    flush_langfuse()
    return GenerateEmbeddingsResponse(
        success=stats.get("success", False),
        processed=stats.get("processed", 0),
        failed=stats.get("failed", 0),
        totalFound=stats.get("total_found", 0),
        error=stats.get("error"),
    )


@router.get("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search_files(
    q: str = "",
    file_type: Optional[str] = None,
    limit: int = 20,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Search indexed files using semantic similarity.
    
    Uses vector embeddings to find files with semantically similar filenames
    to the search query. This enables finding files even when the query
    doesn't exactly match the filename.
    
    Args:
        q: Search query (natural language description of what you're looking for)
        file_type: Optional filter by file type ("video" or "image")
        limit: Maximum number of results to return (default 20, max 100)
    
    Returns:
        List of matching files with similarity scores, ordered by relevance
    """
    if not q or not q.strip():
        return SemanticSearchResponse(
            files=[],
            totalCount=0,
            query=q,
        )
    
    # Cap limit at 100
    limit = min(limit, 100)

    with trace_retrieval("semantic_search", metadata={"query": q[:100], "limit": limit}):
        indexing_service = IndexingService(session)
        results = await indexing_service.semantic_search(
            user_id=user.id,
            query=q,
            file_type=file_type,
            limit=limit,
        )

    # Convert to response format
    response_files = [
        SemanticSearchFileInfo(
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
            similarityScore=score,
        )
        for f, score in results
    ]
    flush_langfuse()
    return SemanticSearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
    )


class HybridSearchFileInfo(BaseModel):
    """File information for hybrid search results with all score components."""
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
    semanticScore: float
    hybridScore: float

    class Config:
        from_attributes = True


class HybridSearchResponse(BaseModel):
    """Response for hybrid file search."""
    files: list[HybridSearchFileInfo]
    totalCount: int
    query: str
    textWeight: float
    semanticWeight: float


@router.get("/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_files(
    q: str = "",
    file_type: Optional[str] = None,
    limit: int = 50,
    text_weight: float = 0.7,
    semantic_weight: float = 0.3,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Hybrid search combining text-based and semantic search with reranking.
    
    Combines results from substring/trigram text search and vector semantic search
    using weighted score fusion for better relevance ranking.
    
    Formula: hybrid_score = text_weight * normalized_text_score + semantic_weight * normalized_semantic_score
    
    Args:
        q: Search query (required)
        file_type: Optional filter by file type ("video" or "image")
        limit: Maximum number of results to return (default 50, max 100)
        text_weight: Weight for text search scores (default 0.7, range 0-1)
        semantic_weight: Weight for semantic search scores (default 0.3, range 0-1)
    
    Returns:
        List of matching files with text, semantic, and hybrid scores,
        ordered by hybrid score from highest to lowest
    """
    if not q or not q.strip():
        return HybridSearchResponse(
            files=[],
            totalCount=0,
            query=q,
            textWeight=text_weight,
            semanticWeight=semantic_weight,
        )
    
    # Validate weights
    if text_weight < 0 or text_weight > 1:
        text_weight = 0.7
    if semantic_weight < 0 or semantic_weight > 1:
        semantic_weight = 0.3
    
    # Cap limit at 100
    limit = min(limit, 100)

    with trace_retrieval("hybrid_search", metadata={"query": q[:100], "limit": limit}):
        indexing_service = IndexingService(session)
        results = await indexing_service.hybrid_search(
            user_id=user.id,
            query=q,
            file_type=file_type,
            limit=limit,
            text_weight=text_weight,
            semantic_weight=semantic_weight,
        )

    # Convert to response format
    response_files = [
        HybridSearchFileInfo(
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
            textScore=text_score,
            semanticScore=semantic_score,
            hybridScore=hybrid_score,
        )
        for f, text_score, semantic_score, hybrid_score in results
    ]
    flush_langfuse()
    return HybridSearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
        textWeight=text_weight,
        semanticWeight=semantic_weight,
    )


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


class VisionHybridSearchFileInfo(BaseModel):
    """File information for vision hybrid search results with all score components."""
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
    hybridScore: float
    matchedFrame: Optional[MatchedFrameInfo] = None

    class Config:
        from_attributes = True


class VisionHybridSearchResponse(BaseModel):
    """Response for vision hybrid file search."""
    files: list[VisionHybridSearchFileInfo]
    totalCount: int
    query: str
    textWeight: float
    visionWeight: float


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
    CLIP embeddings using Azure AI Vision.
    
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
    
    Uses CLIP embeddings to find files with visually similar thumbnails
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
    text_weight: float = 0.5,
    vision_weight: float = 0.5,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Hybrid search combining text-based and vision semantic search with reranking.
    
    Combines results from substring/trigram text search and CLIP vision search
    using weighted score fusion for better relevance ranking.
    
    Formula: hybrid_score = text_weight * normalized_text_score + vision_weight * normalized_vision_score
    
    Args:
        q: Search query (required)
        file_type: Optional filter by file type ("video" or "image")
        limit: Maximum number of results to return (default 50, max 100)
        text_weight: Weight for text search scores (default 0.5, range 0-1)
        vision_weight: Weight for vision search scores (default 0.5, range 0-1)
    
    Returns:
        List of matching files with text, vision, and hybrid scores,
        ordered by hybrid score from highest to lowest
    """
    if not q or not q.strip():
        return VisionHybridSearchResponse(
            files=[],
            totalCount=0,
            query=q,
            textWeight=text_weight,
            visionWeight=vision_weight,
        )
    
    # Validate weights
    if text_weight < 0 or text_weight > 1:
        text_weight = 0.5
    if vision_weight < 0 or vision_weight > 1:
        vision_weight = 0.5
    
    # Cap limit at 100
    limit = min(limit, 100)

    with trace_retrieval("vision_hybrid_search", metadata={"query": q[:100], "limit": limit}):
        indexing_service = IndexingService(session)
        results = await indexing_service.vision_hybrid_search_unified(
            user_id=user.id,
            query=q,
            file_type=file_type,
            limit=limit,
            text_weight=text_weight,
            vision_weight=vision_weight,
        )
    
    # Convert to response format (include matchedFrame for video frame hits)
    response_files = [
        VisionHybridSearchFileInfo(
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
            textScore=text_score,
            visionScore=vision_score,
            hybridScore=hybrid_score,
            matchedFrame=MatchedFrameInfo(**mf) if mf else None,
        )
        for f, text_score, vision_score, hybrid_score, mf in results
    ]
    flush_langfuse()
    return VisionHybridSearchResponse(
        files=response_files,
        totalCount=len(response_files),
        query=q,
        textWeight=text_weight,
        visionWeight=vision_weight,
    )