"""Indexing service for managing indexed files in the database."""

from datetime import datetime
from typing import Optional
from uuid import UUID
import logging
import re

from sqlalchemy import select, and_, or_, func, desc, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.models.indexed_file import IndexedFile, IndexingStatus
from app.models.video_frame_embedding import VideoFrameEmbedding
from app.models.video_transcript_segment import VideoTranscriptSegment
from app.observability import (
    get_current_observation_id,
    get_current_trace_id,
    trace_index_file,
    trace_vector_search,
)
from app.services.video_frame_indexing import get_blob_url_with_sas
from app.services.vision_embedding import get_vision_embedding_service

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for managing indexed files in Supabase/PostgreSQL."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize the indexing service.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def save_files_for_indexing(
        self,
        user_id: UUID,
        google_account_id: str,
        folder_id: str,
        files: list[dict],
    ) -> tuple[list[IndexedFile], list[dict], list[dict]]:
        """
        Bulk insert/upsert files for indexing (metadata only; vision jobs enqueued separately).
        
        Uses ON CONFLICT to handle re-indexing scenarios:
        - New files are inserted with status='pending'
        - Existing files are updated with status='pending' (reset for re-indexing)
        
        Args:
            user_id: User UUID
            google_account_id: Google account ID from OAuth
            folder_id: Google Drive folder ID
            files: List of file metadata dicts from GoogleDriveService
        
        Returns:
            Tuple of (indexed_files, video_trace_contexts, image_trace_contexts).
            Trace context lists align with videos/images in the batch (same order as filtered file_type).
        """
        if not files:
            return [], [], []
        
        values = []
        video_trace_contexts: list[dict] = []
        image_trace_contexts: list[dict] = []
        for i, file in enumerate(files):
            file_type = file.get("mediaType", "image")
            file_meta = {
                "file_id": file.get("id", ""),
                "filename": file.get("name", ""),
                "index_in_batch": i,
            }
            with trace_index_file(file_type, metadata=file_meta):
                modified_time = None
                if file.get("modifiedTime"):
                    try:
                        parsed_time = datetime.fromisoformat(
                            file["modifiedTime"].replace("Z", "+00:00")
                        )
                        modified_time = parsed_time.replace(tzinfo=None)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse modifiedTime: {file.get('modifiedTime')}")

                trace_id = get_current_trace_id()
                parent_span_id = get_current_observation_id()
                if trace_id and parent_span_id:
                    ctx = {"trace_id": trace_id, "parent_span_id": parent_span_id}
                    if file_type == "video":
                        video_trace_contexts.append(ctx)
                    elif file_type == "image" and file.get("thumbnailUrl"):
                        image_trace_contexts.append(ctx)

                has_thumbnail = bool(file.get("thumbnailUrl"))
                vision_indexing_status = None
                if file_type == "image" and has_thumbnail:
                    vision_indexing_status = IndexingStatus.PENDING.value

                values.append({
                    "user_id": user_id,
                    "google_account_id": google_account_id,
                    "folder_id": folder_id,
                    "drive_file_id": file["id"],
                    "filename": file["name"],
                    "mime_type": file["mimeType"],
                    "file_type": file["mediaType"],
                    "size_bytes": file["size"],
                    "duration_seconds": file.get("durationSeconds"),
                    "width": file.get("width"),
                    "height": file.get("height"),
                    "modified_time": modified_time,
                    "thumbnail_url": file.get("thumbnailUrl"),
                    "drive_url": file.get("driveUrl"),
                    "indexing_status": IndexingStatus.PENDING.value,
                    "error_message": None,
                    "vision_embedding": None,
                    "vision_indexing_status": vision_indexing_status,
                    "vision_indexed_at": None,
                    "transcript_status": (
                        IndexingStatus.PENDING.value if file_type == "video" else None
                    ),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "indexed_at": None,
                })
        
        # Use PostgreSQL upsert (INSERT ... ON CONFLICT)
        stmt = insert(IndexedFile).values(values)
        
        # On conflict, update to reset for re-indexing
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_drive_file",
            set_={
                "folder_id": stmt.excluded.folder_id,
                "filename": stmt.excluded.filename,
                "mime_type": stmt.excluded.mime_type,
                "file_type": stmt.excluded.file_type,
                "size_bytes": stmt.excluded.size_bytes,
                "duration_seconds": stmt.excluded.duration_seconds,
                "width": stmt.excluded.width,
                "height": stmt.excluded.height,
                "modified_time": stmt.excluded.modified_time,
                "thumbnail_url": stmt.excluded.thumbnail_url,
                "drive_url": stmt.excluded.drive_url,
                "indexing_status": IndexingStatus.PENDING.value,
                "error_message": None,
                "vision_embedding": stmt.excluded.vision_embedding,
                "vision_indexing_status": stmt.excluded.vision_indexing_status,
                "vision_indexed_at": stmt.excluded.vision_indexed_at,
                "transcript_status": stmt.excluded.transcript_status,
                "frames_total": None,
                "frames_completed": 0,
                "frames_failed": 0,
                "updated_at": datetime.utcnow(),
                "indexed_at": None,
            }
        ).returning(IndexedFile)
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        indexed_files = list(result.scalars().all())
        return indexed_files, video_trace_contexts, image_trace_contexts
    
    async def update_indexing_status(
        self,
        file_id: UUID,
        status: IndexingStatus,
        error_message: Optional[str] = None
    ) -> Optional[IndexedFile]:
        """
        Update the indexing status of a file.
        
        Args:
            file_id: IndexedFile UUID
            status: New indexing status
            error_message: Optional error message (for failed status)
        
        Returns:
            Updated IndexedFile or None if not found
        """
        stmt = select(IndexedFile).where(IndexedFile.id == file_id)
        result = await self.session.execute(stmt)
        indexed_file = result.scalar_one_or_none()
        
        if not indexed_file:
            return None
        
        indexed_file.indexing_status = status.value
        indexed_file.updated_at = datetime.utcnow()
        
        if error_message:
            indexed_file.error_message = error_message
        
        if status == IndexingStatus.COMPLETED:
            indexed_file.indexed_at = datetime.utcnow()
            indexed_file.error_message = None
        
        await self.session.commit()
        await self.session.refresh(indexed_file)
        
        return indexed_file
    
    async def get_files_by_folder(
        self,
        user_id: UUID,
        folder_id: str
    ) -> list[IndexedFile]:
        """
        Get all indexed files for a specific folder.
        
        Args:
            user_id: User UUID
            folder_id: Google Drive folder ID
        
        Returns:
            List of IndexedFile records
        """
        stmt = select(IndexedFile).where(
            and_(
                IndexedFile.user_id == user_id,
                IndexedFile.folder_id == folder_id
            )
        ).order_by(IndexedFile.filename)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_pending_files(
        self,
        user_id: UUID,
        limit: int = 100
    ) -> list[IndexedFile]:
        """
        Get files awaiting indexing for a user.
        
        Args:
            user_id: User UUID
            limit: Maximum number of files to return
        
        Returns:
            List of IndexedFile records with pending status
        """
        stmt = select(IndexedFile).where(
            and_(
                IndexedFile.user_id == user_id,
                IndexedFile.indexing_status == IndexingStatus.PENDING.value
            )
        ).order_by(IndexedFile.created_at).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_file_by_drive_id(
        self,
        user_id: UUID,
        drive_file_id: str
    ) -> Optional[IndexedFile]:
        """
        Get an indexed file by its Drive file ID.
        
        Args:
            user_id: User UUID
            drive_file_id: Google Drive file ID
        
        Returns:
            IndexedFile or None if not found
        """
        stmt = select(IndexedFile).where(
            and_(
                IndexedFile.user_id == user_id,
                IndexedFile.drive_file_id == drive_file_id
            )
        )
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_indexing_stats(
        self,
        user_id: UUID,
        folder_id: Optional[str] = None
    ) -> dict:
        """
        Get indexing statistics for a user (optionally filtered by folder).
        
        Args:
            user_id: User UUID
            folder_id: Optional folder ID to filter by
        
        Returns:
            Dict with counts by status
        """
        base_condition = IndexedFile.user_id == user_id
        if folder_id:
            base_condition = and_(base_condition, IndexedFile.folder_id == folder_id)
        
        stmt = select(IndexedFile).where(base_condition)
        result = await self.session.execute(stmt)
        files = list(result.scalars().all())
        
        stats = {
            "total": len(files),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }
        
        for f in files:
            if f.indexing_status in stats:
                stats[f.indexing_status] += 1
        
        return stats
    
    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """
        Tokenize a search query into individual terms.
        
        Splits on whitespace and filters out empty strings.
        
        Args:
            query: Raw search query string
        
        Returns:
            List of non-empty search terms
        """
        if not query:
            return []
        return [term.strip() for term in query.split() if term.strip()]
    
    async def search_files(
        self,
        user_id: UUID,
        query: str = "",
        file_type: Optional[str] = None,
        fuzzy: bool = False,
        similarity_threshold: float = 0.3,
        limit: int = 100
    ) -> list[IndexedFile]:
        """
        Search indexed files by filename with multi-term matching and optional fuzzy search.
        
        Args:
            user_id: User UUID
            query: Search query (supports multiple space-separated terms)
            file_type: Optional filter by file type ("video" or "image")
            fuzzy: If True, use trigram similarity for typo-tolerant matching.
                   If False (default), require exact substring matches.
            similarity_threshold: Minimum similarity score for fuzzy matches (0.0-1.0).
                                  Only used when fuzzy=True. Default is 0.3.
            limit: Maximum number of results to return
        
        Returns:
            List of matching IndexedFile records, ordered by relevance (similarity score)
        """
        conditions = [IndexedFile.user_id == user_id]
        
        # Tokenize the query into individual terms
        tokens = self._tokenize_query(query)
        
        # Add file type filter if provided
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)
        
        # Build the search conditions based on mode
        if tokens:
            if fuzzy:
                # Fuzzy mode: use trigram similarity for each token
                # All tokens must have similarity above threshold
                for token in tokens:
                    # Use the % operator which checks if similarity > pg_trgm.similarity_threshold
                    # We use a raw text expression for the similarity check
                    conditions.append(
                        text(f"similarity(filename, :token_{tokens.index(token)}) > :threshold")
                        .bindparams(**{f"token_{tokens.index(token)}": token, "threshold": similarity_threshold})
                    )
            else:
                # Exact mode: require all tokens to match via ILIKE (substring match)
                for token in tokens:
                    conditions.append(IndexedFile.filename.ilike(f"%{token}%"))
        
        # Calculate similarity score for ranking
        # Use the full query for overall similarity scoring
        if query:
            similarity_score = func.similarity(IndexedFile.filename, query)
            stmt = select(IndexedFile, similarity_score.label("score")).where(
                and_(*conditions)
            ).order_by(desc("score"), IndexedFile.filename).limit(limit)
        else:
            # No query - just return all files ordered by filename
            stmt = select(IndexedFile).where(
                and_(*conditions)
            ).order_by(IndexedFile.filename).limit(limit)
        
        result = await self.session.execute(stmt)
        
        # Extract just the IndexedFile objects (not the score tuples)
        if query:
            return [row[0] for row in result.all()]
        else:
            return list(result.scalars().all())
    
    async def search_files_with_scores(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 100,
        min_similarity: float = 0.01
    ) -> list[tuple[IndexedFile, float]]:
        """
        Search indexed files by filename and return results with similarity scores.
        
        Uses trigram similarity for scoring without requiring exact substring matches.
        This method is designed for use in hybrid search where scores need to be 
        combined with other search methods.
        
        Args:
            user_id: User UUID
            query: Search query (required for scoring)
            file_type: Optional filter by file type ("video" or "image")
            limit: Maximum number of results to return
            min_similarity: Minimum trigram similarity score (default 0.05, very lenient)
        
        Returns:
            List of tuples (IndexedFile, similarity_score), ordered by score descending
        """
        if not query or not query.strip():
            return []
        
        conditions = [IndexedFile.user_id == user_id]
        
        # Add file type filter if provided
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)
        
        # Calculate similarity score for ranking using trigram similarity
        similarity_score = func.similarity(IndexedFile.filename, query)
        
        # Only require minimum similarity threshold (very lenient for hybrid search)
        # This allows semantic search to contribute even when text match is weak
        conditions.append(similarity_score >= min_similarity)
        
        stmt = select(IndexedFile, similarity_score.label("score")).where(
            and_(*conditions)
        ).order_by(desc("score"), IndexedFile.filename).limit(limit)

        with trace_vector_search("trigram_filename_search", metadata={"limit": limit}) as span:
            result = await self.session.execute(stmt)
            rows = result.all()
            results = [(row[0], float(row[1])) for row in rows]
            if span:
                span.update(metadata={"result_count": len(results)})
            return results
    
    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores (0-1 range)
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return 1.0 for all
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    # ==========================================================================
    # Vision Embedding Methods
    # ==========================================================================
    
    async def get_files_without_vision_embeddings(
        self,
        user_id: Optional[UUID] = None,
        limit: int = 100
    ) -> list[IndexedFile]:
        """
        Get files that don't have vision embeddings yet but have thumbnails.
        
        Args:
            user_id: Optional user UUID to filter by (if None, gets all users)
            limit: Maximum number of files to return
        
        Returns:
            List of IndexedFile records without vision embeddings
        """
        conditions = [
            IndexedFile.vision_embedding.is_(None),
            IndexedFile.thumbnail_url.isnot(None),
        ]
        
        if user_id:
            conditions.append(IndexedFile.user_id == user_id)
        
        stmt = select(IndexedFile).where(
            and_(*conditions)
        ).order_by(IndexedFile.created_at).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def update_vision_indexing_status(
        self,
        file_id: UUID,
        status: IndexingStatus,
        error_message: Optional[str] = None
    ) -> Optional[IndexedFile]:
        """
        Update the vision indexing status of a file.
        
        Args:
            file_id: IndexedFile UUID
            status: New vision indexing status
            error_message: Optional error message (for failed status)
        
        Returns:
            Updated IndexedFile or None if not found
        """
        stmt = select(IndexedFile).where(IndexedFile.id == file_id)
        result = await self.session.execute(stmt)
        indexed_file = result.scalar_one_or_none()
        
        if not indexed_file:
            return None
        
        indexed_file.vision_indexing_status = status.value
        indexed_file.updated_at = datetime.utcnow()
        
        if error_message:
            # Store error in the main error_message field with prefix
            if indexed_file.error_message:
                indexed_file.error_message = f"{indexed_file.error_message}\nVision: {error_message}"
            else:
                indexed_file.error_message = f"Vision: {error_message}"
        
        if status == IndexingStatus.COMPLETED:
            indexed_file.vision_indexed_at = datetime.utcnow()
        
        await self.session.commit()
        await self.session.refresh(indexed_file)
        
        return indexed_file
    
    async def generate_missing_vision_embeddings(
        self,
        user_id: Optional[UUID] = None,
        google_access_token: Optional[str] = None,
        batch_size: int = 50
    ) -> dict:
        """
        Generate vision embeddings for files that don't have them yet.
        
        Args:
            user_id: Optional user UUID to filter by (if None, processes all users)
            google_access_token: Optional Google OAuth token for fetching fresh thumbnail URLs
            batch_size: Number of files to process in each batch (smaller due to image downloads)
        
        Returns:
            Dict with processing statistics
        """
        vision_service = get_vision_embedding_service()
        
        if not vision_service.is_configured:
            logger.warning("Vision embedding service not configured")
            return {
                "success": False,
                "error": "Vision embedding service not configured",
                "processed": 0,
                "failed": 0,
            }
        
        stats = {
            "success": True,
            "processed": 0,
            "failed": 0,
            "total_found": 0,
        }
        
        # Get files without vision embeddings
        files = await self.get_files_without_vision_embeddings(user_id, limit=batch_size)
        stats["total_found"] = len(files)
        
        if not files:
            logger.info("No files found without vision embeddings")
            return stats
        
        # Extract thumbnail URLs and Drive file IDs
        thumbnail_urls = [f.thumbnail_url for f in files]
        drive_file_ids = [f.drive_file_id for f in files]
        
        # Generate vision embeddings in batch (with fresh thumbnail URLs if token provided)
        logger.info(f"Generating vision embeddings for {len(thumbnail_urls)} files")
        embeddings = await vision_service.generate_embeddings_batch(
            thumbnail_urls,
            drive_file_ids=drive_file_ids,
            google_access_token=google_access_token
        )
        
        # Update files with their embeddings
        for i, file in enumerate(files):
            embedding = embeddings[i]
            if embedding:
                file.vision_embedding = embedding
                file.vision_indexing_status = IndexingStatus.COMPLETED.value
                file.vision_indexed_at = datetime.utcnow()
                file.updated_at = datetime.utcnow()
                stats["processed"] += 1
            else:
                file.vision_indexing_status = IndexingStatus.FAILED.value
                file.updated_at = datetime.utcnow()
                stats["failed"] += 1
                logger.warning(f"Failed to generate vision embedding for file: {file.filename}")
        
        await self.session.commit()
        
        logger.info(
            f"Vision embedding generation complete: {stats['processed']} processed, "
            f"{stats['failed']} failed"
        )
        
        return stats
    
    async def vision_semantic_search(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.0
    ) -> list[tuple[IndexedFile, float]]:
        """
        Search indexed files using vision semantic similarity on vision embeddings.
        
        Uses cosine similarity to find files with visually similar thumbnails
        to the query (converted to embedding via CLIP text encoder).
        
        Args:
            user_id: User UUID
            query: Search query text (will be converted to CLIP embedding)
            file_type: Optional filter by file type ("video" or "image")
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0, where 1.0 is identical)
        
        Returns:
            List of tuples (IndexedFile, similarity_score), ordered by similarity descending
        """
        if not query or not query.strip():
            return []
        
        # Generate vision embedding for the query
        vision_service = get_vision_embedding_service()
        
        if not vision_service.is_configured:
            logger.warning("Vision embedding service not configured for vision search")
            return []
        
        # For CLIP, we use the text encoder to search images
        # This generates a text embedding that can be compared with image embeddings
        query_embedding = await vision_service.generate_text_embedding(query)
        
        if not query_embedding:
            logger.warning(f"Failed to generate vision embedding for query: {query}")
            return []
        
        return await self.vision_semantic_search_by_embedding(
            user_id=user_id,
            query_embedding=query_embedding,
            file_type=file_type,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
    
    async def vision_semantic_search_by_embedding(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.0
    ) -> list[tuple[IndexedFile, float]]:
        """
        Search indexed files using a pre-computed vision embedding.
        
        Args:
            user_id: User UUID
            query_embedding: Pre-computed CLIP embedding vector
            file_type: Optional filter by file type ("video" or "image")
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of tuples (IndexedFile, similarity_score), ordered by similarity descending
        """
        # Build conditions
        conditions = [
            IndexedFile.user_id == user_id,
            IndexedFile.vision_embedding.isnot(None),
        ]
        
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)
        
        # Use cosine distance for similarity search
        # pgvector uses <=> for cosine distance (lower is more similar)
        # We convert to similarity score: 1 - distance
        cosine_distance = IndexedFile.vision_embedding.cosine_distance(query_embedding)
        similarity_score = (1 - cosine_distance).label("similarity")
        
        stmt = select(IndexedFile, similarity_score).where(
            and_(*conditions)
        ).order_by(cosine_distance).limit(limit)

        with trace_vector_search("vision_semantic_search", metadata={"limit": limit}):
            result = await self.session.execute(stmt)
            rows = result.all()

        # Filter by similarity threshold and return results
        results = []
        for row in rows:
            file, score = row
            if score >= similarity_threshold:
                results.append((file, float(score)))

        return results

    async def _vision_search_video_frames(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, "VideoFrameEmbedding", float]]:
        """
        Search video_frame_embeddings by query embedding; join to indexed_files.
        Returns (IndexedFile, VideoFrameEmbedding, similarity_score).
        """
        conditions = [IndexedFile.user_id == user_id]
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)
        cosine_distance = VideoFrameEmbedding.embedding.cosine_distance(query_embedding)
        similarity_score = (1 - cosine_distance).label("similarity")
        stmt = (
            select(IndexedFile, VideoFrameEmbedding, similarity_score)
            .select_from(VideoFrameEmbedding)
            .join(IndexedFile, IndexedFile.id == VideoFrameEmbedding.video_id)
            .where(and_(*conditions))
            .order_by(cosine_distance)
            .limit(limit)
        )
        with trace_vector_search("vision_video_frame_search", metadata={"limit": limit}) as span:
            result = await self.session.execute(stmt)
            rows = result.all()
            out = []
            for row in rows:
                file, frame, score = row
                if score >= similarity_threshold:
                    out.append((file, frame, float(score)))
            if span:
                span.update(metadata={"result_count": len(out)})
            return out

    async def vision_semantic_search_unified(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, float, Optional[dict]]]:
        """
        Vision search over both indexed_files.vision_embedding and video_frame_embeddings.
        Returns list of (IndexedFile, similarity_score, matched_frame | None).
        matched_frame = { "frameImageUrl", "timeSeconds", "frameIndex" } when hit is a video frame.
        """
        vision_service = get_vision_embedding_service()
        if not vision_service.is_configured:
            return []
        query_embedding = await vision_service.generate_text_embedding(query)
        if not query_embedding:
            return []

        return await self.vision_semantic_search_unified_by_embedding(
            user_id=user_id,
            query_embedding=query_embedding,
            file_type=file_type,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    async def vision_semantic_search_unified_by_embedding(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 20,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, float, Optional[dict]]]:
        """
        Vision search (files + video frames) using a pre-computed query embedding.
        Returns list of (IndexedFile, similarity_score, matched_frame | None).
        """
        file_results = await self.vision_semantic_search_by_embedding(
            user_id=user_id,
            query_embedding=query_embedding,
            file_type=file_type,
            limit=limit * 2,
            similarity_threshold=similarity_threshold,
        )
        frame_results = await self._vision_search_video_frames(
            user_id=user_id,
            query_embedding=query_embedding,
            file_type=file_type,
            limit=limit * 2,
            similarity_threshold=similarity_threshold,
        )

        by_file_id: dict[UUID, tuple[IndexedFile, float, Optional[dict]]] = {}
        for f, score in file_results:
            by_file_id[f.id] = (f, score, None)
        for f, frame, score in frame_results:
            if f.id not in by_file_id or score > by_file_id[f.id][1]:
                by_file_id[f.id] = (
                    f,
                    score,
                    {
                        "frameImageUrl": get_blob_url_with_sas(frame.frame_image_url) if frame.frame_image_url else frame.frame_image_url,
                        "timeSeconds": frame.time_seconds,
                        "frameIndex": frame.frame_index,
                    },
                )

        merged = list(by_file_id.values())
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:limit]

    # ==========================================================================
    # Transcript Search Methods
    # ==========================================================================

    @staticmethod
    def _transcript_match_info(
        segment: VideoTranscriptSegment,
        query: Optional[str] = None,
    ) -> dict:
        """
        Build the matched-transcript payload returned with search results.

        When the segment has WhisperX word-level timestamps and a query token
        matches a word, startSeconds is refined to that word's timestamp so the
        deep link lands on the exact spoken word.
        """
        info = {
            "text": segment.text,
            "startSeconds": segment.start_seconds,
            "endSeconds": segment.end_seconds,
            "segmentIndex": segment.segment_index,
            "matchedWord": None,
        }
        if query and segment.words:
            tokens = {
                re.sub(r"[^\w']", "", t.lower())
                for t in query.split()
            } - {""}
            for word in segment.words:
                clean = re.sub(r"[^\w']", "", str(word.get("word", "")).lower())
                if clean and clean in tokens and word.get("start") is not None:
                    info["matchedWord"] = str(word["word"]).strip()
                    info["startSeconds"] = float(word["start"])
                    break
        return info

    async def transcript_lexical_search(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[tuple[IndexedFile, VideoTranscriptSegment, float]]:
        """
        Full-text search over video transcript segments (Postgres FTS).

        Returns at most one (best-ranked) segment per file:
        list of (IndexedFile, VideoTranscriptSegment, ts_rank score), ordered by rank.
        """
        if not query or not query.strip():
            return []
        if file_type == "image":
            return []

        tsvector = func.to_tsvector("english", VideoTranscriptSegment.text)
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(tsvector, tsquery).label("rank")

        stmt = (
            select(IndexedFile, VideoTranscriptSegment, rank)
            .select_from(VideoTranscriptSegment)
            .join(IndexedFile, IndexedFile.id == VideoTranscriptSegment.video_id)
            .where(
                and_(
                    IndexedFile.user_id == user_id,
                    tsvector.op("@@")(tsquery),
                )
            )
            .order_by(desc("rank"), VideoTranscriptSegment.start_seconds)
            .limit(limit * 4)
        )

        with trace_vector_search("transcript_lexical_search", metadata={"limit": limit}) as span:
            result = await self.session.execute(stmt)
            rows = result.all()
            seen: set[UUID] = set()
            results: list[tuple[IndexedFile, VideoTranscriptSegment, float]] = []
            for file, segment, score in rows:
                if file.id in seen:
                    continue
                seen.add(file.id)
                results.append((file, segment, float(score)))
                if len(results) >= limit:
                    break
            if span:
                span.update(metadata={"result_count": len(results)})
            return results

    async def transcript_semantic_search_by_embedding(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 50,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, VideoTranscriptSegment, float]]:
        """
        Semantic search over transcript segment embeddings (pgvector cosine).

        Returns at most one (most similar) segment per file:
        list of (IndexedFile, VideoTranscriptSegment, similarity), ordered by similarity.
        """
        if file_type == "image":
            return []

        cosine_distance = VideoTranscriptSegment.text_embedding.cosine_distance(query_embedding)
        similarity_score = (1 - cosine_distance).label("similarity")

        stmt = (
            select(IndexedFile, VideoTranscriptSegment, similarity_score)
            .select_from(VideoTranscriptSegment)
            .join(IndexedFile, IndexedFile.id == VideoTranscriptSegment.video_id)
            .where(
                and_(
                    IndexedFile.user_id == user_id,
                    VideoTranscriptSegment.text_embedding.isnot(None),
                )
            )
            .order_by(cosine_distance)
            .limit(limit * 4)
        )

        with trace_vector_search("transcript_semantic_search", metadata={"limit": limit}) as span:
            result = await self.session.execute(stmt)
            rows = result.all()
            seen: set[UUID] = set()
            results: list[tuple[IndexedFile, VideoTranscriptSegment, float]] = []
            for file, segment, score in rows:
                if score < similarity_threshold or file.id in seen:
                    continue
                seen.add(file.id)
                results.append((file, segment, float(score)))
                if len(results) >= limit:
                    break
            if span:
                span.update(metadata={"result_count": len(results)})
            return results

    # ==========================================================================
    # Thumbnail + caption search
    # ==========================================================================

    async def thumbnail_semantic_search_by_embedding(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 50,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, float]]:
        """Search poster thumbnail embeddings (pgvector cosine). One score per file."""
        conditions = [
            IndexedFile.user_id == user_id,
            IndexedFile.thumbnail_embedding.isnot(None),
        ]
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)

        cosine_distance = IndexedFile.thumbnail_embedding.cosine_distance(query_embedding)
        similarity_score = (1 - cosine_distance).label("similarity")
        stmt = (
            select(IndexedFile, similarity_score)
            .where(and_(*conditions))
            .order_by(cosine_distance)
            .limit(limit)
        )
        with trace_vector_search("thumbnail_semantic_search", metadata={"limit": limit}) as span:
            rows = (await self.session.execute(stmt)).all()
            results = [
                (file, float(score))
                for file, score in rows
                if score >= similarity_threshold
            ]
            if span:
                span.update(metadata={"result_count": len(results)})
            return results

    async def description_lexical_search(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[tuple[IndexedFile, float]]:
        """Full-text search over indexed_files.description (Instagram captions)."""
        if not query or not query.strip():
            return []

        tsvector = func.to_tsvector("english", IndexedFile.description)
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(tsvector, tsquery).label("rank")
        conditions = [
            IndexedFile.user_id == user_id,
            IndexedFile.description.isnot(None),
            IndexedFile.description != "",
            tsvector.op("@@")(tsquery),
        ]
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)

        stmt = (
            select(IndexedFile, rank)
            .where(and_(*conditions))
            .order_by(desc("rank"))
            .limit(limit)
        )
        with trace_vector_search("description_lexical_search", metadata={"limit": limit}) as span:
            rows = (await self.session.execute(stmt)).all()
            results = [(file, float(score)) for file, score in rows]
            if span:
                span.update(metadata={"result_count": len(results)})
            return results

    async def description_semantic_search_by_embedding(
        self,
        user_id: UUID,
        query_embedding: list[float],
        file_type: Optional[str] = None,
        limit: int = 50,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[IndexedFile, float]]:
        """Semantic search over caption/description embeddings."""
        conditions = [
            IndexedFile.user_id == user_id,
            IndexedFile.description_embedding.isnot(None),
        ]
        if file_type and file_type in ("video", "image"):
            conditions.append(IndexedFile.file_type == file_type)

        cosine_distance = IndexedFile.description_embedding.cosine_distance(query_embedding)
        similarity_score = (1 - cosine_distance).label("similarity")
        stmt = (
            select(IndexedFile, similarity_score)
            .where(and_(*conditions))
            .order_by(cosine_distance)
            .limit(limit)
        )
        with trace_vector_search("description_semantic_search", metadata={"limit": limit}) as span:
            rows = (await self.session.execute(stmt)).all()
            results = [
                (file, float(score))
                for file, score in rows
                if score >= similarity_threshold
            ]
            if span:
                span.update(metadata={"result_count": len(results)})
            return results

    # ==========================================================================
    # Hybrid Search (Reciprocal Rank Fusion)
    # ==========================================================================

    RRF_K = 60

    async def hybrid_search_rrf(
        self,
        user_id: UUID,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 50,
        rrf_k: int = RRF_K,
    ) -> list[dict]:
        """
        Hybrid search fusing retrieval legs with Reciprocal Rank Fusion:

        1. Filename trigram similarity (lexical)
        2. Poster thumbnail embeddings (semantic)
        3. Video frame embeddings (semantic, with timestamps)
        4. Caption/description full-text + embeddings
        5. Transcript full-text + segment embeddings (with timestamps)

        Each leg contributes 1 / (rrf_k + rank) per file. Thumbnail and frame
        scores are kept separate; vision_score is their sum for older callers.

        Returns dicts ordered by hybrid_score desc.
        """
        if not query or not query.strip():
            return []
        fetch_limit = max(limit * 2, 50)

        filename_results = await self.search_files_with_scores(
            user_id=user_id,
            query=query,
            file_type=file_type,
            limit=fetch_limit,
        )
        caption_lexical_results = await self.description_lexical_search(
            user_id=user_id,
            query=query,
            file_type=file_type,
            limit=fetch_limit,
        )
        transcript_lexical_results = await self.transcript_lexical_search(
            user_id=user_id,
            query=query,
            file_type=file_type,
            limit=fetch_limit,
        )

        vision_service = get_vision_embedding_service()
        query_embedding: Optional[list[float]] = None
        if vision_service.is_configured:
            query_embedding = await vision_service.generate_text_embedding(query)

        thumbnail_results: list[tuple[IndexedFile, float]] = []
        frame_results: list[tuple[IndexedFile, VideoFrameEmbedding, float]] = []
        caption_semantic_results: list[tuple[IndexedFile, float]] = []
        transcript_semantic_results: list[tuple[IndexedFile, VideoTranscriptSegment, float]] = []
        if query_embedding:
            thumbnail_results = await self.thumbnail_semantic_search_by_embedding(
                user_id=user_id,
                query_embedding=query_embedding,
                file_type=file_type,
                limit=fetch_limit,
            )
            frame_results = await self._vision_search_video_frames(
                user_id=user_id,
                query_embedding=query_embedding,
                file_type=file_type,
                limit=fetch_limit,
            )
            seen_frame_files: set[UUID] = set()
            unique_frames: list[tuple[IndexedFile, VideoFrameEmbedding, float]] = []
            for item in frame_results:
                if item[0].id in seen_frame_files:
                    continue
                seen_frame_files.add(item[0].id)
                unique_frames.append(item)
            frame_results = unique_frames
            caption_semantic_results = await self.description_semantic_search_by_embedding(
                user_id=user_id,
                query_embedding=query_embedding,
                file_type=file_type,
                limit=fetch_limit,
            )
            transcript_semantic_results = await self.transcript_semantic_search_by_embedding(
                user_id=user_id,
                query_embedding=query_embedding,
                file_type=file_type,
                limit=fetch_limit,
            )

        files_by_id: dict[UUID, IndexedFile] = {}
        scores: dict[UUID, dict[str, float]] = {}
        matched_frames: dict[UUID, dict] = {}
        matched_thumbnails: dict[UUID, dict] = {}
        matched_captions: dict[UUID, tuple[float, dict]] = {}
        matched_transcripts: dict[UUID, tuple[float, dict]] = {}

        def leg_scores(file_id: UUID) -> dict[str, float]:
            if file_id not in scores:
                scores[file_id] = {
                    "text": 0.0,
                    "thumbnail": 0.0,
                    "frame": 0.0,
                    "caption": 0.0,
                    "transcript": 0.0,
                }
            return scores[file_id]

        for rank, (file, _score) in enumerate(filename_results):
            files_by_id[file.id] = file
            leg_scores(file.id)["text"] += 1.0 / (rrf_k + rank + 1)

        for rank, (file, _score) in enumerate(thumbnail_results):
            files_by_id[file.id] = file
            leg_scores(file.id)["thumbnail"] += 1.0 / (rrf_k + rank + 1)
            if file.blob_thumbnail_url:
                matched_thumbnails[file.id] = {
                    "thumbnailImageUrl": get_blob_url_with_sas(file.blob_thumbnail_url),
                }

        for rank, (file, frame, _score) in enumerate(frame_results):
            files_by_id[file.id] = file
            contribution = 1.0 / (rrf_k + rank + 1)
            existing_frame = matched_frames.get(file.id)
            # Keep the best-ranked frame hit per file
            if existing_frame is None:
                leg_scores(file.id)["frame"] += contribution
                matched_frames[file.id] = {
                    "frameImageUrl": (
                        get_blob_url_with_sas(frame.frame_image_url)
                        if frame.frame_image_url
                        else frame.frame_image_url
                    ),
                    "timeSeconds": frame.time_seconds,
                    "frameIndex": frame.frame_index,
                }

        for caption_results in (caption_lexical_results, caption_semantic_results):
            for rank, (file, _score) in enumerate(caption_results):
                files_by_id[file.id] = file
                contribution = 1.0 / (rrf_k + rank + 1)
                leg_scores(file.id)["caption"] += contribution
                existing = matched_captions.get(file.id)
                if existing is None or contribution > existing[0]:
                    matched_captions[file.id] = (
                        contribution,
                        {"text": file.description or ""},
                    )

        for transcript_results in (transcript_lexical_results, transcript_semantic_results):
            for rank, (file, segment, _score) in enumerate(transcript_results):
                files_by_id[file.id] = file
                contribution = 1.0 / (rrf_k + rank + 1)
                leg_scores(file.id)["transcript"] += contribution
                existing = matched_transcripts.get(file.id)
                if existing is None or contribution > existing[0]:
                    matched_transcripts[file.id] = (
                        contribution,
                        self._transcript_match_info(segment, query=query),
                    )

        combined = [
            {
                "file": files_by_id[file_id],
                "hybrid_score": sum(legs.values()),
                "text_score": legs["text"],
                "thumbnail_score": legs["thumbnail"],
                "frame_score": legs["frame"],
                "caption_score": legs["caption"],
                "transcript_score": legs["transcript"],
                "vision_score": legs["thumbnail"] + legs["frame"],
                "matched_thumbnail": matched_thumbnails.get(file_id),
                "matched_frame": matched_frames.get(file_id),
                "matched_caption": (
                    matched_captions[file_id][1]
                    if file_id in matched_captions
                    else None
                ),
                "matched_transcript": (
                    matched_transcripts[file_id][1]
                    if file_id in matched_transcripts
                    else None
                ),
            }
            for file_id, legs in scores.items()
        ]
        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:limit]

    async def get_full_transcript(
        self,
        video_id: UUID,
        user_id: UUID,
    ) -> Optional[dict]:
        """
        Fetch a video's full transcript: all segments ordered by segment_index.

        Returns {"file": IndexedFile, "segments": list[VideoTranscriptSegment]},
        or None when the video doesn't exist or isn't owned by user_id.
        """
        file_row = (
            await self.session.execute(
                select(IndexedFile).where(
                    IndexedFile.id == video_id,
                    IndexedFile.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        if not file_row:
            return None

        segments = (
            await self.session.execute(
                select(VideoTranscriptSegment)
                .where(VideoTranscriptSegment.video_id == video_id)
                .order_by(VideoTranscriptSegment.segment_index)
            )
        ).scalars().all()
        return {"file": file_row, "segments": list(segments)}

    async def get_vision_indexing_stats(
        self,
        user_id: UUID,
        folder_id: Optional[str] = None
    ) -> dict:
        """
        Get vision indexing statistics for a user (optionally filtered by folder).
        
        Args:
            user_id: User UUID
            folder_id: Optional folder ID to filter by
        
        Returns:
            Dict with counts by vision indexing status
        """
        base_condition = IndexedFile.user_id == user_id
        if folder_id:
            base_condition = and_(base_condition, IndexedFile.folder_id == folder_id)
        
        stmt = select(IndexedFile).where(base_condition)
        result = await self.session.execute(stmt)
        files = list(result.scalars().all())
        
        stats = {
            "total": len(files),
            "with_thumbnail": 0,
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "not_started": 0,
        }
        
        for f in files:
            if f.thumbnail_url:
                stats["with_thumbnail"] += 1
            
            if f.vision_indexing_status is None:
                stats["not_started"] += 1
            elif f.vision_indexing_status == IndexingStatus.PENDING.value:
                stats["pending"] += 1
            elif f.vision_indexing_status == IndexingStatus.PROCESSING.value:
                stats["processing"] += 1
            elif f.vision_indexing_status == IndexingStatus.COMPLETED.value:
                stats["completed"] += 1
            elif f.vision_indexing_status == IndexingStatus.FAILED.value:
                stats["failed"] += 1
        
        return stats