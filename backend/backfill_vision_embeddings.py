"""
Script to backfill vision embeddings for all records in indexed_files table.

This script generates Gemini Embedding 2 vectors from thumbnail images for files
that don't have vision embeddings yet.

Usage:
    cd backend
    python backfill_vision_embeddings.py [batch_size]

Make sure your .env file has Gemini API credentials set:
    - GEMINI_API_KEY (or GOOGLE_AI_VISION_API_KEY)
    - GEMINI_EMBEDDING_MODEL (optional, defaults to gemini-embedding-2)
    - GEMINI_EMBEDDING_DIMENSION (optional, defaults to 768)

Note: This script requires users to have valid Google access tokens stored
in the database to fetch fresh thumbnail URLs from Google Drive.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select, func
from app.database import async_session_maker
from app.services.indexing import IndexingService
from app.services.vision_embedding import get_vision_embedding_service
from app.services.google_auth import get_valid_access_token
from app.models.user import User
from app.models.indexed_file import IndexedFile


async def get_users_with_pending_vision_embeddings():
    """Get all users who have files needing vision embeddings."""
    async with async_session_maker() as session:
        # Get users who have indexed files without vision embeddings
        stmt = select(User).where(
            User.id.in_(
                select(IndexedFile.user_id).where(
                    IndexedFile.vision_embedding.is_(None),
                    IndexedFile.thumbnail_url.isnot(None)
                ).distinct()
            )
        )
        result = await session.execute(stmt)
        # Need unique() because User model has joined eager loads (oauth_accounts)
        return list(result.unique().scalars().all())


async def backfill_user_vision_embeddings(user_id, google_access_token: str, batch_size: int = 50):
    """
    Backfill vision embeddings for a specific user.
    
    Args:
        user_id: User UUID
        google_access_token: User's Google OAuth access token
        batch_size: Number of files to process in each batch
        
    Returns:
        Tuple of (total_processed, total_failed)
    """
    total_processed = 0
    total_failed = 0
    batch_num = 0
    
    while True:
        batch_num += 1
        
        async with async_session_maker() as session:
            service = IndexingService(session)
            stats = await service.generate_missing_vision_embeddings(
                user_id=user_id,
                google_access_token=google_access_token,
                batch_size=batch_size
            )
        
        if not stats.get("success", False):
            print(f"    ERROR: {stats.get('error', 'Unknown error')}")
            break
        
        found = stats.get("total_found", 0)
        processed = stats.get("processed", 0)
        failed = stats.get("failed", 0)
        
        total_processed += processed
        total_failed += failed
        
        if batch_num == 1 or batch_num % 5 == 0:
            print(f"    Batch {batch_num}: processed {processed}, failed {failed}")
        
        # If no files found, we're done with this user
        if found == 0:
            break
    
    return total_processed, total_failed


async def backfill_all_vision_embeddings(batch_size: int = 50):
    """
    Backfill vision embeddings for all records that don't have them yet.
    
    Processes users one at a time, using their stored Google access tokens
    to fetch fresh thumbnail URLs from Google Drive.
    
    Args:
        batch_size: Number of files to process in each batch (default: 50)
    """
    # Check if vision embedding service is configured
    vision_service = get_vision_embedding_service()
    if not vision_service.is_configured:
        print("ERROR: Gemini Embedding 2 credentials not configured in .env file.")
        print("Required environment variables:")
        print("  - GEMINI_API_KEY (or GOOGLE_AI_VISION_API_KEY)")
        print("  - GEMINI_EMBEDDING_MODEL (optional, defaults to gemini-embedding-2)")
        print("  - GEMINI_EMBEDDING_DIMENSION (optional, defaults to 768)")
        return
    
    print("Starting vision embedding backfill process...")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Get users with pending vision embeddings
    users = await get_users_with_pending_vision_embeddings()
    
    if not users:
        print("No users with pending vision embeddings found.")
        return
    
    print(f"Found {len(users)} users with files needing vision embeddings")
    
    total_processed = 0
    total_failed = 0
    users_without_token = 0
    
    for i, user in enumerate(users):
        print(f"\n[{i+1}/{len(users)}] Processing user: {user.email}")
        
        if not user.google_access_token and not user.google_refresh_token:
            print(f"  WARNING: User has no Google tokens, skipping")
            users_without_token += 1
            continue
        
        # Get a valid (possibly refreshed) access token
        async with async_session_maker() as session:
            access_token = await get_valid_access_token(
                user_id=user.id,
                access_token=user.google_access_token,
                refresh_token=user.google_refresh_token,
                expires_at=user.google_token_expires_at,
                session=session
            )
        
        if not access_token:
            print(f"  WARNING: Could not get valid access token, skipping")
            users_without_token += 1
            continue
        
        processed, failed = await backfill_user_vision_embeddings(
            user_id=user.id,
            google_access_token=access_token,
            batch_size=batch_size
        )
        
        total_processed += processed
        total_failed += failed
        print(f"  User complete: {processed} processed, {failed} failed")
    
    print("-" * 50)
    print(f"Vision embedding backfill complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    if users_without_token > 0:
        print(f"Users skipped (no token): {users_without_token}")


async def show_stats():
    """Show current vision embedding statistics."""
    print("Fetching vision embedding statistics...")
    print("-" * 50)
    
    async with async_session_maker() as session:
        service = IndexingService(session)
        
        # Get stats for all users (pass None to get_vision_indexing_stats would need modification)
        # For now, let's count files directly
        from sqlalchemy import select, func
        from app.models.indexed_file import IndexedFile
        
        # Total files
        result = await session.execute(select(func.count(IndexedFile.id)))
        total = result.scalar()
        
        # Files with thumbnails
        result = await session.execute(
            select(func.count(IndexedFile.id)).where(IndexedFile.thumbnail_url.isnot(None))
        )
        with_thumbnail = result.scalar()
        
        # Files with vision embeddings
        result = await session.execute(
            select(func.count(IndexedFile.id)).where(IndexedFile.vision_embedding.isnot(None))
        )
        with_embedding = result.scalar()
        
        # Files needing embeddings (have thumbnail but no embedding)
        result = await session.execute(
            select(func.count(IndexedFile.id)).where(
                IndexedFile.thumbnail_url.isnot(None),
                IndexedFile.vision_embedding.is_(None)
            )
        )
        needs_embedding = result.scalar()
        
        print(f"Total indexed files: {total}")
        print(f"Files with thumbnails: {with_thumbnail}")
        print(f"Files with vision embeddings: {with_embedding}")
        print(f"Files needing vision embeddings: {needs_embedding}")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            asyncio.run(show_stats())
            sys.exit(0)
        
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid batch size: {sys.argv[1]}")
            print("Usage: python backfill_vision_embeddings.py [batch_size]")
            print("       python backfill_vision_embeddings.py --stats")
            sys.exit(1)
    else:
        batch_size = 50
    
    asyncio.run(backfill_all_vision_embeddings(batch_size))
