"""
Script to backfill filename embeddings for all records in indexed_files table.

Usage:
    cd backend
    python backfill_embeddings.py

Make sure your .env file has the Azure OpenAI credentials set:
    - AZURE_OPENAI_ENDPOINT_SAMPLE_FULL
    - AZURE_OPENAI_API_KEY
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.database import async_session_maker
from app.services.indexing import IndexingService
from app.services.embedding import get_embedding_service


async def backfill_all_embeddings(batch_size: int = 100):
    """
    Backfill embeddings for all records that don't have them yet.
    
    Processes files in batches until all files have embeddings.
    """
    # Check if embedding service is configured
    embedding_service = get_embedding_service()
    if not embedding_service.is_configured:
        print("ERROR: Azure OpenAI credentials not configured in .env file.")
        print("Required environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT_SAMPLE_FULL")
        print("  - AZURE_OPENAI_API_KEY")
        return
    
    print("Starting embedding backfill process...")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    total_processed = 0
    total_failed = 0
    batch_num = 0
    
    while True:
        batch_num += 1
        print(f"\nProcessing batch {batch_num}...")
        
        async with async_session_maker() as session:
            service = IndexingService(session)
            stats = await service.generate_missing_embeddings(
                user_id=None,  # Process all users
                batch_size=batch_size
            )
        
        if not stats.get("success", False):
            print(f"ERROR: {stats.get('error', 'Unknown error')}")
            break
        
        found = stats.get("total_found", 0)
        processed = stats.get("processed", 0)
        failed = stats.get("failed", 0)
        
        total_processed += processed
        total_failed += failed
        
        print(f"  Found: {found} files without embeddings")
        print(f"  Processed: {processed}")
        print(f"  Failed: {failed}")
        
        # If no files found, we're done
        if found == 0:
            print("\nNo more files to process.")
            break
        
        # If we processed fewer than found, there might be more
        # Continue until we find no files without embeddings
    
    print("-" * 50)
    print(f"Backfill complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")


if __name__ == "__main__":
    # Parse optional batch size argument
    batch_size = 100
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid batch size: {sys.argv[1]}")
            print("Usage: python backfill_embeddings.py [batch_size]")
            sys.exit(1)
    
    asyncio.run(backfill_all_embeddings(batch_size))
