"""add filename_embedding column with pgvector

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-01-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension for vector embeddings
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add filename_embedding column for storing 1536-dimensional embeddings
    # text-embedding-3-small produces 1536-dimensional vectors
    op.execute('''
        ALTER TABLE indexed_files 
        ADD COLUMN filename_embedding vector(1536)
    ''')
    
    # Create HNSW index for fast approximate nearest neighbor search
    # HNSW (Hierarchical Navigable Small World) provides excellent query performance
    op.execute('''
        CREATE INDEX ix_indexed_files_filename_embedding_hnsw 
        ON indexed_files 
        USING hnsw (filename_embedding vector_cosine_ops)
    ''')


def downgrade() -> None:
    # Drop the HNSW index
    op.execute('DROP INDEX IF EXISTS ix_indexed_files_filename_embedding_hnsw')
    
    # Drop the embedding column
    op.execute('ALTER TABLE indexed_files DROP COLUMN IF EXISTS filename_embedding')
    
    # Note: We don't drop the vector extension as other tables might use it
