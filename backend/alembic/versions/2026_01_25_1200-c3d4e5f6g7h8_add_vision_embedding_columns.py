"""add vision_embedding columns with pgvector

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-01-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6g7h8'
down_revision: Union[str, None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add vision_embedding column for storing 768-dimensional CLIP embeddings
    op.execute('''
        ALTER TABLE indexed_files 
        ADD COLUMN vision_embedding vector(768)
    ''')
    
    # Add vision_indexing_status column to track vision embedding generation
    op.add_column(
        'indexed_files',
        sa.Column('vision_indexing_status', sa.String(20), nullable=True)
    )
    
    # Add vision_indexed_at column to track when vision embedding was generated
    op.add_column(
        'indexed_files',
        sa.Column('vision_indexed_at', sa.DateTime(), nullable=True)
    )
    
    # Create HNSW index for fast approximate nearest neighbor search on vision embeddings
    op.execute('''
        CREATE INDEX ix_indexed_files_vision_embedding_hnsw 
        ON indexed_files 
        USING hnsw (vision_embedding vector_cosine_ops)
    ''')
    
    # Create index for filtering by vision indexing status
    op.create_index(
        'ix_indexed_files_user_vision_status',
        'indexed_files',
        ['user_id', 'vision_indexing_status']
    )


def downgrade() -> None:
    # Drop the indexes
    op.drop_index('ix_indexed_files_user_vision_status', table_name='indexed_files')
    op.execute('DROP INDEX IF EXISTS ix_indexed_files_vision_embedding_hnsw')
    
    # Drop the columns
    op.drop_column('indexed_files', 'vision_indexed_at')
    op.drop_column('indexed_files', 'vision_indexing_status')
    op.execute('ALTER TABLE indexed_files DROP COLUMN IF EXISTS vision_embedding')
