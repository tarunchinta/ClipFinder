"""add video_frame_embeddings table with pgvector

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-02-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6g7h8i9'
down_revision: Union[str, None] = 'c3d4e5f6g7h8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'video_frame_embeddings',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('video_id', sa.Uuid(), nullable=False),
        sa.Column('frame_index', sa.Integer(), nullable=False),
        sa.Column('time_seconds', sa.Float(), nullable=False),
        sa.Column('frame_image_url', sa.String(1000), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['indexed_files.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.execute('''
        ALTER TABLE video_frame_embeddings
        ADD COLUMN embedding vector(768)
    ''')
    op.create_index(
        'ix_video_frame_embeddings_video_id',
        'video_frame_embeddings',
        ['video_id'],
    )
    op.execute('''
        CREATE INDEX ix_video_frame_embeddings_embedding_hnsw
        ON video_frame_embeddings
        USING hnsw (embedding vector_cosine_ops)
    ''')


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS ix_video_frame_embeddings_embedding_hnsw')
    op.drop_index('ix_video_frame_embeddings_video_id', table_name='video_frame_embeddings')
    op.drop_table('video_frame_embeddings')
