"""add video_transcript_segments table and transcript_status column

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-07-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g7h8i9j0k1l2"
down_revision: Union[str, None] = "f6g7h8i9j0k1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indexed_files",
        sa.Column("transcript_status", sa.String(20), nullable=True),
    )

    op.create_table(
        "video_transcript_segments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["indexed_files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "video_id",
            "segment_index",
            name="uq_video_transcript_video_id_segment_index",
        ),
    )
    op.execute("""
        ALTER TABLE video_transcript_segments
        ADD COLUMN text_embedding vector(768)
    """)
    op.create_index(
        "ix_video_transcript_segments_video_id",
        "video_transcript_segments",
        ["video_id"],
    )
    op.execute("""
        CREATE INDEX ix_video_transcript_segments_embedding_hnsw
        ON video_transcript_segments
        USING hnsw (text_embedding vector_cosine_ops)
    """)
    op.execute("""
        CREATE INDEX ix_video_transcript_segments_text_fts
        ON video_transcript_segments
        USING gin (to_tsvector('english', text))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_video_transcript_segments_text_fts")
    op.execute("DROP INDEX IF EXISTS ix_video_transcript_segments_embedding_hnsw")
    op.drop_index(
        "ix_video_transcript_segments_video_id",
        table_name="video_transcript_segments",
    )
    op.drop_table("video_transcript_segments")
    op.drop_column("indexed_files", "transcript_status")
