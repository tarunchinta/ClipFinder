"""add frame completion tracking and unique video frame constraint

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-06-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5f6g7h8i9j0"
down_revision: Union[str, None] = "d4e5f6g7h8i9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indexed_files",
        sa.Column("frames_total", sa.Integer(), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column(
            "frames_completed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "indexed_files",
        sa.Column(
            "frames_failed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.alter_column("indexed_files", "frames_completed", server_default=None)
    op.alter_column("indexed_files", "frames_failed", server_default=None)

    # Remove duplicate (video_id, frame_index) rows before adding the constraint.
    op.execute("""
        DELETE FROM video_frame_embeddings a
        USING video_frame_embeddings b
        WHERE a.video_id = b.video_id
          AND a.frame_index = b.frame_index
          AND a.id > b.id
    """)
    op.create_unique_constraint(
        "uq_video_frame_video_id_frame_index",
        "video_frame_embeddings",
        ["video_id", "frame_index"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_video_frame_video_id_frame_index",
        "video_frame_embeddings",
        type_="unique",
    )
    op.drop_column("indexed_files", "frames_failed")
    op.drop_column("indexed_files", "frames_completed")
    op.drop_column("indexed_files", "frames_total")
