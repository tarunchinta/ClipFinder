"""add instagram source columns to indexed_files

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2026-07-11 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "i9j0k1l2m3n4"
down_revision: Union[str, None] = "h8i9j0k1l2m3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indexed_files",
        sa.Column("source_type", sa.String(20), nullable=False, server_default="drive"),
    )
    op.add_column(
        "indexed_files",
        sa.Column("source_url", sa.String(1000), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("blob_video_url", sa.String(1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("indexed_files", "blob_video_url")
    op.drop_column("indexed_files", "source_url")
    op.drop_column("indexed_files", "source_type")
