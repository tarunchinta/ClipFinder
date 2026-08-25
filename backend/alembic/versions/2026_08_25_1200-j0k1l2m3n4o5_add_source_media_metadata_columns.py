"""add source media metadata columns to indexed_files

Revision ID: j0k1l2m3n4o5
Revises: i9j0k1l2m3n4
Create Date: 2026-08-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "j0k1l2m3n4o5"
down_revision: Union[str, None] = "i9j0k1l2m3n4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indexed_files",
        sa.Column("title", sa.String(500), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("channel", sa.String(255), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("uploader", sa.String(255), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("uploader_id", sa.String(255), nullable=True),
    )
    op.add_column(
        "indexed_files",
        sa.Column("published_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("indexed_files", "published_at")
    op.drop_column("indexed_files", "uploader_id")
    op.drop_column("indexed_files", "uploader")
    op.drop_column("indexed_files", "channel")
    op.drop_column("indexed_files", "description")
    op.drop_column("indexed_files", "title")
