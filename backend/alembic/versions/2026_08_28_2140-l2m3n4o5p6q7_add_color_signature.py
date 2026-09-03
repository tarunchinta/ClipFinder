"""add per-file color signature columns

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-08-28 21:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "l2m3n4o5p6q7"
down_revision: Union[str, None] = "k1l2m3n4o5p6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE indexed_files
        ADD COLUMN color_histogram vector(256)
    """)
    op.add_column(
        "indexed_files",
        sa.Column("color_palette", postgresql.JSONB(), nullable=True),
    )
    op.add_column("indexed_files", sa.Column("color_mean_l", sa.Float(), nullable=True))
    op.add_column("indexed_files", sa.Column("color_std_l", sa.Float(), nullable=True))
    op.add_column("indexed_files", sa.Column("color_mean_a", sa.Float(), nullable=True))
    op.add_column("indexed_files", sa.Column("color_mean_b", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("indexed_files", "color_mean_b")
    op.drop_column("indexed_files", "color_mean_a")
    op.drop_column("indexed_files", "color_std_l")
    op.drop_column("indexed_files", "color_mean_l")
    op.drop_column("indexed_files", "color_palette")
    op.execute("ALTER TABLE indexed_files DROP COLUMN IF EXISTS color_histogram")
