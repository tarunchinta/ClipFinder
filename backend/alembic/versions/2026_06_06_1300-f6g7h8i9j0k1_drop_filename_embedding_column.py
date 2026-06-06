"""drop filename_embedding column

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-06-06 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = "f6g7h8i9j0k1"
down_revision: Union[str, None] = "e5f6g7h8i9j0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_indexed_files_filename_embedding_hnsw")
    op.execute("ALTER TABLE indexed_files DROP COLUMN IF EXISTS filename_embedding")


def downgrade() -> None:
    op.execute("""
        ALTER TABLE indexed_files
        ADD COLUMN filename_embedding vector(1536)
    """)
    op.execute("""
        CREATE INDEX ix_indexed_files_filename_embedding_hnsw
        ON indexed_files
        USING hnsw (filename_embedding vector_cosine_ops)
    """)
