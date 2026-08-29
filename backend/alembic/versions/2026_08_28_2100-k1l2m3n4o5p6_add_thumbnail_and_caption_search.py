"""add thumbnail blob/embedding and caption search indexes

Revision ID: k1l2m3n4o5p6
Revises: j0k1l2m3n4o5
Create Date: 2026-08-28 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "k1l2m3n4o5p6"
down_revision: Union[str, None] = "j0k1l2m3n4o5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indexed_files",
        sa.Column("blob_thumbnail_url", sa.String(1000), nullable=True),
    )
    op.execute("""
        ALTER TABLE indexed_files
        ADD COLUMN thumbnail_embedding vector(768)
    """)
    op.execute("""
        ALTER TABLE indexed_files
        ADD COLUMN description_embedding vector(768)
    """)
    op.execute("""
        CREATE INDEX ix_indexed_files_thumbnail_embedding_hnsw
        ON indexed_files
        USING hnsw (thumbnail_embedding vector_cosine_ops)
    """)
    op.execute("""
        CREATE INDEX ix_indexed_files_description_embedding_hnsw
        ON indexed_files
        USING hnsw (description_embedding vector_cosine_ops)
    """)
    op.execute("""
        CREATE INDEX ix_indexed_files_description_fts
        ON indexed_files
        USING gin (to_tsvector('english', description))
        WHERE description IS NOT NULL AND description <> ''
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_indexed_files_description_fts")
    op.execute("DROP INDEX IF EXISTS ix_indexed_files_description_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_indexed_files_thumbnail_embedding_hnsw")
    op.execute("ALTER TABLE indexed_files DROP COLUMN IF EXISTS description_embedding")
    op.execute("ALTER TABLE indexed_files DROP COLUMN IF EXISTS thumbnail_embedding")
    op.drop_column("indexed_files", "blob_thumbnail_url")
