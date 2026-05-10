"""add pg_trgm extension and filename trigram index

Revision ID: a1b2c3d4e5f6
Revises: 83d3b5b95773
Create Date: 2026-01-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '83d3b5b95773'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pg_trgm extension for trigram similarity searches
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    
    # Create GIN trigram index on filename for fast similarity queries
    op.execute('''
        CREATE INDEX ix_indexed_files_filename_trgm 
        ON indexed_files 
        USING GIN (filename gin_trgm_ops)
    ''')


def downgrade() -> None:
    # Drop the trigram index
    op.execute('DROP INDEX IF EXISTS ix_indexed_files_filename_trgm')
    
    # Note: We don't drop the pg_trgm extension as other tables might use it


