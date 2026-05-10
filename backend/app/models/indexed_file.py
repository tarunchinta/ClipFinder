"""IndexedFile model for tracking files selected for indexing."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from sqlalchemy import (
    Column, String, BigInteger, Integer, Float, Text, DateTime,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class IndexingStatus(str, Enum):
    """Status of file indexing process."""
    PENDING = "pending"        # File queued for indexing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"    # Successfully indexed
    FAILED = "failed"          # Indexing failed


class IndexedFile(Base):
    """
    Model for tracking files selected for indexing from Google Drive.
    
    Stores metadata about each file and tracks indexing progress.
    """
    __tablename__ = "indexed_files"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to User
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Google account identifier (for multi-account support)
    google_account_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    
    # Google Drive identifiers
    folder_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    drive_file_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    
    # File metadata
    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )
    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False  # "video" or "image"
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False
    )
    
    # Media-specific metadata
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True  # Only for videos
    )
    width: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    height: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Drive timestamps and URLs
    modified_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    thumbnail_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    drive_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    
    # Indexing status
    indexing_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=IndexingStatus.PENDING.value
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Filename embedding for semantic search (1536 dimensions for text-embedding-3-small)
    filename_embedding: Mapped[Optional[list]] = mapped_column(
        Vector(1536),
        nullable=True
    )
    
    # Vision embedding for image-based semantic search (768 dimensions for CLIP)
    vision_embedding: Mapped[Optional[list]] = mapped_column(
        Vector(768),
        nullable=True
    )
    
    # Vision indexing status tracking
    vision_indexing_status: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        default=None
    )
    
    vision_indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    
    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint: one entry per file per user
        UniqueConstraint('user_id', 'drive_file_id', name='uq_user_drive_file'),
        # Index for folder queries (re-indexing)
        Index('ix_indexed_files_user_folder', 'user_id', 'folder_id'),
        # Index for status filtering
        Index('ix_indexed_files_user_status', 'user_id', 'indexing_status'),
    )
    
    def __repr__(self) -> str:
        return f"<IndexedFile {self.filename} ({self.indexing_status})>"
