"""VideoFrameEmbedding model for per-frame video embeddings."""

import uuid

from sqlalchemy import ForeignKey, Index, Integer, Float, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class VideoFrameEmbedding(Base):
    """
    One row per indexed video frame (e.g. every 5th frame).
    Linked to indexed_files via video_id for videos.
    """
    __tablename__ = "video_frame_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("indexed_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    time_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    embedding: Mapped[list] = mapped_column(
        Vector(768),
        nullable=False,
    )
    frame_image_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "video_id",
            "frame_index",
            name="uq_video_frame_video_id_frame_index",
        ),
        Index("ix_video_frame_embeddings_video_id", "video_id"),
    )

    def __repr__(self) -> str:
        return f"<VideoFrameEmbedding video_id={self.video_id} frame_index={self.frame_index}>"
