"""VideoTranscriptSegment model for per-segment video transcripts."""

import uuid

from sqlalchemy import Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.database import Base


class VideoTranscriptSegment(Base):
    """
    One row per transcribed speech segment (WhisperX output) of a video.
    Linked to indexed_files via video_id. Holds the segment text, its
    start/end timestamps, per-word timestamps (WhisperX alignment), and
    a text embedding for semantic search.
    """
    __tablename__ = "video_transcript_segments"

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
    segment_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    start_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    end_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    # Word-level timestamps from WhisperX alignment:
    # [{"word": str, "start": float, "end": float}, ...] (start/end may be absent
    # for words the aligner could not place)
    words: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    text_embedding: Mapped[list | None] = mapped_column(
        Vector(768),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "video_id",
            "segment_index",
            name="uq_video_transcript_video_id_segment_index",
        ),
        Index("ix_video_transcript_segments_video_id", "video_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<VideoTranscriptSegment video_id={self.video_id} "
            f"segment_index={self.segment_index} [{self.start_seconds:.1f}-{self.end_seconds:.1f}s]>"
        )
