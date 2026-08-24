"""Video audio transcription with WhisperX for word-level transcript search.

Runs during video indexing: extracts audio with ffmpeg, transcribes locally on CPU
with WhisperX (batched Whisper ASR + wav2vec2 forced alignment for word-level
timestamps), and stores per-segment text + segment/word timestamps (plus Gemini
text embeddings for semantic search) in video_transcript_segments.
"""

import asyncio
import functools
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_maker
from app.models.indexed_file import IndexedFile, IndexingStatus
from app.models.video_transcript_segment import VideoTranscriptSegment
from app.services.vision_embedding import get_vision_embedding_service

logger = logging.getLogger(__name__)

# Whisper expects 16 kHz mono PCM input
AUDIO_SAMPLE_RATE = 16000

# WhisperX batched inference size (CPU)
WHISPERX_BATCH_SIZE = 8

# Cached WhisperX models (loaded once per process; ASR + one align model per language)
_asr_model = None
_align_models: dict[str, tuple] = {}
_whisper_lock = threading.Lock()


@dataclass(frozen=True)
class TranscriptSegment:
    """One transcribed speech segment with segment- and word-level timestamps."""

    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str
    # [{"word": str, "start": float, "end": float}, ...]; start/end absent for
    # words the aligner could not place
    words: list[dict]


def _get_asr_model():
    """Lazily load and cache the WhisperX ASR model (thread-safe)."""
    global _asr_model
    with _whisper_lock:
        if _asr_model is None:
            import whisperx

            settings = get_settings()
            logger.info("Loading WhisperX model '%s' (cpu/int8)", settings.whisper_model_size)
            _asr_model = whisperx.load_model(
                settings.whisper_model_size,
                device="cpu",
                compute_type="int8",
            )
        return _asr_model


def _get_align_model(language: str):
    """Lazily load and cache the WhisperX alignment model for a language."""
    import whisperx

    with _whisper_lock:
        if language not in _align_models:
            logger.info("Loading WhisperX alignment model for language '%s'", language)
            _align_models[language] = whisperx.load_align_model(
                language_code=language,
                device="cpu",
            )
        return _align_models[language]


def _video_has_audio_stream(video_path: str) -> bool:
    """Check with ffprobe whether the video contains an audio stream."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.returncode == 0 and "audio" in out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"ffprobe audio check failed: {e}")
        return False


def _extract_audio_ffmpeg(video_path: str, wav_path: str) -> bool:
    """Extract 16 kHz mono WAV audio from the video for Whisper input."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                wav_path,
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"ffmpeg audio extract failed: {e}")
        return False
    return os.path.exists(wav_path) and os.path.getsize(wav_path) > 44  # > WAV header


def _transcribe_wav(wav_path: str) -> list[TranscriptSegment]:
    """
    Run WhisperX on the WAV file; return non-empty segments with word-level timestamps.

    Falls back to segment-level timestamps if forced alignment fails (e.g. no
    alignment model for the detected language).
    """
    import whisperx

    model = _get_asr_model()
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=WHISPERX_BATCH_SIZE)
    language = result.get("language") or "en"
    segments = result.get("segments") or []

    if segments:
        try:
            align_model, align_metadata = _get_align_model(language)
            aligned = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                "cpu",
                return_char_alignments=False,
            )
            segments = aligned.get("segments") or segments
        except Exception as e:
            logger.warning(
                "WhisperX alignment failed (language=%s): %s; keeping segment-level timestamps",
                language,
                e,
            )

    results: list[TranscriptSegment] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words: list[dict] = []
        for w in seg.get("words") or []:
            word_text = (w.get("word") or "").strip()
            if not word_text:
                continue
            entry: dict = {"word": word_text}
            if w.get("start") is not None:
                entry["start"] = round(float(w["start"]), 3)
            if w.get("end") is not None:
                entry["end"] = round(float(w["end"]), 3)
            words.append(entry)
        results.append(
            TranscriptSegment(
                segment_index=len(results),
                start_seconds=float(seg.get("start") or 0.0),
                end_seconds=float(seg.get("end") or 0.0),
                text=text,
                words=words,
            )
        )
    logger.info(
        "WhisperX transcribed %d segment(s) (detected language=%s)",
        len(results),
        language,
    )
    return results


async def _set_transcript_status(
    session: AsyncSession,
    video_id: UUID,
    status: IndexingStatus,
) -> None:
    await session.execute(
        update(IndexedFile)
        .where(IndexedFile.id == video_id)
        .values(transcript_status=status.value, updated_at=datetime.utcnow())
    )
    await session.commit()


async def transcribe_and_index_video(
    video_id: UUID,
    video_path: str,
    session: AsyncSession,
    filename: str | None = None,
) -> dict:
    """
    Transcribe one downloaded video file and store transcript segments in the DB.

    Replaces any existing segments for the video. Each segment gets a Gemini text
    embedding when the embedding service is configured (lexical transcript search
    still works without embeddings).

    Returns { "segments": int, "error": str | None }.
    """
    result = {"segments": 0, "error": None}
    settings = get_settings()
    if not settings.transcription_enabled:
        return result

    try:
        await _set_transcript_status(session, video_id, IndexingStatus.PROCESSING)
        loop = asyncio.get_running_loop()

        has_audio = await loop.run_in_executor(
            None, functools.partial(_video_has_audio_stream, video_path)
        )
        if not has_audio:
            logger.info("No audio stream in video_id=%s, skipping transcription", video_id)
            await _set_transcript_status(session, video_id, IndexingStatus.COMPLETED)
            return result

        wav_path = video_path + ".wav"
        extracted = await loop.run_in_executor(
            None, functools.partial(_extract_audio_ffmpeg, video_path, wav_path)
        )
        if not extracted:
            result["error"] = "Audio extraction failed"
            await _set_transcript_status(session, video_id, IndexingStatus.FAILED)
            return result

        segments = await loop.run_in_executor(
            None, functools.partial(_transcribe_wav, wav_path)
        )

        # Embed each segment for semantic transcript search (best-effort)
        vision_service = get_vision_embedding_service()
        embeddings: list[list[float] | None] = [None] * len(segments)
        if vision_service.is_configured:
            for i, seg in enumerate(segments):
                embeddings[i] = await vision_service.generate_document_text_embedding(seg.text)

        # Replace existing segments for this video (re-indexing)
        await session.execute(
            delete(VideoTranscriptSegment).where(VideoTranscriptSegment.video_id == video_id)
        )
        for seg, embedding in zip(segments, embeddings):
            session.add(
                VideoTranscriptSegment(
                    video_id=video_id,
                    segment_index=seg.segment_index,
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    text=seg.text,
                    words=seg.words or None,
                    text_embedding=embedding,
                )
            )
        await session.commit()

        await _set_transcript_status(session, video_id, IndexingStatus.COMPLETED)
        result["segments"] = len(segments)
        logger.info(
            "Transcript indexed for video_id=%s filename=%s: %d segment(s)",
            video_id,
            filename or "",
            len(segments),
        )
    except Exception as e:
        logger.exception("Transcription failed for video_id=%s: %s", video_id, e)
        await session.rollback()
        result["error"] = str(e)
        try:
            await _set_transcript_status(session, video_id, IndexingStatus.FAILED)
        except Exception:
            await session.rollback()
    return result


async def transcribe_video_with_own_session(
    video_id: UUID,
    video_path: str,
    filename: str | None = None,
) -> dict:
    """Run transcribe_and_index_video with a dedicated DB session (for use alongside frame indexing)."""
    async with async_session_maker() as session:
        return await transcribe_and_index_video(
            video_id,
            video_path,
            session,
            filename=filename,
        )
