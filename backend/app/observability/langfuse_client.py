"""Optional Langfuse client for tracing embeddings and retrieval. No-ops when not configured."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry.context import attach as _otel_attach, detach as _otel_detach, Context as _OtelContext

from app.config import get_settings

logger = logging.getLogger(__name__)

_langfuse_client: Any = None


def get_langfuse_client():
    """
    Return the Langfuse client if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set, else None.
    Safe to call repeatedly; client is cached.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    settings = get_settings()
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        return None
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host or "https://cloud.langfuse.com",
        )
        return _langfuse_client
    except Exception as e:
        logger.warning("Langfuse client init failed, tracing disabled: %s", e)
        return None


def flush_langfuse() -> None:
    """Flush pending Langfuse events. Call after request handlers for short-lived processes."""
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.debug("Langfuse flush failed: %s", e)


@contextmanager
def trace_retrieval(
    name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager for a top-level retrieval trace (e.g. vision_hybrid_search).
    Uses start_as_current_observation so the first span is the trace root. Records duration.
    No-ops when Langfuse is not configured.
    """
    client = get_langfuse_client()
    if not client:
        yield None
        return
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name=name,
            as_type="retriever",
            metadata=metadata or {},
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms, **(metadata or {})})
    except Exception as e:
        logger.debug("Langfuse trace_retrieval failed: %s", e)
        raise


@contextmanager
def trace_embedding_generation(
    name: str,
    model: str = "",
    input_summary: Optional[str] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager for an embedding-generation span (text or vision).
    Use as a child span under a trace; records duration and optional model/input summary.
    No-ops when Langfuse is not configured.
    """
    client = get_langfuse_client()
    if not client:
        yield None
        return
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name=name,
            as_type="embedding",
            model=model or None,
            input=input_summary,
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms})
    except Exception as e:
        logger.debug("Langfuse trace_embedding_generation failed: %s", e)
        raise


@contextmanager
def trace_vector_search(
    name: str = "vector_search",
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager for a vector/DB search span (e.g. pgvector query). Use under a retrieval trace.
    Records duration. No-ops when Langfuse is not configured.
    """
    client = get_langfuse_client()
    if not client:
        yield None
        return
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name=name,
            as_type="span",
            metadata=metadata or {},
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms, **(metadata or {})})
    except Exception as e:
        logger.debug("Langfuse trace_vector_search failed: %s", e)
        raise


def get_current_trace_id() -> Optional[str]:
    """Return the current trace id from Langfuse context, or None if disabled."""
    client = get_langfuse_client()
    if not client:
        return None
    try:
        return client.get_current_trace_id()
    except Exception as e:
        logger.debug("Langfuse get_current_trace_id failed: %s", e)
        return None


def get_current_observation_id() -> Optional[str]:
    """Return the current observation (span) id from Langfuse context, or None if disabled."""
    client = get_langfuse_client()
    if not client:
        return None
    try:
        return client.get_current_observation_id()
    except Exception as e:
        logger.debug("Langfuse get_current_observation_id failed: %s", e)
        return None


@contextmanager
def trace_batch_upload(
    name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager for a top-level batch-upload trace (e.g. folder_index).
    Wraps the actual work so Langfuse captures real start/end timestamps
    and populates the latency column.  Child calls that use
    ``trace_index_file`` will NOT nest under this trace because
    ``trace_index_file`` resets the OTel context before creating its span.
    No-ops when Langfuse is not configured.
    """
    client = get_langfuse_client()
    if not client:
        yield None
        return
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name=name,
            as_type="span",
            metadata=metadata or {},
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms, **(metadata or {})})
    except Exception as e:
        logger.warning("trace_batch_upload failed: %s", e)
        raise


@contextmanager
def trace_index_file(
    file_type: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager that creates its own independent top-level trace per file.
    Span name is 'image_index' or 'video_index' based on *file_type*.

    Resets to the OTel ROOT_CONTEXT before creating the observation so it
    is never a child of an enclosing ``trace_batch_upload``.  Child
    observations created *inside* this context (e.g.
    ``trace_embedding_generation``) still nest under it normally.
    No-ops when Langfuse is not configured.
    """
    span_name = "video_index" if file_type == "video" else "image_index"
    merged = {"file_type": file_type, **(metadata or {})}
    client = get_langfuse_client()
    if not client:
        yield None
        return
    token = _otel_attach(_OtelContext())
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name=span_name,
            as_type="span",
            metadata=merged,
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms, **merged})
    except Exception as e:
        logger.debug("Langfuse trace_index_file failed: %s", e)
        raise
    finally:
        _otel_detach(token)


@contextmanager
def trace_video_frame(
    metadata: Optional[dict[str, Any]] = None,
    trace_context: Optional[dict[str, str]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager for a per-frame child span under video_index.
    Span name is always 'video_frame_index'. file_type is always 'video'.
    If trace_context (trace_id, parent_span_id) is provided, the span attaches to the
    existing video span in the batch trace. No-ops when Langfuse is not configured.
    """
    merged = {"file_type": "video", **(metadata or {})}
    client = get_langfuse_client()
    if not client:
        yield None
        return
    try:
        start = time.perf_counter()
        with client.start_as_current_observation(
            name="video_frame_index",
            as_type="span",
            metadata=merged,
            trace_context=trace_context,
        ) as span:
            yield span
            duration_ms = (time.perf_counter() - start) * 1000
            span.update(metadata={"duration_ms": duration_ms, **merged})
    except Exception as e:
        logger.debug("Langfuse trace_video_frame failed: %s", e)
        raise


def emit_video_frame_trace(
    metadata: dict[str, Any],
    duration_ms: float,
) -> None:
    """
    Create a standalone top-level trace named 'video_frame_index'.
    Called *after* each frame is processed so per-frame latency is queryable
    at the same level as image_index traces in Langfuse dashboards.
    """
    client = get_langfuse_client()
    if not client:
        return
    try:
        merged = {"file_type": "video", "duration_ms": duration_ms, **metadata}
        with client.start_as_current_observation(
            name="video_frame_index",
            as_type="span",
            metadata=merged,
        ) as span:
            span.update(metadata=merged)
    except Exception as e:
        logger.warning("emit_video_frame_trace failed: %s", e)


