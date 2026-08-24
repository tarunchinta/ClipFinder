"""Azure Service Bus publishers for image and video indexing jobs."""

import json
import logging

from app.config import get_settings
from app.models.indexed_file import IndexedFile

logger = logging.getLogger(__name__)


def is_service_bus_configured() -> bool:
    """Return True when SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING is set."""
    return bool(get_settings().service_bus_connection_string)


def _send_messages(queue_name: str, messages: list[str], label: str) -> None:
    """Send JSON message bodies to a Service Bus queue."""
    if not messages:
        return

    settings = get_settings()
    from azure.servicebus import ServiceBusClient, ServiceBusMessage

    with ServiceBusClient.from_connection_string(
        conn_str=settings.service_bus_connection_string,
        logging_enable=False,
    ) as sb_client:
        sender = sb_client.get_queue_sender(queue_name=queue_name)
        with sender:
            for body in messages:
                sender.send_messages(ServiceBusMessage(body))

    logger.info(
        "Published %d %s message(s) to queue '%s'",
        len(messages),
        label,
        queue_name,
    )


def publish_video_indexing_jobs(
    videos: list[IndexedFile],
    trace_contexts: list[dict],
) -> None:
    """
    Publish one frame-indexing job per video.

    Raises on send failure (caller may fall back to in-process indexing).
    """
    if not videos:
        return

    settings = get_settings()
    messages: list[str] = []
    for i, video in enumerate(videos):
        ctx = trace_contexts[i] if i < len(trace_contexts) else {}
        messages.append(
            json.dumps({
                "video_id": str(video.id),
                "trace_id": ctx.get("trace_id"),
                "parent_span_id": ctx.get("parent_span_id"),
            })
        )
        logger.info(
            "Queueing video frame indexing: video_id=%s filename=%s",
            video.id,
            video.filename,
        )

    _send_messages(settings.video_indexing_queue_name, messages, "video indexing")


def publish_image_indexing_jobs(
    images: list[IndexedFile],
    trace_contexts: list[dict],
) -> None:
    """
    Publish one image vision-indexing job per image.

    Raises on send failure (caller may fall back to in-process indexing).
    """
    if not images:
        return

    settings = get_settings()
    messages: list[str] = []
    for i, image in enumerate(images):
        ctx = trace_contexts[i] if i < len(trace_contexts) else {}
        messages.append(
            json.dumps({
                "file_id": str(image.id),
                "trace_id": ctx.get("trace_id"),
                "parent_span_id": ctx.get("parent_span_id"),
            })
        )
        logger.info(
            "Queueing image vision indexing: file_id=%s filename=%s",
            image.id,
            image.filename,
        )

    _send_messages(settings.image_indexing_queue_name, messages, "image indexing")
