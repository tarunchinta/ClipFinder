"""Azure Functions Service Bus consumers for ClipFinder indexing jobs."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import azure.functions as func

# Backend package: repo_root/backend/app (copied beside function_app.py in container)
_backend_dir = Path(__file__).resolve().parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.tasks import _run_frame_indexing_async, _run_image_indexing_async

logger = logging.getLogger(__name__)

app = func.FunctionApp()


@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name="%VIDEO_INDEXING_QUEUE%",
    connection="SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING",
)
def video_frame_index_trigger(msg: func.ServiceBusMessage) -> None:
    """Process a video frame-indexing job from the frame-indexing queue."""
    try:
        body = json.loads(msg.get_body().decode("utf-8"))
        video_id = body["video_id"]
        logger.info("Processing video frame indexing job: video_id=%s", video_id)
        result = asyncio.run(
            _run_frame_indexing_async(
                video_id,
                trace_id=body.get("trace_id"),
                parent_span_id=body.get("parent_span_id"),
            )
        )
        if result.get("error"):
            raise RuntimeError(result["error"])
        logger.info("Completed video frame indexing: video_id=%s", video_id)
    except Exception:
        logger.exception("Video frame indexing trigger failed")
        raise


@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name="%IMAGE_INDEXING_QUEUE%",
    connection="SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING",
)
def image_vision_index_trigger(msg: func.ServiceBusMessage) -> None:
    """Process an image thumbnail vision-indexing job from the image-indexing queue."""
    try:
        body = json.loads(msg.get_body().decode("utf-8"))
        file_id = body["file_id"]
        logger.info("Processing image vision indexing job: file_id=%s", file_id)
        result = asyncio.run(
            _run_image_indexing_async(
                file_id,
                trace_id=body.get("trace_id"),
                parent_span_id=body.get("parent_span_id"),
            )
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "Image vision indexing failed")
        logger.info("Completed image vision indexing: file_id=%s", file_id)
    except Exception:
        logger.exception("Image vision indexing trigger failed")
        raise
