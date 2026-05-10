"""Celery app for background tasks (e.g. video frame indexing)."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "clipfinder",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
