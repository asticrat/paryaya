"""
paryaya.worker.celery_app — Celery application instance.

Environment variables:
    REDIS_URL  — broker + result backend (default: redis://localhost:6379/0)
"""
from __future__ import annotations

import os

from celery import Celery

_REDIS = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "paryaya",
    broker=_REDIS,
    backend=_REDIS,
    include=["paryaya.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
