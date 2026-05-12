"""
paryaya.worker.tasks — Celery task definitions.

Run workers:
    celery -A paryaya.worker.celery_app worker --loglevel=info --concurrency=2
"""
from __future__ import annotations

import logging

from paryaya.worker.celery_app import celery_app

logger = logging.getLogger("paryaya.worker")


@celery_app.task(
    name="paryaya.transcribe",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    track_started=True,
)
def transcribe_task(self, job_id: str, audio_bytes: bytes, beam_width: int = 10) -> dict:
    """Transcribe audio bytes and return the result dict.

    Stores result in the Celery result backend (Redis).
    Retries up to 2× on transient errors.
    """
    try:
        from paryaya.api.model_loader import get_model, get_tokenizer
        from paryaya.inference.transcribe import transcribe_bytes

        model, device = get_model()
        tok = get_tokenizer()
        result = transcribe_bytes(audio_bytes, model, tok, device, beam_width)
        logger.info("job=%s dur=%.2fs transcript_len=%d", job_id, result["duration_sec"], result["word_count"])
        return result
    except Exception as exc:
        logger.exception("job=%s failed: %s", job_id, exc)
        raise self.retry(exc=exc)
