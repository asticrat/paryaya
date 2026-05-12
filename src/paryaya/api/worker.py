"""
paryaya.api.worker — Celery worker for async (>60 s) transcription jobs.

Run:
    celery -A paryaya.api.worker worker --concurrency=4 --loglevel=info

Environment:
    REDIS_URL          — broker + backend (default: redis://localhost:6379/0)
    MODEL_PATH         — path to best_model.pt
    VOCAB_PATH         — path to nepali_vocab.json
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

import redis as sync_redis
from celery import Celery

from paryaya.inference.transcribe import transcribe_bytes
from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer

logger = logging.getLogger("paryaya.worker")

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery("paryaya", broker=_REDIS_URL, backend=_REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Lazy singletons — loaded once per worker process on first task
_model: ParyayaASR | None = None
_tok:   NepaliTokenizer | None = None
_device: str = "cpu"


def _load() -> tuple[ParyayaASR, NepaliTokenizer, str]:
    global _model, _tok, _device
    if _model is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = os.getenv("VOCAB_PATH", "data/vocab/nepali_vocab.json")
        _tok  = NepaliTokenizer(vocab if Path(vocab).exists() else None)
        ckpt  = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
        _model = ParyayaASR.load_from_checkpoint(ckpt, _tok.vocab_size).to(_device)
        _model.eval()
    return _model, _tok, _device


@app.task(name="paryaya.transcribe_async", bind=True, max_retries=2, default_retry_delay=5)
def transcribe_async(self, audio_bytes_b64: str, key_info: dict) -> dict:
    """Transcribe audio (passed as base64) and update Redis usage counters."""
    try:
        model, tok, device = _load()
        audio_bytes = base64.b64decode(audio_bytes_b64)
        result = transcribe_bytes(audio_bytes, model, tok, device)

        # Increment usage in Redis
        api_key = key_info.get("api_key", "")
        if api_key:
            r = sync_redis.from_url(_REDIS_URL, decode_responses=True)
            try:
                raw = r.get(f"key:{api_key}")
                if raw:
                    data = json.loads(raw)
                    data["usage_minutes"] = round(
                        data.get("usage_minutes", 0.0) + result["duration_sec"] / 60, 4
                    )
                    data["requests_today"] = data.get("requests_today", 0) + 1
                    r.set(f"key:{api_key}", json.dumps(data))
            finally:
                r.close()

        logger.info("transcribed %.1fs audio for key %s", result["duration_sec"], api_key[:20])
        return result

    except Exception as exc:
        logger.exception("transcribe_async failed: %s", exc)
        raise self.retry(exc=exc)
