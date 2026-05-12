"""
POST /v1/transcribe — synchronous audio transcription.

Accepts: .wav .mp3 .flac .ogg .m4a .webm
Limits:  MAX_AUDIO_MB (default 50)
Returns: TranscribeResponse
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from paryaya import __version__
from paryaya.api.schemas import TranscribeResponse
from paryaya.api.state import MODULE_STATE

router = APIRouter(prefix="/v1", tags=["transcription"])

_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
_MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "50"))
_BEAM_WIDTH   = int(os.getenv("BEAM_WIDTH", "10"))


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request, file: UploadFile) -> TranscribeResponse:
    # Extension check
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {sorted(_ALLOWED_EXT)}",
        )

    audio_bytes = await file.read()

    # Size check
    max_bytes = int(_MAX_AUDIO_MB * 1024 * 1024)
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {_MAX_AUDIO_MB} MB limit. Use /v1/transcribe/async for large files.",
        )

    backend = MODULE_STATE.get("backend", "whisper")
    model   = MODULE_STATE.get("model")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    t0 = time.perf_counter()
    try:
        if backend == "whisper":
            result = model.transcribe_bytes(audio_bytes)
        else:
            from paryaya.inference.transcribe import transcribe_bytes as _tb
            tok    = MODULE_STATE.get("tokenizer")
            device = MODULE_STATE.get("device", "cpu")
            result = _tb(audio_bytes, model, tok, device, _BEAM_WIDTH)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc

    processing_ms = (time.perf_counter() - t0) * 1000

    return TranscribeResponse(
        transcript=result["transcript"],
        confidence=result["confidence"],
        duration_sec=result["duration_sec"],
        processing_ms=round(processing_ms, 1),
        model_version=__version__,
    )
