"""
WebSocket /v1/stream — real-time partial + final transcription.

Protocol:
  Client text  → {"type":"start","sample_rate":16000}
  Client binary → raw PCM int16 chunks
  Client text  → {"type":"stop"}

  Server text  → {"type":"partial","transcript":"..."}        every 5 chunks (~1s)
  Server text  → {"type":"final","transcript":"...","duration_sec":X}  on stop
"""
from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from paryaya.api.state import MODULE_STATE

router = APIRouter(tags=["streaming"])

_CHUNK_BATCH  = 5       # emit partial every N binary chunks (~1.3 s at 16 kHz / 4096 samples)
_INT16_MAX    = 32768.0
_SILENCE_RMS  = 0.01    # skip transcription if audio is below this energy level
_MIN_DURATION = 1.5     # skip transcription if audio is shorter than this (seconds)


@router.websocket("/v1/stream")
async def stream(ws: WebSocket, task: str = Query("transcribe")) -> None:
    await ws.accept()

    backend = MODULE_STATE.get("backend", "whisper")
    model   = MODULE_STATE.get("model")

    if model is None:
        await ws.close(code=4003, reason="Model not loaded")
        return

    pcm_buffer: list[np.ndarray] = []
    chunk_count = 0
    sample_rate = 16_000

    try:
        while True:
            msg = await ws.receive()

            # ── text control frame ────────────────────────────────────────────
            if "text" in msg:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") == "start":
                    sample_rate = int(ctrl.get("sample_rate", 16_000))
                    pcm_buffer.clear()
                    chunk_count = 0

                elif ctrl.get("type") == "stop":
                    if pcm_buffer:
                        audio  = _concat(pcm_buffer)
                        result = _transcribe(audio, sample_rate, backend, model, task)
                    else:
                        result = {"transcript": "", "duration_sec": 0.0}
                    await ws.send_text(json.dumps({
                        "type":        "final",
                        "transcript":  result["transcript"],
                        "duration_sec": result["duration_sec"],
                    }))
                    break

            # ── binary audio chunk ────────────────────────────────────────────
            elif "bytes" in msg:
                raw = msg["bytes"]
                if not raw:
                    continue
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_MAX
                pcm_buffer.append(samples)
                chunk_count += 1

                if chunk_count % _CHUNK_BATCH == 0:
                    audio = _concat(pcm_buffer)
                    if _is_speech(audio, sample_rate):
                        result = _transcribe(audio, sample_rate, backend, model, task)
                        if result["transcript"]:
                            await ws.send_text(json.dumps({
                                "type":       "partial",
                                "transcript": result["transcript"],
                            }))

    except WebSocketDisconnect:
        pass


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks).astype(np.float32)


def _is_speech(audio: np.ndarray, sample_rate: int) -> bool:
    if len(audio) / sample_rate < _MIN_DURATION:
        return False
    return float(np.sqrt(np.mean(audio ** 2))) >= _SILENCE_RMS


def _transcribe(audio: np.ndarray, sample_rate: int, backend: str, model, task: str = "transcribe") -> dict:
    if backend == "whisper":
        if task == "translate":
            return model.translate(audio, sample_rate=sample_rate)
        return model.transcribe(audio, sample_rate=sample_rate)
    from paryaya.inference.transcribe import transcribe_audio_array
    tok    = MODULE_STATE.get("tokenizer")
    device = MODULE_STATE.get("device", "cpu")
    return transcribe_audio_array(audio, model, tok, device)
