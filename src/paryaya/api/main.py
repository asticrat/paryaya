"""
paryaya.api.main — FastAPI application with lifespan model loading.

Usage (dev):
    uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000 --reload

Production (Docker):
    uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000 --workers 1

Environment:
    ASR_BACKEND         — "whisper" (default) | "paryaya"
    WHISPER_MODEL_PATH  — HF model name or fine-tuned dir (default: openai/whisper-medium)
    MODEL_PATH          — path to ParyayaASR .pt checkpoint (paryaya backend only)
    VOCAB_PATH          — path to nepali_vocab.json (paryaya backend only)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import pathlib

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from paryaya.api.middleware.auth import APIKeyMiddleware
from paryaya.api.middleware.rate_limit import RateLimitMiddleware
from paryaya.api.routes import auth, health, stream, transcribe
from paryaya.api.state import MODULE_STATE

logger = logging.getLogger("paryaya.api")


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    device  = _resolve_device()
    backend = os.getenv("ASR_BACKEND", "whisper").lower()
    logger.info("Starting Paryaya API | backend=%s device=%s", backend, device)

    MODULE_STATE["device"]  = device
    MODULE_STATE["backend"] = backend

    if backend == "whisper":
        from paryaya.inference.faster_whisper_backend import load_faster_whisper_backend
        whisper = load_faster_whisper_backend(device=device)
        MODULE_STATE["model"] = whisper
        logger.info("Whisper backend loaded: %s", os.getenv("WHISPER_MODEL_PATH", "checkpoints/whisper-medium-nepali-ct2"))

    else:  # paryaya (custom conformer)
        from paryaya.model.asr_model import ParyayaASR
        from paryaya.model.tokenizer import NepaliTokenizer

        vocab = os.getenv("VOCAB_PATH", "data/vocab/nepali_vocab.json")
        tok   = NepaliTokenizer(vocab if Path(vocab).exists() else None)
        model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")

        if Path(model_path).exists():
            model = ParyayaASR.load_from_checkpoint(model_path, tok.vocab_size).to(device)
            model.eval()
            MODULE_STATE["model"]     = model
            MODULE_STATE["tokenizer"] = tok
            logger.info("ParyayaASR loaded from %s", model_path)
        else:
            logger.warning("Checkpoint not found at %s — /health will report degraded", model_path)
            MODULE_STATE["device"] = device

    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    MODULE_STATE.clear()
    logger.info("Paryaya API shut down")


_ALLOWED_ORIGINS = [
    "https://paryaya.net",
    "https://www.paryaya.net",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "connect-src 'self' wss://paryaya.net ws://localhost:8000; "
    "img-src 'self' data:; "
    "frame-ancestors 'none';"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]  = "nosniff"
        response.headers["X-Frame-Options"]          = "DENY"
        response.headers["X-XSS-Protection"]         = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]        = "microphone=(self)"
        response.headers["Strict-Transport-Security"] = "max-age=15552000; includeSubDomains"
        response.headers["Content-Security-Policy"]   = _CSP
        response.headers["Server"] = ""
        return response


app = FastAPI(
    title="Paryaya API — Nepali Speech Recognition",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware — added in reverse execution order:
# Security → CORS → RateLimit → APIKey → routes
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

# Routers
app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(stream.router)
app.include_router(auth.router)

# Prometheus metrics — exposed at /metrics (public, no auth)
Instrumentator().instrument(app).expose(app)

# Static files
_static = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(_static / "index.html"))
