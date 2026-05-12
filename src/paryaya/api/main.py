"""
paryaya.api.main — FastAPI application with lifespan model loading.

Usage (dev):
    uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000 --reload

Production (Docker):
    uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000 --workers 1

Environment:
    MODEL_PATH     — path to best_model.pt  (default: checkpoints/best_model.pt)
    VOCAB_PATH     — path to nepali_vocab.json (default: data/vocab/nepali_vocab.json)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from paryaya.api.middleware.auth import APIKeyMiddleware
from paryaya.api.middleware.rate_limit import RateLimitMiddleware
from paryaya.api.routes import auth, health, stream, transcribe
from paryaya.api.state import MODULE_STATE
from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer

logger = logging.getLogger("paryaya.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting Paryaya API on device=%s", device)

    vocab = os.getenv("VOCAB_PATH", "data/vocab/nepali_vocab.json")
    tok   = NepaliTokenizer(vocab if Path(vocab).exists() else None)

    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
    if Path(model_path).exists():
        model = ParyayaASR.load_from_checkpoint(model_path, tok.vocab_size).to(device)
        model.eval()
        MODULE_STATE["model"]     = model
        MODULE_STATE["tokenizer"] = tok
        MODULE_STATE["device"]    = device
        logger.info("Model loaded from %s", model_path)
    else:
        logger.warning("Checkpoint not found at %s — /health will report degraded", model_path)
        MODULE_STATE["device"] = device

    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    MODULE_STATE.clear()
    logger.info("Paryaya API shut down")


app = FastAPI(
    title="Paryaya API — Nepali Speech Recognition",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware — added in reverse execution order:
# CORS → RateLimit → APIKey → routes
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

# Routers
app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(stream.router)
app.include_router(auth.router)

# Prometheus metrics — exposed at /metrics (public, no auth)
Instrumentator().instrument(app).expose(app)
