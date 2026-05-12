"""
paryaya.api.app — FastAPI application factory.

Usage (development):
    uvicorn paryaya.api.app:app --host 0.0.0.0 --port 8000 --reload

Production (gunicorn + uvicorn workers):
    gunicorn paryaya.api.app:app -k uvicorn.workers.UvicornWorker \
        -w 4 --bind 0.0.0.0:8000
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from paryaya import __version__
from paryaya.api.routes import router
from paryaya.api.schemas import ErrorResponse

logger = logging.getLogger("paryaya.api")

app = FastAPI(
    title="Paryaya ASR API",
    description="Nepali speech-to-text REST API powered by ParyayaASR",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.middleware("http")
async def _attach_api_key(request: Request, call_next):
    """Store the raw API key on request.state so rate limiter can read it."""
    request.state.api_key = request.headers.get("X-API-Key", "anonymous")
    return await call_next(request)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_server_error", code=500).model_dump(),
    )


app.include_router(router)
