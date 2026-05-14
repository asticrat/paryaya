"""
APIKeyMiddleware — validates "Authorization: Bearer sk-paryaya-XXX" on every
non-public path. Stores parsed key_info JSON on request.state.key_info.

Skipped paths: /health, /health/ready, /docs, /openapi.json, /redoc
"""
from __future__ import annotations

import json
import os

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_SKIP = {"/health", "/health/ready", "/docs", "/openapi.json", "/redoc", "/metrics", "/", "/favicon.ico"}
# Auth routes authenticate themselves via ADMIN-SECRET-KEY header
_SKIP_PREFIX = ("/auth/", "/static/")
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _SKIP or path.startswith("/docs") or any(path.startswith(p) for p in _SKIP_PREFIX):
            return await call_next(request)

        # WebSocket upgrade — key checked inside the WS handler
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer sk-paryaya-"):
            return JSONResponse({"detail": "Missing or invalid Authorization header"}, status_code=401)

        api_key = auth.removeprefix("Bearer ")

        r = aioredis.from_url(_REDIS_URL, decode_responses=True)
        try:
            raw = await r.get(f"key:{api_key}")
        finally:
            await r.aclose()

        if not raw:
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)

        request.state.key_info = json.loads(raw)
        request.state.api_key  = api_key
        return await call_next(request)
