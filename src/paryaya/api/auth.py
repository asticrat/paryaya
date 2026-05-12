"""
paryaya.api.auth — API key authentication middleware.

Keys have the form  sk-paryaya-<base62-32chars>.
Loaded from the PARYAYA_API_KEYS environment variable (comma-separated)
or from REDIS (key set  paryaya:api_keys).

Usage:
    from paryaya.api.auth import require_api_key
    @app.get("/v1/transcribe")
    async def route(api_key: str = Depends(require_api_key)):
        ...
"""
from __future__ import annotations

import os
import secrets
import string

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

_KEY_PREFIX = "sk-paryaya-"
_ALPHABET = string.ascii_letters + string.digits
_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key() -> str:
    """Generate a new API key with the sk-paryaya- prefix."""
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(32))
    return f"{_KEY_PREFIX}{suffix}"


def _load_env_keys() -> set[str]:
    raw = os.getenv("PARYAYA_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def _get_redis_keys() -> set[str]:
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url, decode_responses=True)
        members = await r.smembers("paryaya:api_keys")
        await r.aclose()
        return members
    except Exception:
        return set()


async def require_api_key(api_key: str | None = Security(_HEADER)) -> str:
    """FastAPI dependency that validates the X-API-Key header.

    Checks env-loaded keys first (fast path), then Redis.
    Raises 401 if key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    env_keys = _load_env_keys()
    if api_key in env_keys:
        return api_key

    redis_keys = await _get_redis_keys()
    if api_key in redis_keys:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
