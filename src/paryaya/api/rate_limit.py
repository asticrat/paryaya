"""
paryaya.api.rate_limit — Token-bucket rate limiter backed by Redis.

Limits are per API key:
  - Default tier: 10 requests/minute
  - Premium tier: 100 requests/minute

Tier membership is stored in Redis set  paryaya:premium_keys.

If Redis is unreachable the check passes (fail-open).
"""
from __future__ import annotations

import os
import time

from fastapi import Depends, HTTPException, Request, status

_DEFAULT_RPM = int(os.getenv("RATE_LIMIT_DEFAULT_RPM", "10"))
_PREMIUM_RPM = int(os.getenv("RATE_LIMIT_PREMIUM_RPM", "100"))
_WINDOW_SEC  = 60


async def _redis():
    try:
        import redis.asyncio as aioredis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return aioredis.from_url(url, decode_responses=True)
    except Exception:
        return None


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency. Raises 429 when the per-key rate limit is exceeded."""
    api_key: str = getattr(request.state, "api_key", "anonymous")
    r = await _redis()
    if r is None:
        return  # fail-open: Redis unavailable

    try:
        is_premium = await r.sismember("paryaya:premium_keys", api_key)
        rpm = _PREMIUM_RPM if is_premium else _DEFAULT_RPM

        window = int(time.time()) // _WINDOW_SEC
        redis_key = f"paryaya:rl:{api_key}:{window}"

        count = await r.incr(redis_key)
        if count == 1:
            await r.expire(redis_key, _WINDOW_SEC * 2)

        if count > rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({rpm} req/min). Retry after {_WINDOW_SEC}s.",
                headers={"Retry-After": str(_WINDOW_SEC)},
            )
    finally:
        await r.aclose()
