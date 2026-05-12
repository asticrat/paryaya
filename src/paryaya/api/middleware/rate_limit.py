"""
RateLimitMiddleware — per-API-key rate limiting backed by Redis.

Plans:
  starter    → 60 req/min,   1 000 req/day
  business   → 200 req/min,  10 000 req/day
  enterprise → 9 999 req/min, 9 999 999 req/day

Redis keys:
  rl:min:{key}:{minute_bucket}  — expire 60 s
  rl:day:{key}:{date}           — expire 86 400 s

Returns 429 with plan name and limit on exceed.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_SKIP = {"/health", "/health/ready", "/docs", "/openapi.json", "/redoc", "/metrics"}
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_PLANS: dict[str, dict[str, int]] = {
    "starter":    {"per_min": 60,   "per_day": 1_000},
    "business":   {"per_min": 200,  "per_day": 10_000},
    "enterprise": {"per_min": 9_999, "per_day": 9_999_999},
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP:
            return await call_next(request)

        key_info = getattr(request.state, "key_info", None)
        api_key  = getattr(request.state, "api_key", "anonymous")

        plan_name = (key_info or {}).get("plan", "starter")
        limits    = _PLANS.get(plan_name, _PLANS["starter"])

        r = aioredis.from_url(_REDIS_URL, decode_responses=True)
        try:
            now          = int(time.time())
            minute_bucket = now // 60
            today_str    = datetime.now(timezone.utc).strftime("%Y%m%d")

            min_key = f"rl:min:{api_key}:{minute_bucket}"
            day_key = f"rl:day:{api_key}:{today_str}"

            pipe = r.pipeline()
            pipe.incr(min_key)
            pipe.expire(min_key, 60)
            pipe.incr(day_key)
            pipe.expire(day_key, 86_400)
            results = await pipe.execute()

            min_count, _, day_count, _ = results

            if min_count > limits["per_min"]:
                return JSONResponse(
                    {
                        "detail": f"Rate limit exceeded for plan '{plan_name}': "
                                  f"{limits['per_min']} req/min",
                        "plan":   plan_name,
                        "limit":  limits["per_min"],
                        "window": "minute",
                    },
                    status_code=429,
                    headers={"Retry-After": "60"},
                )

            if day_count > limits["per_day"]:
                return JSONResponse(
                    {
                        "detail": f"Rate limit exceeded for plan '{plan_name}': "
                                  f"{limits['per_day']} req/day",
                        "plan":   plan_name,
                        "limit":  limits["per_day"],
                        "window": "day",
                    },
                    status_code=429,
                    headers={"Retry-After": "86400"},
                )
        except Exception:
            pass  # fail-open: Redis unreachable
        finally:
            await r.aclose()

        return await call_next(request)
