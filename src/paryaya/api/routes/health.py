"""
GET /health        → status, model_loaded, device, version, uptime
GET /health/ready  → 200 if model loaded, 503 if not
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Response

from paryaya.api.schemas import HealthResponse
from paryaya.api.state import MODULE_STATE, _START_TIME

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = "model" in MODULE_STATE
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=MODULE_STATE.get("device", "unknown"),
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/health/ready")
async def health_ready(response: Response) -> dict:
    if "model" not in MODULE_STATE:
        response.status_code = 503
        return {"ready": False}
    return {"ready": True}
