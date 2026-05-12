"""
API key management routes.

POST /auth/keys                   → create key (requires ADMIN_SECRET_KEY header)
GET  /auth/keys/{key}/usage       → usage stats
DELETE /auth/keys/{key}           → revoke key

Redis schema:
  key:{api_key} → JSON { company, plan, usage_minutes, requests_today, created_at }
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Header, HTTPException, status

from paryaya.api.schemas import (
    DeleteKeyResponse,
    KeyCreateRequest,
    KeyCreateResponse,
    KeyUsageResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _redis() -> aioredis.Redis:
    return aioredis.from_url(_REDIS_URL, decode_responses=True)


def _new_key() -> str:
    return f"sk-paryaya-{secrets.token_hex(16)}"


@router.post("/keys", response_model=KeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: KeyCreateRequest,
    admin_secret_key: str = Header(..., alias="ADMIN-SECRET-KEY"),
) -> KeyCreateResponse:
    expected = os.getenv("ADMIN_SECRET_KEY", "")
    if not expected or admin_secret_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin secret")

    if body.plan not in {"starter", "business", "enterprise"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid plan")

    api_key = _new_key()
    record = {
        "company":        body.company,
        "plan":           body.plan,
        "usage_minutes":  0.0,
        "requests_today": 0,
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }

    r = _redis()
    try:
        await r.set(f"key:{api_key}", json.dumps(record))
    finally:
        await r.aclose()

    return KeyCreateResponse(api_key=api_key, company=body.company, plan=body.plan)


@router.get("/keys/{key}/usage", response_model=KeyUsageResponse)
async def key_usage(
    key: str,
    admin_secret_key: str = Header(..., alias="ADMIN-SECRET-KEY"),
) -> KeyUsageResponse:
    expected = os.getenv("ADMIN_SECRET_KEY", "")
    if not expected or admin_secret_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin secret")

    r = _redis()
    try:
        raw = await r.get(f"key:{key}")
    finally:
        await r.aclose()

    if not raw:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    data = json.loads(raw)
    return KeyUsageResponse(api_key=key, **data)


@router.delete("/keys/{key}", response_model=DeleteKeyResponse)
async def delete_key(
    key: str,
    admin_secret_key: str = Header(..., alias="ADMIN-SECRET-KEY"),
) -> DeleteKeyResponse:
    expected = os.getenv("ADMIN_SECRET_KEY", "")
    if not expected or admin_secret_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin secret")

    r = _redis()
    try:
        deleted = await r.delete(f"key:{key}")
    finally:
        await r.aclose()

    return DeleteKeyResponse(deleted=bool(deleted))
