"""
paryaya.api.schemas — Pydantic v2 request/response models for the Paryaya API.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    transcript: str
    confidence: float
    duration_sec: float
    processing_ms: float
    language: str = "ne"
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
    uptime_seconds: float


class KeyCreateRequest(BaseModel):
    company: str
    plan: str  # starter | business | enterprise


class KeyCreateResponse(BaseModel):
    api_key: str
    company: str
    plan: str


class KeyUsageResponse(BaseModel):
    api_key: str
    company: str
    plan: str
    usage_minutes: float
    requests_today: int
    created_at: str


class DeleteKeyResponse(BaseModel):
    deleted: bool
