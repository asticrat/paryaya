"""
tests/test_api.py — API integration tests using FastAPI TestClient.

Model and Redis are fully mocked; no GPU or running Redis required.
"""
import io
import json
import os
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared mock state
# ---------------------------------------------------------------------------

_VALID_KEY  = "sk-paryaya-testkey1234567890abcdef"
_ADMIN_KEY  = "test-admin-secret-000"
_KEY_RECORD = json.dumps({
    "company":        "TestCo",
    "plan":           "starter",
    "usage_minutes":  0.0,
    "requests_today": 0,
    "created_at":     "2026-01-01T00:00:00+00:00",
})

_TRANSCRIBE_RESULT = {
    "transcript":   "नमस्ते",
    "confidence":   0.92,
    "duration_sec": 1.0,
    "word_count":   1,
}


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    n     = int(duration_s * sr)
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n)) * 0.5).astype(np.float32)
    buf   = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Redis mock factory
# ---------------------------------------------------------------------------

def _make_redis_mock(key_raw: str | None = _KEY_RECORD, pipeline_counts=(1, True, 1, True)):
    mock_r = AsyncMock()
    mock_r.aclose = AsyncMock()
    mock_r.get    = AsyncMock(return_value=key_raw)
    mock_r.set    = AsyncMock(return_value=True)
    mock_r.delete = AsyncMock(return_value=1)
    mock_r.sismember = AsyncMock(return_value=False)

    pipe = MagicMock()                                      # pipeline queues sync
    pipe.incr    = MagicMock()
    pipe.expire  = MagicMock()
    pipe.execute = AsyncMock(return_value=list(pipeline_counts))
    mock_r.pipeline = MagicMock(return_value=pipe)
    return mock_r


# ---------------------------------------------------------------------------
# App fixture — patches redis + model before the app is imported
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    mock_r = _make_redis_mock()

    sentinel_model = MagicMock()
    sentinel_tok   = MagicMock()

    with patch("redis.asyncio.from_url", return_value=mock_r):
        from paryaya.api.main import app
        from paryaya.api.state import MODULE_STATE

        MODULE_STATE["model"]     = sentinel_model
        MODULE_STATE["device"]    = "cpu"
        MODULE_STATE["tokenizer"] = sentinel_tok

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        MODULE_STATE.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"
    assert "uptime_seconds" in data


def test_health_ready_ok(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_transcribe_rejects_no_auth(client):
    wav = _make_wav_bytes()
    resp = client.post(
        "/v1/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 401


def test_transcribe_with_valid_key(client):
    wav = _make_wav_bytes()

    with patch("paryaya.api.routes.transcribe.transcribe_bytes", return_value=_TRANSCRIBE_RESULT):
        resp = client.post(
            "/v1/transcribe",
            headers={"Authorization": f"Bearer {_VALID_KEY}"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["transcript"] == "नमस्ते"
    assert data["language"]   == "ne"
    assert "processing_ms"    in data
    assert "model_version"    in data


def test_invalid_format_rejected(client):
    resp = client.post(
        "/v1/transcribe",
        headers={"Authorization": f"Bearer {_VALID_KEY}"},
        files={"file": ("audio.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 400
    assert "format" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_triggered():
    # Force pipeline to report 61 calls in the current minute (starter limit = 60)
    mock_r = _make_redis_mock(pipeline_counts=(61, True, 1, True))

    with patch("redis.asyncio.from_url", return_value=mock_r):
        from paryaya.api.main import app
        from paryaya.api.state import MODULE_STATE

        MODULE_STATE["model"]     = MagicMock()
        MODULE_STATE["tokenizer"] = MagicMock()
        MODULE_STATE["device"]    = "cpu"

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/v1/transcribe",
                headers={"Authorization": f"Bearer {_VALID_KEY}"},
                files={"file": ("test.wav", _make_wav_bytes(), "audio/wav")},
            )

        MODULE_STATE.clear()

    assert resp.status_code == 429
    data = resp.json()
    assert "starter" in data["detail"]
    assert data["plan"] == "starter"


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def test_create_api_key_and_use_it():
    mock_r = _make_redis_mock()

    with patch("redis.asyncio.from_url", return_value=mock_r), \
         patch.dict(os.environ, {"ADMIN_SECRET_KEY": _ADMIN_KEY}):
        from paryaya.api.main import app
        from paryaya.api.state import MODULE_STATE

        MODULE_STATE["model"]     = MagicMock()
        MODULE_STATE["tokenizer"] = MagicMock()
        MODULE_STATE["device"]    = "cpu"

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/auth/keys",
                headers={"ADMIN-SECRET-KEY": _ADMIN_KEY},
                json={"company": "TestCo", "plan": "starter"},
            )

        MODULE_STATE.clear()

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["api_key"].startswith("sk-paryaya-")
    assert data["company"] == "TestCo"
    assert data["plan"]    == "starter"
