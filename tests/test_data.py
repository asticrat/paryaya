"""
tests/test_data.py — Unit tests for the data pipeline.

All tests use tmp_path and synthetic audio; no network access required.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_s: float, sr: int = TARGET_SR) -> Path:
    """Write a synthetic sine-wave WAV at the given path."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _make_manifest(tmp_path: Path, samples: list[dict]) -> Path:
    p = tmp_path / "manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# Preprocess tests
# ---------------------------------------------------------------------------

def test_preprocess_valid_audio_passes(tmp_path):
    from paryaya.data.preprocess import process_file

    wav = _make_wav(tmp_path / "sample.wav", duration_s=2.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = process_file(wav, "नमस्ते मेरो नाम राम हो", out_dir, source="test")

    assert result is not None, "Valid clip should not be rejected"
    assert Path(result["audio_path"]).exists()
    assert result["duration"] > 0
    assert result["source"] == "test"
    assert result["sample_rate"] == TARGET_SR


def test_preprocess_rejects_too_short(tmp_path):
    from paryaya.data.preprocess import process_file

    # 0.1 s is below MIN_DUR (0.5 s)
    wav = _make_wav(tmp_path / "short.wav", duration_s=0.1)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = process_file(wav, "नमस्ते", out_dir, source="test")

    assert result is None, "Sub-0.5s clip should be rejected"


def test_preprocess_rejects_non_nepali(tmp_path):
    from paryaya.data.preprocess import process_file

    wav = _make_wav(tmp_path / "latin.wav", duration_s=2.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = process_file(wav, "Hello world in English only", out_dir, source="test")

    assert result is None, "Latin-only transcript should be rejected"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_augment_speed_changes_length():
    from paryaya.data.augment import speed_perturb

    sr  = TARGET_SR
    dur = 1.0
    audio = np.random.randn(int(dur * sr)).astype(np.float32)

    slow = speed_perturb(audio, rate=0.9)  # slower → longer
    fast = speed_perturb(audio, rate=1.1)  # faster → shorter

    assert len(slow) > len(audio), "0.9× speed should produce longer audio"
    assert len(fast) < len(audio), "1.1× speed should produce shorter audio"


# ---------------------------------------------------------------------------
# Manifest split
# ---------------------------------------------------------------------------

def test_manifest_split_ratios(tmp_path):
    from paryaya.data.manifest import build_manifests

    samples = [
        {"audio_path": f"dummy_{i}.wav", "transcript": "क", "duration": 1.0, "source": "test"}
        for i in range(200)
    ]
    splits = build_manifests(samples, out_dir=tmp_path / "manifests", seed=42)

    n     = len(samples)
    train = splits["train"]
    valid = splits["valid"]
    test  = splits["test"]

    assert len(train) + len(valid) + len(test) == n, "All samples must be assigned"

    train_ratio = len(train) / n
    valid_ratio = len(valid) / n

    assert abs(train_ratio - 0.90) < 0.02, f"Train ratio {train_ratio:.2f} far from 0.90"
    assert abs(valid_ratio - 0.05) < 0.02, f"Valid ratio {valid_ratio:.2f} far from 0.05"

    # JSONL files should exist
    assert (tmp_path / "manifests" / "train.json").exists()
    assert (tmp_path / "manifests" / "valid.json").exists()
    assert (tmp_path / "manifests" / "test.json").exists()


# ---------------------------------------------------------------------------
# Dataset __getitem__ shapes
# ---------------------------------------------------------------------------

def test_dataset_getitem_shapes(tmp_path):
    from paryaya.data.dataset import NepaliASRDataset
    from paryaya.model.tokenizer import NepaliTokenizer

    wav = _make_wav(tmp_path / "speech.wav", duration_s=2.0)
    manifest = _make_manifest(tmp_path, [{
        "audio_path": str(wav),
        "transcript": "नमस्ते",
        "duration": 2.0,
        "source": "test",
    }])

    tok = NepaliTokenizer(vocab_file=None)
    ds  = NepaliASRDataset(manifest, tok, max_dur=30.0)

    assert len(ds) == 1

    feats, tokens = ds[0]

    assert isinstance(feats, torch.Tensor)
    assert isinstance(tokens, torch.Tensor)
    assert feats.ndim == 2, f"Features should be 2D [T, 80], got shape {list(feats.shape)}"
    assert feats.shape[1] == 80, f"Mel bins should be 80, got {feats.shape[1]}"
    assert tokens.ndim == 1, "Tokens should be 1D"
    assert tokens[0].item()  == 1, "<sos> should be first token"
    assert tokens[-1].item() == 2, "<eos> should be last token"
