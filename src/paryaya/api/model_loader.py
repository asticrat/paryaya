"""
paryaya.api.model_loader — Singleton model + tokenizer loader for the API process.

Call get_model() / get_tokenizer() anywhere; they load once on first access
and are reused for the lifetime of the worker process.

Environment variables:
    MODEL_CHECKPOINT  — path to best_model.pt  (default: checkpoints/best_model.pt)
    VOCAB_PATH        — path to nepali_vocab.json (default: data/vocab/nepali_vocab.json)
    DEVICE            — "cuda" | "cpu" | "auto" (default: auto)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer

_tokenizer: NepaliTokenizer | None = None
_model: ParyayaASR | None = None
_device: str | None = None


def _resolve_device() -> str:
    d = os.getenv("DEVICE", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def get_tokenizer() -> NepaliTokenizer:
    global _tokenizer
    if _tokenizer is None:
        vocab = os.getenv("VOCAB_PATH", "data/vocab/nepali_vocab.json")
        _tokenizer = NepaliTokenizer(vocab if Path(vocab).exists() else None)
    return _tokenizer


def get_model() -> tuple[ParyayaASR, str]:
    """Return (model, device). Loads from checkpoint on first call."""
    global _model, _device
    if _model is None:
        ckpt = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
        _device = _resolve_device()
        tok = get_tokenizer()
        if Path(ckpt).exists():
            _model = ParyayaASR.load_from_checkpoint(ckpt, tok.vocab_size).to(_device)
        else:
            raise RuntimeError(
                f"Model checkpoint not found: {ckpt}. "
                "Set MODEL_CHECKPOINT env var or train the model first."
            )
        _model.eval()
    return _model, _device
