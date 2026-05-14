"""
paryaya.inference.faster_whisper_backend — faster-whisper inference wrapper.

4-8x faster than the HuggingFace transformers backend.
Requires the model to be in CTranslate2 format (see scripts/convert_to_ct2.py).

Environment:
    WHISPER_MODEL_PATH  — path to CTranslate2 model dir (default: checkpoints/whisper-medium-nepali-ct2)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


class FasterWhisperBackend:
    def __init__(self, model_path: str, device: str) -> None:
        from faster_whisper import WhisperModel

        ct2_device   = "cuda" if device == "cuda" else "cpu"
        compute_type = "int8_float16" if ct2_device == "cuda" else "int8"

        self.device = device
        self.model  = WhisperModel(
            model_path,
            device=ct2_device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=1,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> dict:
        duration_sec = len(audio) / sample_rate
        segments, _ = self.model.transcribe(
            audio,
            language="ne",
            task="transcribe",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        transcript = " ".join(s.text for s in segments).strip()
        return {
            "transcript":   transcript,
            "confidence":   1.0,
            "duration_sec": round(duration_sec, 3),
            "word_count":   len(transcript.split()) if transcript else 0,
        }

    def translate(self, audio: np.ndarray, sample_rate: int = 16_000) -> dict:
        duration_sec = len(audio) / sample_rate
        segments, _ = self.model.transcribe(
            audio,
            language="ne",
            task="translate",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text for s in segments).strip()
        return {
            "transcript":   text,
            "confidence":   1.0,
            "duration_sec": round(duration_sec, 3),
            "word_count":   len(text.split()) if text else 0,
        }

    def transcribe_bytes(self, audio_bytes: bytes) -> dict:
        import tempfile
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            audio, _ = librosa.load(tmp, sr=16_000, mono=True)
            return self.transcribe(audio)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def translate_bytes(self, audio_bytes: bytes) -> dict:
        import tempfile
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            audio, _ = librosa.load(tmp, sr=16_000, mono=True)
            return self.translate(audio)
        finally:
            Path(tmp).unlink(missing_ok=True)


def load_faster_whisper_backend(device: str | None = None) -> FasterWhisperBackend:
    import torch
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model_path = os.getenv("WHISPER_MODEL_PATH", "checkpoints/whisper-medium-nepali-ct2")
    return FasterWhisperBackend(model_path=model_path, device=device)
