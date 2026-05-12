"""
paryaya.inference.whisper_backend — Whisper inference wrapper.

Drop-in replacement for the ParyayaASR transcription path.
Supports both the base openai/whisper-medium and fine-tuned HF checkpoints.

Environment:
    WHISPER_MODEL_PATH  — path to fine-tuned checkpoint dir, or HF model name
                          default: "openai/whisper-medium"
    ASR_BACKEND         — "whisper" to activate this backend (vs "paryaya")
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import torch


_DEVANAGARI_RE = re.compile(r"[।,?!.\-:\"'॥‌‍]")


def _normalize(text: str) -> str:
    text = _DEVANAGARI_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class WhisperBackend:
    """Loads a Whisper model once and exposes a transcribe() method.

    Works with:
    - "openai/whisper-medium"   (base, no fine-tuning)
    - "openai/whisper-large-v3" (base, best quality)
    - "./checkpoints/whisper-medium-nepali/best"  (fine-tuned)
    """

    def __init__(self, model_path: str, device: str) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device    = device
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model     = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        # Force Nepali transcription (prevent language detection drift)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="ne", task="transcribe"
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> dict:
        """Transcribe a mono float32 numpy array.

        Returns same dict shape as paryaya.inference.transcribe.transcribe_audio_array:
          {transcript, confidence, duration_sec, word_count}
        """
        t0           = time.perf_counter()
        duration_sec = len(audio) / sample_rate

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
            )

        raw_text   = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcript = _normalize(raw_text)

        # Rough confidence: mean max-prob over generated tokens
        with torch.no_grad():
            out = self.model(
                input_features,
                decoder_input_ids=predicted_ids[:, :-1],
            )
            probs      = out.logits.softmax(-1)
            max_probs  = probs.max(-1).values
            confidence = float(max_probs.mean())

        return {
            "transcript":   transcript,
            "confidence":   round(min(confidence, 1.0), 4),
            "duration_sec": round(duration_sec, 3),
            "word_count":   len(transcript.split()) if transcript else 0,
        }

    def transcribe_file(self, path: str | Path) -> dict:
        import librosa
        audio, _ = librosa.load(str(path), sr=16_000, mono=True)
        return self.transcribe(audio)

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


def load_whisper_backend(device: str | None = None) -> WhisperBackend:
    """Load the WhisperBackend from env-configured model path."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model_path = os.getenv("WHISPER_MODEL_PATH", "openai/whisper-medium")
    return WhisperBackend(model_path=model_path, device=device)
