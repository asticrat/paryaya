"""
paryaya.inference.transcribe — Audio-to-Devanagari transcription pipelines.

Three entry points for different input forms:
  transcribe_file(path, ...)       — audio file (WAV/MP3/FLAC/etc.)
  transcribe_bytes(bytes, ...)     — raw file bytes (for API uploads)
  transcribe_audio_array(ndarray)  — pre-loaded 16 kHz mono numpy array

All return:
    {
        "transcript":   str,    # Devanagari text
        "confidence":   float,  # mean max-probability over non-blank frames
        "duration_sec": float,
        "word_count":   int,
    }

Usage:
    from paryaya.inference.transcribe import transcribe_audio_array
    result = transcribe_audio_array(audio, model, tokenizer, device)
"""
import io
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch import Tensor

from paryaya.inference.beam_search import ctc_beam_search
from paryaya.inference.postprocess import normalize_transcript

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=16_000, n_fft=400, hop_length=160,
    win_length=400, n_mels=80, f_min=0.0, f_max=8_000.0,
)
_TO_DB = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

TARGET_SR = 16_000


def _audio_to_features(audio: np.ndarray) -> tuple[Tensor, Tensor]:
    """Convert a mono float32 waveform to log-mel feature tensor.

    Returns:
        features: [1, T, 80]
        feat_lens: [1]
    """
    wav = torch.from_numpy(audio).float()
    mel = _TO_DB(_MEL(wav))        # [80, T_frames]
    features = mel.T.unsqueeze(0)  # [1, T_frames, 80]
    feat_lens = torch.tensor([features.shape[1]], dtype=torch.long)
    return features, feat_lens


def _confidence(ctc_logits: Tensor) -> float:
    """Mean max-prob of non-blank frames as a rough confidence score."""
    probs = ctc_logits.softmax(-1)          # [T, V]
    max_probs = probs.max(-1).values        # [T]
    blank_id = 4
    non_blank = probs.argmax(-1) != blank_id
    if non_blank.any():
        return float(max_probs[non_blank].mean())
    return float(max_probs.mean())


def transcribe_audio_array(
    audio: np.ndarray,
    model,
    tokenizer,
    device: str,
    beam_width: int = 10,
) -> dict:
    """Transcribe a pre-loaded 16 kHz mono numpy array.

    Runs entirely under torch.no_grad(); model is temporarily set to eval.
    """
    model.eval()
    duration_sec = len(audio) / TARGET_SR

    features, feat_lens = _audio_to_features(audio)
    features  = features.to(device)
    feat_lens = feat_lens.to(device)

    with torch.no_grad():
        enc_out, _ = model.encoder(features, feat_lens)
        ctc_logits = model.ctc_head(enc_out)   # [1, T, V]

    log_probs  = ctc_logits[0].log_softmax(-1)  # [T, V]
    token_ids  = ctc_beam_search(log_probs, beam_width=beam_width)

    confidence = _confidence(ctc_logits[0])
    transcript = normalize_transcript(tokenizer.decode(token_ids, skip_special=True))

    return {
        "transcript":   transcript,
        "confidence":   round(confidence, 4),
        "duration_sec": round(duration_sec, 3),
        "word_count":   len(transcript.split()) if transcript else 0,
    }


def transcribe_file(
    audio_path: str | Path,
    model,
    tokenizer,
    device: str,
    beam_width: int = 10,
) -> dict:
    """Transcribe an audio file (WAV, MP3, FLAC, OGG, M4A, WEBM)."""
    audio, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    return transcribe_audio_array(audio, model, tokenizer, device, beam_width)


def transcribe_bytes(
    audio_bytes: bytes,
    model,
    tokenizer,
    device: str,
    beam_width: int = 10,
) -> dict:
    """Transcribe raw audio bytes (multipart upload from API clients)."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return transcribe_file(tmp_path, model, tokenizer, device, beam_width)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
