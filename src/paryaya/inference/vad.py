"""
paryaya.inference.vad — Voice Activity Detection for chunking long audio.

Splits a long recording into model-sized chunks (≤28 s) while preserving
natural speech boundaries using librosa energy-based VAD.

Usage:
    vad = VoiceActivityDetector()
    chunks = vad.split_audio(audio, sr=16000)
    for start_sample, end_sample, chunk in chunks:
        result = transcribe_audio_array(chunk, model, tokenizer, device)
"""
import math

import librosa
import numpy as np


class VoiceActivityDetector:
    """Energy-based VAD using librosa.effects.split + silence-gap merging."""

    def split_audio(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        max_chunk_sec: float = 28.0,
        min_silence_ms: float = 300.0,
        top_db: float = 30.0,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Split audio into speech chunks suitable for the model (≤28 s each).

        Args:
            audio:          Mono float32 waveform at `sr` Hz.
            sr:             Sample rate.
            max_chunk_sec:  Hard ceiling on chunk duration (s).
            min_silence_ms: Silence gaps shorter than this are bridged.
            top_db:         librosa silence threshold (dB below peak).

        Returns:
            List of (start_sample, end_sample, chunk_array) tuples.
            Indices are into the original `audio` array.
        """
        if len(audio) == 0:
            return []

        intervals = librosa.effects.split(audio, top_db=top_db)

        if len(intervals) == 0:
            return [(0, len(audio), audio)]

        # Merge neighbouring intervals separated by short silence
        min_gap = int(min_silence_ms * sr / 1000)
        merged: list[list[int]] = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start - merged[-1][1] <= min_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        # Split any merged segment that still exceeds max_chunk_sec
        max_samples = int(max_chunk_sec * sr)
        chunks: list[tuple[int, int, np.ndarray]] = []

        for seg_start, seg_end in merged:
            duration = seg_end - seg_start
            if duration <= max_samples:
                chunks.append((seg_start, seg_end, audio[seg_start:seg_end]))
            else:
                n_parts = math.ceil(duration / max_samples)
                part_len = duration // n_parts
                for i in range(n_parts):
                    s = seg_start + i * part_len
                    e = seg_start + (i + 1) * part_len if i < n_parts - 1 else seg_end
                    chunks.append((s, e, audio[s:e]))

        return chunks
