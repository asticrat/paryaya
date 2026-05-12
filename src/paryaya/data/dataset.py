"""
paryaya.data.dataset — PyTorch Dataset for Nepali ASR training.

Loads 16 kHz WAV, computes 80-dim log-mel spectrogram, tokenises transcript.

Feature shape:  [T, 80]   — T = ceil(samples / hop_length)
Token shape:    [S]        — S = len(encoded transcript) including <sos>/<eos>

Usage:
    from paryaya.data.dataset import NepaliASRDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = NepaliASRDataset("data/manifests/train.json", tokenizer)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)
    feats, feat_lens, toks, tok_lens = next(iter(loader))
"""
import json
from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class NepaliASRDataset(Dataset):
    """PyTorch Dataset that returns (log_mel_features, token_ids) pairs.

    Args:
        manifest_path: Path to JSONL manifest produced by paryaya.data.manifest.
        tokenizer:     NepaliTokenizer instance (from paryaya.model.tokenizer).
        max_dur:       Clips longer than this (seconds) are excluded at load time.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer,
        max_dur: float = 30.0,
    ) -> None:
        with open(manifest_path, encoding="utf-8") as f:
            self.samples = [
                obj
                for line in f
                if line.strip()
                for obj in [json.loads(line)]
                if obj.get("duration", 0.0) <= max_dur
            ]

        self.tokenizer = tokenizer

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=0.0,
            f_max=8_000.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[idx]

        # soundfile is used instead of torchaudio.load because torchaudio ≥2.4
        # requires torchcodec which may not be installed in all environments.
        # All audio in manifests is already 16 kHz mono WAV (preprocessed).
        import soundfile as _sf
        arr, sr = _sf.read(sample["audio_path"], dtype="float32", always_2d=True)  # always [T, C]
        waveform = torch.from_numpy(arr.copy()).T  # [C, T]
        if sr != 16_000:
            waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)

        # Ensure mono: average channels → [T]
        waveform = waveform.mean(dim=0)

        # [T_frames, 80]
        features: Tensor = self.to_db(self.mel(waveform)).T

        tokens: Tensor = torch.tensor(
            self.tokenizer.encode(sample["transcript"]),
            dtype=torch.long,
        )
        return features, tokens


def collate_fn(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad a variable-length batch to uniform length for DataLoader.

    Returns:
        padded_feats:   [B, T_max, 80]  — log-mel features, zero-padded
        feat_lens:      [B]              — true frame count per item
        padded_tokens:  [B, S_max]      — token ids, pad_id=0
        token_lens:     [B]              — true token count per item
    """
    feats, toks = zip(*batch)

    feat_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tok_lens = torch.tensor([t.shape[0] for t in toks], dtype=torch.long)

    padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    padded_toks = torch.nn.utils.rnn.pad_sequence(
        toks, batch_first=True, padding_value=0
    )

    return padded_feats, feat_lens, padded_toks, tok_lens
