"""
paryaya.training.specaugment — SpecAugment data augmentation for log-mel features.

Applies random frequency and time masking (Park et al. 2019).
Masking is only active in training mode; passes through unchanged at eval.

Each item in the batch gets independently sampled masks.

Usage:
    spec_aug = SpecAugment(freq_masks=2, freq_width=27, time_masks=2, time_width=100)
    x = spec_aug(x)   # x: [B, T, F], in-place zero masking
"""
import torch
import torch.nn as nn
from torch import Tensor


class SpecAugment(nn.Module):
    """Random frequency and time masking as an nn.Module (no learned parameters)."""

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 30,
        time_masks: int = 2,
        time_width: int = 50,
    ) -> None:
        super().__init__()
        self.freq_masks  = freq_masks
        self.freq_width  = freq_width
        self.time_masks  = time_masks
        self.time_width  = time_width

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:  x: [B, T, F]
        Returns:   [B, T, F]  — same tensor with bands zeroed (training only)
        """
        if not self.training:
            return x

        B, T, F = x.shape
        out = x.clone()

        for b in range(B):
            # Frequency masking
            for _ in range(self.freq_masks):
                f = torch.randint(0, self.freq_width + 1, (1,)).item()
                f0 = torch.randint(0, max(1, F - f), (1,)).item()
                out[b, :, f0 : f0 + f] = 0.0

            # Time masking
            for _ in range(self.time_masks):
                t = torch.randint(0, min(self.time_width + 1, T), (1,)).item()
                t0 = torch.randint(0, max(1, T - t), (1,)).item()
                out[b, t0 : t0 + t, :] = 0.0

        return out
