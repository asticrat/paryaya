"""
paryaya.model.subsampling — 4× Conv2D subsampling layer.

Reduces the time axis by 4× using two stride-2 Conv2D layers so the
18 Conformer blocks operate on a 4× shorter sequence, cutting compute ~4×.

Input  mel features:  [B, T, in_dim]      (in_dim=80)
Output encoder input: [B, T/4, d_model]   lengths scaled by //4

Usage:
    sub = ConvSubsampling(in_dim=80, d_model=512)
    out, out_lens = sub(x, lengths)
"""
import torch.nn as nn
from torch import Tensor


class ConvSubsampling(nn.Module):
    """Two stride-2 Conv2D layers followed by a linear projection.

    The frequency dimension goes 80 → 40 → 20, so the linear input is
    d_model * (in_dim // 4) channels.
    """

    def __init__(self, in_dim: int = 80, d_model: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(d_model * (in_dim // 4), d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:       [B, T, in_dim]
            lengths: [B] — true frame counts before padding

        Returns:
            x:       [B, T/4, d_model]
            lengths: [B] — lengths // 4  (long)
        """
        x = x.unsqueeze(1)                    # [B, 1, T, in_dim]
        x = self.conv(x)                       # [B, d_model, T/4, in_dim/4]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)   # [B, T/4, d_model*(in_dim/4)]
        x = self.proj(x)                       # [B, T/4, d_model]
        return x, (lengths // 4).long()
