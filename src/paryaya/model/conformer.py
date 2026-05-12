"""
paryaya.model.conformer — Conformer encoder: ConformerBlock + ConformerEncoder.

Macaron feed-forward structure (Gulati et al. 2020, arXiv 2005.08100):
    x = x + 0.5 * FF1(x)
    x = x + MHSA(x, mask)
    x = x + Conv(x)
    x = x + 0.5 * FF2(x)
    x = LayerNorm(x)

Usage:
    enc = ConformerEncoder(input_dim=80, d_model=512, num_blocks=18, num_heads=8)
    out, out_lens = enc(x, lengths)   # x: [B, T, 80]
"""
import torch
import torch.nn as nn
from torch import Tensor

from paryaya.model.attention import MultiHeadSelfAttention
from paryaya.model.convolution import ConvolutionModule
from paryaya.model.feed_forward import FeedForwardModule
from paryaya.model.subsampling import ConvSubsampling


def _make_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Build a key-padding mask where True marks pad positions.

    Shape: [B, max_len]
    """
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, max_len]
    return idx >= lengths.unsqueeze(1)                                 # [B, max_len]


class ConformerBlock(nn.Module):
    """Single Conformer block with macaron feed-forward structure."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_exp: int = 4,
        kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1  = FeedForwardModule(d_model, ff_exp, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel, dropout)
        self.ff2  = FeedForwardModule(d_model, ff_exp, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x:            [B, T, D]
            padding_mask: [B, T] bool — True at pad positions
        Returns:
            [B, T, D]
        """
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Stack of ConformerBlocks preceded by 4× conv subsampling."""

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_blocks: int = 18,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsampling = ConvSubsampling(input_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:       [B, T, 80]    — log-mel features
            lengths: [B]           — true frame counts

        Returns:
            enc_out:     [B, T/4, d_model]
            out_lengths: [B]
        """
        x, lengths = self.subsampling(x, lengths)
        mask = _make_padding_mask(lengths, x.size(1))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mask)
        return x, lengths
