"""
paryaya.model.attention — Multi-Head Self-Attention module for ConformerBlock.

Pre-LayerNorm variant with key-padding mask support.

Usage:
    mhsa = MultiHeadSelfAttention(d_model=512, num_heads=8)
    y = mhsa(x, padding_mask)   # x: [B, T, 512]  mask: [B, T] bool
"""
import torch.nn as nn
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention.

    padding_mask follows PyTorch convention: True = position is padding (ignore).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x:            [B, T, D]
            padding_mask: [B, T] bool — True at pad positions
        Returns:
            [B, T, D]
        """
        xn = self.norm(x)
        out, _ = self.attn(xn, xn, xn, key_padding_mask=padding_mask)
        return self.drop(out)
