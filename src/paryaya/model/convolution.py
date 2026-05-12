"""
paryaya.model.convolution — Depthwise-separable Convolution Module for Conformer.

Architecture (operates on [B, T, D]):
  LayerNorm
  → Conv1d(d, d*2, 1)          pointwise expand
  → GLU(dim=1)                  gates half the channels, output [B, d, T]
  → DepthwiseConv1d(d, d, k)   per-channel temporal convolution
  → BatchNorm1d
  → SiLU
  → Conv1d(d, d, 1)             pointwise contract
  → Dropout

kernel_size must be odd so padding = (k-1)//2 gives same-length output.

Usage:
    conv = ConvolutionModule(d_model=512, kernel_size=31)
    y = conv(x)   # x, y: [B, T, 512]
"""
import torch.nn as nn
from torch import Tensor


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        pad = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_expand = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=pad, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_contract = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:  x: [B, T, D]
        Returns:   [B, T, D]
        """
        x = self.norm(x)
        x = x.transpose(1, 2)            # [B, D, T]
        x = self.pointwise_expand(x)     # [B, D*2, T]
        x = self.glu(x)                  # [B, D, T]
        x = self.depthwise(x)            # [B, D, T]
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise_contract(x)   # [B, D, T]
        x = self.drop(x)
        return x.transpose(1, 2)         # [B, T, D]
