"""
paryaya.model.decoder — Transformer decoder with sinusoidal positional encoding.

Architecture:
    Embedding(vocab_size, d_model)
    → SinusoidalPositionalEncoding
    → nn.TransformerDecoder (batch_first=True)
    → Linear(d_model, vocab_size)   — logits, no softmax

Causal mask is generated inside forward() so callers need not manage it.

Usage:
    dec = TransformerDecoder(vocab_size=120, d_model=512, num_layers=6, num_heads=8)
    logits = dec(tgt, memory)   # tgt: [B,S]  memory: [B,T,D]  → [B,S,V]
"""
import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE added to token embeddings (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder that cross-attends to encoder output."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
        Args:
            tgt:    [B, S]       — target token ids (teacher-forced, no <eos>)
            memory: [B, T, D]   — ConformerEncoder output

        Returns:
            logits: [B, S, vocab_size]
        """
        S = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            S, device=tgt.device
        )
        tgt_emb = self.pe(self.embed(tgt))                        # [B, S, D]
        out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask) # [B, S, D]
        return self.out_proj(out)                                  # [B, S, V]
