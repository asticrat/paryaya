"""
paryaya.model.ctc_head — CTC projection head.

Maps encoder hidden states to per-frame vocabulary logits used by CTCLoss.
No softmax — nn.CTCLoss expects raw logits and applies log_softmax internally
when zero_infinity=True.

Usage:
    head = CTCHead(d_model=512, vocab_size=120)
    logits = head(enc_out)   # [B, T/4, 120]
"""
import torch.nn as nn
from torch import Tensor


class CTCHead(nn.Module):
    """Single linear projection from encoder dim to vocab size."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:  x: [B, T, d_model]
        Returns:   [B, T, vocab_size]  — raw logits
        """
        return self.linear(x)
