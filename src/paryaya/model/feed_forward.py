"""
paryaya.model.feed_forward — Conformer Feed-Forward Module.

Macaron-style: applied at half residual weight (0.5) at the start and end
of each ConformerBlock.

Architecture: LayerNorm → Linear(d, d*exp) → SiLU → Dropout
              → Linear(d*exp, d) → Dropout

Usage:
    ff = FeedForwardModule(d_model=512)
    y = ff(x)   # x, y: [B, T, 512]
"""
import torch.nn as nn
from torch import Tensor


class FeedForwardModule(nn.Module):
    """Positionwise feed-forward with pre-LayerNorm and SiLU activation."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
