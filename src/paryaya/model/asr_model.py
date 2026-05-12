"""
paryaya.model.asr_model — ParyayaASR: full end-to-end ASR model.

Combines:
  ConformerEncoder  — extracts high-level acoustic representations
  CTCHead           — auxiliary CTC loss during training
  TransformerDecoder — autoregressive text generation

Forward pass (training):
    ctc_logits, attn_logits = model(features, feat_lens, targets)
    loss = 0.3 * CTC(ctc_logits) + 0.7 * CE(attn_logits)

Usage:
    model = ParyayaASR(vocab_size=120)
    ctc, attn = model(x, lengths, targets)
    # ctc:  [B, T/4, vocab_size]
    # attn: [B, S-1, vocab_size]

    model.save_checkpoint("checkpoints/best_model.pt", epoch=10, optimizer=opt, loss=0.4)
    model = ParyayaASR.load_from_checkpoint("checkpoints/best_model.pt", vocab_size=120)
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from paryaya.model.conformer import ConformerEncoder
from paryaya.model.ctc_head import CTCHead
from paryaya.model.decoder import TransformerDecoder


class ParyayaASR(nn.Module):
    """Conformer-based ASR model for Nepali Devanagari speech recognition."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_enc: int = 18,
        num_dec: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder  = ConformerEncoder(80, d_model, num_enc, num_heads, dropout)
        self.ctc_head = CTCHead(d_model, vocab_size)
        self.decoder  = TransformerDecoder(vocab_size, d_model, num_dec, num_heads, dropout)

    def forward(
        self,
        features: Tensor,
        feat_lens: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            features:  [B, T, 80]  — log-mel features
            feat_lens: [B]         — true frame counts per sample
            targets:   [B, S]      — token ids including <sos> and <eos>

        Returns:
            ctc_logits:  [B, T/4, vocab_size]  — raw logits for CTCLoss
            attn_logits: [B, S-1, vocab_size]  — logits for CrossEntropyLoss
        """
        enc_out, _ = self.encoder(features, feat_lens)     # [B, T/4, d_model]
        ctc_logits  = self.ctc_head(enc_out)               # [B, T/4, V]
        # Teacher forcing: feed targets without the last token
        attn_logits = self.decoder(targets[:, :-1], enc_out)  # [B, S-1, V]
        return ctc_logits, attn_logits

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss: float,
    ) -> None:
        """Persist model + optimiser state to disk."""
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "config": {
                    "vocab_size": self.vocab_size,
                    "d_model":    self.encoder.blocks[0].ff1.net[1].in_features,
                    "num_enc":    len(self.encoder.blocks),
                    "num_dec":    self.decoder.decoder.num_layers,
                    "num_heads":  self.encoder.blocks[0].mhsa.attn.num_heads,
                },
            },
            path,
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | Path,
        vocab_size: int,
        **kwargs,
    ) -> "ParyayaASR":
        """Restore a ParyayaASR from a checkpoint saved by save_checkpoint().

        kwargs override the config stored in the checkpoint (e.g. dropout=0).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = {**ckpt.get("config", {}), **kwargs}
        cfg.pop("vocab_size", None)
        model = cls(vocab_size, **cfg)
        model.load_state_dict(ckpt["model"])
        return model
