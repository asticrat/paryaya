"""
paryaya.training.loss — Joint CTC + Attention cross-entropy loss.

total = 0.3 * CTC + 0.7 * CrossEntropy  (weights configurable)

CTC path:
  - log_softmax applied here; nn.CTCLoss receives log-probs
  - targets[:,1:-1] strips <sos> and <eos> (CTC needs raw chars only)
  - input_lens are post-subsampling frame counts (feat_lens // 4)

Attention path:
  - attn_logits shape [B, S-1, V]  (decoder was fed targets[:,:-1])
  - compared against targets[:,1:]  (shift left, skip <sos>)
  - ignore_index=0 (<pad>) so padded positions don't contribute

Usage:
    loss_fn = JointCTCAttentionLoss(vocab_size=120)
    losses = loss_fn(ctc_logits, attn_logits, targets, enc_lens, tok_lens)
    losses["total"].backward()
"""
import torch.nn as nn
from torch import Tensor


class JointCTCAttentionLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 4,
        ctc_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.ctc_w = ctc_weight
        self.ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
        self.ce  = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    def forward(
        self,
        ctc_logits:  Tensor,  # [B, T, V]  — raw logits from CTC head
        attn_logits: Tensor,  # [B, S-1, V] — logits from decoder (teacher-forced)
        targets:     Tensor,  # [B, S]  — full token ids incl. <sos> and <eos>
        input_lens:  Tensor,  # [B] — encoder output lengths (after 4× subsampling)
        target_lens: Tensor,  # [B] — full target lengths incl. <sos> and <eos>
    ) -> dict[str, Tensor]:
        B, T, V = ctc_logits.shape

        # --- CTC loss ---
        log_probs = ctc_logits.log_softmax(-1).permute(1, 0, 2)   # [T, B, V]
        ctc_targets    = targets[:, 1:-1]                          # strip <sos>/<eos>
        ctc_tgt_lens   = (target_lens - 2).clamp(min=1)
        l_ctc = self.ctc(log_probs, ctc_targets, input_lens, ctc_tgt_lens)

        # --- Attention CE loss ---
        S = attn_logits.size(1)
        l_attn = self.ce(
            attn_logits.reshape(B * S, V),
            targets[:, 1 : S + 1].reshape(B * S),
        )

        total = self.ctc_w * l_ctc + (1.0 - self.ctc_w) * l_attn
        return {"total": total, "ctc": l_ctc, "attn": l_attn}
