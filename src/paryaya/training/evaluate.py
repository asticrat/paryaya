"""
paryaya.training.evaluate — WER and CER evaluation utilities.

Both functions run the model in eval mode with torch.no_grad().
Transcription uses CTC greedy decode (argmax → collapse repeats → remove blanks).

Usage:
    wer = compute_wer(model, valid_loader, tokenizer, device)
    cer = compute_cer(model, valid_loader, tokenizer, device)
"""
import torch
from torch.utils.data import DataLoader

import jiwer


def _ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 4) -> list[int]:
    """Argmax, collapse consecutive duplicates, remove blanks."""
    ids = logits.argmax(-1).tolist()
    collapsed = [ids[0]] if ids else []
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            collapsed.append(ids[i])
    return [i for i in collapsed if i != blank_id]


def _decode_batch(
    model,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    toks: torch.Tensor,
    tok_lens: torch.Tensor,
    tokenizer,
    device: str,
) -> tuple[list[str], list[str]]:
    feats     = feats.to(device)
    feat_lens = feat_lens.to(device)

    enc_out, _ = model.encoder(feats, feat_lens)
    ctc_logits = model.ctc_head(enc_out)   # [B, T, V]

    refs, hyps = [], []
    for i in range(ctc_logits.size(0)):
        ids  = _ctc_greedy_decode(ctc_logits[i])
        hyp  = tokenizer.decode(ids, skip_special=True)
        ref  = tokenizer.decode(toks[i, : tok_lens[i]].tolist(), skip_special=True)
        refs.append(ref if ref else " ")
        hyps.append(hyp if hyp else " ")
    return refs, hyps


def compute_wer(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: str,
) -> float:
    """Word Error Rate via CTC greedy decoding."""
    model.eval()
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in dataloader:
            refs, hyps = _decode_batch(model, feats, feat_lens, toks, tok_lens,
                                       tokenizer, device)
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    if not all_refs:
        return 1.0
    return float(jiwer.wer(all_refs, all_hyps))


def compute_cer(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: str,
) -> float:
    """Character Error Rate via CTC greedy decoding."""
    model.eval()
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in dataloader:
            refs, hyps = _decode_batch(model, feats, feat_lens, toks, tok_lens,
                                       tokenizer, device)
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    if not all_refs:
        return 1.0
    return float(jiwer.cer(all_refs, all_hyps))
