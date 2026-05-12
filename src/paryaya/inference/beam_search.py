"""
paryaya.inference.beam_search — CTC greedy and beam-search decoders.

Both return lists of token IDs (integers), not text.
Callers pass the output through NepaliTokenizer.decode().

ctc_greedy_decode: O(T·V)   — fast, used during training eval
ctc_beam_search:   O(T·V·W) — slower but more accurate at inference

Usage:
    from paryaya.inference.beam_search import ctc_greedy_decode, ctc_beam_search
    import torch

    log_probs = ctc_logits[0].log_softmax(-1)   # [T, V]
    ids = ctc_beam_search(log_probs, beam_width=10)
    text = tokenizer.decode(ids)
"""
import numpy as np
import torch
from torch import Tensor


def ctc_greedy_decode(logits: Tensor, blank_id: int = 4) -> list[int]:
    """Greedy CTC decode: argmax → collapse consecutive duplicates → remove blanks.

    Args:
        logits:   [T, V] — raw or log-softmax logits
        blank_id: CTC blank token id

    Returns:
        list of decoded token ids (no blanks, no duplicates)
    """
    ids: list[int] = logits.argmax(-1).tolist()
    if not ids:
        return []
    collapsed = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            collapsed.append(ids[i])
    return [i for i in collapsed if i != blank_id]


def ctc_beam_search(
    log_probs: Tensor,
    beam_width: int = 10,
    blank_id: int = 4,
) -> list[int]:
    """CTC prefix beam search with per-step prefix scoring.

    Maintains `beam_width` candidate prefixes, each tracked by two
    log-probabilities: probability of ending in blank (log_pb) and
    probability of ending in a non-blank token (log_pnb).

    Args:
        log_probs: [T, V] — log-softmax output of CTC head (one utterance)
        beam_width: number of beams to keep at each step
        blank_id:   CTC blank token id

    Returns:
        list of decoded token ids for the best beam
    """
    lp = log_probs.detach().cpu().numpy()   # [T, V]
    T, V = lp.shape
    NEG_INF = float("-inf")

    # beams: prefix_tuple → (log_pb, log_pnb)
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams: dict[tuple, tuple[float, float]] = {}

        for prefix, (log_pb, log_pnb) in beams.items():
            log_p_total = np.logaddexp(log_pb, log_pnb)

            for c in range(V):
                log_pc = lp[t, c]

                if c == blank_id:
                    # Blank: prefix unchanged; only blank prob updated
                    new_pb = np.logaddexp(
                        new_beams.get(prefix, (NEG_INF, NEG_INF))[0],
                        log_p_total + log_pc,
                    )
                    nb = new_beams.get(prefix, (NEG_INF, NEG_INF))[1]
                    new_beams[prefix] = (new_pb, nb)
                else:
                    new_prefix = prefix + (c,)
                    old_pb, old_pnb = new_beams.get(new_prefix, (NEG_INF, NEG_INF))

                    if prefix and prefix[-1] == c:
                        # Same char repeated: only the blank-ending path can extend
                        # without collapsing (non-blank path would collapse the char)
                        new_pnb = np.logaddexp(old_pnb, log_pb + log_pc)
                    else:
                        new_pnb = np.logaddexp(old_pnb, log_p_total + log_pc)

                    new_beams[new_prefix] = (old_pb, new_pnb)

        # Prune: keep top-k by total log-probability
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    if not beams:
        return []
    best = max(beams, key=lambda p: np.logaddexp(beams[p][0], beams[p][1]))
    return list(best)
