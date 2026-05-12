"""
tests/test_model.py — Unit tests for model architecture and checkpointing.

Uses a tiny config (d_model=64, 2 enc blocks, 2 dec blocks, 2 heads)
so tests run on CPU in seconds.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from paryaya.model.asr_model import ParyayaASR
from paryaya.model.conformer import ConformerEncoder
from paryaya.model.ctc_head import CTCHead
from paryaya.model.decoder import TransformerDecoder
from paryaya.model.tokenizer import NepaliTokenizer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
D_MODEL    = 64
NUM_ENC    = 2
NUM_DEC    = 2
NUM_HEADS  = 2
BATCH      = 2
T_FRAMES   = 80   # raw frame count before subsampling
S_TOKS     = 10   # sequence length including <sos>/<eos>


@pytest.fixture(scope="module")
def tok():
    return NepaliTokenizer(vocab_file=None)


@pytest.fixture(scope="module")
def model():
    return ParyayaASR(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_enc=NUM_ENC,
        num_dec=NUM_DEC,
        num_heads=NUM_HEADS,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def tiny_batch():
    feats     = torch.randn(BATCH, T_FRAMES, 80)
    feat_lens = torch.tensor([T_FRAMES, T_FRAMES // 2])
    targets   = torch.randint(1, VOCAB_SIZE, (BATCH, S_TOKS))
    targets[:, 0]  = 1   # <sos>
    targets[:, -1] = 2   # <eos>
    return feats, feat_lens, targets


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip(tok):
    texts = [
        "नमस्ते",
        "मेरो नाम राम हो।",
        "यो एक परीक्षण हो",
    ]
    for text in texts:
        ids    = tok.encode(text)
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == text, f"Roundtrip failed: {text!r} → {decoded!r}"


def test_tokenizer_special_tokens(tok):
    assert tok.encode("क")[0]  == 1, "<sos> should be first token"
    assert tok.encode("क")[-1] == 2, "<eos> should be last token"


def test_tokenizer_unknown_char(tok):
    ids = tok.encode("ABC")  # Latin → <unk>
    inner = ids[1:-1]        # strip sos/eos
    assert all(i == 3 for i in inner), "Non-Devanagari chars should map to <unk> (id=3)"


# ---------------------------------------------------------------------------
# Component shapes
# ---------------------------------------------------------------------------

def test_conformer_output_shape(model, tiny_batch):
    feats, feat_lens, _ = tiny_batch
    with torch.no_grad():
        enc_out, enc_lens = model.encoder(feats, feat_lens)
    expected_t = T_FRAMES // 4
    assert enc_out.shape == (BATCH, expected_t, D_MODEL), (
        f"Expected [{BATCH}, {expected_t}, {D_MODEL}], got {list(enc_out.shape)}"
    )


def test_ctc_head_shape(model, tiny_batch):
    feats, feat_lens, _ = tiny_batch
    with torch.no_grad():
        enc_out, _ = model.encoder(feats, feat_lens)
        ctc        = model.ctc_head(enc_out)
    expected_t = T_FRAMES // 4
    assert ctc.shape == (BATCH, expected_t, VOCAB_SIZE), (
        f"Expected [{BATCH}, {expected_t}, {VOCAB_SIZE}], got {list(ctc.shape)}"
    )


def test_decoder_shape(model, tiny_batch):
    feats, feat_lens, targets = tiny_batch
    with torch.no_grad():
        enc_out, _ = model.encoder(feats, feat_lens)
        dec_out    = model.decoder(targets[:, :-1], enc_out)
    assert dec_out.shape == (BATCH, S_TOKS - 1, VOCAB_SIZE), (
        f"Expected [{BATCH}, {S_TOKS - 1}, {VOCAB_SIZE}], got {list(dec_out.shape)}"
    )


# ---------------------------------------------------------------------------
# Full forward
# ---------------------------------------------------------------------------

def test_full_model_forward(model, tiny_batch):
    feats, feat_lens, targets = tiny_batch
    model.eval()
    with torch.no_grad():
        ctc_logits, attn_logits = model(feats, feat_lens, targets)

    expected_t = T_FRAMES // 4
    assert ctc_logits.shape  == (BATCH, expected_t, VOCAB_SIZE)
    assert attn_logits.shape == (BATCH, S_TOKS - 1, VOCAB_SIZE)

    assert not torch.isnan(ctc_logits).any(),  "NaN in CTC logits"
    assert not torch.isnan(attn_logits).any(), "NaN in attention logits"
    assert not torch.isinf(ctc_logits).any(),  "Inf in CTC logits"
    assert not torch.isinf(attn_logits).any(), "Inf in attention logits"


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def test_model_save_load_roundtrip(model, tiny_batch):
    feats, feat_lens, targets = tiny_batch
    model.eval()

    opt = torch.optim.Adam(model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        model.save_checkpoint(ckpt_path, epoch=1, optimizer=opt, loss=0.5)

        restored = ParyayaASR.load_from_checkpoint(ckpt_path, vocab_size=VOCAB_SIZE)
        restored.eval()

        with torch.no_grad():
            ctc_orig, attn_orig = model(feats, feat_lens, targets)
            ctc_rest, attn_rest = restored(feats, feat_lens, targets)

        assert torch.allclose(ctc_orig, ctc_rest,  atol=1e-5), "CTC outputs differ after reload"
        assert torch.allclose(attn_orig, attn_rest, atol=1e-5), "Attn outputs differ after reload"
    finally:
        Path(ckpt_path).unlink(missing_ok=True)
