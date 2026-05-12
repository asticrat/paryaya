"""
paryaya.training.train — Main training loop for ParyayaASR.

Features: fp16 mixed-precision, Noam LR, SpecAugment, W&B logging,
          gradient clipping, best/last checkpoint saving, --resume support.

Usage (RunPod A100):
    WANDB_API_KEY=xxx python -m paryaya.training.train \\
        --config  configs/model_medium.yaml \\
        --train   data/manifests/train.json \\
        --valid   data/manifests/valid.json \\
        --out_dir checkpoints/ \\
        --epochs  100

Smoke test (CPU, no W&B):
    WANDB_MODE=offline python -m paryaya.training.train \\
        --config configs/model_small.yaml \\
        --train  test_manifest.json --valid test_manifest.json \\
        --epochs 2 --batch 2
"""
import argparse
import os
from pathlib import Path

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from paryaya.data.dataset import NepaliASRDataset, collate_fn
from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer
from paryaya.training.callbacks import CheckpointManager, EarlyStopping
from paryaya.training.evaluate import compute_cer, compute_wer
from paryaya.training.loss import JointCTCAttentionLoss
from paryaya.training.optimizer import get_noam_optimizer
from paryaya.training.specaugment import SpecAugment


def _device_autocast(device: str):
    """Return an autocast context for the active device."""
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype)


def train_epoch(
    model: ParyayaASR,
    loader: DataLoader,
    loss_fn: JointCTCAttentionLoss,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    specaugment: SpecAugment,
) -> float:
    model.train()
    specaugment.train()
    total_loss = 0.0

    for step, (feats, feat_lens, toks, tok_lens) in enumerate(loader, 1):
        feats, feat_lens = feats.to(device), feat_lens.to(device)
        toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

        feats = specaugment(feats)

        opt.zero_grad(set_to_none=True)

        with _device_autocast(device):
            ctc_logits, attn_logits = model(feats, feat_lens, toks)
            enc_lens = (feat_lens // 4).long()
            losses   = loss_fn(ctc_logits, attn_logits, toks, enc_lens, tok_lens)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        total_loss += losses["total"].item()

        if step % 50 == 0:
            wandb.log({
                "train/loss":      losses["total"].item(),
                "train/ctc_loss":  losses["ctc"].item(),
                "train/attn_loss": losses["attn"].item(),
                "train/lr":        sched.get_last_lr()[0],
            })

    return total_loss / len(loader)


def valid_epoch(
    model: ParyayaASR,
    loader: DataLoader,
    loss_fn: JointCTCAttentionLoss,
    tokenizer: NepaliTokenizer,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in loader:
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)
            ctc_logits, attn_logits = model(feats, feat_lens, toks)
            enc_lens = (feat_lens // 4).long()
            losses   = loss_fn(ctc_logits, attn_logits, toks, enc_lens, tok_lens)
            total_loss += losses["total"].item()

    wer = compute_wer(model, loader, tokenizer, device)
    cer = compute_cer(model, loader, tokenizer, device)

    return {
        "valid_loss": total_loss / max(len(loader), 1),
        "wer":        wer,
        "cer":        cer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ParyayaASR")
    parser.add_argument("--config",   required=True,         help="YAML config file")
    parser.add_argument("--train",    required=True,         help="Train manifest (JSONL)")
    parser.add_argument("--valid",    required=True,         help="Valid manifest (JSONL)")
    parser.add_argument("--out_dir",  default="checkpoints/")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=None, help="Override config batch size")
    parser.add_argument("--resume",   default=None,          help="Checkpoint path to resume from")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "paryaya-asr"),
        config=cfg,
        mode=os.getenv("WANDB_MODE", "online"),
    )

    # Tokenizer
    vocab_path = os.getenv("VOCAB_PATH", "data/vocab/nepali_vocab.json")
    tok = NepaliTokenizer(vocab_path if Path(vocab_path).exists() else None)

    # Model
    m_cfg  = cfg["model"]
    model  = ParyayaASR(tok.vocab_size, **m_cfg).to(device)
    wandb.watch(model, log_freq=200) if os.getenv("WANDB_WATCH") else None

    # Loss + optimiser
    t_cfg    = cfg["training"]
    loss_fn  = JointCTCAttentionLoss(
        tok.vocab_size,
        ctc_weight=t_cfg["ctc_weight"],
        label_smoothing=t_cfg["label_smooth"],
    )
    opt, sched = get_noam_optimizer(
        model.parameters(),
        d_model=m_cfg["d_model"],
        warmup_steps=t_cfg["warmup_steps"],
    )
    use_amp = t_cfg.get("mixed_precision", True) and device == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    specaugment = SpecAugment(freq_masks=2, freq_width=27, time_masks=2, time_width=100)

    # Data
    d_cfg      = cfg.get("data", {})
    batch_size = args.batch or t_cfg["batch_size"]
    max_dur    = d_cfg.get("max_duration", 30.0)
    train_ds   = NepaliASRDataset(args.train, tok, max_dur=max_dur)
    valid_ds   = NepaliASRDataset(args.valid, tok, max_dur=max_dur)
    n_workers  = 4 if device == "cuda" else 0
    tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_fn, num_workers=n_workers, pin_memory=(device == "cuda"))
    va_dl = DataLoader(valid_ds, batch_size=batch_size,
                       collate_fn=collate_fn, num_workers=min(n_workers, 2))

    # Callbacks
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(out_dir, keep_top_k=3)
    es       = EarlyStopping(patience=t_cfg.get("early_stopping_patience", 15))

    # Resume
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = train_epoch(model, tr_dl, loss_fn, opt, sched, scaler, device, specaugment)
        val     = valid_epoch(model, va_dl, loss_fn, tok, device)

        wandb.log({
            "epoch":         epoch,
            "train/loss":    tr_loss,
            "valid/loss":    val["valid_loss"],
            "valid/wer":     val["wer"],
            "valid/cer":     val["cer"],
        })
        print(
            f"Epoch {epoch:3d} | train {tr_loss:.4f} | "
            f"val {val['valid_loss']:.4f} | WER {val['wer']:.4f} | CER {val['cer']:.4f}"
        )

        # Always save last
        model.save_checkpoint(str(out_dir / "last_model.pt"), epoch, opt, tr_loss)

        # Save among top-k; copy to best_model.pt if WER improved
        ckpt_mgr.save_checkpoint(model, epoch, opt, val["wer"])
        if val["wer"] <= ckpt_mgr.best_wer:
            model.save_checkpoint(str(out_dir / "best_model.pt"), epoch, opt, tr_loss)
            print(f"  ✅ Best checkpoint saved (WER={val['wer']:.4f})")

        if es.update(val["wer"]):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    wandb.finish()


if __name__ == "__main__":
    main()
