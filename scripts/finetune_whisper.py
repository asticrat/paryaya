#!/usr/bin/env python3
"""
Fine-tune openai/whisper-medium on Nepali using Google FLEURS (ne_np).

NOTE: Mozilla Common Voice moved off HuggingFace in October 2025.
      We now use google/fleurs (ne_np) which is free, public, and requires no account.

Prerequisites:
  None required — FLEURS is a public dataset. HF_TOKEN is optional.

RunPod A100 40GB — estimated cost ~$10-15 for full run:
  bash scripts/setup_runpod_whisper.sh
  python scripts/finetune_whisper.py --config configs/finetune_whisper.yaml

Local smoke test (verifies pipeline, not full training):
  python scripts/finetune_whisper.py --config configs/finetune_whisper.yaml --smoke_test
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import evaluate


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Split audio and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]}               for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove decoder_start_token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"[।,?!.\-:\"'॥‌‍]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(batch, feature_extractor, tokenizer, max_dur: float,
                    text_col: str = "transcription"):
    audio = batch["audio"]
    arr   = np.array(audio["array"], dtype=np.float32)

    # Skip if too long / too short
    dur = len(arr) / audio["sampling_rate"]
    if dur > max_dur or dur < 0.5:
        batch["input_features"] = None
        batch["labels"]         = None
        return batch

    # Compute log-mel spectrogram
    batch["input_features"] = feature_extractor(
        arr, sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Tokenise transcript — column name differs by dataset
    transcript = normalize_text(batch.get(text_col, "") or "")
    batch["labels"] = tokenizer(transcript).input_ids
    return batch


def compute_metrics(pred, tokenizer, metric):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [normalize_text(t) for t in pred_str]
    label_str = [normalize_text(t) for t in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/finetune_whisper.yaml")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 2 steps to verify pipeline (no full training)")
    args = parser.parse_args()

    cfg      = yaml.safe_load(open(args.config))
    m_cfg    = cfg["model"]
    d_cfg    = cfg["data"]
    t_cfg    = cfg["training"]
    exp_cfg  = cfg.get("export", {})

    hf_token = os.getenv("HF_TOKEN") or None  # optional — FLEURS is public

    # ── Load processor ────────────────────────────────────────────────────────
    base_model = m_cfg["base"]
    print(f"Loading processor from {base_model} ...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        base_model, language=m_cfg["language"], task=m_cfg["task"]
    )
    processor = WhisperProcessor.from_pretrained(
        base_model, language=m_cfg["language"], task=m_cfg["task"]
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    text_col = d_cfg.get("text_column", "transcription")
    print(f"\nLoading {d_cfg['dataset']} ({d_cfg['language_code']}) ...")
    load_kwargs: dict = {"token": hf_token} if hf_token else {}
    raw = load_dataset(d_cfg["dataset"], d_cfg["language_code"], **load_kwargs)

    # Normalise split names — some datasets use "validation", others "dev"
    eval_split = "validation" if "validation" in raw else "dev"
    ds = DatasetDict({
        "train": raw["train"],
        "test":  raw[eval_split],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    if args.smoke_test:
        ds["train"] = ds["train"].select(range(8))
        ds["test"]  = ds["test"].select(range(4))
        t_cfg["max_steps"]  = 2
        t_cfg["eval_steps"] = 2
        t_cfg["save_steps"] = 2
        t_cfg["warmup_steps"] = 0
        print("  ⚡ Smoke test mode — 2 steps only")

    # ── Pre-process ───────────────────────────────────────────────────────────
    print("\nPre-processing audio + tokenising transcripts ...")
    ds = ds.map(
        lambda b: prepare_dataset(
            b, feature_extractor, tokenizer, d_cfg["max_duration_sec"], text_col
        ),
        remove_columns=ds.column_names["train"],
        num_proc=1,
    )
    # Drop None entries (too long / too short)
    ds = ds.filter(lambda x: x["input_features"] is not None)
    print(f"  train={len(ds['train'])}  eval={len(ds['test'])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {base_model} weights ...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model  = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = m_cfg["language"]
    model.generation_config.task     = m_cfg["task"]
    model.generation_config.forced_decoder_ids = None

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── WER metric ────────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")
    _compute   = lambda pred: compute_metrics(pred, tokenizer, wer_metric)

    # ── Training args ─────────────────────────────────────────────────────────
    use_fp16 = t_cfg.get("fp16", True) and device == "cuda"
    out_dir  = t_cfg["output_dir"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        max_steps=t_cfg["max_steps"],
        warmup_steps=t_cfg["warmup_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "linear"),
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        fp16=use_fp16,
        bf16=(device == "cuda" and not use_fp16),
        eval_strategy=t_cfg["evaluation_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        logging_steps=t_cfg["logging_steps"],
        report_to=t_cfg.get("report_to", "none"),
        predict_with_generate=t_cfg["predict_with_generate"],
        generation_max_length=t_cfg["generation_max_length"],
        push_to_hub=False,
        dataloader_num_workers=4 if device == "cuda" else 0,
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=t_cfg.get("early_stopping_patience", 5)
    )]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        compute_metrics=_compute,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n🚀 Training on {device} | steps={t_cfg['max_steps']} | fp16={use_fp16}")
    print(f"   Checkpoint dir: {out_dir}\n")

    # Auto-resume from latest checkpoint if one exists
    checkpoints = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    resume_from = str(checkpoints[-1]) if checkpoints else None
    if resume_from:
        print(f"   Resuming from {resume_from}\n")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save best model ───────────────────────────────────────────────────────
    best_dir = Path(exp_cfg.get("hf_dir", f"{out_dir}/best"))
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    print(f"\n✅ Training complete.")
    print(f"   Best model saved → {best_dir}")
    print(f"\n   Dataset used: {d_cfg['dataset']} ({d_cfg['language_code']})")
    print(f"\n   Deploy to Paryaya API:")
    print(f"   export ASR_BACKEND=whisper")
    print(f"   export WHISPER_MODEL_PATH={best_dir}")
    print(f"   uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
