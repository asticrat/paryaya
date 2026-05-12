#!/usr/bin/env python3
"""
Evaluate a trained ParyayaASR model on a test manifest.

Computes WER, CER, RTF; per-source breakdown; worst 20 samples by WER.
Saves evaluation_report_{timestamp}.json.

Usage:
    python scripts/evaluate_model.py \
        --checkpoint checkpoints/best_model.pt \
        --manifest   data/manifests/test.json \
        --vocab      data/vocab/nepali_vocab.json
"""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import librosa
import numpy as np
import torch

from paryaya.inference.transcribe import transcribe_audio_array
from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer


def evaluate(
    model: ParyayaASR,
    manifest_path: str | Path,
    tokenizer: NepaliTokenizer,
    device: str,
    beam_width: int = 10,
) -> dict:
    model.eval()

    with open(manifest_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    global_refs:  list[str]  = []
    global_hyps:  list[str]  = []
    per_sample:   list[dict] = []
    total_audio_s = 0.0
    total_infer_s = 0.0

    by_source: dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": []})

    print(f"\n{'#':>5}  {'Source':12}  {'Ref (30 chars)':30}  {'Hyp (30 chars)':30}  {'ms':>7}  {'RTF':>6}")
    print("─" * 100)

    for i, sample in enumerate(samples):
        try:
            audio, _ = librosa.load(sample["audio_path"], sr=16_000, mono=True)
        except Exception as e:
            print(f"  [{i}] skip: {e}")
            continue

        ref = (sample.get("transcript") or "").strip()
        dur = len(audio) / 16_000

        t0 = time.perf_counter()
        result = transcribe_audio_array(audio, model, tokenizer, device, beam_width)
        elapsed = time.perf_counter() - t0

        hyp = result["transcript"]
        total_audio_s += dur
        total_infer_s += elapsed

        ref_safe = ref if ref else " "
        hyp_safe = hyp if hyp else " "
        global_refs.append(ref_safe)
        global_hyps.append(hyp_safe)

        source = sample.get("source", "unknown")
        by_source[source]["refs"].append(ref_safe)
        by_source[source]["hyps"].append(hyp_safe)

        sample_wer = jiwer.wer([ref_safe], [hyp_safe])
        per_sample.append({
            "index":        i,
            "audio_path":   sample["audio_path"],
            "source":       source,
            "reference":    ref,
            "hypothesis":   hyp,
            "duration_sec": round(dur, 3),
            "latency_ms":   round(elapsed * 1000, 1),
            "rtf":          round(elapsed / max(dur, 1e-6), 4),
            "wer":          round(sample_wer, 4),
        })

        print(
            f"{i:>5}  {source:12}  {ref[:30]:30}  {hyp[:30]:30}  "
            f"{elapsed*1000:>7.1f}  {elapsed/max(dur,1e-6):>6.3f}"
        )

    if not global_refs:
        print("No samples evaluated.")
        return {}

    print("─" * 100)

    wer = float(jiwer.wer(global_refs, global_hyps))
    cer = float(jiwer.cer(global_refs, global_hyps))
    rtf = total_infer_s / max(total_audio_s, 1e-6)

    print(f"\n  Samples : {len(global_refs)}")
    print(f"  WER     : {wer:.4f}  ({wer*100:.2f} %)")
    print(f"  CER     : {cer:.4f}  ({cer*100:.2f} %)")
    print(f"  RTF     : {rtf:.4f}  {'✅' if rtf < 0.3 else '❌'} (target < 0.3)")

    # Per-source breakdown
    print("\n  Per-source breakdown:")
    source_stats: list[dict] = []
    for src, data in sorted(by_source.items()):
        s_wer = float(jiwer.wer(data["refs"], data["hyps"]))
        s_cer = float(jiwer.cer(data["refs"], data["hyps"]))
        n     = len(data["refs"])
        print(f"    {src:20}  n={n:6}  WER={s_wer:.4f}  CER={s_cer:.4f}")
        source_stats.append({"source": src, "n": n, "wer": round(s_wer, 4), "cer": round(s_cer, 4)})

    # Worst 20 by WER
    worst = sorted(per_sample, key=lambda x: x["wer"], reverse=True)[:20]
    print("\n  Worst 20 samples (by WER):")
    for row in worst:
        print(
            f"    [{row['index']:>5}] wer={row['wer']:.2f}  "
            f"src={row['source']:12}  ref={row['reference'][:40]!r}"
        )

    report = {
        "wer":             round(wer, 4),
        "cer":             round(cer, 4),
        "rtf":             round(rtf, 4),
        "n_samples":       len(global_refs),
        "total_audio_sec": round(total_audio_s, 1),
        "per_source":      source_stats,
        "worst_samples":   worst,
        "all_samples":     per_sample,
    }

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(f"evaluation_report_{ts}.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Report → {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ParyayaASR on a test manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--vocab",      default="data/vocab/nepali_vocab.json")
    parser.add_argument("--beam_width", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok   = NepaliTokenizer(args.vocab if Path(args.vocab).exists() else None)
    model = ParyayaASR.load_from_checkpoint(args.checkpoint, tok.vocab_size).to(device)

    evaluate(model, args.manifest, tok, device, args.beam_width)


if __name__ == "__main__":
    main()
