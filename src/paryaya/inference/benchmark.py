"""
paryaya.inference.benchmark — End-to-end model evaluation and benchmarking.

Measures WER, CER, Real-Time Factor (RTF), and per-sample latency against a
test manifest. Prints a formatted table and saves evaluation_report.json.

RTF = total_inference_time / total_audio_duration
    < 0.3 means the model runs 3× faster than real-time.

Usage:
    python -m paryaya.inference.benchmark \\
        --checkpoint checkpoints/best_model.pt \\
        --test       data/manifests/test.json \\
        --vocab      data/vocab/nepali_vocab.json
"""
import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from paryaya.inference.transcribe import transcribe_audio_array
from paryaya.model.asr_model import ParyayaASR
from paryaya.model.tokenizer import NepaliTokenizer


def run_benchmark(
    model: ParyayaASR,
    test_manifest_path: str | Path,
    tokenizer: NepaliTokenizer,
    device: str,
    beam_width: int = 10,
) -> dict:
    """Benchmark model on a JSONL test manifest.

    Returns a dict with wer, cer, rtf, avg_latency_ms, n_samples.
    Also prints a formatted per-sample table and saves evaluation_report.json.
    """
    model.eval()

    samples = []
    with open(test_manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    refs:       list[str]   = []
    hyps:       list[str]   = []
    latencies:  list[float] = []
    total_audio_sec  = 0.0
    total_infer_sec  = 0.0
    per_sample: list[dict]  = []

    print(f"\n{'#':>4}  {'Ref (first 30)':30}  {'Hyp (first 30)':30}  {'Lat ms':>7}  {'RTF':>5}")
    print("─" * 80)

    import librosa
    for i, sample in enumerate(samples):
        try:
            audio, _ = librosa.load(sample["audio_path"], sr=16_000, mono=True)
        except Exception:
            continue

        ref = sample.get("transcript", "")
        dur = len(audio) / 16_000

        t0 = time.perf_counter()
        result = transcribe_audio_array(audio, model, tokenizer, device, beam_width)
        latency_s = time.perf_counter() - t0

        hyp = result["transcript"]
        refs.append(ref if ref else " ")
        hyps.append(hyp if hyp else " ")
        latencies.append(latency_s * 1000)
        total_audio_sec  += dur
        total_infer_sec  += latency_s

        rtf_sample = latency_s / max(dur, 1e-6)
        per_sample.append({
            "index":       i,
            "audio_path":  sample["audio_path"],
            "reference":   ref,
            "hypothesis":  hyp,
            "duration_sec": round(dur, 3),
            "latency_ms":  round(latency_s * 1000, 1),
            "rtf":         round(rtf_sample, 3),
        })

        print(
            f"{i:>4}  {ref[:30]:30}  {hyp[:30]:30}  "
            f"{latency_s*1000:>7.1f}  {rtf_sample:>5.3f}"
        )

    if not refs:
        print("No samples evaluated.")
        return {}

    wer = float(jiwer.wer(refs, hyps))
    cer = float(jiwer.cer(refs, hyps))
    rtf = total_infer_sec / max(total_audio_sec, 1e-6)
    avg_lat = float(np.mean(latencies))

    print("─" * 80)
    print(f"\n  Samples : {len(refs)}")
    print(f"  WER     : {wer:.4f}  ({wer*100:.2f} %)")
    print(f"  CER     : {cer:.4f}  ({cer*100:.2f} %)")
    print(f"  RTF     : {rtf:.4f}  {'✅' if rtf < 0.3 else '❌'} (target < 0.3)")
    print(f"  Avg lat : {avg_lat:.1f} ms")

    report = {
        "wer":             wer,
        "cer":             cer,
        "rtf":             round(rtf, 4),
        "avg_latency_ms":  round(avg_lat, 1),
        "n_samples":       len(refs),
        "total_audio_sec": round(total_audio_sec, 1),
        "samples":         per_sample,
    }

    report_path = Path("evaluation_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Report saved → {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ParyayaASR on a test manifest")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--test",       required=True, help="Test JSONL manifest")
    parser.add_argument("--vocab",      default="data/vocab/nepali_vocab.json")
    parser.add_argument("--beam_width", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok    = NepaliTokenizer(args.vocab if Path(args.vocab).exists() else None)
    model  = ParyayaASR.load_from_checkpoint(args.checkpoint, tok.vocab_size).to(device)

    run_benchmark(model, args.test, tok, device, beam_width=args.beam_width)


if __name__ == "__main__":
    main()
