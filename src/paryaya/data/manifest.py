"""
paryaya.data.manifest — Combine all data sources into train/valid/test JSONL manifests.

Manifest format — one JSON object per line:
    {"audio_path": "data/processed/abc.wav",
     "transcript": "नमस्ते, मेरो नाम राम हो।",
     "duration":   4.23,
     "source":     "openslr"}

Split: 90 % train / 5 % valid / 5 % test (deterministic with seed=42).

Usage:
    python -m paryaya.data.manifest                          # auto-discover
    python -m paryaya.data.manifest --manifests a.json b.json --out_dir data/manifests/
"""
import argparse
import json
import random
from pathlib import Path

TRAIN_RATIO = 0.90
VALID_RATIO = 0.05

_DEFAULT_MANIFEST_PATHS = [
    "data/processed/processed_manifest.json",
    "data/augmented/augmented_manifest.json",
    "data/synthetic/synthetic_manifest.json",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_manifests(
    all_samples: list[dict],
    out_dir: str | Path,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Shuffle all_samples with seed, split 90/5/5, write JSONL manifests.

    Prints per-split summary: clip count + total hours.
    Returns the three splits as a dict keyed by "train"/"valid"/"test".
    """
    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    splits = {
        "train": all_samples[:n_train],
        "valid": all_samples[n_train : n_train + n_valid],
        "test":  all_samples[n_train + n_valid :],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, samples in splits.items():
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        hours = sum(s["duration"] for s in samples) / 3600
        print(f"  {name:5s}: {len(samples):7,} clips  {hours:7.1f} h  → {out_path}")

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/valid/test manifests")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=_DEFAULT_MANIFEST_PATHS,
        metavar="FILE",
        help="Source JSONL manifest files to combine (default: auto-discover)",
    )
    parser.add_argument("--out_dir", default="data/manifests",
                        help="Output directory (default: data/manifests/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_samples: list[dict] = []
    for mp in args.manifests:
        p = Path(mp)
        if p.exists():
            loaded = load_jsonl(p)
            all_samples.extend(loaded)
            hrs = sum(s["duration"] for s in loaded) / 3600
            print(f"  Loaded {len(loaded):7,} samples ({hrs:.1f} h) from {p}")
        else:
            print(f"  Skip (not found): {p}")

    if not all_samples:
        print("❌ No samples found. Run download + preprocess first.")
        raise SystemExit(1)

    total_hrs = sum(s["duration"] for s in all_samples) / 3600
    print(f"\n  Total: {len(all_samples):,} samples  ({total_hrs:.1f} h)\n")
    build_manifests(all_samples, args.out_dir, seed=args.seed)
    print(f"\n✅ Manifests written to {args.out_dir}")


if __name__ == "__main__":
    main()
