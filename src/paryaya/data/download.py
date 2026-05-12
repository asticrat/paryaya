"""
paryaya.data.download — Auto-download all free Nepali ASR datasets.

Downloads three sources in priority order:
  1. OpenSLR SLR54   — ~157 h, direct wget from openslr.org/54
  2. Common Voice 13 — ~10 h,  HuggingFace mozilla-foundation/common_voice_13_0
  3. Google FLEURS   — ~10 h,  HuggingFace google/fleurs ne_np

Usage:
    python -m paryaya.data.download --output data/raw/
    python -m paryaya.data.download --output data/raw/ --sources openslr fleurs
"""
import argparse
import subprocess
from pathlib import Path


def download_openslr(out_dir: Path) -> None:
    """Download and unzip OpenSLR SLR54 Nepali (male + female, ~157 h)."""
    urls = [
        "https://www.openslr.org/resources/54/ne_np_female.zip",
        "https://www.openslr.org/resources/54/ne_np_male.zip",
    ]
    for url in urls:
        fname = Path(url).name
        dest = out_dir / fname
        print(f"  Downloading {url} …")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
        subprocess.run(["unzip", "-o", str(dest), "-d", str(out_dir)], check=True)
        dest.unlink()
        print(f"  ✅ {fname} extracted")
    print("✅ OpenSLR SLR54 ready")


def download_common_voice(out_dir: Path) -> None:
    """Download Mozilla Common Voice 13 Nepali via HuggingFace (~10 h)."""
    from datasets import load_dataset

    print("  Downloading Common Voice 13 ne …")
    ds = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "ne",
        split="train+validation+test",
        trust_remote_code=True,
    )
    save_path = out_dir / "common_voice_ne"
    ds.save_to_disk(str(save_path))
    print(f"✅ Common Voice saved → {save_path}")


def download_fleurs(out_dir: Path) -> None:
    """Download Google FLEURS Nepali via HuggingFace (~10 h)."""
    from datasets import load_dataset

    print("  Downloading FLEURS ne_np …")
    ds = load_dataset(
        "google/fleurs",
        "ne_np",
        split="train+validation+test",
        trust_remote_code=True,
    )
    save_path = out_dir / "fleurs_ne"
    ds.save_to_disk(str(save_path))
    print(f"✅ FLEURS saved → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free Nepali ASR datasets")
    parser.add_argument("--output", default="data/raw", help="Root output directory")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["openslr", "common_voice", "fleurs"],
        default=["openslr", "common_voice", "fleurs"],
        help="Datasets to download (default: all three)",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if "openslr" in args.sources:
        download_openslr(out)
    if "common_voice" in args.sources:
        download_common_voice(out)
    if "fleurs" in args.sources:
        download_fleurs(out)

    print(f"\n✅ All requested datasets downloaded to {out}")


if __name__ == "__main__":
    main()
