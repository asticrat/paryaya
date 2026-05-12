"""
paryaya.data.augment — Data augmentation: speed perturbation + noise injection.

Generates 3–5× more training samples from existing clean audio.
Each augmented clip shares the exact same transcript as its source.

Variants per source clip:
  speed_0.9   — 0.9× time-stretch (same pitch, slower)
  speed_1.1   — 1.1× time-stretch (same pitch, faster)
  noise_10db  — SNR 10 dB background noise
  noise_20db  — SNR 20 dB background noise  (lighter noise)

Usage:
    python -m paryaya.data.augment \
        --input     data/processed/ \
        --output    data/augmented/ \
        --noise_dir data/noise_samples/
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16_000


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix background noise into audio at a target signal-to-noise ratio.

    Tiles noise if it is shorter than the signal.
    """
    sig_power = np.mean(audio ** 2)
    nse_power = np.mean(noise ** 2)
    if nse_power == 0:
        return audio
    scale = np.sqrt(sig_power / (nse_power * 10 ** (snr_db / 10.0)))
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    return np.clip(audio + scale * noise[: len(audio)], -1.0, 1.0)


def speed_perturb(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch audio without changing pitch."""
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_file(
    audio_path: Path,
    out_dir: Path,
    noise_files: list[Path],
    transcript: str,
    source: str = "augmented",
) -> list[dict]:
    """Produce all augmented variants for one audio file.

    Returns list of metadata dicts (one per variant successfully written).
    """
    try:
        audio, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception:
        return []

    stem = audio_path.stem
    results: list[dict] = []

    def _save(arr: np.ndarray, suffix: str) -> Optional[dict]:
        out_path = out_dir / f"{stem}_{suffix}.wav"
        try:
            sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")
            return {
                "audio_path": str(out_path),
                "transcript": transcript,
                "duration": round(len(arr) / TARGET_SR, 3),
                "sample_rate": TARGET_SR,
                "source": source,
            }
        except Exception:
            return None

    # Speed perturbation — transcript unchanged because only tempo shifts
    for rate in (0.9, 1.1):
        tag = f"spd{int(rate * 10)}"
        r = _save(speed_perturb(audio, rate), tag)
        if r:
            results.append(r)

    # Noise injection — use only the first noise file to keep I/O bounded
    if noise_files:
        try:
            noise, _ = librosa.load(str(noise_files[0]), sr=TARGET_SR, mono=True)
            for snr in (10, 20):
                r = _save(add_noise(audio, noise, snr), f"noise{snr}db")
                if r:
                    results.append(r)
        except Exception:
            pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment processed Nepali audio")
    parser.add_argument("--input", required=True, help="Directory of clean processed WAVs")
    parser.add_argument("--output", required=True, help="Output directory for augmented WAVs")
    parser.add_argument("--noise_dir", default="data/noise_samples", help="Directory of noise WAVs")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Source JSONL manifest (defaults to <input>/processed_manifest.json)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_dir = Path(args.noise_dir)
    noise_files = sorted(noise_dir.glob("*.wav")) if noise_dir.exists() else []
    if not noise_files:
        print("  ⚠ No noise WAVs found — noise variants will be skipped")

    manifest_path = Path(args.manifest) if args.manifest else in_dir / "processed_manifest.json"
    if not manifest_path.exists():
        print(f"  ⚠ Manifest not found at {manifest_path}; scanning {in_dir} for WAVs")
        samples = [
            {"audio_path": str(p), "transcript": "", "source": "processed"}
            for p in sorted(in_dir.glob("*.wav"))
        ]
    else:
        with open(manifest_path, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

    all_results: list[dict] = []
    for sample in tqdm(samples, desc="Augmenting"):
        audio_path = Path(sample["audio_path"])
        if not audio_path.exists():
            continue
        variants = augment_file(
            audio_path,
            out_dir,
            noise_files,
            transcript=sample.get("transcript", ""),
            source="augmented",
        )
        all_results.extend(variants)

    aug_manifest = out_dir / "augmented_manifest.json"
    with open(aug_manifest, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_hrs = sum(r["duration"] for r in all_results) / 3600
    print(f"\n✅ {len(all_results)} augmented clips ({total_hrs:.1f} h) → {aug_manifest}")


if __name__ == "__main__":
    main()
