"""
paryaya.data.preprocess — Standardise raw Nepali audio to 16 kHz mono WAV.

Per-file pipeline:
  1. librosa.load(sr=16 000, mono=True)
  2. Strip leading/trailing silence — librosa.effects.split(top_db=30)
  3. Reject if duration < 0.5 s or > 30 s
  4. Validate Devanagari Unicode in transcript — regex [\\u0900-\\u097F]
  5. Peak-normalise to −3 dBFS
  6. soundfile.write 16-bit PCM WAV

Returns {audio_path, transcript, duration, sample_rate, source} or None if rejected.

Usage:
    python -m paryaya.data.preprocess --input data/raw/ --output data/processed/
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16_000
MIN_DUR = 0.5
MAX_DUR = 30.0
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ---------------------------------------------------------------------------
# Core per-file functions
# ---------------------------------------------------------------------------

def strip_silence(audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    return audio[intervals[0][0] : intervals[-1][1]]


def peak_normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (10 ** (target_db / 20.0) / peak)


def is_valid_nepali(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def process_file(
    audio_path: Path,
    transcript: str,
    out_dir: Path,
    source: str = "unknown",
) -> Optional[dict]:
    """Run the full cleaning pipeline on one audio file.

    Returns a metadata dict on success, None if the clip is rejected.
    """
    if not is_valid_nepali(transcript):
        return None

    try:
        audio, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception:
        return None

    audio = strip_silence(audio)
    duration = len(audio) / TARGET_SR

    if not (MIN_DUR <= duration <= MAX_DUR):
        return None

    audio = peak_normalize(audio)

    out_path = out_dir / (Path(audio_path).stem + ".wav")
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

    return {
        "audio_path": str(out_path),
        "transcript": transcript.strip(),
        "duration": round(duration, 3),
        "sample_rate": TARGET_SR,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Source-specific iterators
# ---------------------------------------------------------------------------

def _iter_openslr(raw_dir: Path):
    """Yield (audio_path, transcript, source) from OpenSLR SLR54 directory tree.

    SLR54 layout:
        ne_np_female/ne_NP_f/line_index.tsv   — <stem>\\t<transcript>
        ne_np_female/ne_NP_f/wavs/<stem>.wav
        ne_np_male/ne_NP_m/line_index.tsv
        ne_np_male/ne_NP_m/wavs/<stem>.wav
    """
    for tsv in raw_dir.rglob("line_index.tsv"):
        wav_dir = tsv.parent / "wavs"
        if not wav_dir.exists():
            wav_dir = tsv.parent  # some releases put wavs alongside the index
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    continue
                stem, transcript = parts
                # Strip any path prefix inside the stem field
                stem = Path(stem).stem if "/" in stem else stem
                for ext in (".wav", ".flac", ".mp3"):
                    wav = wav_dir / f"{stem}{ext}"
                    if wav.exists():
                        yield wav, transcript, "openslr"
                        break


def _iter_hf_dataset(dataset_path: Path, source: str):
    """Yield (array, sr, uid, transcript) from a HuggingFace saved-to-disk dataset.

    HuggingFace audio columns are decoded to {"array": ndarray, "sampling_rate": int, "path": str}.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))

    audio_col = next((c for c in ("audio",) if c in ds.column_names), None)
    text_col = next(
        (c for c in ("sentence", "transcription", "raw_transcription", "transcript")
         if c in ds.column_names),
        None,
    )
    if audio_col is None or text_col is None:
        print(f"  ⚠ {source}: cannot find audio/text columns in {ds.column_names}")
        return

    for i, sample in enumerate(ds):
        audio_data = sample[audio_col]
        transcript = sample.get(text_col, "") or ""
        if not transcript:
            continue
        if isinstance(audio_data, dict):
            arr = np.asarray(audio_data["array"], dtype=np.float32)
            sr = int(audio_data["sampling_rate"])
            uid = Path(audio_data.get("path", f"{source}_{i}")).stem
        else:
            continue
        yield arr, sr, uid, transcript, source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw Nepali audio to clean 16 kHz WAV")
    parser.add_argument("--input", required=True, help="Raw data root directory")
    parser.add_argument("--output", required=True, help="Output directory for clean WAVs")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    skipped = 0

    # --- OpenSLR ---
    openslr_clips = list(_iter_openslr(in_dir))
    if openslr_clips:
        for audio_path, transcript, source in tqdm(openslr_clips, desc="OpenSLR"):
            result = process_file(audio_path, transcript, out_dir, source)
            if result:
                manifest.append(result)
            else:
                skipped += 1

    # --- HuggingFace datasets ---
    for hf_name, source in [("common_voice_ne", "common_voice"), ("fleurs_ne", "fleurs")]:
        hf_path = in_dir / hf_name
        if not hf_path.exists():
            continue
        tmp_path = out_dir / "_tmp_hf.wav"
        for arr, sr, uid, transcript, src in tqdm(
            _iter_hf_dataset(hf_path, source), desc=source
        ):
            sf.write(str(tmp_path), arr, sr, subtype="PCM_16")
            result = process_file(tmp_path, transcript, out_dir, src)
            if result:
                # Rename to avoid collisions across sources
                unique_path = out_dir / f"{src}_{uid}.wav"
                Path(result["audio_path"]).rename(unique_path)
                result["audio_path"] = str(unique_path)
                manifest.append(result)
            else:
                skipped += 1
        tmp_path.unlink(missing_ok=True)

    manifest_path = out_dir / "processed_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_hrs = sum(r["duration"] for r in manifest) / 3600
    print(f"\n✅ Processed {len(manifest)} clips ({total_hrs:.1f} h), skipped {skipped}")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
