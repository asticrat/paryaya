#!/usr/bin/env python3
"""
Test Whisper large-v3 on 20 Nepali sentences.

Generates audio via gTTS (known ground truth) → runs Whisper → computes WER/CER.
Gives a clear recommendation: fine-tune vs use as backend.

Usage:
    python scripts/test_whisper.py                  # uses large-v3 on MPS/GPU/CPU
    python scripts/test_whisper.py --model medium   # faster, less accurate
    python scripts/test_whisper.py --save_audio     # keep WAV files for inspection
"""
import argparse
import io
import re
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 20 Nepali test sentences — varied length, topic, complexity
# ---------------------------------------------------------------------------
TEST_SENTENCES = [
    # Greetings / basics
    "नमस्ते मेरो नाम राम हो",
    "तपाईंलाई कस्तो छ",
    "तपाईंको नाम के हो",
    "धन्यवाद",

    # Daily life
    "आज मौसम राम्रो छ",
    "मलाई नेपाली खाना मन पर्छ",
    "पानी पर्दैछ",
    "म स्कुल जान्छु",
    "मैले भात खाएँ",
    "मलाई पानी चाहियो",

    # Places / geography
    "काठमाडौं नेपालको राजधानी हो",
    "नेपाल एउटा सुन्दर देश हो",
    "नेपालमा धेरै सुन्दर ठाउँहरू छन्",
    "मेरो घर काठमाडौंमा छ",

    # Numbers / transactions
    "मेरो उमेर पच्चीस वर्ष छ",
    "सय रुपैयाँ दिनुस्",

    # Longer sentences
    "उसले किताब पढ्दैछ",
    "बिहान उठेर व्यायाम गर्नु राम्रो हो",
    "हामी सँगै खाना खाऔं",
    "मेरो बाबा किसान हुनुहुन्छ",
]

assert len(TEST_SENTENCES) == 20, "Must have exactly 20 sentences"


def normalize(text: str) -> str:
    """Strip punctuation and extra whitespace for fair WER comparison."""
    text = re.sub(r"[।,?!.\-:\"'॥]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(ref: str, hyp: str) -> float:
    """Simple word-level edit distance WER."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def char_error_rate(ref: str, hyp: str) -> float:
    """Character-level CER — more informative for Devanagari."""
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def generate_audio(text: str, path: Path) -> bool:
    """Generate gTTS Nepali audio and save as WAV via pydub."""
    try:
        from gtts import gTTS
        import subprocess

        mp3_path = path.with_suffix(".mp3")
        gTTS(text=text, lang="ne").save(str(mp3_path))

        # Convert mp3 → wav with ffmpeg (already installed)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(path)],
            capture_output=True,
        )
        mp3_path.unlink(missing_ok=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  Audio generation failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Whisper on 20 Nepali sentences")
    parser.add_argument("--model",      default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--save_audio", action="store_true",
                        help="Keep generated WAV files in ./whisper_test_audio/")
    parser.add_argument("--device",    default="auto",
                        help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    # Device selection
    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"\n{'='*65}")
    print(f"  Paryaya — Whisper Nepali Benchmark")
    print(f"  Model: whisper-{args.model}   Device: {device}")
    print(f"{'='*65}\n")

    # Load model
    import whisper
    print(f"Loading whisper-{args.model} (first run downloads ~{_model_size(args.model)})...")
    t0 = time.perf_counter()
    model = whisper.load_model(args.model, device=device)
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    # Audio directory
    if args.save_audio:
        audio_dir = Path("whisper_test_audio")
        audio_dir.mkdir(exist_ok=True)
    else:
        _tmp = tempfile.TemporaryDirectory()
        audio_dir = Path(_tmp.name)

    # Run evaluation
    results = []
    print(f"  {'#':>2}  {'Reference':35}  {'Hypothesis':35}  {'WER':>6}  {'CER':>6}")
    print("  " + "─" * 90)

    total_audio_s = 0.0
    total_infer_s = 0.0

    for i, sentence in enumerate(TEST_SENTENCES, 1):
        wav_path = audio_dir / f"test_{i:02d}.wav"

        # Generate audio
        if not generate_audio(sentence, wav_path):
            print(f"  {i:>2}  SKIPPED (audio generation failed)")
            continue

        # Measure duration
        import librosa
        audio, _ = librosa.load(str(wav_path), sr=16000)
        dur = len(audio) / 16000
        total_audio_s += dur

        # Transcribe
        t_start = time.perf_counter()
        result = model.transcribe(str(wav_path), language="ne", task="transcribe")
        elapsed = time.perf_counter() - t_start
        total_infer_s += elapsed

        hyp = normalize(result["text"])
        ref = normalize(sentence)

        wer = word_error_rate(ref, hyp)
        cer = char_error_rate(ref, hyp)

        results.append({"ref": ref, "hyp": hyp, "wer": wer, "cer": cer, "dur": dur})

        # Colour-code: green ≤15%, yellow ≤30%, red >30%
        flag = "✅" if wer <= 0.15 else ("⚠️ " if wer <= 0.30 else "❌")
        print(
            f"  {i:>2}  {ref[:35]:35}  {hyp[:35]:35}  "
            f"{wer:>5.1%}  {cer:>5.1%}  {flag}"
        )

    if not results:
        print("No samples completed.")
        return

    # Summary stats
    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_cer = sum(r["cer"] for r in results) / len(results)
    rtf     = total_infer_s / max(total_audio_s, 1e-6)
    green   = sum(1 for r in results if r["wer"] <= 0.15)
    yellow  = sum(1 for r in results if 0.15 < r["wer"] <= 0.30)
    red     = sum(1 for r in results if r["wer"] > 0.30)

    print("\n  " + "─" * 90)
    print(f"\n  Samples   : {len(results)}/20")
    print(f"  Avg WER   : {avg_wer:.1%}")
    print(f"  Avg CER   : {avg_cer:.1%}")
    print(f"  RTF       : {rtf:.3f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"  ✅ ≤15%   : {green}   ⚠️  ≤30%: {yellow}   ❌ >30%: {red}")

    # Recommendation
    print(f"\n{'='*65}")
    if avg_wer <= 0.15:
        print("  VERDICT: ✅ USE WHISPER AS BACKEND")
        print()
        print("  WER is under 15%. Whisper large-v3 already handles Nepali")
        print("  well enough. Ship the Paryaya API with Whisper inside —")
        print("  skip $800+ scratch training entirely.")
        print()
        print("  Next step → swap ParyayaASR backend for Whisper (I can do this now)")
    elif avg_wer <= 0.30:
        print("  VERDICT: ⚠️  FINE-TUNE WHISPER (don't train from scratch)")
        print()
        print(f"  WER is {avg_wer:.0%} — gap exists but scratch training won't fix it.")
        print("  Fine-tune whisper-medium on Common Voice Nepali (~8h A100 = ~$20).")
        print("  That will likely get you under 10% WER.")
        print()
        print("  Next step → set up whisper fine-tuning script")
    else:
        print("  VERDICT: ❌ GAP IS LARGE — FINE-TUNE OR TRAIN")
        print()
        print(f"  WER is {avg_wer:.0%}. Whisper struggles with this audio type.")
        print("  Fine-tune on real Nepali speech data first.")

    print(f"{'='*65}\n")

    if not args.save_audio:
        _tmp.cleanup()


def _model_size(name: str) -> str:
    sizes = {
        "tiny": "75 MB", "base": "145 MB", "small": "460 MB",
        "medium": "1.5 GB", "large": "2.9 GB",
        "large-v2": "2.9 GB", "large-v3": "2.9 GB",
    }
    return sizes.get(name, "?")


if __name__ == "__main__":
    main()
