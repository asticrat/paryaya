"""
paryaya.data.synthetic_tts — Generate synthetic Nepali speech from text.

Two backends:
  gtts   — gTTS library, lang="ne". Free, zero cost, lower quality.
           Saves to MP3 then converts to 16 kHz WAV via librosa.
  google — Google Cloud Text-to-Speech, LINEAR16 at 16 kHz.
           Rotates between ne-NP-Standard-A and ne-NP-Standard-B for
           speaker diversity. Requires GOOGLE_APPLICATION_CREDENTIALS.

Input:  one Devanagari sentence per line in text_file (skips lines < 5 chars)
Output: WAV files + synthetic_manifest.json in out_dir
Logs:   progress every 100 items

Usage:
    python -m paryaya.data.synthetic_tts \
        --text_file data/text_corpus/nepali_sentences.txt \
        --output    data/synthetic/ \
        --backend   gtts
"""
import argparse
import json
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

TARGET_SR = 16_000
GOOGLE_VOICES = ["ne-NP-Standard-A", "ne-NP-Standard-B"]


def synthesize_gtts(text: str, out_path: Path) -> float:
    """Synthesise text with gTTS → MP3 → 16 kHz WAV. Returns duration (s)."""
    from gtts import gTTS

    tts = gTTS(text=text, lang="ne")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = Path(tmp.name)
        tts.save(str(mp3_path))

    audio, _ = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
    mp3_path.unlink(missing_ok=True)
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


def synthesize_google(text: str, out_path: Path, voice: str = "ne-NP-Standard-A") -> float:
    """Synthesise text with Google Cloud TTS → 16 kHz WAV. Returns duration (s)."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(language_code="ne-NP", name=voice),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=TARGET_SR,
        ),
    )
    out_path.write_bytes(response.audio_content)
    audio, _ = librosa.load(str(out_path), sr=TARGET_SR, mono=True)
    return len(audio) / TARGET_SR


def batch_synthesize(
    text_file: str | Path,
    out_dir: str | Path,
    backend: str = "gtts",
) -> list[dict]:
    """Synthesise every sentence in text_file and write a manifest JSON.

    Skips lines shorter than 5 characters and logs progress every 100 items.
    Returns list of metadata dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = Path(text_file).read_text(encoding="utf-8").splitlines()
    results: list[dict] = []
    errors = 0

    for i, raw in enumerate(texts):
        text = raw.strip()
        if len(text) < 5:
            continue

        out_path = out_dir / f"tts_{i:06d}.wav"
        try:
            if backend == "google":
                voice = GOOGLE_VOICES[i % len(GOOGLE_VOICES)]
                duration = synthesize_google(text, out_path, voice)
            else:
                duration = synthesize_gtts(text, out_path)

            results.append({
                "audio_path": str(out_path),
                "transcript": text,
                "duration": round(duration, 3),
                "sample_rate": TARGET_SR,
                "source": f"synthetic_tts_{backend}",
            })
        except Exception as exc:
            errors += 1
            print(f"  Skip {i}: {exc}")

        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(texts)} synthesised …")

    manifest_path = out_dir / "synthetic_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_hrs = sum(r["duration"] for r in results) / 3600
    print(f"\n✅ Synthesised {len(results)} clips ({total_hrs:.1f} h), {errors} errors")
    print(f"   Manifest: {manifest_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Nepali TTS audio")
    parser.add_argument("--text_file", required=True, help="Input text file (one sentence per line)")
    parser.add_argument("--output", required=True, help="Output directory for WAVs + manifest")
    parser.add_argument("--backend", choices=["gtts", "google"], default="gtts")
    args = parser.parse_args()
    batch_synthesize(args.text_file, args.output, args.backend)


if __name__ == "__main__":
    main()
