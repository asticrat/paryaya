"""
paryaya — Nepali Automatic Speech Recognition.

Converts spoken Nepali audio to written Devanagari text.
Architecture: Conformer encoder (18 blocks) + CTC/Attention decoder.

Usage:
    from paryaya.inference.transcribe import transcribe_file
    result = transcribe_file("audio.wav", model, tokenizer, device)
    print(result["transcript"])  # नमस्ते, मेरो नाम राम हो।
"""

__version__ = "1.0.0"
__author__ = "Paryaya ASR"
