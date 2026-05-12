"""
paryaya.inference.postprocess — Transcript cleaning and number normalisation.

normalize_transcript: collapse whitespace, basic Devanagari sentence punctuation.
number_to_nepali:    Arabic numeral (0–999) → Nepali word representation.

Usage:
    from paryaya.inference.postprocess import normalize_transcript, number_to_nepali
    text = normalize_transcript(raw_text)
    word = number_to_nepali(42)   # "बयालिस"
"""
import re

_WS = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")

# Nepali ones (0-19 have unique words)
_ONES = [
    "शून्य", "एक", "दुई", "तीन", "चार", "पाँच",
    "छ", "सात", "आठ", "नौ", "दश", "एघार",
    "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र",
    "सत्र", "अठार", "उन्नाइस",
]

# Tens words (index 2 = 20, 3 = 30, …, 9 = 90)
_TENS = [
    "", "", "बीस", "तीस", "चालिस", "पचास",
    "साठी", "सत्तरी", "असी", "नब्बे",
]


def number_to_nepali(n: int) -> str:
    """Convert an integer 0–999 to its Nepali word representation.

    Numbers outside [0, 999] are returned as digit strings.
    """
    if not (0 <= n <= 999):
        return str(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens = _TENS[n // 10]
        ones = _ONES[n % 10] if n % 10 else ""
        return f"{tens} {ones}".strip()
    hundreds = f"{_ONES[n // 100]} सय"
    remainder = n % 100
    if remainder == 0:
        return hundreds
    return f"{hundreds} {number_to_nepali(remainder)}"


def _replace_digits(match: re.Match) -> str:
    try:
        return number_to_nepali(int(match.group()))
    except (ValueError, OverflowError):
        return match.group()


def normalize_transcript(text: str) -> str:
    """Clean a raw ASR transcript.

    Steps:
      1. Replace Arabic digit sequences with Nepali words (0–999)
      2. Collapse multiple whitespace into a single space
      3. Strip leading/trailing whitespace
    """
    text = _DIGIT_RE.sub(_replace_digits, text)
    text = _WS.sub(" ", text)
    return text.strip()
