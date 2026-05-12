"""
paryaya.model.tokenizer — Nepali Devanagari character-level tokenizer.

Vocabulary (≈120 tokens):
  Special : <pad>=0  <sos>=1  <eos>=2  <unk>=3  <blank>=4
  Script  : Devanagari Unicode U+0900–U+097F
  Extras  : digits 0–9, space, punctuation  .,?!-:()\"/

Usage:
    tok = NepaliTokenizer()
    ids = tok.encode("नमस्ते")   # [1, 45, 78, ...., 2]
    text = tok.decode(ids)        # "नमस्ते"
    tok.save("data/vocab/nepali_vocab.json")
    tok2 = NepaliTokenizer.load("data/vocab/nepali_vocab.json")
"""
import json
from pathlib import Path

_SPECIALS: dict[str, int] = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<blank>": 4,
}
_EXTRA_CHARS = "0123456789 .,?!-:()\"/'"


class NepaliTokenizer:
    """Character-level tokenizer for Nepali Devanagari text."""

    def __init__(self, vocab_file: str | Path | None = None) -> None:
        if vocab_file and Path(vocab_file).exists():
            self.char2id: dict[str, int] = json.loads(
                Path(vocab_file).read_text(encoding="utf-8")
            )
        else:
            self.char2id = dict(_SPECIALS)
            for cp in range(0x0900, 0x0980):
                ch = chr(cp)
                if ch not in self.char2id:
                    self.char2id[ch] = len(self.char2id)
            for ch in _EXTRA_CHARS:
                if ch not in self.char2id:
                    self.char2id[ch] = len(self.char2id)

        self.id2char: dict[int, str] = {v: k for k, v in self.char2id.items()}

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids, prepending <sos> and appending <eos>."""
        ids = [_SPECIALS["<sos>"]]
        for ch in text:
            ids.append(self.char2id.get(ch, _SPECIALS["<unk>"]))
        ids.append(_SPECIALS["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to a string.

        Args:
            ids:          Sequence of token ids.
            skip_special: If True, omit <pad>, <sos>, <eos>, <unk>, <blank>.
        """
        skip = set(_SPECIALS.values()) if skip_special else set()
        return "".join(self.id2char.get(i, "?") for i in ids if i not in skip)

    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def save(self, path: str | Path) -> None:
        """Serialise vocabulary to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.char2id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NepaliTokenizer":
        """Deserialise vocabulary from JSON produced by save()."""
        obj = cls.__new__(cls)
        obj.char2id = json.loads(Path(path).read_text(encoding="utf-8"))
        obj.id2char = {v: k for k, v in obj.char2id.items()}
        return obj
