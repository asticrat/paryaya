"""
paryaya.data.text_corpus_builder — Build a Nepali Devanagari text corpus for TTS.

Sources:
  1. Wikipedia Nepali — MediaWiki API (ne.wikipedia.org), no auth required.
     Fetches random article extracts, splits on ।/\\n, filters by length + script.
  2. Nepali news RSS — public RSS feeds, no auth required.
     Extracts <title> and <description> tags, strips HTML.

Output: deduplicated sentences 5–100 chars, one per line, Devanagari only.

Usage:
    python -m paryaya.data.text_corpus_builder \
        --output data/text_corpus/nepali_sentences.txt \
        --n_articles 5000
"""
import argparse
import re
import time
from pathlib import Path

import requests

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_NOISE_RE = re.compile(r"[^ऀ-ॿ\s.,!?।\-0-9]")
_MIN_LEN = 5
_MAX_LEN = 100

_RSS_FEEDS = [
    "https://www.onlinekhabar.com/feed",
    "https://ekantipur.com/rss",
    "https://ratopati.com/rss",
    "https://www.setopati.com/rss",
    "https://nayapatrikadaily.com/feed",
]


def _clean(text: str) -> str:
    text = _NOISE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_valid(sentence: str) -> bool:
    return _MIN_LEN <= len(sentence) <= _MAX_LEN and bool(_DEVANAGARI_RE.search(sentence))


def fetch_nepali_wikipedia(n_articles: int = 5000) -> list[str]:
    """Fetch Nepali article extracts via the Wikipedia API and split into sentences.

    Respects a 0.5 s crawl delay between batch requests.
    """
    sentences: list[str] = []
    session = requests.Session()
    api = "https://ne.wikipedia.org/w/api.php"
    collected = 0

    while collected < n_articles:
        # Step 1: get a batch of random page IDs
        try:
            resp = session.get(
                api,
                params={"action": "query", "list": "random", "rnnamespace": 0,
                        "rnlimit": 50, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            pages = resp.json()["query"]["random"]
        except Exception as exc:
            print(f"  Wikipedia list error: {exc}")
            break

        page_ids = "|".join(str(p["id"]) for p in pages)

        # Step 2: fetch plain-text extracts (20 sentences per page)
        try:
            eresp = session.get(
                api,
                params={"action": "query", "pageids": page_ids, "prop": "extracts",
                        "explaintext": True, "exsentences": 20, "format": "json"},
                timeout=15,
            )
            eresp.raise_for_status()
            page_data = eresp.json()["query"]["pages"]
        except Exception as exc:
            print(f"  Wikipedia extract error: {exc}")
            time.sleep(2)
            continue

        for page in page_data.values():
            extract = page.get("extract", "")
            for raw in re.split(r"[।\n]", extract):
                cleaned = _clean(raw)
                if _is_valid(cleaned):
                    sentences.append(cleaned)

        collected += len(pages)
        if collected % 500 == 0:
            print(f"  Wikipedia: {collected}/{n_articles} articles, {len(sentences)} sentences …")
        time.sleep(0.5)

    print(f"  Wikipedia done: {len(sentences)} sentences from {collected} articles")
    return sentences


def fetch_nepali_news() -> list[str]:
    """Scrape Nepali news headlines + descriptions from public RSS feeds."""
    sentences: list[str] = []
    for feed_url in _RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=10)
            resp.raise_for_status()
            # Extract text without adding feedparser as a dependency
            for tag in ("title", "description", "summary"):
                for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", resp.text, re.DOTALL):
                    raw = re.sub(r"<[^>]+>", "", m.group(1))  # strip HTML tags
                    raw = raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    for part in re.split(r"[।\n]", raw):
                        cleaned = _clean(part)
                        if _is_valid(cleaned):
                            sentences.append(cleaned)
            print(f"  RSS OK: {feed_url}")
        except Exception as exc:
            print(f"  RSS skip {feed_url}: {exc}")

    print(f"  News done: {len(sentences)} sentences")
    return sentences


def save_corpus(sentences: list[str], path: Path) -> None:
    """Deduplicate and write sentences, one per line."""
    unique = list(dict.fromkeys(s for s in sentences if _is_valid(s)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(unique), encoding="utf-8")
    print(f"✅ Saved {len(unique):,} unique sentences → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nepali text corpus for TTS synthesis")
    parser.add_argument("--output", default="data/text_corpus/nepali_sentences.txt")
    parser.add_argument("--n_articles", type=int, default=5000,
                        help="Target number of Wikipedia articles to scrape")
    parser.add_argument("--skip_wikipedia", action="store_true")
    parser.add_argument("--skip_news", action="store_true")
    args = parser.parse_args()

    sentences: list[str] = []

    if not args.skip_wikipedia:
        sentences.extend(fetch_nepali_wikipedia(args.n_articles))

    if not args.skip_news:
        sentences.extend(fetch_nepali_news())

    save_corpus(sentences, Path(args.output))


if __name__ == "__main__":
    main()
