"""Headline sentiment scoring: FinBERT when available, lexicon when not, always labelled.

score = P(positive) - P(negative), in [-1, +1].

Three practical constraints shape this module:
  - torch here is a CPU-only build, so scoring must be one batched forward pass, not a
    call per headline, and must not run across the whole watchlist on a page load.
  - ProsusAI/finbert is a ~440MB download. If it is unavailable for any reason the
    scorer degrades to the keyword lexicon and says so via `scorer`, because a caller
    must be able to tell a model reading from a keyword guess.
  - FinBERT was trained on analyst-style financial English and is least reliable on
    exactly this corpus of Indian market headlines, which is why the composite engine
    caps the sentiment penalty low.
"""
import hashlib
import logging
import re
import sqlite3
import threading

from src.paths import DB_PATH

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 96
LEXICON_SCORER = "lexicon_fallback"
FINBERT_SCORER = "finbert"

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS headline_sentiment (
    cache_key TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    score      REAL NOT NULL,
    scorer     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# Weighted financial lexicon used only for the fallback. Corrected from the original:
# 'high'/'low' fired indiscriminately on "52-week low", and 'fine' matched the adjective.
BULLISH_TERMS = {
    "surge": 1.0, "jump": 0.9, "gain": 0.8, "growth": 0.8, "profit": 0.9, "rally": 1.0,
    "bullish": 0.8, "upgrade": 1.0, "beats": 0.9, "outperform": 1.0, "soar": 1.0,
    "dividend": 0.6, "breakout": 0.9, "acquisition": 0.6, "expansion": 0.7,
    "record high": 0.9, "order win": 0.9, "wins order": 0.9, "bags order": 0.9,
    "stake buy": 0.7, "buyback": 0.8, "fully subscribed": 0.7, "premium listing": 0.8,
}
BEARISH_TERMS = {
    "fall": 0.8, "drop": 0.9, "loss": 1.0, "crash": 1.0, "bearish": 0.8,
    "downgrade": 1.0, "misses": 0.9, "probe": 1.0, "slump": 1.0, "plunge": 1.0,
    "decline": 0.8, "penalty": 0.9, "fraud": 1.0, "investigation": 1.0, "weak": 0.7,
    "headwind": 0.8, "default": 1.0, "lawsuit": 0.9, "52-week low": 0.9,
    "record low": 0.9, "profit warning": 1.0, "cut target": 0.8, "stake sale": 0.5,
    "resigns": 0.7, "raid": 1.0, "ban": 0.8, "recall": 0.7,
}

_LOCK = threading.Lock()
_MODEL = None
_TOKENIZER = None
_LOAD_FAILED = None


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(term)}", re.IGNORECASE)


_BULLISH_PATTERNS = {t: (_term_pattern(t), w) for t, w in BULLISH_TERMS.items()}
_BEARISH_PATTERNS = {t: (_term_pattern(t), w) for t, w in BEARISH_TERMS.items()}


def lexicon_score(title: str) -> float:
    """Keyword fallback. Prefix-anchored so 'profit' does not fire inside 'nonprofit'."""
    positive = sum(w for pattern, w in _BULLISH_PATTERNS.values() if pattern.search(title))
    negative = sum(w for pattern, w in _BEARISH_PATTERNS.values() if pattern.search(title))
    total = positive + negative
    if total == 0:
        return 0.0
    return float(max(min((positive - negative) / (total + 0.5), 1.0), -1.0))


def load_finbert():
    """Lazy module-level singleton. Returns (tokenizer, model) or (None, None)."""
    global _MODEL, _TOKENIZER, _LOAD_FAILED
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    if _LOAD_FAILED is not None:
        return None, None
    with _LOCK:
        if _MODEL is not None:
            return _TOKENIZER, _MODEL
        if _LOAD_FAILED is not None:
            return None, None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model.eval()
            torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
            _TOKENIZER, _MODEL = tokenizer, model
            logger.info("FinBERT loaded (%s)", MODEL_NAME)
            return _TOKENIZER, _MODEL
        except Exception as exc:
            _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
            logger.warning("FinBERT unavailable, falling back to lexicon: %s", _LOAD_FAILED)
            return None, None


def finbert_available() -> bool:
    return load_finbert()[0] is not None


def _label_indices(model) -> dict:
    """Map 'positive'/'negative' onto column indices from the model's own config."""
    mapping = {}
    for index, label in (getattr(model.config, "id2label", None) or {}).items():
        mapping[str(label).lower()] = int(index)
    return mapping


def score_batch(titles: list) -> tuple:
    """(scores, scorer_name). One forward pass for the whole batch."""
    if not titles:
        return [], FINBERT_SCORER if finbert_available() else LEXICON_SCORER
    tokenizer, model = load_finbert()
    if tokenizer is None:
        return [lexicon_score(t) for t in titles], LEXICON_SCORER
    try:
        import torch

        encoded = tokenizer(titles, padding=True, truncation=True,
                            max_length=MAX_TOKENS, return_tensors="pt")
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        indices = _label_indices(model)
        positive, negative = indices.get("positive", 0), indices.get("negative", 1)
        return [float(row[positive] - row[negative]) for row in probabilities], FINBERT_SCORER
    except Exception as exc:
        logger.warning("FinBERT inference failed, falling back to lexicon: %s", exc)
        return [lexicon_score(t) for t in titles], LEXICON_SCORER


class HeadlineScorer:
    """Batched scoring with a SQLite cache keyed on the title and the scorer identity."""

    def __init__(self, db_path=None, use_cache: bool = True):
        self.db_path = str(db_path or DB_PATH)
        self.use_cache = use_cache
        self.conn = None
        if use_cache:
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.executescript(CACHE_SCHEMA)
                self.conn.commit()
            except Exception as exc:
                logger.warning("Headline cache unavailable: %s", exc)
                self.conn = None

    def close(self):
        if self.conn is not None:
            self.conn.close()

    @staticmethod
    def cache_key(title: str) -> str:
        """Keyed on the title alone; the row records which scorer produced the value.

        Keying on scorer too would force a 440MB model load just to compute the lookup
        key, even when every headline is already cached - which is the common case for
        the watchlist scanner.
        """
        return hashlib.sha1(title.encode("utf-8")).hexdigest()

    def score(self, titles: list, rescore: bool = False) -> tuple:
        """(scores aligned to titles, scorer name, cache_hits).

        The model is loaded only when something is actually missing from the cache.
        Pass rescore=True to discard lexicon-scored rows and retry with FinBERT.
        """
        if not titles:
            return [], LEXICON_SCORER, 0
        cached, used = {}, {}
        if self.conn is not None and not rescore:
            for title in titles:
                row = self.conn.execute(
                    "SELECT score, scorer FROM headline_sentiment WHERE cache_key = ?",
                    (self.cache_key(title),)).fetchone()
                if row is not None:
                    cached[title] = float(row[0])
                    used[title] = row[1]
        pending = [t for t in titles if t not in cached]
        scorer = None
        if pending:
            fresh, scorer = score_batch(pending)
            for title, value in zip(pending, fresh):
                cached[title] = value
                used[title] = scorer
            if self.conn is not None:
                from datetime import datetime, timezone
                stamp = datetime.now(timezone.utc).isoformat()
                self.conn.executemany(
                    "INSERT OR REPLACE INTO headline_sentiment "
                    "(cache_key, title, score, scorer, created_at) VALUES (?, ?, ?, ?, ?)",
                    [(self.cache_key(t), t, cached[t], scorer, stamp) for t in pending])
                self.conn.commit()
        names = set(used.values())
        if scorer is None:
            # Everything came from cache; report what produced it, flagging any mix.
            scorer = names.pop() if len(names) == 1 else "mixed"
        elif len(names) > 1:
            scorer = "mixed"
        return [cached[t] for t in titles], scorer, len(titles) - len(pending)


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    samples = [
        "Reliance Industries shares jump 3% as Jio IPO fully subscribed",
        "Tata Steel slumps 6% after Q2 loss widens on weak demand",
        "SBI raises Rs 10,000 crore via infrastructure bonds",
        "Sebi opens probe into alleged disclosure lapses at the company",
        "Analysts downgrade the stock, cut target price by 15%",
    ]
    print(f"FinBERT available: {finbert_available()}")
    if not finbert_available():
        print(f"reason: {_LOAD_FAILED}")
    scorer = HeadlineScorer()
    values, used, hits = scorer.score(samples)
    print(f"scorer: {used}  cache hits: {hits}\n")
    for title, value in zip(samples, values):
        print(f"  {value:+.3f}  (lexicon {lexicon_score(title):+.3f})  {title[:70]}")
    scorer.close()
