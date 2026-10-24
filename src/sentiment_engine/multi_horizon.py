"""Multi-horizon (1h / 1d / 7d) sentiment readings for the signal engine.

Separate from the legacy MultiHorizonSentimentAnalyzer, which belongs to the deprecated
experimental dashboard, writes to SQLite on every Streamlit rerun, and scores with a
keyword lexicon over unfiltered headlines.

Two corrections matter here beyond the scorer swap:

  - Divergence is computed from the last COMPLETED bar's own return, not from a live
    quote. The legacy version used a live `pct_change` field, which would import
    non-causal data into a score attached to a bar close.
  - A reading is cached with a TTL, so opening a page does not re-fetch news.

Horizon buckets are cumulative, matching spec.md section 4.2: an article from 20 minutes
ago counts in all three windows.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import numpy as np
import pandas as pd

from src.paths import DB_PATH
from src.sentiment_engine.finbert_scorer import HeadlineScorer
from src.sentiment_engine.news_sources import EntityMatcher, fetch_articles, load_instruments

HORIZON_SECONDS = {"1h": 3600, "1d": 86400, "1w": 604800}
BULLISH_CUTOFF = 0.20
BEARISH_CUTOFF = -0.20
DEFAULT_TTL_MINUTES = 15

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_readings (
    ticker       TEXT PRIMARY KEY,
    fetched_at   TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def label_for(score: float) -> str:
    if score is None:
        return "NO_DATA"
    if score >= BULLISH_CUTOFF:
        return "BULLISH"
    if score <= BEARISH_CUTOFF:
        return "BEARISH"
    return "NEUTRAL"


@dataclass(frozen=True)
class SentimentReading:
    ticker: str
    sentiment_1h: float = None
    sentiment_1d: float = None
    sentiment_1w: float = None
    counts: dict = field(default_factory=dict)
    articles_count: int = 0
    scorer: str = "none"
    divergence_flag: str = "CONGRUENT"
    headlines: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    fetched_at: str = None

    def to_dict(self) -> dict:
        """Shape consumed by composite.aggregate_sentiment and the UI."""
        return {
            "ticker": self.ticker,
            "sentiment_1h": self.sentiment_1h, "label_1h": label_for(self.sentiment_1h),
            "sentiment_1d": self.sentiment_1d, "label_1d": label_for(self.sentiment_1d),
            "sentiment_1w": self.sentiment_1w, "label_1w": label_for(self.sentiment_1w),
            "counts": dict(self.counts), "articles_count": self.articles_count,
            "scorer": self.scorer, "divergence_flag": self.divergence_flag,
            "latest_headlines": list(self.headlines), "diagnostics": dict(self.diagnostics),
            "fetched_at": self.fetched_at,
        }


def detect_divergence(score_1h, score_1d, bar_return_pct: float) -> str:
    """Sentiment-price disagreement, measured against a completed bar's own return.

    This flags a disagreement worth a second look; it is not evidence of institutional
    activity, and the composite engine treats it only as a risk term.
    """
    if bar_return_pct is None or not np.isfinite(bar_return_pct):
        return "UNKNOWN_NO_PRICE"
    if score_1h is not None:
        if score_1h >= 0.40 and bar_return_pct < -0.40:
            return "BEARISH_DIVERGENCE (price fell on positive news)"
        if score_1h <= -0.40 and bar_return_pct > 0.40:
            return "BULLISH_ABSORPTION (price rose on negative news)"
    if score_1d is not None:
        if score_1d >= 0.35 and bar_return_pct < -0.75:
            return "BEARISH_DRIFT_DIVERGENCE"
        if score_1d <= -0.35 and bar_return_pct > 0.75:
            return "BULLISH_RESILIENCE"
    return "CONGRUENT"


def bar_return_pct(bars: pd.DataFrame) -> float:
    """Percent return of the last completed bar. None when it cannot be computed."""
    if bars is None or len(bars) < 2 or "close" not in bars:
        return None
    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if len(closes) < 2 or closes.iloc[-2] <= 0:
        return None
    return float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)


class SentimentService:
    """Fetch, score, bucket and cache sentiment for one ticker at a time."""

    def __init__(self, db_path=None, ttl_minutes: int = DEFAULT_TTL_MINUTES,
                 use_cache: bool = True):
        self.db_path = str(db_path or DB_PATH)
        self.ttl = timedelta(minutes=ttl_minutes)
        self.instruments = load_instruments()
        self.matcher = EntityMatcher(self.instruments)
        self.scorer = HeadlineScorer(self.db_path, use_cache=use_cache)
        self.conn = None
        if use_cache:
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.executescript(CACHE_SCHEMA)
                self.conn.commit()
            except Exception:
                self.conn = None

    def close(self):
        self.scorer.close()
        if self.conn is not None:
            self.conn.close()

    def cached(self, ticker: str) -> dict:
        """A reading younger than the TTL, else None. Prevents per-rerun refetching."""
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT fetched_at, payload_json FROM sentiment_readings WHERE ticker = ?",
            (ticker,)).fetchone()
        if not row:
            return None
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(row[0])
        except (TypeError, ValueError):
            return None
        if age > self.ttl:
            return None
        payload = json.loads(row[1])
        payload["cache_age_seconds"] = int(age.total_seconds())
        return payload

    def _store(self, ticker: str, payload: dict):
        if self.conn is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO sentiment_readings (ticker, fetched_at, payload_json) "
            "VALUES (?, ?, ?)",
            (ticker, datetime.now(timezone.utc).isoformat(), json.dumps(payload, allow_nan=False)))
        self.conn.commit()

    def reading(self, ticker: str, bars: pd.DataFrame = None, force: bool = False,
                now: datetime = None) -> dict:
        """Cached-or-fresh sentiment reading for one ticker."""
        if not force:
            hit = self.cached(ticker)
            if hit is not None:
                hit["divergence_flag"] = detect_divergence(
                    hit.get("sentiment_1h"), hit.get("sentiment_1d"), bar_return_pct(bars))
                return hit
        now = now or datetime.now(timezone.utc)
        articles, diagnostics = fetch_articles(ticker, self.instruments, self.matcher)
        titles = [a.title for a in articles]
        scores, scorer, cache_hits = self.scorer.score(titles)
        diagnostics["headline_cache_hits"] = cache_hits

        buckets = {name: [] for name in HORIZON_SECONDS}
        for article, score in zip(articles, scores):
            age = article.age_seconds(now)
            for name, limit in HORIZON_SECONDS.items():
                if age <= limit:
                    buckets[name].append(score)

        values = {name: (float(np.mean(v)) if v else None) for name, v in buckets.items()}
        reading = SentimentReading(
            ticker=ticker,
            sentiment_1h=values["1h"], sentiment_1d=values["1d"], sentiment_1w=values["1w"],
            counts={name: len(v) for name, v in buckets.items()},
            articles_count=len(articles), scorer=scorer,
            divergence_flag=detect_divergence(values["1h"], values["1d"], bar_return_pct(bars)),
            headlines=[a.to_dict() for a in articles[:8]], diagnostics=diagnostics,
            fetched_at=now.isoformat(),
        )
        payload = reading.to_dict()
        self._store(ticker, payload)
        return payload


if __name__ == "__main__":
    import argparse
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Show a multi-horizon sentiment reading")
    parser.add_argument("--ticker", default="RELIANCE")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from src.data_pipeline.broker_client import MarketDataFetcher

    candles = MarketDataFetcher().fetch_intraday_data(args.ticker, "5d", "1d")
    service = SentimentService()
    result = service.reading(args.ticker, bars=candles, force=args.force)
    print(f"ticker     : {result['ticker']}")
    print(f"scorer     : {result['scorer']}  (cache hits "
          f"{result['diagnostics'].get('headline_cache_hits')})")
    print(f"articles   : {result['articles_count']}  buckets {result['counts']}")
    for horizon in ("1h", "1d", "1w"):
        value = result[f"sentiment_{horizon}"]
        shown = f"{value:+.3f}" if value is not None else "  none"
        print(f"  {horizon:3s} {shown}  {result[f'label_{horizon}']}")
    print(f"divergence : {result['divergence_flag']}")
    print(f"dropped    : {result['diagnostics'].get('dropped')}")
    print()
    for headline in result["latest_headlines"]:
        print(f"  ({headline['matched_alias']}) {headline['title'][:88]}")
    service.close()
