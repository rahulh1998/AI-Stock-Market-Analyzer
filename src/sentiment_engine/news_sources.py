"""Company-aware news retrieval with entity matching, dedup and real timestamps.

The classifier was never the hard part of this layer. The failures that actually corrupt a
sentiment reading are upstream:

  - A bare-ticker query returns the wrong company. Searching "RELIANCE" on Google News
    returns "Reliance Power" and generic Sensex round-ups. Verified against live RSS.
  - Short aliases overlap. "SBI" is a substring of "SBI Cards" and "SBI Life", which are
    separate instruments in this watchlist, so a naive match attributes their news to
    SBIN. Resolved by longest-alias-wins.
  - Index round-ups mention many companies and are about none of them.
  - yfinance returns 0 news items for .NS tickers (verified), so RSS is the real source.
  - pubDate was previously discarded and replaced with "now - 4 hours", which silently
    destroyed the 1h/1d/7d bucketing this engine is built around.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import re
import xml.etree.ElementTree as ET

import requests

from src.paths import INSTRUMENTS_PATH, WATCHLIST_PATH

logger = logging.getLogger(__name__)

RSS_TEMPLATE = ("https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
REQUEST_TIMEOUT = 8

# Quote and listing pages carry a company name but no news. They are pure noise.
BOILERPLATE = (
    "share price", "stock price", "live nse", "live bse", "quote & history",
    "quote and history", "share price today", "stock quote", "price target",
    "historical data", "financial statements",
)
# Unambiguous round-up markers. These headlines list many tickers and are about none of
# them, and they slip under the company-count threshold when most names listed are
# outside this watchlist.
ROUNDUP_MARKERS = (
    "top stocks to watch", "stocks to watch", "shares in focus", "stocks to buy",
    "market wrap", "closing bell", "opening bell", "market live", "markets live",
    "top gainers", "top losers", "f&o ban", "stocks in news", "buzzing stocks",
)
MAX_COMPANIES_PER_TITLE = 3


@dataclass(frozen=True)
class Article:
    title: str
    published: datetime
    source: str
    link: str
    ticker: str
    matched_alias: str

    def age_seconds(self, now: datetime = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.published).total_seconds())

    def to_dict(self) -> dict:
        return {"title": self.title, "published": self.published.isoformat(),
                "source": self.source, "link": self.link,
                "matched_alias": self.matched_alias}


def load_instruments() -> dict:
    if INSTRUMENTS_PATH.exists():
        return json.loads(INSTRUMENTS_PATH.read_text(encoding="utf-8")).get("instruments", {})
    tickers = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8")).get("watchlist", [])
    return {t: {"company_name": t, "aliases": [t]} for t in tickers}


def _alias_pattern(alias: str, case_sensitive: bool = False) -> re.Pattern:
    """Word-boundary match, so 'ITC' cannot match 'switch' and 'UPL' cannot match 'upload'.

    Curated aliases match case-insensitively. The bare ticker SYMBOL matches
    case-sensitively, because "RELIANCE" as a symbol is a different thing from the word
    "Reliance" in "Reliance Power" - a company that is not in this watchlist and whose
    news would otherwise be attributed to RELIANCE. A missed mention costs at most a
    zero sentiment penalty; a misattributed one corrupts the reading.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags)


class EntityMatcher:
    """Attributes a headline to at most one instrument, preferring the longest alias.

    Longest-alias-wins is what keeps "SBI Cards and Payment Services Q2 profit" with
    SBICARD instead of also matching SBIN's "SBI" alias.
    """

    def __init__(self, instruments: dict = None):
        self.instruments = instruments or load_instruments()
        self._patterns = []
        for ticker, meta in self.instruments.items():
            for alias in list(meta.get("aliases") or []):
                if alias:
                    self._patterns.append((len(alias), alias, ticker, _alias_pattern(alias)))
            # The symbol itself, matched case-sensitively (see _alias_pattern).
            self._patterns.append((len(ticker), ticker, ticker, _alias_pattern(ticker, True)))
        # Longest first: the most specific alias claims the headline.
        self._patterns.sort(key=lambda item: -item[0])

    def matches(self, title: str) -> list:
        """All instruments named in the title, each via its longest matching alias."""
        found, claimed = {}, []
        for _, alias, ticker, pattern in self._patterns:
            match = pattern.search(title)
            if not match:
                continue
            span = match.span()
            # A shorter alias inside an already-claimed span is the same mention.
            if any(span[0] >= s and span[1] <= e for s, e in claimed):
                continue
            claimed.append(span)
            found.setdefault(ticker, alias)
        return list(found.items())

    def attribute(self, title: str, ticker: str) -> tuple:
        """(is_relevant, matched_alias, reason)."""
        lowered = title.lower()
        if any(phrase in lowered for phrase in BOILERPLATE):
            return False, None, "boilerplate quote/listing page"
        if any(marker in lowered for marker in ROUNDUP_MARKERS):
            return False, None, "market round-up headline"
        matches = self.matches(title)
        if not matches:
            return False, None, "no instrument alias in title"
        if len(matches) >= MAX_COMPANIES_PER_TITLE:
            return False, None, f"index round-up naming {len(matches)} companies"
        owners = dict(matches)
        if ticker not in owners:
            return False, None, f"headline is about {', '.join(sorted(owners))}"
        return True, owners[ticker], "matched"


def normalise_title(title: str) -> str:
    """Key for dedup: lowercase, publisher suffix dropped, punctuation and spacing flattened."""
    text = re.sub(r"\s+[-|–—]\s+[^-|–—]{1,40}$", "", title.strip())
    # Apostrophes are DELETED, not spaced: "Steel's" and "Steels" must collide, or the
    # same story from two publishers survives dedup twice.
    text = re.sub(r"[’'`]", "", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_published(raw: str) -> datetime:
    """RFC 2822 from RSS, ISO 8601 fallback. Raises rather than inventing a timestamp."""
    if not raw:
        raise ValueError("missing pubDate")
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_queries(ticker: str, meta: dict) -> list:
    """Company-name queries first; the bare ticker is never queried alone."""
    aliases = [a for a in (meta.get("aliases") or []) if len(a.split()) >= 2]
    primary = meta.get("company_name") or ticker
    queries = []
    for alias in [primary] + aliases:
        candidate = f'"{alias}" NSE share'
        if candidate not in queries:
            queries.append(candidate)
    return queries[:2]


def parse_rss(content: bytes) -> list:
    """(title, pubDate, link, publisher) tuples from a Google News RSS payload."""
    items = []
    for item in ET.fromstring(content).findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append((title, item.findtext("pubDate") or "",
                      item.findtext("link") or "",
                      item.findtext("source") or "Google News"))
    return items


def fetch_articles(ticker: str, instruments: dict = None, matcher: EntityMatcher = None,
                   session=None, max_articles: int = 25) -> tuple:
    """(articles, diagnostics). Never raises; diagnostics explain every drop.

    Diagnostics matter here: "no articles" and "20 articles, all about other companies"
    are very different states and the UI must be able to tell them apart.
    """
    instruments = instruments or load_instruments()
    matcher = matcher or EntityMatcher(instruments)
    meta = instruments.get(ticker, {})
    getter = session.get if session else requests.get
    diagnostics = {"ticker": ticker, "queries": [], "fetched": 0, "kept": 0,
                   "dropped": {}, "errors": []}

    raw_items = []
    for query in build_queries(ticker, meta):
        url = RSS_TEMPLATE.format(query=requests.utils.quote(query))
        diagnostics["queries"].append(query)
        try:
            response = getter(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                diagnostics["errors"].append(f"{query}: HTTP {response.status_code}")
                continue
            raw_items.extend(parse_rss(response.content))
        except Exception as exc:
            diagnostics["errors"].append(f"{query}: {type(exc).__name__}: {exc}")

    diagnostics["fetched"] = len(raw_items)
    articles, seen = [], set()
    for title, published_raw, link, publisher in raw_items:
        relevant, alias, reason = matcher.attribute(title, ticker)
        if not relevant:
            diagnostics["dropped"][reason] = diagnostics["dropped"].get(reason, 0) + 1
            continue
        key = normalise_title(title)
        if key in seen:
            diagnostics["dropped"]["duplicate title"] = (
                diagnostics["dropped"].get("duplicate title", 0) + 1)
            continue
        try:
            published = parse_published(published_raw)
        except (ValueError, TypeError) as exc:
            diagnostics["dropped"][f"unparseable date ({exc})"] = (
                diagnostics["dropped"].get(f"unparseable date ({exc})", 0) + 1)
            continue
        seen.add(key)
        articles.append(Article(title=title, published=published, source=publisher,
                                link=link, ticker=ticker, matched_alias=alias))

    articles.sort(key=lambda a: a.published, reverse=True)
    articles = articles[:max_articles]
    diagnostics["kept"] = len(articles)
    return articles, diagnostics


if __name__ == "__main__":
    import argparse
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Inspect news retrieval for one ticker")
    parser.add_argument("--ticker", default="RELIANCE")
    args = parser.parse_args()
    found, info = fetch_articles(args.ticker)
    print(f"queries : {info['queries']}")
    print(f"fetched : {info['fetched']}  kept: {info['kept']}")
    print(f"dropped : {json.dumps(info['dropped'], indent=2)}")
    if info["errors"]:
        print(f"errors  : {info['errors']}")
    print()
    for article in found:
        age = article.age_seconds() / 3600
        print(f"  [{age:6.1f}h] ({article.matched_alias}) {article.title[:95]}")
