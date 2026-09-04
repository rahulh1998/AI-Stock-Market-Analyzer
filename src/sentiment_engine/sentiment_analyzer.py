import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import requests
import yfinance as yf

from src.data_pipeline.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.getcwd(), "config", "watchlist.json")

class MultiHorizonSentimentAnalyzer:
    """
    Multi-horizon financial news sentiment analyzer.
    Captures sentiment velocity across:
    - 1 Hour (T <= 1h): Immediate flash catalysts, earnings surprises, breaking events.
    - 24 Hours / 1 Day (1h < T <= 24h): Session sentiment and overnight digests.
    - 7 Days / 1 Week (1d < T <= 7d): Multi-day trend health and macro drift.
    Includes Sentiment-Price Divergence Detection (e.g., Sell-on-Good-News).
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.watchlist = self._load_watchlist()

        # Financial sentiment lexicon with directional weights
        self.bullish_keywords = {
            'surge': 1.0, 'jump': 0.9, 'gain': 0.8, 'growth': 0.8, 'profit': 0.9,
            'rally': 1.0, 'bull': 0.8, 'upgrade': 1.0, 'beat': 0.9, 'outperform': 1.0,
            'soar': 1.0, 'high': 0.6, 'dividend': 0.7, 'breakout': 0.9, 'acquisition': 0.7,
            'expansion': 0.7, 'revenue up': 1.0, 'orders': 0.7, 'positive': 0.6, 'record': 0.8
        }
        self.bearish_keywords = {
            'fall': 0.8, 'drop': 0.9, 'loss': 1.0, 'crash': 1.0, 'bear': 0.8,
            'downgrade': 1.0, 'miss': 0.9, 'probe': 1.0, 'slump': 1.0, 'plunge': 1.0,
            'decline': 0.8, 'low': 0.6, 'fine': 0.9, 'fraud': 1.0, 'investigation': 1.0,
            'weak': 0.7, 'headwind': 0.8, 'debt': 0.8, 'default': 1.0, 'lawsuit': 0.9
        }

    def _load_watchlist(self) -> List[str]:
        if not os.path.exists(CONFIG_PATH):
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f).get("watchlist", [])
        except Exception:
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

    def _score_text(self, text: str) -> float:
        """Calculates normalized sentiment score from -1.0 to +1.0 using weighted financial terms."""
        text_lower = text.lower()
        pos_score = 0.0
        neg_score = 0.0

        for word, weight in self.bullish_keywords.items():
            if word in text_lower:
                pos_score += weight

        for word, weight in self.bearish_keywords.items():
            if word in text_lower:
                neg_score += weight

        total = pos_score + neg_score
        if total == 0:
            return 0.0

        # Normalization with soft ceiling
        raw_score = (pos_score - neg_score) / (total + 0.5)
        return float(max(min(raw_score, 1.0), -1.0))

    def _score_to_label(self, score: float) -> str:
        if score >= 0.20:
            return "BULLISH"
        elif score <= -0.20:
            return "BEARISH"
        return "NEUTRAL"

    def fetch_news_articles(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Gathers time-stamped news articles from yfinance and Google News RSS for the ticker.
        """
        clean_ticker = ticker.split('.')[0].upper()
        yf_symbol = f"{clean_ticker}.NS"
        articles: List[Dict[str, Any]] = []

        now_ts = time.time()

        # 1. Fetch yfinance news
        try:
            t = yf.Ticker(yf_symbol)
            yf_news = getattr(t, "news", []) or []
            for item in yf_news:
                title = item.get('title') or item.get('content', {}).get('title', '')
                pub_ts = item.get('providerPublishTime') or item.get('content', {}).get('pubDate')

                if isinstance(pub_ts, str):
                    try:
                        dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                        pub_ts = dt.timestamp()
                    except Exception:
                        pub_ts = now_ts - 3600
                elif not isinstance(pub_ts, (int, float)):
                    pub_ts = now_ts - 3600

                if title:
                    articles.append({
                        "title": title,
                        "timestamp": float(pub_ts),
                        "source": item.get('publisher', 'Yahoo Finance'),
                        "link": item.get('link', '')
                    })
        except Exception as e:
            logger.warning(f"yfinance news error for {ticker}: {e}")

        # 2. Complement with Google News RSS if yfinance returns < 3 articles
        if len(articles) < 3:
            try:
                rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+share+NSE&hl=en-IN&gl=IN&ceid=IN:en"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                resp = requests.get(rss_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item")[:5]:
                        title = item.findtext("title", "")
                        pub_str = item.findtext("pubDate", "")
                        pub_ts = now_ts - 14400 # Default ~4h ago
                        if pub_str:
                            try:
                                dt = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S %Z")
                                pub_ts = dt.replace(tzinfo=timezone.utc).timestamp()
                            except Exception:
                                pass
                        if title:
                            articles.append({
                                "title": title,
                                "timestamp": float(pub_ts),
                                "source": "Google News",
                                "link": item.findtext("link", "")
                            })
            except Exception as e:
                logger.warning(f"Google RSS fetch warning for {ticker}: {e}")

        return articles

    def analyze_ticker(self, ticker: str, live_pct_change: float = 0.0) -> Dict[str, Any]:
        """
        Segments news into 1h, 1d, 1w buckets, calculates sentiment, and flags divergences.
        """
        clean_ticker = ticker.split('.')[0].upper()
        articles = self.fetch_news_articles(clean_ticker)
        now_ts = time.time()

        articles_1h = []
        articles_1d = []
        articles_1w = []

        for art in articles:
            delta = now_ts - art["timestamp"]
            if delta <= 3600: # 1 Hour
                articles_1h.append(art)
                articles_1d.append(art)
                articles_1w.append(art)
            elif delta <= 86400: # 24 Hours
                articles_1d.append(art)
                articles_1w.append(art)
            elif delta <= 604800: # 7 Days
                articles_1w.append(art)

        # Compute sentiment per bucket
        def compute_bucket_score(arts: List[Dict[str, Any]]) -> float:
            if not arts:
                return 0.0
            scores = [self._score_text(a["title"]) for a in arts]
            return float(sum(scores) / len(scores))

        score_1h = compute_bucket_score(articles_1h)
        score_1d = compute_bucket_score(articles_1d)
        score_1w = compute_bucket_score(articles_1w)

        label_1h = self._score_to_label(score_1h)
        label_1d = self._score_to_label(score_1d)
        label_1w = self._score_to_label(score_1w)

        # Detect Divergence between sentiment and live price change
        divergence = "CONGRUENT"
        if score_1h >= 0.40 and live_pct_change < -0.40:
            divergence = "BEARISH_DIVERGENCE (Sell the News / Distribution)"
        elif score_1h <= -0.40 and live_pct_change > 0.40:
            divergence = "BULLISH_ABSORPTION (Accumulation into Weak News)"
        elif score_1d >= 0.35 and live_pct_change < -0.75:
            divergence = "BEARISH_DRIFT_DIVERGENCE"
        elif score_1d <= -0.35 and live_pct_change > 0.75:
            divergence = "BULLISH_RESILIENCE"

        summary = (
            f"1h Sentiment: {label_1h} ({score_1h:.2f}, {len(articles_1h)} articles) | "
            f"1d: {label_1d} ({score_1d:.2f}, {len(articles_1d)} articles) | "
            f"1w: {label_1w} ({score_1w:.2f}, {len(articles_1w)} articles). "
            f"Divergence: {divergence}."
        )

        now_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to database
        self.db.save_multi_horizon_sentiment(
            ticker=clean_ticker,
            timestamp=now_formatted,
            s_1h=round(score_1h, 2),
            s_1d=round(score_1d, 2),
            s_1w=round(score_1w, 2),
            l_1h=label_1h,
            l_1d=label_1d,
            l_1w=label_1w,
            divergence=divergence,
            summary=summary,
            count=len(articles)
        )

        # Also save backward-compatible record for stock_sentiment table
        self.db.save_sentiment(
            ticker=clean_ticker,
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            score=round(score_1d, 2),
            label=label_1d,
            summary=summary
        )

        return {
            "ticker": clean_ticker,
            "sentiment_1h": round(score_1h, 2),
            "label_1h": label_1h,
            "sentiment_1d": round(score_1d, 2),
            "label_1d": label_1d,
            "sentiment_1w": round(score_1w, 2),
            "label_1w": label_1w,
            "divergence_flag": divergence,
            "summary": summary,
            "articles_count": len(articles),
            "latest_headlines": [a["title"] for a in articles[:5]],
            "timestamp": now_formatted
        }

    def scan_watchlist_batch(self, tickers: Optional[List[str]] = None, max_workers: int = 8) -> Dict[str, Any]:
        """Runs concurrent multi-horizon sentiment analysis across watchlist."""
        target_tickers = tickers or self.watchlist
        logger.info(f"Scanning multi-horizon sentiment for {len(target_tickers)} stocks...")
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_ticker, t): t for t in target_tickers}
            for future in as_completed(future_to_ticker):
                try:
                    res = future.result()
                    results[res["ticker"]] = res
                except Exception as e:
                    t = future_to_ticker[future]
                    logger.error(f"Error scanning sentiment for {t}: {e}")

        logger.info(f"Completed multi-horizon scan for {len(results)} stocks.")
        return results

if __name__ == "__main__":
    analyzer = MultiHorizonSentimentAnalyzer()
    print("Testing multi-horizon sentiment analysis on sample tickers...")
    test_stocks = ["RELIANCE", "TCS", "HDFCBANK"]
    for s in test_stocks:
        res = analyzer.analyze_ticker(s, live_pct_change=-0.5)
        print(f"\n--- {s} ---")
        print(f"1h Sentiment: {res['label_1h']} ({res['sentiment_1h']})")
        print(f"1d Sentiment: {res['label_1d']} ({res['sentiment_1d']})")
        print(f"1w Sentiment: {res['label_1w']} ({res['sentiment_1w']})")
        print(f"Divergence: {res['divergence_flag']}")
        print(f"Summary: {res['summary']}")