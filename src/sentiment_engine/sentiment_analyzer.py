import os
import json
import sqlite3
import logging
import yfinance as yf
from datetime import date
from src.data_pipeline.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.getcwd(), "config", "watchlist.json")

class SentimentAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()
        self.watchlist = self._load_watchlist()

    def _load_watchlist(self) -> list:
        if not os.path.exists(CONFIG_PATH):
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
        with open(CONFIG_PATH, "r") as f:
            return json.load(f).get("watchlist", [])

    def fetch_and_analyze_ticker(self, ticker: str):
        """Fetches news headlines and computes sentiment score."""
        yf_ticker = f"{ticker}.NS"
        try:
            t = yf.Ticker(yf_ticker)
            news_list = t.news
            
            if not news_list:
                logger.warning(f"No news found for {ticker}.")
                return

            headlines = []
            for item in news_list[:5]: # Analyze top 5 recent news items
                # yfinance news structure varies slightly across versions
                title = item.get('title') or item.get('content', {}).get('title', '')
                if title:
                    headlines.append(title)

            if not headlines:
                return

            # Simple robust sentiment heuristic (or plug in OpenAI/LangChain LLM call here)
            # Positive/Negative keyword scoring weighted for financial markets
            bullish_keywords = ['surge', 'jump', 'gain', 'growth', 'profit', 'rally', 'bull', 'upgrade', 'beat']
            bearish_keywords = ['fall', 'drop', 'loss', 'crash', 'bear', 'downgrade', 'miss', 'probe', 'slump']

            score = 0.0
            text_corpus = " ".join(headlines).lower()
            
            for word in bullish_keywords:
                score += text_corpus.count(word) * 0.2
            for word in bearish_keywords:
                score -= text_corpus.count(word) * 0.2
                
            # Normalize score between -1.0 and 1.0
            score = max(min(score, 1.0), -1.0)
            
            label = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")
            summary = f"Analyzed {len(headlines)} headlines. Key themes reflect {label.lower()} market sentiment."
            
            today_str = date.today().strftime("%Y-%m-%d")
            self.db.save_sentiment(ticker, today_str, score, label, summary)
            logger.info(f"Saved sentiment for {ticker}: Score={score:.2f} ({label})")

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {e}")

    def run_daily_scan(self):
        logger.info(f"Starting daily sentiment scan for {len(self.watchlist)} stocks...")
        for ticker in self.watchlist:
            self.fetch_and_analyze_ticker(ticker)

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.run_daily_scan()