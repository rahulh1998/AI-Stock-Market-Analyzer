import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """
    High-speed, multi-threaded market data provider for NSE India equities.
    Features:
    - Multi-threaded concurrent batch fetching for real-time snapshots (LTP, % change, High, Low, Volume)
    - Multi-timeframe intraday candle retrieval (1m, 5m, 15m, 1h, 1d)
    - In-memory TTL cache to eliminate redundant network hits
    - Full technical indicator enrichment via pandas-ta
    - Ticker normalization for NSE (.NS suffix handling & known aliases)
    """

    def __init__(self, cache_ttl_seconds: int = 60):
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Dict[str, float] = {}

        # Ticker aliases mapping
        self.ticker_aliases = {
            "TATAMOTORS": "TMPV",
            "M&M": "M&M"
        }

    def _format_ticker(self, ticker: str) -> str:
        clean = ticker.split('.')[0].upper()
        clean = self.ticker_aliases.get(clean, clean)
        return f"{clean}.NS"

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache or cache_key not in self._cache_timestamp:
            return False
        return (time.time() - self._cache_timestamp[cache_key]) < self.cache_ttl

    def fetch_live_quote_single(self, ticker: str) -> Dict[str, Any]:
        """Fetches live market snapshot for a single ticker with fallback."""
        formatted = self._format_ticker(ticker)
        clean_symbol = ticker.split('.')[0].upper()

        cache_key = f"quote_{clean_symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            t = yf.Ticker(formatted)
            fast_info = getattr(t, "fast_info", None)

            price = 0.0
            prev_close = 0.0
            day_high = 0.0
            day_low = 0.0
            volume = 0

            if fast_info is not None:
                try:
                    price = float(fast_info.last_price or 0.0)
                    prev_close = float(fast_info.previous_close or price)
                    day_high = float(fast_info.day_high or price)
                    day_low = float(fast_info.day_low or price)
                    volume = int(fast_info.last_volume or 0)
                except Exception:
                    pass

            # Fallback to 1d history if fast_info is incomplete
            if price <= 0.0:
                hist = t.history(period="2d", interval="1d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else price
                    day_high = float(hist['High'].iloc[-1])
                    day_low = float(hist['Low'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1])

            change = price - prev_close if prev_close > 0 else 0.0
            pct_change = (change / prev_close * 100) if prev_close > 0 else 0.0

            result = {
                "ticker": clean_symbol,
                "price": round(price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "pct_change": round(pct_change, 2),
                "day_high": round(day_high, 2),
                "day_low": round(day_low, 2),
                "volume": volume,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self._cache[cache_key] = result
            self._cache_timestamp[cache_key] = time.time()
            return result

        except Exception as e:
            logger.warning(f"Failed fetching live quote for {ticker}: {e}")
            return {
                "ticker": clean_symbol,
                "price": 0.0,
                "prev_close": 0.0,
                "change": 0.0,
                "pct_change": 0.0,
                "day_high": 0.0,
                "day_low": 0.0,
                "volume": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def fetch_live_quotes_batch(self, tickers: List[str], max_workers: int = 16) -> Dict[str, Dict[str, Any]]:
        """
        Fetches live market quotes for all tickers concurrently using thread pooling.
        Takes ~2 seconds for 84 stocks instead of 90 seconds.
        """
        results: Dict[str, Dict[str, Any]] = {}
        missing_tickers: List[str] = []

        # Check cache first
        for ticker in tickers:
            clean_symbol = ticker.split('.')[0].upper()
            cache_key = f"quote_{clean_symbol}"
            if self._is_cache_valid(cache_key):
                results[clean_symbol] = self._cache[cache_key]
            else:
                missing_tickers.append(ticker)

        if not missing_tickers:
            return results

        # Fetch missing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_live_quote_single, t): t for t in missing_tickers
            }
            for future in as_completed(future_to_ticker):
                data = future.result()
                results[data["ticker"]] = data

        return results

    def fetch_intraday_data(self, ticker: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        """
        Fetches multi-timeframe candles (1m, 5m, 15m, 1h, 1d) with indicator enrichment.
        """
        formatted = self._format_ticker(ticker)
        clean_symbol = ticker.split('.')[0].upper()

        cache_key = f"candles_{clean_symbol}_{period}_{interval}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key].copy()

        try:
            df = yf.download(formatted, period=period, interval=interval, progress=False, auto_adjust=False)

            if df.empty:
                logger.warning(f"No candlestick data returned for {formatted} ({interval})")
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]

            rename_map = {}
            for col in df.columns:
                if 'date' in col or 'timestamp' in col or 'datetime' in col:
                    rename_map[col] = 'timestamp'
                elif 'open' in col:
                    rename_map[col] = 'open'
                elif 'high' in col:
                    rename_map[col] = 'high'
                elif 'low' in col:
                    rename_map[col] = 'low'
                elif 'close' in col and 'adj' not in col:
                    rename_map[col] = 'close'
                elif 'adj close' in col or 'adj_close' in col:
                    rename_map[col] = 'adj_close'
                elif 'volume' in col:
                    rename_map[col] = 'volume'

            df.rename(columns=rename_map, inplace=True)
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']

            cols = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols].dropna().reset_index(drop=True)

            if not df.empty:
                df = self.enrich_with_indicators(df)

            self._cache[cache_key] = df
            self._cache_timestamp[cache_key] = time.time()
            return df.copy()

        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()

    def enrich_with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates standard technical indicators using pandas-ta."""
        if df.empty or len(df) < 5:
            return df

        try:
            df.ta.ema(length=9, append=True)
            df.ta.ema(length=21, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.atr(length=14, append=True)
            
            # Fill or forward fill indicator initial NaN values to avoid wiping out recent bars
            df.bfill(inplace=True)
            df.ffill(inplace=True)
        except Exception as e:
            logger.warning(f"Indicator calculation warning: {e}")

        return df

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    print("Testing batch live quotes for sample tickers...")
    t0 = time.time()
    quotes = fetcher.fetch_live_quotes_batch(["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"])
    print(f"Fetched {len(quotes)} quotes in {time.time() - t0:.2f}s:")
    for k, v in quotes.items():
        print(f"  {k}: INR {v['price']} ({v['pct_change']}%)")