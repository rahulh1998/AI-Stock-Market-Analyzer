"""Fill the daily_ohlcv gap per ticker, concurrently, and report honestly.

DatabaseManager.update_latest_data only ever asks Yahoo for period="5d", so it cannot
close a multi-week gap, and it sleeps 1s per ticker. This script computes each ticker's
actual gap, fetches only what is missing, and resolves symbols through
config/instruments.json so renamed listings (ZOMATO -> ETERNAL.NS) still load.

Usage, from anywhere:
    python -m scripts.update_market_data                 # incremental
    python -m scripts.update_market_data --seed-years 5  # full history rebuild
    python -m scripts.update_market_data --ticker RELIANCE --ticker TCS
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

from src.data_pipeline.universe import MARKETS, load_market_instruments, split_key
from src.paths import DB_PATH, INSTRUMENTS_PATH, WATCHLIST_PATH

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

COLUMNS = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
UPSERT = """INSERT OR REPLACE INTO daily_ohlcv
            (ticker, timestamp, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


def load_instruments(markets=("NSE",)) -> dict:
    """Instruments across markets, keyed exactly as they are stored in daily_ohlcv.

    NSE rows keep their bare ticker so five years of existing history stays addressable;
    every other market is stored market-qualified ("US:AAPL"). The primary key is
    (ticker, timestamp), so qualifying the key is enough to keep markets apart without a
    schema migration, and split_key() reads both forms.
    """
    combined = {}
    for code in markets:
        for ticker, meta in load_market_instruments(code).items():
            key = ticker if code == "NSE" else f"{code}:{ticker}"
            combined[key] = {**meta, "market": code, "bare_ticker": ticker}
    return combined


def load_watchlist(markets=("NSE",)) -> list:
    return list(load_instruments(markets))


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to the daily_ohlcv column names."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(c).lower() for c in frame.columns]
    rename = {}
    for col in frame.columns:
        if "date" in col or "timestamp" in col:
            rename[col] = "timestamp"
        elif "adj close" in col or "adj_close" in col:
            rename[col] = "adj_close"
        elif "open" in col:
            rename[col] = "open"
        elif "high" in col:
            rename[col] = "high"
        elif "low" in col:
            rename[col] = "low"
        elif "close" in col:
            rename[col] = "close"
        elif "volume" in col:
            rename[col] = "volume"
    frame = frame.rename(columns=rename)
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    if not set(COLUMNS).issubset(frame.columns):
        missing = sorted(set(COLUMNS) - set(frame.columns))
        raise ValueError(f"missing columns {missing}")
    frame = frame[COLUMNS].dropna().copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d")
    return frame


def fetch(ticker: str, symbol: str, start: date, end: date) -> tuple:
    """Return (ticker, frame, error). Never raises; the caller reports failures."""
    try:
        raw = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                          interval="1d", progress=False, auto_adjust=False, timeout=30)
        if raw.empty:
            return ticker, None, "no rows returned"
        return ticker, normalise(raw), None
    except Exception as exc:
        return ticker, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-years", type=float, default=None,
                        help="Rebuild this many years for every ticker instead of filling gaps")
    parser.add_argument("--ticker", action="append", default=None, help="Limit to these tickers")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--markets", default="NSE",
                        help="Comma-separated markets to update, e.g. NSE,US")
    args = parser.parse_args()

    markets = tuple(m.strip().upper() for m in args.markets.split(",") if m.strip())
    unknown = [m for m in markets if m not in MARKETS]
    if unknown:
        print(f"Unknown market(s): {unknown}. Known: {sorted(MARKETS)}")
        return 2
    instruments = load_instruments(markets)
    tickers = args.ticker or list(instruments)
    today = date.today()

    connection = sqlite3.connect(DB_PATH, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    existing = dict(connection.execute(
        "SELECT ticker, MAX(timestamp) FROM daily_ohlcv GROUP BY ticker").fetchall())
    rows_before = connection.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0]
    overall_before = connection.execute("SELECT MAX(timestamp) FROM daily_ohlcv").fetchone()[0]

    jobs, skipped = [], []
    for ticker in tickers:
        meta = instruments.get(ticker, {})
        if not meta.get("feed_available", True):
            skipped.append((ticker, meta.get("note", "marked feed_available: false")))
            continue
        code, bare = split_key(ticker)
        symbol = meta.get("yahoo_symbol") or MARKETS[code].yahoo_symbol(bare)
        if args.seed_years:
            start = today - timedelta(days=int(args.seed_years * 365.25))
        else:
            last = existing.get(ticker)
            # Re-fetch the last stored day so a partial final candle is corrected.
            start = (pd.Timestamp(last).date() if last else today - timedelta(days=1825))
        jobs.append((ticker, symbol, start, today + timedelta(days=1)))

    print(f"Database     : {DB_PATH}")
    print(f"Markets      : {', '.join(markets)}")
    print(f"Rows before  : {rows_before:,}   latest stored date: {overall_before}")
    print(f"Tickers      : {len(jobs)} to fetch, {len(skipped)} skipped")
    if skipped:
        for ticker, why in skipped:
            print(f"  SKIP {ticker}: {why}")
    print()

    written, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda job: fetch(*job), jobs)
        cursor = connection.cursor()
        for index, (ticker, frame, error) in enumerate(results, start=1):
            if error:
                failures.append((ticker, error))
                print(f"[{index:3d}/{len(jobs)}] {ticker:12s} FAILED  {error}")
                continue
            payload = [(ticker, r.timestamp, float(r.open), float(r.high), float(r.low),
                        float(r.close), float(r.adj_close), int(r.volume))
                       for r in frame.itertuples(index=False)]
            cursor.executemany(UPSERT, payload)
            connection.commit()
            written += len(payload)
            print(f"[{index:3d}/{len(jobs)}] {ticker:12s} {len(payload):4d} rows "
                  f"-> {frame.timestamp.iloc[-1]}")

    rows_after = connection.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0]
    overall_after = connection.execute("SELECT MAX(timestamp) FROM daily_ohlcv").fetchone()[0]
    ticker_count = connection.execute(
        "SELECT COUNT(DISTINCT ticker) FROM daily_ohlcv").fetchone()[0]
    connection.close()

    print()
    print(f"Upserted     : {written:,} rows across {len(jobs) - len(failures)} tickers")
    print(f"Rows after   : {rows_after:,} (net new {rows_after - rows_before:,})")
    print(f"Latest date  : {overall_before} -> {overall_after}")
    print(f"Tickers in db: {ticker_count}")
    if failures:
        print(f"\nFailures ({len(failures)}) — these tickers are NOT updated:")
        for ticker, error in failures:
            print(f"  {ticker:12s} {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
