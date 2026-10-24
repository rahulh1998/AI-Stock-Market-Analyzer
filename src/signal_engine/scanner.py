"""Concurrent watchlist scanning, plus the market-hours clock policy.

Two things live here rather than in the UI, so they can be tested without Streamlit.

Scanning is concurrent and two-speed because the cost is real: analyze_market measures
~0.5s on 499 intraday bars and ~0.8s on 1236 daily bars, so a naive sequential scan of 82
tickers is 40-70s. Fast mode skips the nearest-neighbour model and uses only cached
sentiment; the result says which components were deferred, so a conviction number is never
quietly built from fewer inputs than it claims.

The clock policy exists because the engine marks intraday data stale 2 bars after the last
close, which is correct but makes every intraday signal HISTORICAL_ONLY outside
09:15-15:30 IST. "Last close review" evaluates as at the last completed bar instead, and
is labelled as a review of a finished session rather than a live signal.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

import pandas as pd

from src.quant_engine.decision_engine import AccountRisk, SignalPolicy
from src.quant_engine.research import EquityCosts
from src.signal_engine.composite import INTERVAL_DURATIONS, SignalConfig, generate_signal

logger = logging.getLogger(__name__)

# Yahoo caps sub-hour intervals at 60 days, and the nearest-neighbour model needs roughly
# 300-400 bars before it will speak. Measured: 15m+5d = 125 bars (model always abstains),
# 15m+1mo = 499 bars (model available), 15m+2mo = EMPTY (rejected by Yahoo).
PERIOD_FOR_INTERVAL = {"5m": "1mo", "15m": "1mo", "1h": "3mo", "1d": "5y"}

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

LIVE = "live"
LAST_CLOSE_REVIEW = "last_close_review"


def market_is_open(now: pd.Timestamp = None) -> bool:
    """NSE equity session by clock and weekday only. Holidays are NOT modelled."""
    now = (pd.Timestamp.now(tz="Asia/Kolkata") if now is None
           else pd.Timestamp(now).tz_convert("Asia/Kolkata"))
    if now.weekday() >= 5:
        return False
    open_at = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_at = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return open_at <= now <= close_at


def review_clock(bars: pd.DataFrame, interval: str) -> pd.Timestamp:
    """A clock pinned just after the last bar's close, so that bar counts as complete."""
    if bars is None or bars.empty:
        return None
    stamp = pd.Timestamp(bars["timestamp"].iloc[-1])
    stamp = (stamp.tz_localize("Asia/Kolkata") if stamp.tzinfo is None
             else stamp.tz_convert("Asia/Kolkata"))
    return stamp + pd.Timedelta(INTERVAL_DURATIONS[interval])


def resolve_clock(bars: pd.DataFrame, interval: str, mode: str) -> tuple:
    """(clock, label). LIVE uses the wall clock; review pins to the last completed bar."""
    if mode == LIVE:
        return None, "Live wall clock"
    clock = review_clock(bars, interval)
    if clock is None:
        return None, "No bars; falling back to the live clock"
    return clock, f"Reviewed as at {clock.isoformat()} (last completed bar)"


@dataclass(frozen=True)
class ScanRow:
    ticker: str
    signal: object = None
    error: str = None
    bars: int = 0
    deferred: tuple = ()

    def to_row(self) -> dict:
        """Flat record for a table. Missing values stay None, never 0.0 placeholders."""
        if self.error or self.signal is None:
            return {"Ticker": self.ticker, "Action": "UNAVAILABLE", "Conviction": None,
                    "Grade": None, "Risk": None, "Status": self.error or "no signal",
                    "Bars": self.bars, "Deferred": ", ".join(self.deferred) or "-"}
        signal = self.signal
        plan = signal.plan or {}
        return {
            "Ticker": self.ticker,
            "Action": signal.action,
            "Conviction": round(signal.conviction, 1),
            "Grade": (signal.setup or {}).get("letter"),
            "Risk": (signal.risk or {}).get("score"),
            "Status": signal.status,
            "Entry": plan.get("buy_trigger") or plan.get("sell_trigger"),
            "Stop": plan.get("stop"),
            "Target": plan.get("take_profit"),
            "Qty": plan.get("quantity"),
            "Bias": signal.bias,
            "Sentiment": (signal.components.get("sentiment", {}) or {}).get("aggregate"),
            "Algo": (signal.components.get("algo", {}) or {}).get("available"),
            "Bars": self.bars,
            "Deferred": ", ".join(self.deferred) or "-",
        }


ACTION_RANK = {"BUY": 0, "SHORT_ENTRY": 1, "EXIT_LONG": 2, "HOLD": 3, "UNAVAILABLE": 4}


def scan_one(ticker: str, fetcher, interval: str, *, account: AccountRisk = None,
             policy: SignalPolicy = None, costs: EquityCosts = None,
             config: SignalConfig = None, sentiment_service=None, fast: bool = True,
             mode: str = LAST_CLOSE_REVIEW, confidence_lookup=None,
             period: str = None) -> ScanRow:
    """One ticker, never raising: a failure becomes a reported row."""
    deferred = []
    try:
        bars = fetcher.fetch_intraday_data(ticker, period or PERIOD_FOR_INTERVAL[interval],
                                           interval)
        if bars is None or bars.empty:
            return ScanRow(ticker, error="no candles returned")

        sentiment = None
        if sentiment_service is not None:
            if fast:
                sentiment = sentiment_service.cached(ticker)
                if sentiment is None:
                    deferred.append("sentiment (not cached)")
            else:
                sentiment = sentiment_service.reading(ticker, bars=bars)
        if fast:
            deferred.append("algo model")

        clock, _ = resolve_clock(bars, interval, mode)
        signal = generate_signal(
            bars, interval, ticker=ticker, account=account, policy=policy, costs=costs,
            sentiment=sentiment, now=clock, config=config,
            confidence_lookup=confidence_lookup, include_model=not fast)
        return ScanRow(ticker, signal=signal, bars=len(bars), deferred=tuple(deferred))
    except Exception as exc:
        logger.warning("scan failed for %s: %s", ticker, exc)
        return ScanRow(ticker, error=f"{type(exc).__name__}: {exc}", deferred=tuple(deferred))


def scan(tickers, fetcher, interval: str = "15m", *, max_workers: int = 8, **kwargs) -> list:
    """Scan a watchlist concurrently. Rows come back sorted by action then conviction."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(
            lambda ticker: scan_one(ticker, fetcher, interval, **kwargs), tickers))
    return sorted(rows, key=lambda row: (
        ACTION_RANK.get(row.to_row()["Action"], 9),
        -(row.signal.conviction if row.signal else -1)))


def to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([row.to_row() for row in rows])
