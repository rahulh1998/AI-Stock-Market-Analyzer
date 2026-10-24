"""Corporate-action adjustment and trading-calendar derivation.

Two defects were measured in the seeded history, both of which would corrupt a backtest:

1. **Fabricated bars on non-trading days.** 421 bars have open == high == low == close with
   zero volume, concentrated on five dates that ~80 tickers share, plus 81 weekend rows.
   2026-05-01 (Maharashtra Day) is one of them. These pollute ATR, relative volume and the
   bar count every warm-up rule depends on.

2. **Unadjusted prices in the signal path.** The engine signs off `close`, not `adj_close`.
   A 1:5 split looks exactly like a -80% crash to every pattern in decision_engine, and it
   moves a stop to a price that never existed.

The corporate-action detector deliberately does **not** treat a large move as an action.
Measured: of 9 moves above 20% in this history, the Adani/Hindenburg crash, the IndusInd
derivatives crash and the 2024 election-day drop are all *real*. Auto-"correcting" those
would destroy genuine data. The discriminator is the divergence between the raw close ratio
and the adjusted close ratio, which only a split, bonus or dividend produces.
"""
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A real adjustment moves the raw/adjusted ratio by more than rounding noise.
ACTION_TOLERANCE = 0.02
# Below this, a divergence is an ordinary dividend rather than a share-count change.
SPLIT_TOLERANCE = 0.10
# A date is a trading day if at least this share of covered tickers actually traded.
TRADING_DAY_QUORUM = 0.30
REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    date: str
    close_ratio: float
    adjusted_ratio: float
    factor: float
    kind: str

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "date": self.date,
                "close_ratio": round(self.close_ratio, 6),
                "adjusted_ratio": round(self.adjusted_ratio, 6),
                "factor": round(self.factor, 6), "kind": self.kind}


def detect_corporate_actions(frame: pd.DataFrame, ticker: str = "") -> list:
    """Actions implied by raw-vs-adjusted divergence. Needs an adj_close column.

    factor = adjusted_ratio / close_ratio. It is ~1 for an ordinary price move however
    violent, and departs from 1 only when the share count or a distribution changed.
    """
    if "adj_close" not in frame.columns or len(frame) < 2:
        return []
    df = frame.sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    adjusted = pd.to_numeric(df["adj_close"], errors="coerce")
    previous_close, previous_adjusted = close.shift(1), adjusted.shift(1)
    valid = (previous_close > 0) & (previous_adjusted > 0) & (close > 0) & (adjusted > 0)
    close_ratio = (close / previous_close).where(valid)
    adjusted_ratio = (adjusted / previous_adjusted).where(valid)
    factor = adjusted_ratio / close_ratio

    actions = []
    for index in np.flatnonzero((factor - 1.0).abs() > ACTION_TOLERANCE):
        value = float(factor.iloc[index])
        kind = "split_or_bonus" if abs(value - 1.0) > SPLIT_TOLERANCE else "dividend"
        actions.append(CorporateAction(
            ticker=ticker, date=str(pd.Timestamp(df.timestamp.iloc[index]).date()),
            close_ratio=float(close_ratio.iloc[index]),
            adjusted_ratio=float(adjusted_ratio.iloc[index]), factor=value, kind=kind))
    return actions


def adjust_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC onto the adj_close scale, and volume inversely.

    Returns the frame unchanged when adj_close is absent or carries no information
    (it equals close everywhere), which is legitimate for a recent listing that has paid
    no dividend. The caller can tell from the `adjustment_factor` column, which is then 1.

    Note: these are not prices anyone could have traded at. Returns are correct; the fixed
    components of the fee model (the INR 20 brokerage cap, the DP charge) are therefore
    approximate for periods before an adjustment. Splits are rare enough in this universe
    that the effect is negligible, but it is a real limitation.
    """
    df = frame.sort_values("timestamp").reset_index(drop=True).copy()
    if "adj_close" not in df.columns:
        df["adjustment_factor"] = 1.0
        return df
    close = pd.to_numeric(df["close"], errors="coerce")
    adjusted = pd.to_numeric(df["adj_close"], errors="coerce")
    factor = (adjusted / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["adjustment_factor"] = factor
    if bool(np.isclose(factor, 1.0, atol=1e-6).all()):
        return df
    for column in ("open", "high", "low"):
        df[column] = pd.to_numeric(df[column], errors="coerce") * factor
    df["close"] = adjusted
    # Share count scales inversely with price, so notional turnover is preserved.
    df["volume"] = (pd.to_numeric(df["volume"], errors="coerce") / factor).round()
    return df


def flag_fabricated_bars(frame: pd.DataFrame) -> pd.Series:
    """True where a bar is a placeholder rather than a session.

    Zero volume AND a zero range together. Either alone is possible in a thin but real
    session, so both are required before a bar is called fabricated.
    """
    df = frame
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    return flat & (pd.to_numeric(df["volume"], errors="coerce").fillna(0) <= 0)


def derive_trading_calendar(panel: dict, quorum: float = TRADING_DAY_QUORUM) -> set:
    """Trading dates inferred from the panel itself, by quorum across tickers.

    There is no authoritative NSE holiday feed available here, so the calendar is derived
    from observed behaviour: a date is a trading day when a sufficient share of covered
    tickers posted a real (non-fabricated, non-zero-volume) bar. Weekends are excluded
    outright. This is empirical, not official - it will not know about a special session.
    """
    traded, covered = {}, {}
    for frame in panel.values():
        if frame is None or frame.empty:
            continue
        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fabricated = flag_fabricated_bars(df)
        for date, is_fabricated in zip(df["timestamp"].dt.date, fabricated):
            covered[date] = covered.get(date, 0) + 1
            if not is_fabricated:
                traded[date] = traded.get(date, 0) + 1
    calendar = set()
    for date, count in covered.items():
        if date.weekday() >= 5:
            continue
        if traded.get(date, 0) / count >= quorum:
            calendar.add(date)
    return calendar


def clean_frame(frame: pd.DataFrame, calendar: set = None, adjust: bool = True) -> tuple:
    """(cleaned frame, report). Drops fabricated and off-calendar bars, then adjusts.

    Order matters: fabricated bars are removed before adjustment, so a placeholder cannot
    contribute a bogus adjustment factor.
    """
    report = {"input_rows": len(frame), "dropped_fabricated": 0, "dropped_off_calendar": 0,
              "dropped_weekend": 0, "corporate_actions": 0, "adjusted": False}
    if frame is None or frame.empty:
        report["output_rows"] = 0
        return frame, report
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fabricated = flag_fabricated_bars(df)
    report["dropped_fabricated"] = int(fabricated.sum())
    df = df[~fabricated]

    weekend = df["timestamp"].dt.weekday >= 5
    report["dropped_weekend"] = int(weekend.sum())
    df = df[~weekend]

    if calendar is not None:
        off = ~df["timestamp"].dt.date.isin(calendar)
        report["dropped_off_calendar"] = int(off.sum())
        df = df[~off]

    df = df.sort_values("timestamp").reset_index(drop=True)
    report["corporate_actions"] = len(detect_corporate_actions(df))
    if adjust:
        before = df["close"].copy()
        df = adjust_ohlcv(df)
        report["adjusted"] = not bool(np.isclose(df["close"], before, rtol=1e-9).all())
    report["output_rows"] = len(df)
    return df, report


def clean_panel(panel: dict, adjust: bool = True) -> tuple:
    """Clean every ticker against one calendar derived from the whole panel."""
    calendar = derive_trading_calendar(panel)
    cleaned, reports = {}, {}
    for ticker, frame in panel.items():
        cleaned[ticker], reports[ticker] = clean_frame(frame, calendar, adjust=adjust)
    summary = {
        "tickers": len(panel),
        "trading_days": len(calendar),
        "dropped_fabricated": sum(r["dropped_fabricated"] for r in reports.values()),
        "dropped_weekend": sum(r["dropped_weekend"] for r in reports.values()),
        "dropped_off_calendar": sum(r["dropped_off_calendar"] for r in reports.values()),
        "tickers_adjusted": sum(1 for r in reports.values() if r["adjusted"]),
        "corporate_actions": sum(r["corporate_actions"] for r in reports.values()),
    }
    return cleaned, calendar, summary, reports
