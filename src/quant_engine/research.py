"""Deterministic, long-only historical replay. Not a broker or fill guarantee."""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquityCosts:
    """Illustrative NSE cash-equity schedule; verify against your contract note.

    Rates are fractions, not percentages. One executed order per leg is assumed.
    DP is charged per simulated delivery exit (no same-day aggregation).
    """
    intraday: bool = True
    brokerage_rate: float = 0.0003
    brokerage_cap: float = 20.0
    exchange_rate: float = 0.0000307
    sebi_rate: float = 0.000001
    gst_rate: float = 0.18
    delivery_dp: float = 15.34
    slippage_bps: float = 5.0
    # Securities-transaction and stamp rates. Defaulted to the NSE schedule so existing
    # behaviour is unchanged, but parameterised because they are jurisdiction-specific:
    # a US trade pays an SEC/FINRA pass-through on sells and no stamp duty at all, and
    # applying Indian STT to it would overstate costs by about an order of magnitude.
    stt_intraday_sell: float = 0.00025
    stt_delivery: float = 0.001
    # NSE delivery STT is charged on both legs; the US SEC/FINRA pass-through is sell-only.
    stt_delivery_sell_only: bool = False
    stamp_intraday_buy: float = 0.00003
    stamp_delivery_buy: float = 0.00015

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name not in ("intraday", "stt_delivery_sell_only") and (
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid cost parameter: {name}")
        if self.slippage_bps >= 10000:
            raise ValueError("Slippage must be less than 10000 bps")

    def breakdown(self, price: float, quantity: int, side: str) -> dict:
        if side not in ("BUY", "SELL") or not math.isfinite(price) or price <= 0:
            raise ValueError("Positive price and BUY/SELL side required")
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("Quantity must be a non-negative integer")
        turnover = price * quantity
        brokerage = min(turnover * self.brokerage_rate, self.brokerage_cap) if self.intraday else 0.0
        exchange = turnover * self.exchange_rate
        sebi = turnover * self.sebi_rate
        stt = (turnover * (self.stt_intraday_sell if side == "SELL" else 0.0)
               if self.intraday else
               turnover * (self.stt_delivery
                           if side == "SELL" or not self.stt_delivery_sell_only else 0.0))
        stamp = (turnover * (self.stamp_intraday_buy if self.intraday else self.stamp_delivery_buy)
                 if side == "BUY" else 0.0)
        gst = (brokerage + exchange + sebi) * self.gst_rate
        dp = self.delivery_dp if not self.intraday and side == "SELL" and quantity else 0.0
        result = dict(brokerage=brokerage, exchange=exchange, sebi=sebi, stt=stt, stamp=stamp, gst=gst, dp=dp)
        result["total"] = sum(result.values())
        return result

    def fee(self, price: float, quantity: int, side: str) -> float:
        return self.breakdown(price, quantity, side)["total"]


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed rather than repair malformed OHLCV silently. Naive NSE dates -> IST."""
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Non-empty timestamp/open/high/low/close/volume data required")
    df = frame[required].copy()
    ts = pd.to_datetime(df["timestamp"], errors="raise")
    ts = ts.dt.tz_localize("Asia/Kolkata") if ts.dt.tz is None else ts.dt.tz_convert("Asia/Kolkata")
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("Timestamps must be present, unique, and ascending")
    df["timestamp"] = ts
    numeric = df[required[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV contains non-finite values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any() or (numeric.volume < 0).any():
        raise ValueError("Prices must be positive and volume non-negative")
    if (numeric.high < numeric[["open", "close", "low"]].max(axis=1)).any() or (numeric.low > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Inconsistent OHLC geometry")
    df[required[1:]] = numeric
    return df.reset_index(drop=True)


def research_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = validate_bars(frame)
    previous = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - previous).abs(), (df.low - previous).abs()], axis=1).max(axis=1)
    # Explicit SMA ATR definition, not advertised as Wilder's recursive ATR.
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema_fast"] = df.close.ewm(span=9, adjust=False, min_periods=9).mean()
    df["ema_slow"] = df.close.ewm(span=21, adjust=False, min_periods=21).mean()
    df["sma_200"] = df.close.rolling(200, min_periods=200).mean()
    session = df.timestamp.dt.date
    pv = (df.high + df.low + df.close) / 3 * df.volume
    df["vwap"] = pv.groupby(session).cumsum() / df.volume.groupby(session).cumsum().replace(0, np.nan)
    df["relative_volume"] = df.volume / df.volume.shift(1).rolling(20).mean().replace(0, np.nan)
    return df


def size_long(entry: float, stop: float, cash: float, risk_pct: float, costs: EquityCosts,
              allocation_pct: float = 25.0) -> int:
    values = (entry, stop, cash, risk_pct, allocation_pct)
    if not all(math.isfinite(v) for v in values) or not 0 < stop < entry or cash <= 0:
        return 0
    if not 0 < risk_pct <= 3 or not 0 < allocation_pct <= 100:
        raise ValueError("Risk must be (0, 3]% and allocation (0, 100]%")
    budget = cash * risk_pct / 100
    allocation = cash * allocation_pct / 100
    exit_fill = stop * (1 - costs.slippage_bps / 10000)
    lo, hi = 0, int(min(allocation / entry, budget / (entry - exit_fill)))
    while lo < hi:
        q = (lo + hi + 1) // 2
        entry_fee = costs.fee(entry, q, "BUY")
        loss = q * (entry - exit_fill) + entry_fee + costs.fee(exit_fill, q, "SELL")
        if q * entry + entry_fee <= min(cash, allocation) and loss <= budget:
            lo = q
        else:
            hi = q - 1
    return lo


def size_short(entry: float, stop: float, cash: float, risk_pct: float, costs: EquityCosts,
               allocation_pct: float = 25.0, margin_pct: float = 100.0) -> int:
    """Intraday-only short sizing. Mirror of size_long with three real differences.

    The price order is inverted (stop above entry), the adverse exit fill moves UPWARD
    because the exit is a buy, and the fee legs swap: the entry pays SELL-side charges.
    margin_pct defaults to the full notional; modelling a lower broker margin would imply
    leverage on a position whose loss is unbounded, which this tool does not simulate.
    """
    if not costs.intraday:
        return 0
    values = (entry, stop, cash, risk_pct, allocation_pct, margin_pct)
    if not all(math.isfinite(v) for v in values) or not 0 < entry < stop or cash <= 0:
        return 0
    if not 0 < risk_pct <= 3 or not 0 < allocation_pct <= 100:
        raise ValueError("Risk must be (0, 3]% and allocation (0, 100]%")
    if not 0 < margin_pct <= 100:
        raise ValueError("Short margin must be (0, 100]% of notional; leverage is not modelled")
    budget = cash * risk_pct / 100
    allocation = cash * allocation_pct / 100
    exit_fill = stop * (1 + costs.slippage_bps / 10000)
    margin_per_share = entry * margin_pct / 100
    lo, hi = 0, int(min(allocation / margin_per_share, budget / (exit_fill - entry)))
    while lo < hi:
        q = (lo + hi + 1) // 2
        entry_fee = costs.fee(entry, q, "SELL")
        loss = q * (exit_fill - entry) + entry_fee + costs.fee(exit_fill, q, "BUY")
        if q * margin_per_share + entry_fee <= min(cash, allocation) and loss <= budget:
            lo = q
        else:
            hi = q - 1
    return lo


def resolve_exit(row, stop: float, target: float, direction: str = "LONG") -> tuple:
    """(raw_exit_price, reason) or (None, None) if neither barrier is touched.

    Single source of truth for barrier resolution, shared by the historical replay and the
    paper book so the two can never drift apart. Two rules that matter:
    a gap through the stop fills at the worse OPEN, not at the stop; and if both barriers
    are touched inside one bar the STOP wins, because bar data cannot say which came first.
    """
    if direction == "LONG":
        if row.open <= stop:
            return row.open, "gap_stop"
        if row.low <= stop:
            return stop, "stop"
        if row.high >= target:
            return target, "target"
        return None, None
    if row.open >= stop:
        return row.open, "gap_stop"
    if row.high >= stop:
        return stop, "stop"
    if row.low <= target:
        return target, "target"
    return None, None


def replay(frame: pd.DataFrame, capital: float = 200000.0, risk_pct: float = 0.5,
           allocation_pct: float = 25.0, intraday: bool = True,
           costs: EquityCosts = None, reward_risk: float = 2.0,
           daily_loss_pct: float = 2.0) -> dict:
    """Signals at t close, entry at t+1 open; stop first if both barriers touch.

    Intraday exits use the last observed session bar (retrospective proxy, not a
    scheduled square-off service). Feed gaps/halts and liquidity are NOT modeled.
    """
    if not math.isfinite(capital) or capital <= 0 or not math.isfinite(reward_risk) or reward_risk <= 0:
        raise ValueError("Positive capital and reward/risk required")
    if not 0 < risk_pct <= 3 or not 0 < allocation_pct <= 100 or not 0 < daily_loss_pct <= 100:
        raise ValueError("Invalid risk, allocation or daily loss setting")
    costs = costs or EquityCosts(intraday=intraday)
    if costs.intraday != intraday:
        raise ValueError("Cost product must match replay horizon")
    df = research_features(frame)
    slip = costs.slippage_bps / 10000
    cash, position = capital, None
    trades, curve = [], []
    session, session_equity = None, capital
    for i, row in df.iterrows():
        day = row.timestamp.date()
        if day != session:
            session = day
            session_equity = cash + (position["quantity"] * row.open if position else 0)
        if position is None and i > 0 and cash > session_equity * (1 - daily_loss_pct / 100):
            prev = df.iloc[i - 1]
            signal = prev.ema_fast > prev.ema_slow and pd.notna(prev.atr) and prev.atr > 0
            if intraday:
                signal = signal and prev.timestamp.date() == day and prev.close > prev.vwap and prev.relative_volume >= 1.25
            else:
                signal = signal and pd.notna(prev.sma_200) and prev.close > prev.sma_200
            # Current bar's total volume is not known at its opening fill.
            if signal and prev.volume > 0:
                entry = row.open * (1 + slip)
                stop = entry - 1.5 * prev.atr
                q = size_long(entry, stop, cash, risk_pct, costs, allocation_pct)
                if q:
                    fee = costs.fee(entry, q, "BUY")
                    position = dict(entry_time=row.timestamp, entry=entry, stop=stop,
                                    target=entry + reward_risk * (entry - stop), quantity=q, entry_fee=fee)
                    cash -= entry * q + fee
        if position:
            p = position
            raw_exit, reason = resolve_exit(row, p["stop"], p["target"], "LONG")
            last = i == len(df) - 1
            session_end = last or df.iloc[i + 1].timestamp.date() != day
            if raw_exit is None and (last or (intraday and session_end)):
                raw_exit, reason = row.close, "sample_end" if last else "session_end_proxy"
            if raw_exit is not None:
                exit_price = raw_exit * (1 - slip)
                fee = costs.fee(exit_price, p["quantity"], "SELL")
                cash += exit_price * p["quantity"] - fee
                gross = (exit_price - p["entry"]) * p["quantity"]
                trades.append({**p, "exit_time": row.timestamp, "exit": exit_price, "reason": reason,
                               "fees": fee + p["entry_fee"], "net_pnl": gross - fee - p["entry_fee"]})
                position = None
        equity = cash + (position["quantity"] * row.close if position else 0)
        curve.append({"timestamp": row.timestamp, "equity": equity})
    ledger = pd.DataFrame(trades)
    equity = pd.DataFrame(curve)
    peak = equity.equity.cummax().clip(lower=capital)
    pnl = ledger.net_pnl if not ledger.empty else pd.Series(dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return dict(trades=ledger, equity=equity, metrics={
        "net_return_pct": (cash / capital - 1) * 100,
        "max_drawdown_pct": float(((peak - equity.equity) / peak).max() * 100),
        "trades": len(ledger), "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "expectancy_rupees": float(pnl.mean()) if len(pnl) else None,
        "total_fees": float(ledger.fees.sum()) if len(ledger) else 0.0,
    })