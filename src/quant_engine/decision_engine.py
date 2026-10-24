"""Causal cash-equity research decisions. No broker permissions or optimal-price claims.

All patterns are known only at candle close. Swing pivots require two later bars
and are reported at confirmation, never backdated. Thresholds are conventions,
not empirically established trading edges. Inputs must be a single instrument.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np
import pandas as pd

from src.quant_engine.research import (EquityCosts, research_features, size_long,
                                       size_short, validate_bars)


PATTERNS = {
    "bullish_engulfing": ("bullish", "Green body engulfs a red body after a local decline."),
    "bearish_engulfing": ("bearish", "Red body engulfs a green body after a local advance."),
    "hammer": ("bullish", "Lower wick >= twice the body after a decline; requires confirmation."),
    "shooting_star": ("bearish", "Upper wick >= twice the body after an advance."),
    "morning_reversal": ("bullish", "Three-bar, gap-optional morning reversal; not a strict gapped star."),
    "evening_reversal": ("bearish", "Three-bar, gap-optional evening reversal; not a strict gapped star."),
    "doji": ("neutral", "Body <= 10% of nonzero range; indecision, not a directional signal."),
    "inside_bar": ("neutral", "Range inside the preceding bar; direction remains unconfirmed."),
    "range_breakout": ("bullish", "Close above all highs of the preceding 20 completed bars."),
    "range_breakdown": ("bearish", "Close below all lows of the preceding 20 completed bars."),
    "vwap_reclaim": ("bullish", "Close crosses above same-session cumulative VWAP."),
    "vwap_rejection": ("bearish", "Close crosses below same-session cumulative VWAP."),
    "double_bottom_break": ("bullish", "Two confirmed similar swing lows; close crosses their neckline."),
    "double_top_break": ("bearish", "Two confirmed similar swing highs; close crosses their neckline."),
}
BULLISH = [name for name, (direction, _) in PATTERNS.items() if direction == "bullish"]
BEARISH = [name for name, (direction, _) in PATTERNS.items() if direction == "bearish"]


def wilder_mean(values: pd.Series, length: int = 14) -> pd.Series:
    """SMA seed over length consecutive finite observations, then Wilder recursion."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("Wilder length must be a positive integer")
    result = np.full(len(values), np.nan)
    window, previous = [], None
    for i, value in enumerate(values.to_numpy(dtype=float)):
        if not np.isfinite(value):
            window, previous = [], None
        elif previous is None:
            window.append(value)
            if len(window) == length:
                previous = float(np.mean(window))
                result[i] = previous
        else:
            previous = (previous * (length - 1) + value) / length
            result[i] = previous
    return pd.Series(result, index=values.index)


def pattern_features(frame: pd.DataFrame, intraday: bool = True) -> pd.DataFrame:
    df = research_features(frame)
    o, h, l, c = (df[name] for name in ("open", "high", "low", "close"))
    delta = c.diff()
    gain, loss = wilder_mean(delta.clip(lower=0)), wilder_mean(-delta.clip(upper=0))
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    df.loc[(loss == 0) & (gain == 0), "rsi"] = 50.0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["wilder_atr"] = wilder_mean(tr)
    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    plus.iloc[0] = minus.iloc[0] = np.nan
    denominator = wilder_mean(tr.where(delta.notna()))
    pdi = 100 * wilder_mean(plus) / denominator.replace(0, np.nan)
    mdi = 100 * wilder_mean(minus) / denominator.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    dx = dx.mask((pdi + mdi) == 0, 0.0)
    df["adx"] = wilder_mean(dx)
    df["support_20"] = l.shift().rolling(20).min()
    df["resistance_20"] = h.shift().rolling(20).max()
    df["resistance_60"] = h.shift().rolling(60).max()
    # Shifted like its neighbours: the current bar must not cap its own target.
    df["support_60"] = l.shift().rolling(60).min()
    df["trend_atr"] = (df.ema_fast - df.ema_slow) / df.wilder_atr.replace(0, np.nan)
    df["return_3"] = c.pct_change(3)
    df["atr_fraction"] = df.wilder_atr / c
    body, span = (c - o).abs(), h - l
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    declining, advancing = c.shift() < c.shift(4), c.shift() > c.shift(4)
    session = df.timestamp.dt.date
    same = session == session.shift() if intraday else pd.Series(True, index=df.index)
    same_three = same & same.shift(fill_value=False)
    df["bullish_engulfing"] = same & declining & (c.shift() < o.shift()) & (c > o) & (o <= c.shift()) & (c >= o.shift()) & (body > body.shift())
    df["bearish_engulfing"] = same & advancing & (c.shift() > o.shift()) & (c < o) & (o >= c.shift()) & (c <= o.shift()) & (body > body.shift())
    df["hammer"] = declining & (body > span * 0.05) & (lower >= 2 * body) & (upper <= body * 0.3)
    df["shooting_star"] = advancing & (body > span * 0.05) & (upper >= 2 * body) & (lower <= body * 0.3)
    small_middle = body.shift() < body.shift(2) * 0.3
    df["morning_reversal"] = same_three & declining & (c.shift(2) < o.shift(2)) & small_middle & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
    df["evening_reversal"] = same_three & advancing & (c.shift(2) > o.shift(2)) & small_middle & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)
    df["doji"] = (span > 0) & (body <= span * 0.1)
    df["inside_bar"] = same & (h < h.shift()) & (l > l.shift())
    df["range_breakout"] = c > df.resistance_20
    df["range_breakdown"] = c < df.support_20
    df["vwap_reclaim"] = same & (c > df.vwap) & (c.shift() <= df.vwap.shift()) & intraday
    df["vwap_rejection"] = same & (c < df.vwap) & (c.shift() >= df.vwap.shift()) & intraday
    df["double_bottom_break"] = False
    df["double_top_break"] = False
    df["confirmed_swing_low"] = np.nan
    df["confirmed_swing_high"] = np.nan
    lows, highs = [], []
    for i in range(4, len(df)):
        pivot = i - 2
        # Strict extrema avoid declaring every bar on a flat plateau a pivot.
        others = [pivot - 2, pivot - 1, pivot + 1, pivot + 2]
        if l.iloc[pivot] < l.iloc[others].min():
            lows.append(pivot)
            df.loc[i, "confirmed_swing_low"] = l.iloc[pivot]
        if h.iloc[pivot] > h.iloc[others].max():
            highs.append(pivot)
            df.loc[i, "confirmed_swing_high"] = h.iloc[pivot]
        atr = df.wilder_atr.iloc[i]
        for points, is_low in ((lows, True), (highs, False)):
            if len(points) < 2 or not np.isfinite(atr):
                continue
            a, b = points[-2:]
            series = l if is_low else h
            if not 5 <= b - a <= 60 or i - a > 100 or abs(series.iloc[a] - series.iloc[b]) > 0.5 * atr:
                continue
            neckline = h.iloc[a:b + 1].max() if is_low else l.iloc[a:b + 1].min()
            crossed = c.iloc[i - 1] <= neckline < c.iloc[i] if is_low else c.iloc[i - 1] >= neckline > c.iloc[i]
            df.loc[i, "double_bottom_break" if is_low else "double_top_break"] = crossed
    return df


@dataclass(frozen=True)
class AccountRisk:
    equity: float = 200000.0
    cash: float = 200000.0
    day_start_equity: float = 200000.0
    peak_equity: float = 200000.0
    open_risk: float = 0.0
    gross_exposure: float = 0.0
    consecutive_losses: int = 0
    kill_switch: bool = False

    def __post_init__(self):
        values = (self.equity, self.cash, self.day_start_equity, self.peak_equity, self.open_risk, self.gross_exposure)
        if (not all(math.isfinite(v) for v in values)
                or min(self.equity, self.day_start_equity, self.peak_equity) <= 0
                or min(self.cash, self.open_risk, self.gross_exposure) < 0):
            raise ValueError("Account values must be finite; capital positive; risk/exposure non-negative")
        if self.cash > self.equity or self.peak_equity < self.equity:
            raise ValueError("Cash cannot exceed equity and peak equity cannot be below equity")
        if not isinstance(self.consecutive_losses, int) or self.consecutive_losses < 0:
            raise ValueError("Consecutive losses must be a non-negative integer")


@dataclass(frozen=True)
class SignalPolicy:
    risk_pct: float = 0.5
    allocation_pct: float = 25.0
    max_open_risk_pct: float = 2.0
    max_gross_exposure_pct: float = 80.0
    daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    loss_streak_limit: int = 3
    min_net_rr: float = 1.5
    tick_size: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, key)) and getattr(self, key) > 0 for key in self.__dataclass_fields__):
            raise ValueError("Policy limits must be positive and finite")
        if self.risk_pct > 3 or max(self.allocation_pct, self.max_open_risk_pct, self.max_gross_exposure_pct,
                                    self.daily_loss_pct, self.max_drawdown_pct) > 100:
            raise ValueError("Risk exceeds 3% or percentage limit exceeds 100%")
        if not isinstance(self.loss_streak_limit, int):
            raise ValueError("Loss streak limit must be an integer")


def tick_round(price: float, tick: float, upward: bool = False) -> float:
    unit = Decimal(str(tick))
    return float((Decimal(str(price)) / unit).to_integral_value(rounding=ROUND_CEILING if upward else ROUND_FLOOR) * unit)


def build_long_plan(row: pd.Series, account: AccountRisk, policy: SignalPolicy, costs: EquityCosts) -> tuple:
    """Worst entry-zone fill sizing; no assumption that the trigger will execute."""
    reasons = []
    if account.kill_switch:
        reasons.append("Manual kill switch is enabled")
    if account.equity <= account.day_start_equity * (1 - policy.daily_loss_pct / 100):
        reasons.append("Daily equity-loss limit reached (includes unrealized P&L)")
    if account.equity <= account.peak_equity * (1 - policy.max_drawdown_pct / 100):
        reasons.append("Portfolio drawdown limit reached")
    if account.consecutive_losses >= policy.loss_streak_limit:
        reasons.append("Consecutive-loss cooldown requires manual review")
    remaining_risk = account.equity * policy.max_open_risk_pct / 100 - account.open_risk
    remaining_exposure = account.equity * policy.max_gross_exposure_pct / 100 - account.gross_exposure
    budget = min(account.equity * policy.risk_pct / 100, remaining_risk)
    allocation = min(account.cash, account.equity * policy.allocation_pct / 100, remaining_exposure)
    if budget <= 0 or allocation <= 0:
        reasons.append("Portfolio open-risk or gross-exposure capacity exhausted")
    needed = [row.get(key, np.nan) for key in ("high", "low", "close", "wilder_atr")]
    if not all(np.isfinite(v) and v > 0 for v in needed):
        reasons.append("Invalid price/ATR inputs")
    if reasons:
        return None, reasons
    atr = float(row.wilder_atr)
    trigger = tick_round(float(row.high) + policy.tick_size, policy.tick_size, True)
    ceiling = tick_round(trigger + 0.25 * atr, policy.tick_size, True)
    stop = tick_round(min(float(row.low) - 0.1 * atr, trigger - 1.5 * atr), policy.tick_size)
    slip = costs.slippage_bps / 10000
    entry_fill = ceiling * (1 + slip)
    risk_distance = entry_fill - stop
    if stop <= 0 or risk_distance / entry_fill > (0.04 if costs.intraday else 0.12):
        return None, ["Stop distance is invalid or exceeds horizon volatility limit"]
    target = entry_fill + 3 * risk_distance
    for key in ("resistance_20", "resistance_60", "last_swing_high"):
        level = row.get(key, np.nan)
        if np.isfinite(level) and level > trigger:
            target = min(target, float(level) - policy.tick_size)
    target = tick_round(target, policy.tick_size)
    # Existing sizing function caps risk at 3% of cash. Apply equity budget also,
    # conservatively declining to use leverage when available cash is small.
    effective_risk = min(3.0, budget / account.cash * 100)
    quantity = size_long(entry_fill, stop, account.cash, effective_risk, costs, allocation / account.cash * 100)
    if quantity == 0:
        return None, ["No whole share fits cash, fees and risk limits"]
    buy_fee = costs.fee(entry_fill, quantity, "BUY")
    stop_fill, target_fill = stop * (1 - slip), target * (1 - slip)
    if target_fill <= 0:
        return None, ["Invalid resistance-limited target"]
    loss = quantity * (entry_fill - stop_fill) + buy_fee + costs.fee(stop_fill, quantity, "SELL")
    reward = quantity * (target_fill - entry_fill) - buy_fee - costs.fee(target_fill, quantity, "SELL")
    if reward / loss < policy.min_net_rr:
        return None, ["Nearest resistance/costs leave insufficient net reward-to-risk"]
    return {"buy_trigger": trigger, "entry_ceiling": ceiling, "stop": stop, "take_profit": target,
            "quantity": quantity, "worst_modeled_entry_fill": entry_fill, "cash_required": quantity * entry_fill + buy_fee,
            "modeled_stop_loss": loss, "modeled_target_profit": reward, "net_reward_risk": reward / loss,
            "rule": "Only consider a subsequent breakout above trigger; cancel above entry ceiling or below stop. "
                    "Recompute on the next completed bar. No fill or maximum-loss guarantee."}, []


def build_short_plan(row: pd.Series, account: AccountRisk, policy: SignalPolicy,
                     costs: EquityCosts) -> tuple:
    """Worst entry-zone fill sizing for an intraday short.

    Two deliberate asymmetries versus build_long_plan, not oversights:
    shorting is rejected outright on a delivery product, and the stop-distance cap is
    tighter (3% vs 4%) because a short's loss tail is unbounded.
    """
    if not costs.intraday:
        return None, ["Short entries are intraday-only; NSE cash delivery cannot be sold short"]
    reasons = []
    if account.kill_switch:
        reasons.append("Manual kill switch is enabled")
    if account.equity <= account.day_start_equity * (1 - policy.daily_loss_pct / 100):
        reasons.append("Daily equity-loss limit reached (includes unrealized P&L)")
    if account.equity <= account.peak_equity * (1 - policy.max_drawdown_pct / 100):
        reasons.append("Portfolio drawdown limit reached")
    if account.consecutive_losses >= policy.loss_streak_limit:
        reasons.append("Consecutive-loss cooldown requires manual review")
    remaining_risk = account.equity * policy.max_open_risk_pct / 100 - account.open_risk
    remaining_exposure = account.equity * policy.max_gross_exposure_pct / 100 - account.gross_exposure
    budget = min(account.equity * policy.risk_pct / 100, remaining_risk)
    allocation = min(account.cash, account.equity * policy.allocation_pct / 100, remaining_exposure)
    if budget <= 0 or allocation <= 0:
        reasons.append("Portfolio open-risk or gross-exposure capacity exhausted")
    needed = [row.get(key, np.nan) for key in ("high", "low", "close", "wilder_atr")]
    if not all(np.isfinite(v) and v > 0 for v in needed):
        reasons.append("Invalid price/ATR inputs")
    if reasons:
        return None, reasons
    atr = float(row.wilder_atr)
    trigger = tick_round(float(row.low) - policy.tick_size, policy.tick_size)
    entry_floor = tick_round(trigger - 0.25 * atr, policy.tick_size)
    stop = tick_round(max(float(row.high) + 0.1 * atr, trigger + 1.5 * atr), policy.tick_size, True)
    slip = costs.slippage_bps / 10000
    # Selling into the move fills lower than the quoted floor.
    entry_fill = entry_floor * (1 - slip)
    risk_distance = stop - entry_fill
    if entry_fill <= 0 or risk_distance / entry_fill > 0.03:
        return None, ["Stop distance is invalid or exceeds the short volatility limit"]
    target = entry_fill - 3 * risk_distance
    for key in ("support_20", "support_60", "last_swing_low"):
        level = row.get(key, np.nan)
        if np.isfinite(level) and level < trigger:
            target = max(target, float(level) + policy.tick_size)
    target = tick_round(target, policy.tick_size, True)
    if target <= 0:
        return None, ["Invalid support-limited target"]
    effective_risk = min(3.0, budget / account.cash * 100)
    quantity = size_short(entry_fill, stop, account.cash, effective_risk, costs,
                          allocation / account.cash * 100)
    if quantity == 0:
        return None, ["No whole share fits cash, fees and risk limits"]
    entry_fee = costs.fee(entry_fill, quantity, "SELL")
    # Both exits are buys, so adverse slippage pays UP on each.
    stop_fill, target_fill = stop * (1 + slip), target * (1 + slip)
    loss = quantity * (stop_fill - entry_fill) + entry_fee + costs.fee(stop_fill, quantity, "BUY")
    reward = quantity * (entry_fill - target_fill) - entry_fee - costs.fee(target_fill, quantity, "BUY")
    if reward <= 0 or reward / loss < policy.min_net_rr:
        return None, ["Nearest support/costs leave insufficient net reward-to-risk"]
    return {"sell_trigger": trigger, "entry_floor": entry_floor, "stop": stop, "take_profit": target,
            "quantity": quantity, "worst_modeled_entry_fill": entry_fill,
            "margin_required": quantity * entry_fill + entry_fee,
            "modeled_stop_loss": loss, "modeled_target_profit": reward, "net_reward_risk": reward / loss,
            "product": "INTRADAY",
            "rule": "Only consider a subsequent breakdown below trigger; cancel below entry floor or "
                    "above stop. Intraday only: square off before session close. Recompute on the "
                    "next completed bar. No fill or maximum-loss guarantee."}, []


def summarize_patterns(df: pd.DataFrame, last: pd.Series, duration: pd.Timedelta) -> tuple:
    """Active patterns on the latest completed bar, plus the trailing 20-bar timeline.

    Each entry is stamped with the bar close at which it became knowable, never the bar
    open, so a caller cannot mistake it for information available earlier.
    """
    active = [{"name": name, "direction": direction, "rule": rule}
              for name, (direction, rule) in PATTERNS.items() if bool(last[name])]
    recent = []
    for _, row in df.tail(20).iterrows():
        for name, (direction, _) in PATTERNS.items():
            if row[name]:
                recent.append({"known_at": (row.timestamp + duration).isoformat(),
                               "pattern": name, "direction": direction})
    return active, recent


def classify_bias(last: pd.Series) -> tuple:
    """(bull, bear, label). Both directions firing is CONFLICTING, never a net direction."""
    bull = any(bool(last[name]) for name in BULLISH)
    bear = any(bool(last[name]) for name in BEARISH)
    label = "CONFLICTING" if bull and bear else "BULLISH" if bull else "BEARISH" if bear else "NEUTRAL"
    return bull, bear, label


def evaluate_data_quality(df: pd.DataFrame, last: pd.Series, clock: pd.Timestamp,
                          duration: pd.Timedelta, expiry: pd.Timestamp, intraday: bool) -> tuple:
    """(reasons, stale). Feed problems only; says nothing about direction.

    Returned in a fixed order because callers surface the first reason to the user.
    """
    reasons = []
    minimum = 60 if intraday else 200
    if len(df) < minimum:
        reasons.append(f"Need at least {minimum} completed bars for this horizon")
    stale = clock >= expiry
    if intraday and last.timestamp.date() != clock.date():
        stale = True
    if stale:
        reasons.append("Latest data is stale; historical observations only")
    if intraday:
        recent = df[df.timestamp.dt.date == last.timestamp.date()]
        differences = recent.timestamp.diff().dropna()
        if (differences != duration).any():
            reasons.append("Missing or irregular candles in the current observed session")
    elif (df.timestamp.dt.date.duplicated()).any():
        reasons.append("Daily analysis requires at most one candle per date")
    if last.volume <= 0 or (df.volume.tail(5) <= 0).any():
        reasons.append("Zero-volume recent candles; tradability unverified")
    if df.close.pct_change().tail(20).abs().gt(0.2).any():
        reasons.append("Large price discontinuity: check corporate actions or bad data")
    return reasons, stale


def evaluate_long_gates(last: pd.Series, intraday: bool, bull: bool) -> list:
    """Hard long-entry gates. Deliberately boolean: these conditions are near-collinear,
    so weighting them would count one piece of evidence several times.
    """
    reasons = []
    if not bull:
        reasons.append("No confirmed bullish setup on the latest completed bar")
    if not (last.ema_fast > last.ema_slow and last.adx >= 20):
        reasons.append("Trend/ADX confirmation absent")
    if not (40 <= last.rsi <= 75):
        reasons.append("RSI outside long-entry risk band [40, 75]")
    if not last.relative_volume >= 1.25:
        reasons.append("Volume confirmation below 1.25x preceding 20-bar mean")
    if intraday and not last.close > last.vwap:
        reasons.append("Price is not above session VWAP")
    if not intraday and not last.close > last.sma_200:
        reasons.append("Price is not above a warmed-up SMA200")
    return reasons


def evaluate_short_gates(last: pd.Series, intraday: bool, bear: bool) -> list:
    """Mirror of evaluate_long_gates. Intraday only; enforced by build_short_plan."""
    reasons = []
    if not bear:
        reasons.append("No confirmed bearish setup on the latest completed bar")
    if not (last.ema_fast < last.ema_slow and last.adx >= 20):
        reasons.append("Downtrend/ADX confirmation absent")
    if not (25 <= last.rsi <= 60):
        reasons.append("RSI outside short-entry risk band [25, 60]")
    if not last.relative_volume >= 1.25:
        reasons.append("Volume confirmation below 1.25x preceding 20-bar mean")
    if intraday and not last.close < last.vwap:
        reasons.append("Price is not below session VWAP")
    return reasons


def analyze_market(frame: pd.DataFrame, interval: str = "5m", now=None,
                   account: AccountRisk = None, policy: SignalPolicy = None,
                   costs: EquityCosts = None, include_model: bool = True) -> dict:
    """Analyze completed observations automatically; never grant live eligibility."""
    durations = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1d"}
    if interval not in durations:
        raise ValueError("Unsupported analysis interval")
    intraday = interval != "1d"
    account, policy = account or AccountRisk(), policy or SignalPolicy()
    costs = costs or EquityCosts(intraday=intraday)
    if costs.intraday != intraday:
        raise ValueError("Analysis horizon and fee product mismatch")
    clock = pd.Timestamp.now(tz="Asia/Kolkata") if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        raise ValueError("Analysis clock must be timezone aware")
    clock = clock.tz_convert("Asia/Kolkata")
    duration = pd.Timedelta(durations[interval])
    report = {"version": "pattern-risk-v1", "status": "WAIT", "execution_eligible": False,
              "patterns": [], "recent_patterns": [], "rejections": [], "plan": None,
              "model": {"available": False, "reason": "Insufficient completed history"},
              "warnings": ["Research feed; exchange calendar, spread, circuits and corporate actions are unverified.",
                           "Pattern agreement is not a calibrated probability. No model knows the day's best prices."]}
    try:
        raw = validate_bars(frame)
    except (ValueError, TypeError, KeyError) as exc:
        report.update(status="INVALID_DATA", rejections=[str(exc)])
        return report
    if (raw.timestamp > clock).any():
        report.update(status="INVALID_DATA", rejections=["Future timestamps detected"])
        return report
    bars = raw[raw.timestamp + duration <= clock].reset_index(drop=True)
    report["excluded_incomplete_bars"] = len(raw) - len(bars)
    if bars.empty:
        report["rejections"] = ["No completed candles"]
        return report
    df = pattern_features(bars, intraday)
    last = df.iloc[-1].copy()
    swings = df.confirmed_swing_high.dropna()
    last["last_swing_high"] = float(swings.iloc[-1]) if len(swings) else np.nan
    swing_lows = df.confirmed_swing_low.dropna()
    last["last_swing_low"] = float(swing_lows.iloc[-1]) if len(swing_lows) else np.nan
    report["as_of_bar_close"] = (last.timestamp + duration).isoformat()
    expiry = last.timestamp + (2 * duration if intraday else duration + pd.Timedelta(days=7))
    report["valid_until"] = expiry.isoformat()
    report["patterns"], report["recent_patterns"] = summarize_patterns(df, last, duration)
    report["levels"] = {name: float(last[name]) if pd.notna(last[name]) else None for name in
                        ("close", "vwap", "support_20", "resistance_20", "wilder_atr", "rsi", "adx", "relative_volume")}
    reasons = report["rejections"]
    quality_reasons, stale = evaluate_data_quality(df, last, clock, duration, expiry, intraday)
    reasons.extend(quality_reasons)
    bull, bear, report["bias"] = classify_bias(last)
    data_blocked = bool(reasons)
    if bull and bear:
        reasons.append("Conflicting directional patterns")
    reasons.extend(evaluate_long_gates(last, intraday, bull))
    plan, risk_reasons = build_long_plan(last, account, policy, costs)
    reasons.extend(risk_reasons)
    if not reasons:
        report.update(status="CONDITIONAL_BUY", plan=plan)
    elif stale:
        report["status"] = "HISTORICAL_ONLY"
    elif bear and not bull and not data_blocked:
        report["status"] = "EXIT_REVIEW"
    report["exit_guidance"] = ("Bearish pattern: review existing long holdings and their original stop. "
                               "This is not an instruction to open a short." if bear else
                               "Keep the original protective stop; no automatic stop widening or averaging down.")
    # Separate empirical research evidence; it never changes action, prices or risk.
    # include_model=False skips it for fast watchlist scanning; the report then says so
    # rather than implying the model was consulted and had nothing to say.
    if include_model:
        from src.ml_engine.historical_neighbors import historical_outlook
        report["model"] = historical_outlook(df, horizon=3 if intraday else 5, intraday=intraday)
    else:
        report["model"] = {"available": False, "reason": "Not computed in fast scan mode",
                           "skipped": True}
    return report


def analyze_market_short(frame: pd.DataFrame, interval: str = "5m", now=None,
                         account: AccountRisk = None, policy: SignalPolicy = None,
                         costs: EquityCosts = None, include_model: bool = True) -> dict:
    """Intraday-only short counterpart of analyze_market.

    Kept separate rather than folded into analyze_market, whose long-only contract is
    asserted by the existing suite. Statuses are distinct so a caller can never confuse a
    short entry with a long one.
    """
    durations = {"5m": "5min", "15m": "15min", "1h": "1h"}
    if interval not in durations:
        raise ValueError("Short analysis supports intraday intervals only (5m, 15m, 1h)")
    account, policy = account or AccountRisk(), policy or SignalPolicy()
    costs = costs or EquityCosts(intraday=True)
    if not costs.intraday:
        raise ValueError("Short analysis requires an intraday fee product")
    clock = pd.Timestamp.now(tz="Asia/Kolkata") if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        raise ValueError("Analysis clock must be timezone aware")
    clock = clock.tz_convert("Asia/Kolkata")
    duration = pd.Timedelta(durations[interval])
    report = {"version": "pattern-risk-short-v1", "status": "WAIT", "execution_eligible": False,
              "direction": "SHORT", "patterns": [], "recent_patterns": [], "rejections": [],
              "plan": None, "model": {"available": False, "reason": "Insufficient completed history"},
              "warnings": ["Research feed; exchange calendar, spread, circuits and corporate actions are unverified.",
                           "Short selling is intraday-only in NSE cash and must be squared off before close.",
                           "Pattern agreement is not a calibrated probability. No model knows the day's best prices."]}
    try:
        raw = validate_bars(frame)
    except (ValueError, TypeError, KeyError) as exc:
        report.update(status="INVALID_DATA", rejections=[str(exc)])
        return report
    if (raw.timestamp > clock).any():
        report.update(status="INVALID_DATA", rejections=["Future timestamps detected"])
        return report
    bars = raw[raw.timestamp + duration <= clock].reset_index(drop=True)
    report["excluded_incomplete_bars"] = len(raw) - len(bars)
    if bars.empty:
        report["rejections"] = ["No completed candles"]
        return report
    df = pattern_features(bars, True)
    last = df.iloc[-1].copy()
    swing_lows = df.confirmed_swing_low.dropna()
    last["last_swing_low"] = float(swing_lows.iloc[-1]) if len(swing_lows) else np.nan
    report["as_of_bar_close"] = (last.timestamp + duration).isoformat()
    expiry = last.timestamp + 2 * duration
    report["valid_until"] = expiry.isoformat()
    report["patterns"], report["recent_patterns"] = summarize_patterns(df, last, duration)
    report["levels"] = {name: float(last[name]) if pd.notna(last[name]) else None for name in
                        ("close", "vwap", "support_20", "support_60", "resistance_20",
                         "wilder_atr", "rsi", "adx", "relative_volume")}
    reasons = report["rejections"]
    quality_reasons, stale = evaluate_data_quality(df, last, clock, duration, expiry, True)
    reasons.extend(quality_reasons)
    bull, bear, report["bias"] = classify_bias(last)
    data_blocked = bool(reasons)
    if bull and bear:
        reasons.append("Conflicting directional patterns")
    reasons.extend(evaluate_short_gates(last, True, bear))
    plan, risk_reasons = build_short_plan(last, account, policy, costs)
    reasons.extend(risk_reasons)
    if not reasons:
        report.update(status="SHORT_ENTRY", plan=plan)
    elif stale:
        report["status"] = "HISTORICAL_ONLY"
    elif bull and not bear and not data_blocked:
        report["status"] = "EXIT_REVIEW_SHORT"
    report["exit_guidance"] = ("Bullish pattern against a short: review any open short and its "
                               "original stop. This is not an instruction to open a long."
                               if bull else
                               "Keep the original protective stop; no averaging up into a losing short.")
    if include_model:
        from src.ml_engine.historical_neighbors import historical_outlook
        report["model"] = historical_outlook(df, horizon=3, intraday=True)
    else:
        report["model"] = {"available": False, "reason": "Not computed in fast scan mode",
                           "skipped": True}
    return report
