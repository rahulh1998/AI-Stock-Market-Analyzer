"""Portfolio backtest of the actual composite strategy.

The existing `research.replay` tests the old EMA/VWAP baseline on one instrument. This
runs the strategy the app really emits, across the whole universe, under the real
portfolio constraints — which is the only thing that can say whether the signals are worth
anything.

Three design rules keep it honest:

1. **Shared code, not a reimplementation.** Gates come from `evaluate_long_gates`, plans
   from `build_long_plan`, grading from `setup_grade`, risk from `risk_score`, barrier
   resolution from `research.resolve_exit`, and fees from `EquityCosts`. A backtest that
   re-derives the strategy is testing a different strategy.
2. **Features are computed once per ticker, not per bar.** This is sound precisely because
   `pattern_features` is prefix-causal, a property asserted by
   `test_decision_engine.test_prefix_causality_including_pivots`. Without that guarantee
   this optimisation would be look-ahead bias.
3. **Long-only, like `analyze_market`.** Shorts are intraday-only and would need margin and
   borrow modelling this does not have, so the backtest declines to pretend.
4. **Entries follow the plan's own rule.** `build_long_plan` emits a conditional order:
   trigger above the bar high, cancel above the entry ceiling. The backtest fills it that
   way — including declining to fill when the next bar gaps straight past the zone — rather
   than assuming a next-open fill the plan never promised.
"""
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from src.quant_engine.decision_engine import (AccountRisk, SignalPolicy, build_long_plan,
                                              classify_bias, evaluate_long_gates,
                                              pattern_features)
from src.quant_engine.research import EquityCosts, resolve_exit
from src.signal_engine.composite import SignalConfig
from src.signal_engine.scoring import risk_context, risk_score, setup_grade

logger = logging.getLogger(__name__)

MIN_BARS = {"daily": 200, "intraday": 60}


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 200000.0
    policy: SignalPolicy = field(default_factory=SignalPolicy)
    costs: EquityCosts = field(default_factory=lambda: EquityCosts(intraday=False))
    signal: SignalConfig = field(default_factory=SignalConfig)
    intraday: bool = False
    # A conditional order lives this many bars before it is cancelled unfilled.
    order_valid_bars: int = 1
    warmup_bars: int = 200


def prepare_panel(panel: dict, intraday: bool = False) -> dict:
    """Features, risk context and data-quality flags, computed once per ticker.

    Every column here is causal: each value depends only on bars at or before its own row.
    """
    prepared = {}
    for ticker, frame in panel.items():
        if frame is None or len(frame) < 60:
            continue
        try:
            features = pattern_features(frame, intraday)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("skipping %s: %s", ticker, exc)
            continue
        # analyze_market takes the latest confirmed swing; forward-filling the confirmed
        # series reproduces that per-bar without a backward look.
        features["last_swing_high"] = features.confirmed_swing_high.ffill()
        features["last_swing_low"] = features.confirmed_swing_low.ffill()
        context = risk_context(features)
        returns = features.close.pct_change().abs()
        flags = pd.DataFrame({
            "zero_volume_5": (features.volume <= 0).rolling(5, min_periods=1).max().astype(bool),
            "discontinuity_20": (returns > 0.2).rolling(20, min_periods=1).max().fillna(0).astype(bool),
        }, index=features.index)
        features["date"] = features.timestamp.dt.date
        prepared[ticker] = {"features": features, "context": context, "flags": flags,
                            "by_date": {d: i for i, d in enumerate(features["date"])}}
    return prepared


def _fill_price(bar, trigger: float, limit: float, direction: str, slippage: float):
    """Conditional-order fill, or None. `limit` is the entry ceiling (long) / floor (short)."""
    if direction == "LONG":
        if bar.open > limit:
            return None                      # gapped straight past the entry zone
        if bar.high >= trigger:
            price = max(bar.open, trigger)
            return None if price > limit else price * (1 + slippage)
        return None
    if bar.open < limit:
        return None
    if bar.low <= trigger:
        price = min(bar.open, trigger)
        return None if price < limit else price * (1 - slippage)
    return None


@dataclass
class Position:
    ticker: str
    direction: str
    quantity: int
    entry: float
    stop: float
    target: float
    entry_date: object
    entry_fee: float
    grade: str
    conviction: float
    risk_score: float
    risk_amount: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict
    config: dict


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame, capital: float,
             periods_per_year: float = 252.0) -> dict:
    if equity.empty:
        return {"trades": 0, "note": "no equity curve"}
    curve = equity.set_index("date")["equity"]
    returns = curve.pct_change().dropna()
    peak = curve.cummax().clip(lower=capital)
    drawdown = ((peak - curve) / peak).max()
    years = max(len(curve) / periods_per_year, 1e-9)
    total = curve.iloc[-1] / capital
    out = {
        "trades": int(len(trades)),
        "final_equity": float(curve.iloc[-1]),
        "net_return_pct": float((total - 1) * 100),
        "cagr_pct": float((total ** (1 / years) - 1) * 100) if total > 0 else None,
        "max_drawdown_pct": float(drawdown * 100),
        "volatility_annual_pct": float(returns.std() * np.sqrt(periods_per_year) * 100),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
                  if returns.std() > 0 else None,
        "bars": int(len(curve)),
        "years": round(years, 2),
    }
    downside = returns[returns < 0]
    out["sortino"] = (float(returns.mean() / downside.std() * np.sqrt(periods_per_year))
                      if len(downside) > 1 and downside.std() > 0 else None)
    if not trades.empty:
        pnl = trades.net_pnl
        gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out.update({
            "win_rate_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "expectancy_rupees": float(pnl.mean()),
            "total_fees": float(trades.fees.sum()),
            "avg_bars_held": float(trades.bars_held.mean()),
            "exit_reasons": {str(k): int(v) for k, v in trades.reason.value_counts().items()},
        })
    return out


def run_backtest(prepared: dict, calendar, config: BacktestConfig = None) -> BacktestResult:
    """Walk the calendar once, managing a single portfolio across all tickers."""
    config = config or BacktestConfig()
    policy, costs = config.policy, config.costs
    slippage = costs.slippage_bps / 10000
    horizon = "intraday" if config.intraday else "daily"
    minimum_bars = MIN_BARS[horizon]
    dates = sorted(calendar)

    cash = config.capital
    positions: dict = {}
    pending: list = []
    trades, curve, signal_log = [], [], []
    peak_equity = config.capital

    for date in dates:
        # --- 1. Exits on today's bar, for positions opened earlier ---------------
        for ticker in list(positions):
            position = positions[ticker]
            store = prepared.get(ticker)
            index = store["by_date"].get(date) if store else None
            if index is None:
                continue
            bar = store["features"].iloc[index]
            price, reason = resolve_exit(bar, position.stop, position.target, position.direction)
            if price is None:
                continue
            exit_fill = price * (1 - slippage)
            fee = costs.fee(exit_fill, position.quantity, "SELL")
            gross = (exit_fill - position.entry) * position.quantity
            cash += exit_fill * position.quantity - fee
            trades.append({
                "ticker": ticker, "direction": position.direction,
                "entry_date": position.entry_date, "exit_date": date,
                "entry": position.entry, "exit": exit_fill, "quantity": position.quantity,
                "reason": reason, "grade": position.grade, "conviction": position.conviction,
                "risk_score": position.risk_score,
                "fees": position.entry_fee + fee, "net_pnl": gross - position.entry_fee - fee,
                "bars_held": (pd.Timestamp(date) - pd.Timestamp(position.entry_date)).days,
            })
            del positions[ticker]

        # --- 2. Fill yesterday's conditional orders on today's bar ---------------
        still_pending = []
        for order in pending:
            store = prepared.get(order["ticker"])
            index = store["by_date"].get(date) if store else None
            if index is None:
                continue
            bar = store["features"].iloc[index]
            fill = _fill_price(bar, order["trigger"], order["limit"], order["direction"], slippage)
            if fill is None:
                order["bars"] += 1
                if order["bars"] < config.order_valid_bars:
                    still_pending.append(order)
                continue
            if order["ticker"] in positions:
                continue
            fee = costs.fee(fill, order["quantity"], "BUY")
            committed = fill * order["quantity"] + fee
            if committed > cash:
                continue
            cash -= committed
            positions[order["ticker"]] = Position(
                ticker=order["ticker"], direction=order["direction"], quantity=order["quantity"],
                entry=fill, stop=order["stop"], target=order["target"], entry_date=date,
                entry_fee=fee, grade=order["grade"], conviction=order["conviction"],
                risk_score=order["risk_score"],
                risk_amount=abs(fill - order["stop"]) * order["quantity"])
            # A position can gap through its stop on the very bar it opens.
            position = positions[order["ticker"]]
            price, reason = resolve_exit(bar, position.stop, position.target, position.direction)
            if price is not None:
                exit_fill = price * (1 - slippage)
                exit_fee = costs.fee(exit_fill, position.quantity, "SELL")
                cash += exit_fill * position.quantity - exit_fee
                trades.append({
                    "ticker": position.ticker, "direction": position.direction,
                    "entry_date": date, "exit_date": date, "entry": position.entry,
                    "exit": exit_fill, "quantity": position.quantity, "reason": reason,
                    "grade": position.grade, "conviction": position.conviction,
                    "risk_score": position.risk_score,
                    "fees": position.entry_fee + exit_fee,
                    "net_pnl": (exit_fill - position.entry) * position.quantity
                               - position.entry_fee - exit_fee,
                    "bars_held": 0})
                del positions[position.ticker]
        pending = still_pending

        # --- 3. Mark to market ---------------------------------------------------
        holdings = 0.0
        open_risk = 0.0
        for ticker, position in positions.items():
            store = prepared[ticker]
            index = store["by_date"].get(date)
            price = (float(store["features"].close.iloc[index]) if index is not None
                     else position.entry)
            holdings += price * position.quantity
            open_risk += position.risk_amount
        equity = cash + holdings
        peak_equity = max(peak_equity, equity)
        curve.append({"date": date, "equity": equity, "cash": cash,
                      "positions": len(positions), "holdings": holdings})

        # --- 4. Generate signals on today's completed bar -> orders for tomorrow --
        if equity <= 0:
            continue
        account_template = dict(equity=equity, cash=min(cash, equity),
                                day_start_equity=equity, peak_equity=max(peak_equity, equity),
                                open_risk=open_risk, gross_exposure=holdings)
        candidates = []
        for ticker, store in prepared.items():
            if ticker in positions or any(o["ticker"] == ticker for o in pending):
                continue
            index = store["by_date"].get(date)
            if index is None or index < max(minimum_bars, config.warmup_bars):
                continue
            flags = store["flags"].iloc[index]
            if bool(flags.zero_volume_5) or bool(flags.discontinuity_20):
                continue
            row = store["features"].iloc[index]
            bull, bear, _ = classify_bias(row)
            if bull and bear:
                continue
            if evaluate_long_gates(row, config.intraday, bull):
                continue
            try:
                account = AccountRisk(**account_template)
            except ValueError:
                break
            plan, _ = build_long_plan(row, account, policy, costs)
            if not plan:
                continue
            grade = setup_grade(row, plan, "LONG")
            risk = risk_score(row, plan, "LONG", config.intraday,
                              context=store["context"].iloc[index])
            if risk.vetoes:
                continue
            conviction = grade.score   # no live news or model in a historical replay
            signal_log.append({"date": date, "ticker": ticker, "conviction": conviction,
                               "grade": grade.letter, "risk_score": risk.score})
            if conviction < config.signal.min_conviction:
                continue
            candidates.append((conviction, ticker, row, grade, risk))

        # --- 5. Allocate in conviction order, re-sizing against live capacity -----
        candidates.sort(key=lambda item: -item[0])
        committed_cash, committed_risk, committed_exposure = 0.0, 0.0, 0.0
        for conviction, ticker, row, grade, risk in candidates:
            available_cash = cash - committed_cash
            if available_cash <= 0:
                break
            try:
                account = AccountRisk(
                    equity=equity, cash=min(available_cash, equity),
                    day_start_equity=equity, peak_equity=max(peak_equity, equity),
                    open_risk=open_risk + committed_risk,
                    gross_exposure=holdings + committed_exposure)
            except ValueError:
                continue
            plan, _ = build_long_plan(row, account, policy, costs)
            if not plan:
                continue
            cost_now = plan["cash_required"]
            if cost_now > available_cash:
                continue
            committed_cash += cost_now
            committed_risk += plan["modeled_stop_loss"]
            committed_exposure += plan["quantity"] * plan["worst_modeled_entry_fill"]
            pending.append({
                "ticker": ticker, "direction": "LONG", "trigger": plan["buy_trigger"],
                "limit": plan["entry_ceiling"], "stop": plan["stop"],
                "target": plan["take_profit"], "quantity": plan["quantity"],
                "grade": grade.letter, "conviction": conviction,
                "risk_score": risk.score, "bars": 0})

    # --- close anything still open at the final bar ------------------------------
    if positions and dates:
        final = dates[-1]
        for ticker, position in list(positions.items()):
            store = prepared[ticker]
            index = store["by_date"].get(final)
            if index is None:
                continue
            price = float(store["features"].close.iloc[index]) * (1 - slippage)
            fee = costs.fee(price, position.quantity, "SELL")
            cash += price * position.quantity - fee
            trades.append({
                "ticker": ticker, "direction": position.direction,
                "entry_date": position.entry_date, "exit_date": final,
                "entry": position.entry, "exit": price, "quantity": position.quantity,
                "reason": "sample_end", "grade": position.grade,
                "conviction": position.conviction, "risk_score": position.risk_score,
                "fees": position.entry_fee + fee,
                "net_pnl": (price - position.entry) * position.quantity - position.entry_fee - fee,
                "bars_held": (pd.Timestamp(final) - pd.Timestamp(position.entry_date)).days})
            del positions[ticker]
        if curve:
            curve[-1]["equity"] = cash
            curve[-1]["cash"] = cash
            curve[-1]["holdings"] = 0.0

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(curve)
    return BacktestResult(
        trades=trades_frame, equity=equity_frame,
        signals=pd.DataFrame(signal_log),
        metrics=_metrics(trades_frame, equity_frame, config.capital),
        config={"capital": config.capital, "risk_pct": policy.risk_pct,
                "allocation_pct": policy.allocation_pct,
                "max_open_risk_pct": policy.max_open_risk_pct,
                "min_conviction": config.signal.min_conviction,
                "slippage_bps": costs.slippage_bps, "intraday": config.intraday,
                "order_valid_bars": config.order_valid_bars})
