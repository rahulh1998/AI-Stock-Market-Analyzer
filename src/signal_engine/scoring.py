"""Setup grading and risk scoring. Fixed constants, no user-tunable weights.

Two separate numbers with separate jobs:

`setup_grade` answers "how good is this setup, on the rule evidence alone" and is the ONLY
input that can give a signal conviction. It grades five continuous axes rather than
re-weighting the hard gates in decision_engine, because those gates are near-collinear by
construction (`trend_atr` is literally (ema_fast - ema_slow)/atr), so weighting them would
count one piece of evidence five times.

`risk_score` answers "how much can this hurt" and is deliberately independent of every
model output, so a high-conviction signal on a thin, gappy, wide-stop name still reads as
risky. Terms saturate exactly where the engine's own vetoes fire, so the score and the
vetoes agree at the boundary instead of contradicting each other.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.quant_engine.decision_engine import BEARISH, BULLISH

# Grade axes: (low, high) saturating ramp. Lower bound is the gate threshold already
# required by decision_engine, so the grade measures margin ABOVE the pass mark.
GRADE_AXES = {
    "trend": (0.0, 2.0),            # directional (ema_fast - ema_slow) / wilder_atr
    "adx": (20.0, 40.0),            # gate requires >= 20
    "relative_volume": (1.25, 3.0),  # gate requires >= 1.25
    "net_reward_risk": (1.5, 4.0),   # gate requires >= policy.min_net_rr (1.5)
}

# A three-bar break of a 20-bar range is stronger evidence than a single wick.
PATTERN_STRENGTH = {
    "range_breakout": 1.0, "range_breakdown": 1.0,
    "double_bottom_break": 1.0, "double_top_break": 1.0,
    "bullish_engulfing": 0.7, "bearish_engulfing": 0.7,
    "morning_reversal": 0.7, "evening_reversal": 0.7,
    "hammer": 0.4, "shooting_star": 0.4,
    "vwap_reclaim": 0.4, "vwap_rejection": 0.4,
}

GRADE_LETTERS = ((75.0, "A"), (55.0, "B"), (0.0, "C"))

# risk term weights; deliberately sum above 1 because aggregation is a soft-OR, not a mean
RISK_WEIGHTS = {
    "stop_distance": 0.25,
    "liquidity": 0.25,
    "participation": 0.20,
    "reward_risk": 0.20,
    "extension": 0.15,
    "gap_history": 0.15,
    "opposing_pattern": 0.10,
    "sentiment_divergence": 0.10,
}

# Breakpoints per horizon: (low -> 0 risk, high -> 100 risk). Inverted terms pass
# reversed bounds. Calibrated for NSE large caps; documented, not empirically optimal.
RISK_BREAKPOINTS = {
    "stop_distance": {"intraday": (0.005, 0.040), "daily": (0.020, 0.120)},
    "liquidity": {"intraday": (2.0e7, 2.0e6), "daily": (5.0e8, 5.0e7)},
    "participation": {"intraday": (0.002, 0.020), "daily": (0.0005, 0.005)},
    "reward_risk": {"intraday": (4.0, 1.5), "daily": (4.0, 1.5)},
    "extension": {"intraday": (0.5, 4.0), "daily": (0.0, 0.25)},
    "gap_history": {"intraday": (0.0, 4.0), "daily": (0.0, 4.0)},
}

OPPOSING_PATTERN_RISK = 55.0
DIVERGENCE_RISK = 40.0
OPPOSING_LOOKBACK_BARS = 5


def ramp(value: float, low: float, high: float) -> float:
    """Clamped linear ramp to 0..100. Pass high < low for an inverted term."""
    if value is None or not math.isfinite(value) or low == high:
        return 0.0
    fraction = (value - low) / (high - low)
    return float(min(max(fraction, 0.0), 1.0) * 100.0)


def grade_letter(score: float) -> str:
    for threshold, letter in GRADE_LETTERS:
        if score >= threshold:
            return letter
    return "C"


@dataclass(frozen=True)
class SetupGrade:
    score: float
    letter: str
    axes: dict

    def to_dict(self) -> dict:
        return {"score": round(self.score, 1), "letter": self.letter,
                "axes": {k: round(v, 1) for k, v in self.axes.items()},
                "units": "rule_evidence_grade",
                "note": "Margin above the hard gate thresholds, not a probability."}


def setup_grade(row: pd.Series, plan: dict, direction: str) -> SetupGrade:
    """Grade a passing setup on five continuous axes. direction is LONG or SHORT."""
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    sign = 1.0 if direction == "LONG" else -1.0
    axes = {}

    trend = row.get("trend_atr", np.nan)
    axes["trend"] = ramp(sign * float(trend) if pd.notna(trend) else np.nan, *GRADE_AXES["trend"])
    for name in ("adx", "relative_volume"):
        value = row.get(name, np.nan)
        axes[name] = ramp(float(value) if pd.notna(value) else np.nan, *GRADE_AXES[name])
    axes["net_reward_risk"] = ramp(
        float(plan["net_reward_risk"]) if plan and "net_reward_risk" in plan else np.nan,
        *GRADE_AXES["net_reward_risk"])

    names = BULLISH if direction == "LONG" else BEARISH
    strengths = [PATTERN_STRENGTH.get(name, 0.5) for name in names if bool(row.get(name, False))]
    axes["pattern_strength"] = float(max(strengths) * 100.0) if strengths else 0.0

    score = float(np.mean(list(axes.values())))
    return SetupGrade(score=score, letter=grade_letter(score), axes=axes)


@dataclass(frozen=True)
class RiskScore:
    score: float
    terms: dict
    vetoes: list

    def to_dict(self) -> dict:
        return {"score": round(self.score, 1),
                "terms": {k: round(v, 1) for k, v in self.terms.items()},
                "vetoes": list(self.vetoes),
                "units": "risk_0_safe_100_severe",
                "weights": dict(RISK_WEIGHTS),
                "note": "Soft-OR of independent terms, so one severe term dominates. "
                        "Derived only from price, volume and cost inputs - never from a model."}


def _gap_count(df: pd.DataFrame) -> float:
    """Overnight/!bar-open gaps above 1% in the trailing 20 bars."""
    if len(df) < 2:
        return 0.0
    previous = df.close.shift(1)
    gaps = (df.open / previous - 1.0).abs().tail(20)
    return float((gaps > 0.01).sum())


def _recent_opposing(recent_patterns: list, direction: str, as_of: str) -> bool:
    """An opposing-direction pattern inside the lookback window.

    Distinct from decision_engine's CONFLICTING veto, which only fires when both
    directions appear on the SAME bar and already blocks any plan.
    """
    wanted = "bearish" if direction == "LONG" else "bullish"
    tail = recent_patterns[-(OPPOSING_LOOKBACK_BARS * len(BULLISH + BEARISH)):]
    for entry in tail:
        if entry.get("direction") == wanted and entry.get("known_at") != as_of:
            return True
    return False


def risk_score(df: pd.DataFrame, row: pd.Series, plan: dict, direction: str, intraday: bool,
               recent_patterns: list = None, as_of: str = None,
               divergence: bool = False) -> RiskScore:
    """0 (benign) to 100 (severe), plus the per-term table and any hard vetoes."""
    horizon = "intraday" if intraday else "daily"
    terms, vetoes = {}, []

    entry = float(plan["worst_modeled_entry_fill"]) if plan else float("nan")
    stop = float(plan["stop"]) if plan else float("nan")
    quantity = int(plan["quantity"]) if plan else 0

    distance = abs(entry - stop) / entry if entry and math.isfinite(entry) and entry > 0 else np.nan
    terms["stop_distance"] = ramp(distance, *RISK_BREAKPOINTS["stop_distance"][horizon])

    turnover = float((df.close * df.volume).tail(20).median()) if len(df) else np.nan
    terms["liquidity"] = ramp(turnover, *RISK_BREAKPOINTS["liquidity"][horizon])

    participation = (quantity * entry / turnover
                     if turnover and math.isfinite(turnover) and turnover > 0 and quantity else 0.0)
    terms["participation"] = ramp(participation, *RISK_BREAKPOINTS["participation"][horizon])

    terms["reward_risk"] = ramp(float(plan["net_reward_risk"]) if plan else np.nan,
                                *RISK_BREAKPOINTS["reward_risk"][horizon])

    if intraday:
        vwap, atr = row.get("vwap", np.nan), row.get("wilder_atr", np.nan)
        extension = (abs(float(row.close) - float(vwap)) / float(atr)
                     if pd.notna(vwap) and pd.notna(atr) and float(atr) > 0 else np.nan)
    else:
        sma = row.get("sma_200", np.nan)
        extension = ((float(row.close) - float(sma)) / float(sma)
                     if pd.notna(sma) and float(sma) > 0 else np.nan)
    terms["extension"] = ramp(extension, *RISK_BREAKPOINTS["extension"][horizon])

    terms["gap_history"] = ramp(_gap_count(df), *RISK_BREAKPOINTS["gap_history"][horizon])
    terms["opposing_pattern"] = (OPPOSING_PATTERN_RISK
                                 if _recent_opposing(recent_patterns or [], direction, as_of)
                                 else 0.0)
    terms["sentiment_divergence"] = DIVERGENCE_RISK if divergence else 0.0

    if terms["liquidity"] >= 100.0:
        vetoes.append("Median bar turnover below the tradable floor for this horizon")
    if terms["participation"] >= 100.0:
        vetoes.append("Order size is too large a share of observed bar turnover")

    survival = 1.0
    for name, value in terms.items():
        survival *= 1.0 - RISK_WEIGHTS[name] * value / 100.0
    return RiskScore(score=float((1.0 - survival) * 100.0), terms=terms, vetoes=vetoes)
