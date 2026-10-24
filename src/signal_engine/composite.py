"""The end-to-end signal: rules + algo + sentiment -> action, conviction, risk score.

The invariant this module exists to enforce, carried over from the audited half of this
codebase (risk_math.calculate_levels accepts model bounds and ignores them on purpose):

    Sentiment and ML may only REDUCE conviction or veto. They can never create a
    directional signal the rules do not support, and they never alter entry, stop,
    target or position size. Deterministic account-risk vetoes resolve last.

That is structural here, not a convention. Conviction is

    conviction = base * (1 - pi_algo) * (1 - pi_sent)

where `base` comes only from the rule layer and each penalty is clamped to [0, pi_max].
A positive sentiment or model reading maps to a penalty of exactly zero, so it cannot add
conviction; an unavailable component also maps to zero, so absence is not a silent 50%.
Because every non-rule factor is a non-negative penalty, conviction is monotone
non-increasing in each of them and `base == 0` forces `conviction == 0`.

Three numbers are reported, with three different meanings:
  conviction  - a ranking statistic for sorting a watchlist and for scaling size DOWN.
                Not a probability, never rendered with a percent sign.
  confidence  - a measured hit rate with a Wilson interval, from the paper book's own
                closed trades. None until a bucket has enough history.
  risk_score  - 0..100 severity from price/volume/cost inputs only.
"""
from dataclasses import dataclass, field
import json
import math

import numpy as np
import pandas as pd

from src.quant_engine.decision_engine import (AccountRisk, SignalPolicy, analyze_market,
                                              analyze_market_short, pattern_features)
from src.quant_engine.research import EquityCosts, validate_bars
from src.signal_engine.scoring import RiskScore, SetupGrade, risk_score, setup_grade

VERSION = "composite-v1"

# Multi-horizon news weights, from spec.md section 4.2.
SENTIMENT_HORIZON_WEIGHTS = {"sentiment_1h": 0.50, "sentiment_1d": 0.35, "sentiment_1w": 0.15}


@dataclass(frozen=True)
class SignalConfig:
    """Fixed, documented constants. Deliberately not user-tunable: a dial the user can
    turn makes the output a preference rather than a measurement, and makes the paper
    book's own statistics unattributable to any one configuration.
    """
    pi_algo_max: float = 0.35
    pi_sent_max: float = 0.15
    min_conviction: float = 55.0
    # A median analogue return this far against the trade applies the full algo penalty.
    algo_full_penalty_pct: float = 0.5
    sentiment_enabled: bool = True
    min_confidence_samples: int = 30

    def __post_init__(self):
        for name in ("pi_algo_max", "pi_sent_max"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.min_conviction <= 100.0:
            raise ValueError("min_conviction must be in [0, 100]")
        if self.algo_full_penalty_pct <= 0:
            raise ValueError("algo_full_penalty_pct must be positive")


def _clamp01(value: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    # +0.0 normalises negative zero, which max() propagates and json renders as -0.0.
    return float(min(max(value, 0.0), 1.0)) + 0.0


def _json_number(value):
    """Builtin float or None. json.dumps(allow_nan=False) must never see NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_sentiment(sentiment: dict) -> tuple:
    """(aggregate in [-1, 1], article_count) or (None, 0) when there is no reading."""
    if not sentiment:
        return None, 0
    total, weight_used = 0.0, 0.0
    for key, weight in SENTIMENT_HORIZON_WEIGHTS.items():
        value = sentiment.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            total += weight * number
            weight_used += weight
    if weight_used <= 0:
        return None, int(sentiment.get("articles_count") or 0)
    return total / weight_used, int(sentiment.get("articles_count") or 0)


def algo_penalty(model: dict, direction: str, config: SignalConfig) -> tuple:
    """(penalty, detail). Only a self-validated model may penalise.

    historical_outlook abstains with ABSTAIN_NO_BASELINE_IMPROVEMENT when it cannot beat a
    flat-return baseline. An abstaining or unavailable model yields a zero penalty and is
    reported as unavailable, never as a neutral probability.
    """
    detail = {"component": "algo", "available": False, "penalty": 0.0}
    if not model or not model.get("available"):
        detail["reason"] = (model or {}).get("reason") or "No model output"
        return 0.0, detail
    status = model.get("validation_status")
    if status != "EXPERIMENTAL_BASELINE_BEATEN":
        detail["reason"] = f"Model abstains ({status})"
        detail["validation_status"] = status
        return 0.0, detail
    median = model.get("median_return_pct")
    if median is None or not math.isfinite(float(median)):
        detail["reason"] = "Model produced no median analogue return"
        return 0.0, detail
    signed = float(median) * (1.0 if direction == "LONG" else -1.0)
    penalty = config.pi_algo_max * _clamp01(-signed / config.algo_full_penalty_pct)
    detail.update(available=True, penalty=round(penalty, 4),
                  validation_status=status,
                  median_return_pct=_json_number(median),
                  signed_return_pct=_json_number(signed),
                  directional_accuracy_pct=_json_number(model.get("directional_accuracy_pct")),
                  evaluation_windows=model.get("evaluation_windows"),
                  reason=("Historical analogues lean against this direction"
                          if penalty > 0 else "Historical analogues do not oppose this direction"))
    return penalty, detail


def sentiment_penalty(sentiment: dict, direction: str, config: SignalConfig) -> tuple:
    """(penalty, detail). Positive news maps to exactly zero penalty, never a bonus."""
    detail = {"component": "sentiment", "available": False, "penalty": 0.0}
    if not config.sentiment_enabled:
        detail["reason"] = "Sentiment disabled"
        return 0.0, detail
    aggregate, count = aggregate_sentiment(sentiment)
    if aggregate is None:
        detail["reason"] = "No sentiment reading"
        return 0.0, detail
    if count < 1:
        detail["reason"] = "No articles matched this instrument"
        detail["articles_count"] = count
        return 0.0, detail
    signed = aggregate * (1.0 if direction == "LONG" else -1.0)
    penalty = config.pi_sent_max * _clamp01(-signed)
    detail.update(available=True, penalty=round(penalty, 4),
                  aggregate=_json_number(aggregate), signed=_json_number(signed),
                  articles_count=count, scorer=sentiment.get("scorer", "unknown"),
                  horizons={k: _json_number(sentiment.get(k))
                            for k in SENTIMENT_HORIZON_WEIGHTS},
                  reason=("News leans against this direction" if penalty > 0
                          else "News does not oppose this direction"))
    return penalty, detail


@dataclass(frozen=True)
class TradeSignal:
    ticker: str
    interval: str
    horizon: str
    action: str
    direction: str
    conviction: float
    setup: dict
    risk: dict
    components: dict
    plan: dict = None
    confidence: dict = None
    as_of_bar_close: str = None
    valid_until: str = None
    status: str = "WAIT"
    rejections: list = field(default_factory=list)
    vetoes: list = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    bias: str = "NEUTRAL"
    data_quality: str = "UNVERIFIED"
    warnings: list = field(default_factory=list)
    version: str = VERSION

    def to_dict(self) -> dict:
        """JSON-safe: no NaN and no numpy scalars, so allow_nan=False always holds."""
        return {
            "version": self.version, "ticker": self.ticker, "interval": self.interval,
            "horizon": self.horizon, "action": self.action, "direction": self.direction,
            "conviction": round(float(self.conviction), 1),
            "conviction_units": "ranking_score",
            "conviction_note": "Opinion strength for ranking and for scaling size down. "
                               "Not a probability; never read it as a percent chance.",
            "confidence": self.confidence,
            "setup": self.setup, "risk": self.risk, "components": self.components,
            "plan": self.plan, "status": self.status,
            "as_of_bar_close": self.as_of_bar_close, "valid_until": self.valid_until,
            "rejections": list(self.rejections), "vetoes": list(self.vetoes),
            "levels": {k: _json_number(v) for k, v in (self.levels or {}).items()},
            "bias": self.bias, "data_quality": self.data_quality,
            "execution_eligible": False,
            "warnings": list(self.warnings),
        }


INTERVAL_DURATIONS = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1d"}


def _completed_bars(frame: pd.DataFrame, interval: str, now=None) -> pd.DataFrame:
    """Mirror of analyze_market's incomplete-bar exclusion (decision_engine.py:270)."""
    bars = validate_bars(frame)
    clock = pd.Timestamp.now(tz="Asia/Kolkata") if now is None else pd.Timestamp(now)
    clock = clock.tz_convert("Asia/Kolkata")
    duration = pd.Timedelta(INTERVAL_DURATIONS[interval])
    return bars[bars.timestamp + duration <= clock].reset_index(drop=True)


NOT_CONSULTED = ("Not consulted: the rules produced no directional setup, so no penalty "
                 "could apply. Sentiment and the model can only reduce conviction, never "
                 "create a signal.")


def _unconsulted_components() -> dict:
    """Components for the no-setup path, which returns before any penalty is computed.

    They carry an explicit reason rather than an empty dict, so the UI never has to render
    "not available" with nothing to say - and so the invariant is visible to the user.
    """
    return {name: {"component": name, "available": False, "penalty": 0.0,
                   "reason": NOT_CONSULTED}
            for name in ("algo", "sentiment")}


def generate_signal(bars: pd.DataFrame, interval: str = "15m", *, ticker: str = "",
                    account: AccountRisk = None, policy: SignalPolicy = None,
                    costs: EquityCosts = None, sentiment: dict = None, now=None,
                    config: SignalConfig = None, enable_short: bool = True,
                    confidence_lookup=None, include_model: bool = True) -> TradeSignal:
    """Produce one TradeSignal for one instrument on one interval.

    Order of operations IS the invariant: the rule engine decides direction first, then
    penalties scale conviction down, then the conviction floor applies, then risk vetoes
    resolve last and clear the plan.
    """
    config = config or SignalConfig()
    intraday = interval != "1d"
    account, policy = account or AccountRisk(), policy or SignalPolicy()
    costs = costs or EquityCosts(intraday=intraday)
    horizon = "intraday" if intraday else "daily"

    long_report = analyze_market(bars, interval, now=now, account=account, policy=policy,
                                 costs=costs, include_model=include_model)
    short_report = None
    if intraday and enable_short:
        short_report = analyze_market_short(bars, interval, now=now, account=account,
                                           policy=policy, costs=costs,
                                           include_model=include_model)

    # 1. Rules alone choose the direction. Nothing downstream may override this.
    if long_report["status"] == "CONDITIONAL_BUY" and long_report["plan"]:
        direction, action, source = "LONG", "BUY", long_report
    elif short_report and short_report["status"] == "SHORT_ENTRY" and short_report["plan"]:
        direction, action, source = "SHORT", "SHORT_ENTRY", short_report
    elif long_report["status"] == "EXIT_REVIEW":
        direction, action, source = "LONG", "EXIT_LONG", long_report
    else:
        direction, action, source = "NONE", "HOLD", long_report

    base_signal = TradeSignal(
        ticker=ticker, interval=interval, horizon=horizon, action=action, direction=direction,
        conviction=0.0, setup={}, risk={}, components=_unconsulted_components(),
        status=source["status"], as_of_bar_close=source.get("as_of_bar_close"),
        valid_until=source.get("valid_until"), rejections=source.get("rejections", []),
        levels=source.get("levels", {}), bias=source.get("bias", "NEUTRAL"),
        warnings=source.get("warnings", []),
    )

    # No actionable entry: report the rule engine's own verdict and stop. Sentiment and
    # the model are not even consulted, so they cannot manufacture a direction.
    if action in ("HOLD", "EXIT_LONG"):
        return base_signal

    plan = dict(source["plan"])
    # Recompute features over the SAME completed-bar set analyze_market used. Taking the
    # last row of the unfiltered frame would score a still-forming bar against a plan
    # built from the previous one.
    features = pattern_features(_completed_bars(bars, interval, now), intraday)
    row = features.iloc[-1]

    # 2. Rule evidence is the only thing that can give conviction.
    grade = setup_grade(row, plan, direction)

    # 3. Penalties: non-negative, so they can only reduce.
    algo_pi, algo_detail = algo_penalty(source.get("model"), direction, config)
    sent_pi, sent_detail = sentiment_penalty(sentiment, direction, config)
    conviction = grade.score * (1.0 - algo_pi) * (1.0 - sent_pi)

    # 4. Risk is scored independently of every model output.
    divergence = bool(sentiment and "DIVERGENCE" in str(sentiment.get("divergence_flag", "")))
    risk = risk_score(features, row, plan, direction, intraday,
                      recent_patterns=source.get("recent_patterns", []),
                      as_of=source.get("as_of_bar_close"), divergence=divergence)

    rejections = list(source.get("rejections", []))
    vetoes = list(risk.vetoes)
    final_action, final_plan, status = action, plan, source["status"]

    # 5. Conviction floor.
    if conviction < config.min_conviction:
        final_action, final_plan, status = "HOLD", None, "LOW_CONVICTION"
        rejections.append(
            f"Conviction {conviction:.1f} below the {config.min_conviction:.0f} floor "
            f"(setup {grade.score:.1f}, algo penalty {algo_pi:.0%}, news penalty {sent_pi:.0%})")

    # 6. Risk vetoes resolve last and clear the plan, never merely colour a badge.
    if vetoes:
        final_action, final_plan, status = "HOLD", None, "RISK_VETO"
        rejections.extend(vetoes)

    confidence = None
    if confidence_lookup is not None and final_plan:
        confidence = confidence_lookup(direction=direction, interval=interval,
                                       grade=grade.letter)

    return TradeSignal(
        ticker=ticker, interval=interval, horizon=horizon, action=final_action,
        direction=direction, conviction=conviction, setup=grade.to_dict(),
        risk=risk.to_dict(), components={"algo": algo_detail, "sentiment": sent_detail},
        plan=final_plan, confidence=confidence, status=status,
        as_of_bar_close=source.get("as_of_bar_close"), valid_until=source.get("valid_until"),
        rejections=rejections, vetoes=vetoes, levels=source.get("levels", {}),
        bias=source.get("bias", "NEUTRAL"), warnings=source.get("warnings", []),
    )


if __name__ == "__main__":
    import argparse
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from src.data_pipeline.broker_client import MarketDataFetcher
    from src.signal_engine.scanner import PERIOD_FOR_INTERVAL, resolve_clock

    parser = argparse.ArgumentParser(description="Print one TradeSignal as JSON")
    parser.add_argument("--ticker", default="RELIANCE")
    parser.add_argument("--interval", default="15m", choices=sorted(INTERVAL_DURATIONS))
    parser.add_argument("--period", default=None)
    parser.add_argument("--live", action="store_true",
                        help="Use the wall clock instead of reviewing the last completed bar")
    parser.add_argument("--as-of", default=None,
                        help="Truncate history at this date (YYYY-MM-DD) to replay a past bar")
    parser.add_argument("--no-sentiment", action="store_true")
    args = parser.parse_args()

    intraday = args.interval != "1d"
    frame = MarketDataFetcher().fetch_intraday_data(
        args.ticker, args.period or PERIOD_FOR_INTERVAL[args.interval], args.interval)
    if frame is None or frame.empty:
        raise SystemExit(f"No candles returned for {args.ticker}")
    if args.as_of:
        keep = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d") <= args.as_of
        frame = frame[keep].reset_index(drop=True)
        if frame.empty:
            raise SystemExit(f"No candles on or before {args.as_of}")

    reading = None
    if not args.no_sentiment:
        from src.sentiment_engine.multi_horizon import SentimentService

        service = SentimentService()
        reading = service.reading(args.ticker, bars=frame)
        service.close()

    clock, label = resolve_clock(frame, args.interval,
                                 "live" if args.live else "last_close_review")
    result = generate_signal(frame, args.interval, ticker=args.ticker, now=clock,
                             costs=EquityCosts(intraday=intraday), sentiment=reading)
    print(f"# {label}", file=sys.stderr)
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
