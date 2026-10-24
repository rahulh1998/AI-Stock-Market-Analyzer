"""Invariant tests for the composite signal.

The properties asserted here are the whole reason signal_engine is structured the way it
is. If any of these fail, sentiment or a model has gained the ability to manufacture or
reprice a trade, which is the specific failure mode the previous audit of this repo was
written to prevent.
"""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.quant_engine.decision_engine import AccountRisk, SignalPolicy, pattern_features
from src.quant_engine.research import EquityCosts
from src.signal_engine.composite import (SignalConfig, TradeSignal, aggregate_sentiment,
                                         algo_penalty, generate_signal, sentiment_penalty)
from src.signal_engine.scoring import risk_score, setup_grade


def bars(count=320, start="2026-10-01 09:15", freq="15min", trend=0.0015):
    """Rising intraday series with enough history to clear the 60-bar minimum."""
    index = pd.date_range(start, periods=count, freq=freq, tz="Asia/Kolkata")
    close = 1000 * np.cumprod(1 + np.full(count, trend))
    return pd.DataFrame({
        "timestamp": index, "open": close - 0.4, "high": close + 1.2,
        "low": close - 1.2, "close": close,
        "volume": np.full(count, 50000, dtype=np.int64),
    })


def bullish_features(frame, intraday=True):
    """A frame whose final bar satisfies every long gate, so a plan can be built."""
    df = pattern_features(frame, intraday)
    last = len(df) - 1
    for name in ("bullish_engulfing", "hammer", "morning_reversal", "double_bottom_break",
                 "vwap_reclaim", "range_breakdown", "bearish_engulfing", "shooting_star",
                 "evening_reversal", "double_top_break", "vwap_rejection", "doji",
                 "inside_bar"):
        df.loc[last, name] = False
    # Levels are set RELATIVE to the series' own last close. Pinning absolute prices here
    # would create a huge final pct_change and trip the discontinuity data-quality check.
    price = float(df.loc[last, "close"])
    df.loc[last, "range_breakout"] = True
    df.loc[last, "ema_fast"] = price
    df.loc[last, "ema_slow"] = price * 0.955
    df.loc[last, "adx"] = 32.0
    df.loc[last, "rsi"] = 58.0
    df.loc[last, "relative_volume"] = 2.0
    df.loc[last, "wilder_atr"] = price * 0.0055
    df.loc[last, "trend_atr"] = 1.6
    df.loc[last, "high"] = price * 1.0009
    df.loc[last, "low"] = price * 0.9964
    df.loc[last, "vwap"] = price * 0.982
    df.loc[last, "sma_200"] = price * 0.90
    df.loc[last, "support_20"] = price * 0.955
    df.loc[last, "resistance_20"] = price * 1.27
    df.loc[last, "resistance_60"] = price * 1.27
    df.loc[last, "support_60"] = price * 0.945
    return df


def flat_features(frame, intraday=True):
    """No pattern on the final bar: the rules must refuse to trade."""
    df = pattern_features(frame, intraday)
    last = len(df) - 1
    for name, (direction, _) in __import__(
            "src.quant_engine.decision_engine", fromlist=["PATTERNS"]).PATTERNS.items():
        df.loc[last, name] = False
    return df


MODEL_BULLISH = {"available": True, "validation_status": "EXPERIMENTAL_BASELINE_BEATEN",
                 "median_return_pct": 1.2, "directional_accuracy_pct": 61.0,
                 "evaluation_windows": 30}
MODEL_BEARISH = {"available": True, "validation_status": "EXPERIMENTAL_BASELINE_BEATEN",
                 "median_return_pct": -1.2, "directional_accuracy_pct": 59.0,
                 "evaluation_windows": 30}
MODEL_ABSTAIN = {"available": True, "validation_status": "ABSTAIN_NO_BASELINE_IMPROVEMENT",
                 "median_return_pct": -5.0}
MODEL_NONE = {"available": False, "reason": "Insufficient matured history"}


def sentiment(value, count=8, divergence="CONGRUENT"):
    return {"sentiment_1h": value, "sentiment_1d": value, "sentiment_1w": value,
            "articles_count": count, "divergence_flag": divergence, "scorer": "test"}


def run(features_fn=bullish_features, model=MODEL_BULLISH, sent=None, interval="15m",
        config=None, account=None, frame=None, **kwargs):
    frame = bars() if frame is None else frame
    now = frame.timestamp.iloc[-1] + pd.Timedelta(minutes=15)
    with patch("src.quant_engine.decision_engine.pattern_features", side_effect=features_fn), \
         patch("src.ml_engine.historical_neighbors.historical_outlook", return_value=model), \
         patch("src.signal_engine.composite.pattern_features", side_effect=features_fn):
        return generate_signal(frame, interval, ticker="TEST", now=now, sentiment=sent,
                               config=config or SignalConfig(),
                               account=account or AccountRisk(), **kwargs)


class SentimentCannotCreateSignalTests(unittest.TestCase):
    def test_sentiment_cannot_create_direction(self):
        """No rule setup: no amount of bullish news may produce a trade."""
        for value in np.linspace(-1.0, 1.0, 21):
            with self.subTest(sentiment=round(float(value), 2)):
                signal = run(features_fn=flat_features, sent=sentiment(float(value)))
                self.assertEqual(signal.action, "HOLD")
                self.assertIsNone(signal.plan)
                self.assertEqual(signal.conviction, 0.0)

    def test_positive_sentiment_adds_exactly_nothing(self):
        """Conviction must be identical for neutral and maximally bullish news."""
        neutral = run(sent=sentiment(0.0))
        glowing = run(sent=sentiment(1.0))
        self.assertEqual(neutral.conviction, glowing.conviction)
        self.assertEqual(glowing.components["sentiment"]["penalty"], 0.0)

    def test_sentiment_is_monotone_non_decreasing(self):
        """Conviction never falls as news improves."""
        previous = -1.0
        for value in np.linspace(-1.0, 1.0, 21):
            signal = run(sent=sentiment(float(value)))
            self.assertGreaterEqual(round(signal.conviction, 9), round(previous, 9))
            previous = signal.conviction

    def test_sentiment_never_moves_prices_or_size(self):
        """News may remove a plan via the conviction floor, but never reprice or resize it.

        One equality covers entry, stop, target and quantity at once.
        """
        baseline = run(sent=sentiment(0.0)).plan
        self.assertIsNotNone(baseline)
        for value in (-1.0, -0.5, 0.5, 1.0):
            with self.subTest(sentiment=value):
                plan = run(sent=sentiment(value)).plan
                self.assertIn(plan, (None, baseline))
        # And at least one negative case must actually still carry a plan, so this test
        # cannot pass merely because every penalised branch returned None.
        self.assertEqual(run(sent=sentiment(-0.5)).plan, baseline)

    def test_missing_sentiment_equals_neutral(self):
        self.assertEqual(run(sent=None).conviction, run(sent=sentiment(0.0)).conviction)
        self.assertFalse(run(sent=None).components["sentiment"]["available"])

    def test_zero_articles_is_not_a_reading(self):
        signal = run(sent=sentiment(-1.0, count=0))
        self.assertEqual(signal.components["sentiment"]["penalty"], 0.0)
        self.assertFalse(signal.components["sentiment"]["available"])


class ModelCannotCreateSignalTests(unittest.TestCase):
    def test_model_cannot_create_direction(self):
        signal = run(features_fn=flat_features, model=MODEL_BULLISH)
        self.assertEqual(signal.action, "HOLD")
        self.assertIsNone(signal.plan)
        self.assertEqual(signal.conviction, 0.0)

    def test_model_is_penalty_only(self):
        """A bullish model cannot raise conviction above the rule grade."""
        bullish = run(model=MODEL_BULLISH)
        # setup["score"] is rounded for display; conviction is raw.
        self.assertAlmostEqual(bullish.conviction, bullish.setup["score"], places=1)
        self.assertEqual(bullish.components["algo"]["penalty"], 0.0)
        self.assertLess(run(model=MODEL_BEARISH).conviction, bullish.conviction)

    def test_abstaining_model_applies_no_penalty(self):
        """ABSTAIN must behave exactly like absence, despite a -5% median."""
        self.assertEqual(run(model=MODEL_ABSTAIN).conviction, run(model=MODEL_NONE).conviction)
        self.assertFalse(run(model=MODEL_ABSTAIN).components["algo"]["available"])

    def test_no_silent_fifty_percent(self):
        """An unavailable model exposes no probability at all."""
        detail = run(model=MODEL_NONE).components["algo"]
        self.assertFalse(detail["available"])
        self.assertNotIn("p", detail)
        self.assertNotIn("probability", detail)
        self.assertEqual(detail["penalty"], 0.0)

    def test_model_never_moves_prices_or_size(self):
        """Same contract as sentiment: remove the plan, or leave it byte-identical."""
        baseline = run(model=MODEL_NONE).plan
        self.assertIsNotNone(baseline)
        for model in (MODEL_BULLISH, MODEL_BEARISH, MODEL_ABSTAIN):
            with self.subTest(model=model.get("validation_status")):
                self.assertIn(run(model=model).plan, (None, baseline))
        self.assertEqual(run(model=MODEL_BULLISH).plan, baseline)


class VetoPrecedenceTests(unittest.TestCase):
    ACCOUNTS = [
        AccountRisk(kill_switch=True),
        AccountRisk(equity=100000, cash=100000, day_start_equity=200000, peak_equity=200000),
        AccountRisk(equity=100000, cash=100000, day_start_equity=100000, peak_equity=200000),
        AccountRisk(consecutive_losses=5),
        AccountRisk(open_risk=100000),
        AccountRisk(gross_exposure=200000),
        AccountRisk(equity=200000, cash=0.0),
    ]

    def test_account_veto_beats_everything(self):
        """Perfect rules, best model, best news: an account lock still wins."""
        for account in self.ACCOUNTS:
            with self.subTest(account=account):
                signal = run(model=MODEL_BULLISH, sent=sentiment(1.0), account=account)
                self.assertIsNone(signal.plan)
                self.assertEqual(signal.action, "HOLD")

    def test_conviction_floor_clears_the_plan(self):
        signal = run(model=MODEL_BEARISH, sent=sentiment(-1.0),
                     config=SignalConfig(min_conviction=99.9))
        self.assertEqual(signal.action, "HOLD")
        self.assertIsNone(signal.plan)
        self.assertEqual(signal.status, "LOW_CONVICTION")
        self.assertTrue(any("below the" in r for r in signal.rejections))

    def test_risk_veto_clears_the_plan_not_just_the_badge(self):
        thin = bars()
        thin["volume"] = 20  # turnover far below the intraday floor
        signal = run(frame=thin, model=MODEL_NONE)
        self.assertEqual(signal.status, "RISK_VETO")
        self.assertIsNone(signal.plan)
        self.assertTrue(signal.vetoes)

    def test_risk_score_cannot_unblock_a_failing_rule(self):
        signal = run(features_fn=flat_features)
        self.assertEqual(signal.action, "HOLD")
        self.assertIsNone(signal.plan)


class ShapeTests(unittest.TestCase):
    def test_payload_is_json_safe_on_every_branch(self):
        cases = {
            "valid": dict(),
            "no_setup": dict(features_fn=flat_features),
            "abstain": dict(model=MODEL_ABSTAIN),
            "penalised": dict(model=MODEL_BEARISH, sent=sentiment(-1.0)),
            "low_conviction": dict(config=SignalConfig(min_conviction=99.9)),
            "vetoed": dict(account=AccountRisk(kill_switch=True)),
        }
        for name, kwargs in cases.items():
            with self.subTest(branch=name):
                payload = run(**kwargs).to_dict()
                json.dumps(payload, allow_nan=False)
                self.assertIn("conviction_units", payload)
                self.assertFalse(payload["execution_eligible"])

    def test_conviction_is_never_labelled_a_probability(self):
        payload = run().to_dict()
        self.assertEqual(payload["conviction_units"], "ranking_score")
        self.assertIsNone(payload["confidence"])
        self.assertIn("Not a probability", payload["conviction_note"])

    def test_scores_stay_in_bounds(self):
        for kwargs in (dict(), dict(model=MODEL_BEARISH, sent=sentiment(-1.0)),
                       dict(sent=sentiment(1.0))):
            signal = run(**kwargs)
            self.assertGreaterEqual(signal.conviction, 0.0)
            self.assertLessEqual(signal.conviction, 100.0)
            self.assertGreaterEqual(signal.risk.get("score", 0.0), 0.0)
            self.assertLessEqual(signal.risk.get("score", 0.0), 100.0)
            self.assertGreaterEqual(signal.setup["score"], 0.0)
            self.assertLessEqual(signal.setup["score"], 100.0)

    def test_short_is_not_offered_on_the_daily_horizon(self):
        """generate_signal must not attempt a short when the product is delivery."""
        frame = bars(count=420, freq="1D", trend=-0.002)
        now = frame.timestamp.iloc[-1] + pd.Timedelta(days=1)
        with patch("src.ml_engine.historical_neighbors.historical_outlook",
                   return_value=MODEL_NONE):
            signal = generate_signal(frame, "1d", ticker="TEST", now=now,
                                     costs=EquityCosts(intraday=False))
        self.assertNotEqual(signal.action, "SHORT_ENTRY")


class PenaltyUnitTests(unittest.TestCase):
    def test_aggregate_sentiment_weights(self):
        aggregate, count = aggregate_sentiment(
            {"sentiment_1h": 1.0, "sentiment_1d": 0.0, "sentiment_1w": -1.0,
             "articles_count": 3})
        self.assertAlmostEqual(aggregate, 0.50 * 1.0 + 0.35 * 0.0 + 0.15 * -1.0)
        self.assertEqual(count, 3)

    def test_aggregate_handles_partial_horizons(self):
        aggregate, _ = aggregate_sentiment({"sentiment_1d": -1.0, "articles_count": 1})
        self.assertAlmostEqual(aggregate, -1.0)

    def test_penalties_are_bounded_and_signed(self):
        config = SignalConfig()
        self.assertEqual(algo_penalty(MODEL_BULLISH, "LONG", config)[0], 0.0)
        self.assertAlmostEqual(algo_penalty(MODEL_BEARISH, "LONG", config)[0],
                               config.pi_algo_max)
        # The same bearish model is neutral-to-favourable for a short.
        self.assertEqual(algo_penalty(MODEL_BEARISH, "SHORT", config)[0], 0.0)
        self.assertAlmostEqual(sentiment_penalty(sentiment(-1.0), "LONG", config)[0],
                               config.pi_sent_max)
        self.assertEqual(sentiment_penalty(sentiment(-1.0), "SHORT", config)[0], 0.0)

    def test_sentiment_disabled_applies_no_penalty(self):
        config = SignalConfig(sentiment_enabled=False)
        self.assertEqual(sentiment_penalty(sentiment(-1.0), "LONG", config)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
