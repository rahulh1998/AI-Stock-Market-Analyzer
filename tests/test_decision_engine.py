import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.quant_engine.decision_engine import (
    AccountRisk, PATTERNS, SignalPolicy, analyze_market, build_long_plan, pattern_features, tick_round, wilder_mean,
)
from src.quant_engine.research import EquityCosts
from src.ml_engine.historical_neighbors import FEATURES, _neighbors, historical_outlook, outcome_labels


def sample(count=70):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.03, 0.3, count))
    return pd.DataFrame({"timestamp": pd.date_range("2026-09-07 09:15", periods=count, freq="5min", tz="Asia/Kolkata"),
                         "open": close - 0.1, "high": close + 0.4, "low": close - 0.4,
                         "close": close, "volume": rng.integers(1000, 3000, count)})


class PatternTests(unittest.TestCase):
    def test_prefix_causality_including_pivots(self):
        frame = sample(100)
        full = pattern_features(frame)
        for length in (20, 45, 70):
            pd.testing.assert_frame_equal(pattern_features(frame.iloc[:length]), full.iloc[:length])

    def test_wilder_seed_and_recursion(self):
        values = pd.Series([np.nan, 1., 2., 3., 7.])
        actual = wilder_mean(values, 3)
        self.assertTrue(actual.iloc[:3].isna().all())
        self.assertAlmostEqual(actual.iloc[3], 2)
        self.assertAlmostEqual(actual.iloc[4], 11 / 3)

    def test_monotonic_rsi_adx_and_flat_rsi(self):
        frame = sample()
        frame["close"] = np.arange(len(frame)) + 100.0
        frame["open"], frame["high"], frame["low"] = frame.close - 0.1, frame.close + 1, frame.close - 1
        features = pattern_features(frame)
        self.assertEqual(features.rsi.iloc[-1], 100)
        self.assertAlmostEqual(features.adx.iloc[-1], 100)
        self.assertTrue(features.adx.iloc[:27].isna().all())
        frame[["open", "close"]] = 100.0
        frame["high"], frame["low"] = 101.0, 99.0
        features = pattern_features(frame)
        self.assertEqual(features.rsi.iloc[-1], 50)
        self.assertEqual(features.adx.iloc[-1], 0)

    def test_engulfing_requires_directional_context_and_body(self):
        frame = sample(10)
        frame.loc[5:9, "close"] = [108., 106., 104., 102., 105.]
        frame.loc[5:9, "open"] = [109., 107., 105., 104., 101.]
        frame["high"] = frame[["open", "close"]].max(axis=1) + 0.1
        frame["low"] = frame[["open", "close"]].min(axis=1) - 0.1
        self.assertTrue(pattern_features(frame).bullish_engulfing.iloc[-1])
        frame.loc[9, "timestamp"] += pd.Timedelta(days=1)
        self.assertFalse(pattern_features(frame).bullish_engulfing.iloc[-1])

    def test_flat_candle_is_not_doji_or_hammer(self):
        frame = sample()
        frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = 100.
        features = pattern_features(frame)
        self.assertFalse(features.doji.iloc[-1])
        self.assertFalse(features.hammer.iloc[-1])

    def test_breakout_uses_previous_highs(self):
        frame = sample()
        old_high = frame.high.iloc[-21:-1].max()
        frame.loc[len(frame) - 1, ["open", "low", "close", "high"]] = [old_high, old_high - 1, old_high + 1, old_high + 2]
        self.assertTrue(pattern_features(frame).range_breakout.iloc[-1])

    def test_swing_is_reported_two_bars_later(self):
        frame = sample(10)
        frame[["open", "close"]] = 100.
        frame["high"], frame["low"] = 101., 99.
        frame.loc[5, "low"] = 90.
        features = pattern_features(frame)
        self.assertTrue(pd.isna(features.confirmed_swing_low.iloc[5]))
        self.assertTrue(pd.isna(features.confirmed_swing_low.iloc[6]))
        self.assertEqual(features.confirmed_swing_low.iloc[7], 90)


class PlanRiskTests(unittest.TestCase):
    def row(self):
        return pd.Series({"high": 100.5, "low": 99.5, "close": 100., "wilder_atr": 1.,
                          "resistance_20": 98., "resistance_60": 99.})

    def test_cash_risk_tick_and_costs(self):
        account, policy, costs = AccountRisk(), SignalPolicy(), EquityCosts()
        plan, reasons = build_long_plan(self.row(), account, policy, costs)
        self.assertEqual(reasons, [])
        self.assertLessEqual(plan["modeled_stop_loss"], account.equity * policy.risk_pct / 100)
        self.assertLessEqual(plan["cash_required"], account.equity * policy.allocation_pct / 100)
        self.assertGreaterEqual(plan["net_reward_risk"], policy.min_net_rr)
        self.assertLess(plan["stop"], plan["buy_trigger"])
        self.assertLess(plan["entry_ceiling"], plan["take_profit"])
        for key in ("buy_trigger", "entry_ceiling", "stop", "take_profit"):
            self.assertAlmostEqual(plan[key] / policy.tick_size, round(plan[key] / policy.tick_size))
        self.assertEqual(tick_round(100.03, 0.05, True), 100.05)
        self.assertEqual(tick_round(100.03, 0.05), 100.0)

    def test_locks(self):
        for account in (replace(AccountRisk(), kill_switch=True),
                        replace(AccountRisk(), cash=0),
                        replace(AccountRisk(), equity=190000, cash=190000),
                        replace(AccountRisk(), peak_equity=250000),
                        replace(AccountRisk(), consecutive_losses=3),
                        replace(AccountRisk(), open_risk=4000),
                        replace(AccountRisk(), gross_exposure=160000)):
            with self.subTest(account=account):
                plan, reasons = build_long_plan(self.row(), account, SignalPolicy(), EquityCosts())
                self.assertIsNone(plan)
                self.assertTrue(reasons)

    def test_remaining_portfolio_capacity_is_respected(self):
        account = replace(AccountRisk(), open_risk=3900, gross_exposure=159000)
        plan, reasons = build_long_plan(self.row(), account, SignalPolicy(), EquityCosts())
        self.assertFalse(reasons)
        self.assertLessEqual(plan["modeled_stop_loss"], 100)
        self.assertLessEqual(plan["cash_required"], 1000)

    def test_nearby_resistance_veto(self):
        row = self.row()
        row["resistance_60"] = 101.0
        plan, reasons = build_long_plan(row, AccountRisk(), SignalPolicy(), EquityCosts())
        self.assertIsNone(plan)
        self.assertIn("reward-to-risk", reasons[0])

    def test_invalid_account_and_policy(self):
        for kwargs in ({"equity": float("nan")}, {"cash": 300000}, {"open_risk": -1}):
            with self.assertRaises(ValueError):
                AccountRisk(**kwargs)
        with self.assertRaises(ValueError):
            SignalPolicy(tick_size=0)


class DecisionTests(unittest.TestCase):
    def test_stale_future_and_malformed_data(self):
        frame = sample()
        report = analyze_market(frame, now=frame.timestamp.iloc[-1] + pd.Timedelta(days=1))
        self.assertEqual(report["status"], "HISTORICAL_ONLY")
        self.assertIsNone(report["plan"])
        self.assertFalse(report["execution_eligible"])
        self.assertEqual(analyze_market(frame, now=frame.timestamp.iloc[0])["status"], "INVALID_DATA")
        frame.loc[0, "close"] = np.nan
        self.assertEqual(analyze_market(frame)["status"], "INVALID_DATA")

    def test_incomplete_candle_excluded_and_report_serializable(self):
        frame = sample()
        now = frame.timestamp.iloc[-1] + pd.Timedelta(minutes=2)
        report = analyze_market(frame, now=now)
        self.assertEqual(report["excluded_incomplete_bars"], 1)
        self.assertEqual(report["as_of_bar_close"], frame.timestamp.iloc[-1].isoformat())
        json.dumps(report, allow_nan=False)

    def test_missing_session_bar_blocks_analysis(self):
        frame = sample().drop(index=50)
        report = analyze_market(frame, now=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
        self.assertTrue(any("irregular" in item for item in report["rejections"]))
        self.assertIsNone(report["plan"])

    def controlled_features(self, frame):
        features = pattern_features(frame)
        index = len(features) - 1
        for name in PATTERNS:
            features.loc[index, name] = False
        features.loc[index, "range_breakout"] = True
        for name, value in {"ema_fast": 101., "ema_slow": 100., "adx": 30., "rsi": 60.,
                            "relative_volume": 2., "vwap": 99., "high": 100.5, "low": 99.5,
                            "close": 100., "wilder_atr": 1., "resistance_20": 98., "resistance_60": 99.}.items():
            features.loc[index, name] = value
        features["confirmed_swing_high"] = np.nan
        return features

    def test_conditional_plan_and_kill_switch(self):
        frame = sample()
        now = frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5)
        with patch("src.quant_engine.decision_engine.pattern_features", return_value=self.controlled_features(frame)):
            report = analyze_market(frame, now=now)
            self.assertEqual(report["status"], "CONDITIONAL_BUY")
            self.assertIsNotNone(report["plan"])
            self.assertFalse(report["execution_eligible"])
            report = analyze_market(frame, now=now, account=replace(AccountRisk(), kill_switch=True))
            self.assertEqual(report["status"], "WAIT")
            self.assertIsNone(report["plan"])

    def test_conflicting_patterns_veto(self):
        frame = sample()
        features = self.controlled_features(frame)
        features.loc[len(features) - 1, "shooting_star"] = True
        with patch("src.quant_engine.decision_engine.pattern_features", return_value=features):
            report = analyze_market(frame, now=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
        self.assertIsNone(report["plan"])
        self.assertEqual(report["bias"], "CONFLICTING")

    def test_bearish_setup_is_exit_review_not_a_short(self):
        frame = sample()
        features = self.controlled_features(frame)
        features.loc[len(features) - 1, "range_breakout"] = False
        features.loc[len(features) - 1, "range_breakdown"] = True
        with patch("src.quant_engine.decision_engine.pattern_features", return_value=features):
            report = analyze_market(frame, now=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
        self.assertEqual(report["status"], "EXIT_REVIEW")
        self.assertIsNone(report["plan"])
        self.assertFalse(report["execution_eligible"])

    def test_horizon_cost_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            analyze_market(sample(), costs=EquityCosts(intraday=False))


class HistoricalModelTests(unittest.TestCase):
    def test_matured_labels_only_and_non_overlapping_neighbors(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(500, len(FEATURES))), rng.normal(size=(500, 3))
        indices = _neighbors(x, y, 400, 3, 30)
        self.assertTrue((indices + 3 < 400).all())
        self.assertTrue((np.diff(np.sort(indices)) > 3).all())
        y[400:] = 999
        np.testing.assert_array_equal(indices, _neighbors(x, y, 400, 3, 30))

    def test_label_uses_next_open_and_does_not_cross_sessions(self):
        frame = sample(10)
        frame.loc[5:, "timestamp"] += pd.Timedelta(days=1)
        labels = outcome_labels(frame, 3, True)
        self.assertTrue(np.isnan(labels[0]).all())  # No observed interval at first bar.
        self.assertAlmostEqual(labels[1, 0], frame.close.iloc[4] / frame.open.iloc[2] - 1)
        self.assertTrue(np.isnan(labels[2]).all())
        self.assertTrue(np.isnan(labels[-3:]).all())

    def test_future_frequency_change_does_not_rewrite_matured_labels(self):
        frame = sample(60)
        prefix = outcome_labels(frame.iloc[:10], 3, True)
        start = frame.timestamp.iloc[9]
        frame.loc[10:, "timestamp"] = pd.date_range(start + pd.Timedelta(minutes=15), periods=50,
                                                    freq="15min", tz="Asia/Kolkata")
        full = outcome_labels(frame, 3, True)
        np.testing.assert_allclose(prefix[:7], full[:7], equal_nan=True)

    def test_model_abstains_without_evaluation_and_reports_baseline(self):
        self.assertFalse(historical_outlook(pattern_features(sample()))["available"])
        frame = sample(600)
        frame["timestamp"] = pd.date_range("2023-01-01", periods=600, freq="D", tz="Asia/Kolkata")
        result = historical_outlook(pattern_features(frame, False), horizon=5, intraday=False)
        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["evaluation_windows"], 10)
        self.assertIn("flat_baseline_mae_pct", result)
        self.assertLess(pd.Timestamp(result["latest_neighbor_label_end"]), frame.timestamp.iloc[-1])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()