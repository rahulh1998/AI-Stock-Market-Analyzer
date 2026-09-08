import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.quant_engine.research import EquityCosts, replay, research_features, size_long, validate_bars
from src.quant_engine.risk_math import RiskEngine


def bars(count=45):
    close = np.arange(count, dtype=float) + 100
    return pd.DataFrame({"timestamp": pd.date_range("2026-08-10 09:15", periods=count, freq="5min", tz="Asia/Kolkata"),
                         "open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": np.full(count, 1000)})


class DataTests(unittest.TestCase):
    def test_validation_rejects_invalid_data(self):
        for column, value in [("close", np.nan), ("open", 0), ("volume", -1), ("high", 1)]:
            frame = bars()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_bars(frame)

    def test_order_and_duplicates(self):
        with self.assertRaises(ValueError):
            validate_bars(bars().iloc[::-1])
        frame = bars()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        with self.assertRaises(ValueError):
            validate_bars(frame)

    def test_features_are_causal(self):
        frame = bars()
        prefix = research_features(frame.iloc[:30])
        full = research_features(frame)
        pd.testing.assert_frame_equal(prefix, full.iloc[:30])
        self.assertTrue(full.atr.iloc[:13].isna().all())
        self.assertTrue(full.sma_200.isna().all())

    def test_session_vwap_resets_and_zero_volume_is_unknown(self):
        frame = bars()
        frame.loc[30:, "timestamp"] += pd.Timedelta(days=1)
        feature = research_features(frame)
        self.assertAlmostEqual(feature.vwap.iloc[30], frame.close.iloc[30])
        frame.loc[30, "volume"] = 0
        self.assertTrue(pd.isna(research_features(frame).vwap.iloc[30]))


class CostRiskTests(unittest.TestCase):
    def test_intraday_fees(self):
        costs = EquityCosts()
        buy = costs.breakdown(1000, 100, "BUY")
        sell = costs.breakdown(1000, 100, "SELL")
        self.assertEqual(buy["brokerage"], 20)
        self.assertAlmostEqual(buy["stamp"], 3)
        self.assertEqual(buy["stt"], 0)
        self.assertAlmostEqual(sell["stt"], 25)
        self.assertEqual(sell["stamp"], 0)
        self.assertAlmostEqual(buy["gst"], (20 + 3.07 + 0.1) * 0.18)

    def test_delivery_and_zero_order(self):
        costs = EquityCosts(intraday=False)
        self.assertEqual(costs.breakdown(1000, 100, "BUY")["stt"], 100)
        self.assertEqual(costs.breakdown(1000, 100, "SELL")["dp"], 15.34)
        self.assertEqual(costs.fee(1000, 0, "SELL"), 0)
        with self.assertRaises(ValueError):
            EquityCosts(slippage_bps=float("nan"))

    def test_sizing_respects_cash_and_cost_risk(self):
        costs = EquityCosts()
        q = size_long(1000, 980, 200000, 0.5, costs)
        loss = q * (1000 - 980 * 0.9995) + costs.fee(1000, q, "BUY") + costs.fee(980 * 0.9995, q, "SELL")
        self.assertGreater(q, 0)
        self.assertLessEqual(loss, 1000)
        self.assertLessEqual(1000 * q + costs.fee(1000, q, "BUY"), 50000)
        self.assertEqual(size_long(1000, 1000, 200000, 1, costs), 0)

    def test_legacy_sizing_and_veto(self):
        engine = RiskEngine(default_capital=10000)
        self.assertLessEqual(engine.calculate_position_sizing(1000, 999)["total_exposure"], 10000)
        self.assertEqual(engine.calculate_position_sizing(1000, 0)["quantity"], 0)
        self.assertEqual(engine.calculate_levels(100, float("nan"), "BUY")["stop_loss"], 0)
        self.assertFalse(engine.verify_signal_guardrails({"action": "BUY", "confidence_score": 99},
                         {"close": 100, "SMA_200": 90, "RSI_14": 50, "research_only": True})["is_approved"])
        self.assertFalse(engine.verify_signal_guardrails({"action": "BUY", "confidence_score": 99},
                         {"close": 100, "SMA_200": float("nan"), "RSI_14": 50})["is_approved"])


class ReplayTests(unittest.TestCase):
    def controlled(self):
        frame = research_features(bars(3))
        frame["open"] = 100.0
        frame["high"] = 101.0
        frame["low"] = 99.0
        frame["close"] = 100.0
        frame["ema_fast"], frame["ema_slow"] = 2.0, 1.0
        frame["vwap"], frame["relative_volume"], frame["atr"] = 90.0, 2.0, 2.0
        return frame

    def run_frame(self, frame, **kwargs):
        with patch("src.quant_engine.research.research_features", return_value=frame):
            return replay(bars(3), **kwargs)

    def test_next_bar_fill_and_ledger_reconciliation(self):
        frame = self.controlled()
        result = self.run_frame(frame)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade.entry_time, frame.timestamp.iloc[1])
        self.assertGreater(trade.entry, 100)
        self.assertLess(trade.exit, 100)
        self.assertAlmostEqual(result["equity"].equity.iloc[-1], 200000 + result["trades"].net_pnl.sum())

    def test_same_bar_stop_wins(self):
        frame = self.controlled()
        frame.loc[1, ["high", "low"]] = [120, 80]
        result = self.run_frame(frame)
        self.assertEqual(result["trades"].iloc[0].reason, "stop")
        self.assertLess(result["trades"].iloc[0].net_pnl, 0)

    def test_open_entry_does_not_peek_at_current_total_volume(self):
        frame = self.controlled()
        frame.loc[1, "volume"] = 0
        result = self.run_frame(frame)
        self.assertEqual(result["trades"].iloc[0].entry_time, frame.timestamp.iloc[1])

    def test_gap_stop(self):
        frame = self.controlled()
        frame.loc[2, ["open", "low", "close"]] = [90, 89, 90]
        trade = self.run_frame(frame)["trades"].iloc[0]
        self.assertEqual(trade.reason, "gap_stop")
        self.assertLess(trade.exit, 90)

    def test_insufficient_daily_history_has_no_trades(self):
        result = replay(bars(), intraday=False)
        self.assertEqual(result["metrics"]["trades"], 0)
        self.assertIsNone(result["metrics"]["win_rate_pct"])

    def test_product_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            replay(bars(), costs=EquityCosts(intraday=False))

    def test_daily_loss_lock_blocks_reentry(self):
        frame = self.controlled()
        frame.loc[1, ["high", "low"]] = [120, 80]
        result = self.run_frame(frame, risk_pct=3, allocation_pct=100, daily_loss_pct=0.5)
        self.assertEqual(len(result["trades"]), 1)


if __name__ == "__main__":
    unittest.main()