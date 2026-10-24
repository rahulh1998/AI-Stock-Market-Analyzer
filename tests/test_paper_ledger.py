"""Paper book: idempotency, reconciliation after reload, and barrier fill rules."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.paper_trading.ledger import PaperBook, make_event_id, wilson_interval
from src.quant_engine.research import EquityCosts
from src.signal_engine.composite import TradeSignal


def signal(ticker="TEST", direction="LONG", product="DELIVERY", grade="B",
           as_of="2026-10-20T15:30:00+05:30", quantity=10):
    stop, entry, target = (990.0, 1000.0, 1030.0) if direction == "LONG" else (1010.0, 1000.0, 970.0)
    return TradeSignal(
        ticker=ticker, interval="1d", horizon="daily",
        action="BUY" if direction == "LONG" else "SHORT_ENTRY", direction=direction,
        conviction=70.0, setup={"letter": grade, "score": 70.0}, risk={"score": 30.0},
        components={}, as_of_bar_close=as_of, status="CONDITIONAL_BUY",
        plan={"worst_modeled_entry_fill": entry, "stop": stop, "take_profit": target,
              "quantity": quantity, "product": product},
    )


def bars(rows):
    return pd.DataFrame(
        [{"timestamp": pd.Timestamp(ts, tz="Asia/Kolkata"), "open": o, "high": h,
          "low": l, "close": c, "volume": 10000} for ts, o, h, l, c in rows])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "book.db"
        self.book = PaperBook(self.path, starting_capital=200000.0)

    def tearDown(self):
        self.book.close()
        self.directory.cleanup()

    def test_event_id_is_stable(self):
        self.assertEqual(make_event_id("ENTRY", "TCS", "1d"), make_event_id("ENTRY", "TCS", "1d"))
        self.assertNotEqual(make_event_id("ENTRY", "TCS", "1d"), make_event_id("EXIT", "TCS", "1d"))

    def test_repeated_entry_is_idempotent(self):
        """A Streamlit rerun must not be able to double-fill."""
        for _ in range(5):
            self.book.open_from_signal(signal())
        self.assertEqual(len(self.book.events("ENTRY")), 1)
        self.assertEqual(len(self.book.open_positions()), 1)

    def test_repeated_signal_is_idempotent(self):
        for _ in range(4):
            self.book.record_signal(signal())
        self.assertEqual(len(self.book.events("SIGNAL")), 1)

    def test_cash_reconciles_after_reload(self):
        self.book.open_from_signal(signal())
        self.book.record_exit("TEST", "1d", "LONG", price=1030.0, quantity=10,
                              reason="target", ts_ist="2026-10-21T15:30:00+05:30")
        before = self.book.cash()
        self.book.close()
        reopened = PaperBook(self.path, starting_capital=200000.0)
        try:
            self.assertAlmostEqual(reopened.cash(), before, places=6)
            self.assertEqual(len(reopened.closed_trades()), 1)
            self.assertTrue(reopened.open_positions().empty)
        finally:
            reopened.close()
            self.book = PaperBook(self.path)

    def test_realised_pnl_is_net_of_both_fee_legs(self):
        costs = EquityCosts(intraday=False)
        self.book.open_from_signal(signal())
        self.book.record_exit("TEST", "1d", "LONG", price=1030.0, quantity=10,
                              reason="target", ts_ist="2026-10-21T15:30:00+05:30")
        trade = self.book.closed_trades().iloc[0]
        expected_fees = costs.fee(1000.0, 10, "BUY") + costs.fee(1030.0, 10, "SELL")
        self.assertAlmostEqual(trade.fees, expected_fees, places=6)
        self.assertAlmostEqual(trade.net_pnl, 300.0 - expected_fees, places=6)

    def test_short_pnl_sign_is_inverted(self):
        self.book.open_from_signal(signal(direction="SHORT", product="INTRADAY"))
        self.book.record_exit("TEST", "1d", "SHORT", price=970.0, quantity=10,
                              reason="target", ts_ist="2026-10-21T15:30:00+05:30",
                              product="INTRADAY")
        trade = self.book.closed_trades().iloc[0]
        self.assertGreater(trade.net_pnl, 0.0)  # bought back lower

    def test_delivery_short_is_rejected(self):
        """Third enforcement layer for the intraday-only short rule."""
        with self.assertRaises(ValueError):
            self.book.open_from_signal(signal(direction="SHORT", product="DELIVERY"))

    def test_entry_without_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            self.book.open_from_signal(TradeSignal(
                ticker="X", interval="1d", horizon="daily", action="HOLD", direction="NONE",
                conviction=0.0, setup={}, risk={}, components={}))


class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.book = PaperBook(Path(self.directory.name) / "book.db")

    def tearDown(self):
        self.book.close()
        self.directory.cleanup()

    def test_stop_wins_when_both_barriers_touch_in_one_bar(self):
        """Bar data cannot order the touches, so the stop must win."""
        self.book.open_from_signal(signal())
        candles = {"TEST": bars([("2026-10-21 09:15", 1000, 1040, 985, 1000)])}
        closures = self.book.evaluate_open_positions(candles)
        self.assertEqual(closures[0]["reason"], "stop")
        self.assertEqual(closures[0]["price"], 990.0)

    def test_gap_stop_fills_at_the_worse_open(self):
        self.book.open_from_signal(signal())
        candles = {"TEST": bars([("2026-10-21 09:15", 960, 975, 955, 970)])}
        closures = self.book.evaluate_open_positions(candles)
        self.assertEqual(closures[0]["reason"], "gap_stop")
        self.assertEqual(closures[0]["price"], 960.0)

    def test_target_fills_when_only_target_touched(self):
        self.book.open_from_signal(signal())
        candles = {"TEST": bars([("2026-10-21 09:15", 1005, 1035, 1001, 1030)])}
        closures = self.book.evaluate_open_positions(candles)
        self.assertEqual(closures[0]["reason"], "target")
        self.assertEqual(closures[0]["price"], 1030.0)

    def test_untouched_position_stays_open(self):
        self.book.open_from_signal(signal())
        candles = {"TEST": bars([("2026-10-21 09:15", 1001, 1005, 995, 1000)])}
        self.assertEqual(self.book.evaluate_open_positions(candles), [])
        self.assertEqual(len(self.book.open_positions()), 1)

    def test_bars_at_or_before_entry_are_ignored(self):
        """A position cannot be closed by the bar it was opened on."""
        self.book.open_from_signal(signal(as_of="2026-10-21T09:15:00+05:30"))
        candles = {"TEST": bars([("2026-10-21 09:15", 960, 975, 955, 970)])}
        self.assertEqual(self.book.evaluate_open_positions(candles), [])


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.book = PaperBook(Path(self.directory.name) / "book.db")

    def tearDown(self):
        self.book.close()
        self.directory.cleanup()

    def _trade(self, index, outcome):
        as_of = f"2026-01-{index % 28 + 1:02d}T15:30:00+05:30"
        self.book.open_from_signal(signal(ticker=f"T{index}", as_of=as_of))
        self.book.record_exit(f"T{index}", "1d", "LONG", price=1030.0 if outcome == "target" else 990.0,
                              quantity=10, reason=outcome,
                              ts_ist=f"2026-02-{index % 28 + 1:02d}T15:30:00+05:30")

    def test_confidence_abstains_below_the_sample_floor(self):
        for index in range(12):
            self._trade(index, "target")
        result = self.book.hit_rate("LONG", "1d", "B")
        self.assertFalse(result["available"])
        self.assertEqual(result["n"], 12)
        self.assertIn("12 of 30", result["reason"])
        self.assertNotIn("p", result)

    def test_confidence_reports_wilson_interval_once_measured(self):
        for index in range(40):
            self._trade(index, "target" if index % 2 == 0 else "stop")
        result = self.book.hit_rate("LONG", "1d", "B")
        self.assertTrue(result["available"])
        self.assertEqual(result["n"], 40)
        self.assertAlmostEqual(result["p"], 0.5, places=6)
        low, high = result["ci"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(result["event"], "target touched before stop")

    def test_empty_bucket_is_not_an_error(self):
        result = self.book.hit_rate("SHORT", "15m", "A")
        self.assertFalse(result["available"])
        self.assertEqual(result["n"], 0)

    def test_wilson_interval_bounds(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(10, 10)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 1.0)
        low, high = wilson_interval(0, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertGreater(high, 0.0)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.book = PaperBook(Path(self.directory.name) / "book.db")

    def tearDown(self):
        self.book.close()
        self.directory.cleanup()

    def test_empty_summary_is_explicit(self):
        summary = self.book.performance_summary()
        self.assertEqual(summary["trades"], 0)
        self.assertIn("note", summary)

    def test_summary_metrics(self):
        self.book.open_from_signal(signal(ticker="A", as_of="2026-01-05T15:30:00+05:30"))
        self.book.record_exit("A", "1d", "LONG", 1030.0, 10, "target",
                              "2026-01-06T15:30:00+05:30")
        self.book.open_from_signal(signal(ticker="B", as_of="2026-01-07T15:30:00+05:30"))
        self.book.record_exit("B", "1d", "LONG", 990.0, 10, "stop",
                              "2026-01-08T15:30:00+05:30")
        summary = self.book.performance_summary()
        self.assertEqual(summary["trades"], 2)
        self.assertAlmostEqual(summary["win_rate_pct"], 50.0)
        self.assertIsNotNone(summary["profit_factor"])
        self.assertEqual(len(self.book.equity_curve()), 2)


if __name__ == "__main__":
    unittest.main()
