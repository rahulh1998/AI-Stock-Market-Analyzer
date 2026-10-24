"""Corporate-action detection, bar cleaning and calendar derivation.

The central requirement is discrimination, not detection: a 1:5 split and a genuine -80%
crash look identical in the close column alone. Measured in the real history, nine moves
exceeded 20% and almost all were real — the Adani/Hindenburg crash, the IndusInd
derivatives disclosure, the 2024 election-day drop. Auto-"correcting" those would destroy
true data, so the detector keys off raw-vs-adjusted divergence instead.
"""
import unittest

import numpy as np
import pandas as pd

from src.data_pipeline.data_quality import (adjust_ohlcv, clean_frame, clean_panel,
                                            derive_trading_calendar, detect_corporate_actions,
                                            flag_fabricated_bars)


def frame(closes, adj=None, dates=None, volume=10000):
    count = len(closes)
    closes = np.asarray(closes, dtype=float)
    dates = dates if dates is not None else pd.bdate_range("2024-01-01", periods=count)
    return pd.DataFrame({
        "timestamp": dates, "open": closes * 0.998, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes,
        "adj_close": closes if adj is None else np.asarray(adj, dtype=float),
        "volume": [volume] * count})


class CorporateActionTests(unittest.TestCase):
    def test_split_is_detected(self):
        """Raw close halves; the adjusted series does not. That gap is the signal."""
        closes = [100.0, 101.0, 102.0, 51.0, 51.5]
        adjusted = [50.0, 50.5, 51.0, 51.0, 51.5]
        actions = detect_corporate_actions(frame(closes, adjusted), "TEST")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "split_or_bonus")
        self.assertAlmostEqual(actions[0].factor, 2.0, places=3)

    def test_a_real_crash_is_not_flagged(self):
        """Close and adj_close fall together, so nothing is adjusted away."""
        closes = [100.0, 101.0, 102.0, 28.0, 29.0]
        actions = detect_corporate_actions(frame(closes, closes), "CRASH")
        self.assertEqual(actions, [])

    def test_a_real_rally_is_not_flagged(self):
        closes = [100.0, 101.0, 102.0, 160.0, 162.0]
        self.assertEqual(detect_corporate_actions(frame(closes, closes), "RALLY"), [])

    def test_dividend_is_classified_separately_from_a_split(self):
        closes = [100.0, 100.0, 98.0, 98.5]
        adjusted = [97.0, 97.0, 98.0, 98.5]   # ~3% divergence
        actions = detect_corporate_actions(frame(closes, adjusted), "DIV")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "dividend")

    def test_no_adj_close_column_yields_nothing(self):
        plain = frame([100.0, 50.0, 51.0]).drop(columns=["adj_close"])
        self.assertEqual(detect_corporate_actions(plain), [])

    def test_identical_adj_close_yields_nothing(self):
        closes = [100.0, 50.0, 51.0]
        self.assertEqual(detect_corporate_actions(frame(closes, closes)), [])


class AdjustmentTests(unittest.TestCase):
    def test_ohlc_is_scaled_and_volume_inverted(self):
        closes = [100.0, 102.0]
        adjusted = [50.0, 51.0]
        result = adjust_ohlcv(frame(closes, adjusted))
        self.assertAlmostEqual(result.close.iloc[0], 50.0)
        self.assertAlmostEqual(result.open.iloc[0], 99.8 * 0.5)
        self.assertAlmostEqual(result.volume.iloc[0], 20000)
        self.assertAlmostEqual(result.adjustment_factor.iloc[0], 0.5)

    def test_adjustment_preserves_notional_turnover(self):
        closes = [100.0, 102.0]
        result = adjust_ohlcv(frame(closes, [50.0, 51.0]))
        self.assertAlmostEqual(result.close.iloc[0] * result.volume.iloc[0], 100.0 * 10000)

    def test_adjustment_preserves_returns(self):
        closes = np.array([100.0, 104.0, 99.0, 50.0, 52.0])
        adjusted = np.array([50.0, 52.0, 49.5, 50.0, 52.0])
        result = adjust_ohlcv(frame(closes, adjusted))
        expected = pd.Series(adjusted).pct_change().dropna().to_numpy()
        actual = result.close.pct_change().dropna().to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_unadjustable_frame_is_returned_unchanged(self):
        closes = [100.0, 101.0]
        result = adjust_ohlcv(frame(closes, closes))
        np.testing.assert_allclose(result.close.to_numpy(), closes)
        np.testing.assert_allclose(result.adjustment_factor.to_numpy(), [1.0, 1.0])

    def test_ohlc_ordering_survives_adjustment(self):
        result = adjust_ohlcv(frame([100.0, 102.0, 51.0], [50.0, 51.0, 51.0]))
        self.assertTrue((result.high >= result.low).all())
        self.assertTrue((result.high >= result.close).all())
        self.assertTrue((result.low <= result.close).all())


class FabricatedBarTests(unittest.TestCase):
    def test_flat_zero_volume_bar_is_fabricated(self):
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2024-01-01", periods=3),
            "open": [100.0, 100.0, 101.0], "high": [101.0, 100.0, 102.0],
            "low": [99.0, 100.0, 100.0], "close": [100.5, 100.0, 101.5],
            "volume": [1000, 0, 1200]})
        np.testing.assert_array_equal(flag_fabricated_bars(df).to_numpy(),
                                      [False, True, False])

    def test_flat_bar_with_real_volume_is_kept(self):
        """A thin but genuine session can print a zero range; volume decides."""
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2024-01-01", periods=1),
            "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
            "volume": [5000]})
        self.assertFalse(bool(flag_fabricated_bars(df).iloc[0]))

    def test_zero_volume_with_a_real_range_is_kept(self):
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2024-01-01", periods=1),
            "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
            "volume": [0]})
        self.assertFalse(bool(flag_fabricated_bars(df).iloc[0]))


class CalendarTests(unittest.TestCase):
    def panel_with_holiday(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                "2024-01-04", "2024-01-05"])
        panel = {}
        for name in ("A", "B", "C", "D"):
            df = frame([100.0] * 5, dates=dates)
            df.loc[2, ["open", "high", "low", "close"]] = 100.0   # flat
            df.loc[2, "volume"] = 0                               # and no volume
            panel[name] = df
        return panel

    def test_shared_fabricated_date_is_excluded(self):
        calendar = derive_trading_calendar(self.panel_with_holiday())
        self.assertNotIn(pd.Timestamp("2024-01-03").date(), calendar)
        self.assertIn(pd.Timestamp("2024-01-02").date(), calendar)

    def test_weekends_are_never_trading_days(self):
        dates = pd.to_datetime(["2024-01-06", "2024-01-07"])   # Sat, Sun
        calendar = derive_trading_calendar({"A": frame([100.0, 100.0], dates=dates)})
        self.assertEqual(calendar, set())

    def test_a_single_ticker_halt_does_not_remove_the_day(self):
        """One name not trading is a halt, not a holiday; quorum protects the date."""
        panel = self.panel_with_holiday()
        only_one = {k: v.copy() for k, v in panel.items()}
        for name in ("B", "C", "D"):
            only_one[name].loc[2, "volume"] = 9000
            only_one[name].loc[2, "high"] = 101.0
        calendar = derive_trading_calendar(only_one)
        self.assertIn(pd.Timestamp("2024-01-03").date(), calendar)


class CleaningTests(unittest.TestCase):
    def test_clean_frame_reports_what_it_removed(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-03"])
        df = frame([100.0] * 4, dates=dates)
        df.loc[1, ["open", "high", "low", "close"]] = 100.0
        df.loc[1, "volume"] = 0
        cleaned, report = clean_frame(df, calendar=None)
        self.assertEqual(report["dropped_fabricated"], 1)
        self.assertEqual(report["dropped_weekend"], 1)   # 2024-01-06 is a Saturday
        self.assertEqual(report["output_rows"], 2)

    def test_clean_frame_sorts_by_timestamp(self):
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        cleaned, _ = clean_frame(frame([102.0, 100.0, 101.0], dates=dates))
        self.assertTrue(cleaned.timestamp.is_monotonic_increasing)

    def test_clean_panel_applies_one_shared_calendar(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        panel = {}
        for name in ("A", "B"):
            df = frame([100.0] * 3, dates=dates)
            df.loc[1, ["open", "high", "low", "close"]] = 100.0
            df.loc[1, "volume"] = 0
            panel[name] = df
        cleaned, calendar, summary, _ = clean_panel(panel)
        self.assertEqual(summary["dropped_fabricated"], 2)
        self.assertEqual(len(calendar), 2)
        for df in cleaned.values():
            self.assertEqual(len(df), 2)

    def test_empty_frame_is_handled(self):
        cleaned, report = clean_frame(pd.DataFrame())
        self.assertEqual(report["output_rows"], 0)


if __name__ == "__main__":
    unittest.main()
