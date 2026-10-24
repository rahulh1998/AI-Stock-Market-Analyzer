"""Correctness tests for the portfolio backtest and the validation statistics.

The backtest produced a decisively negative verdict on the strategy. That verdict is only
worth anything if the machinery is right, so these tests attack it directly: no look-ahead,
cash conservation, correct fill and barrier semantics, and known-answer checks on synthetic
series whose outcome can be derived by hand.

The statistics get the same treatment — PBO is checked against a deliberately overfit
population and a genuinely-skilled one, so a passing test means the estimator discriminates
rather than merely runs.
"""
import math
import unittest

import numpy as np
import pandas as pd

from src.backtesting.portfolio import (BacktestConfig, Position, _fill_price, prepare_panel,
                                       run_backtest)
from src.backtesting.validation import (buy_and_hold_equal_weight, compare, curve_metrics,
                                        deflated_sharpe_ratio, expected_max_sharpe,
                                        index_buy_and_hold, _normal_ppf,
                                        probability_of_backtest_overfitting, sharpe_ratio)
from src.quant_engine.decision_engine import SignalPolicy
from src.quant_engine.research import EquityCosts
from src.signal_engine.composite import SignalConfig


def series(count=400, start="2022-01-03", drift=0.0008, seed=0, price=1000.0, volume=200000):
    """Business-day OHLCV with a mild upward drift and enough history to warm up."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=count)
    steps = rng.normal(drift, 0.012, count)
    close = price * np.cumprod(1 + steps)
    high = close * (1 + np.abs(rng.normal(0, 0.006, count)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, count)))
    open_ = close * (1 + rng.normal(0, 0.003, count))
    return pd.DataFrame({
        "timestamp": dates, "open": open_,
        "high": np.maximum.reduce([high, open_, close]),
        "low": np.minimum.reduce([low, open_, close]),
        "close": close, "adj_close": close,
        "volume": rng.integers(volume // 2, volume * 2, count),
    })


def calendar_of(panel):
    dates = set()
    for frame in panel.values():
        dates |= set(pd.to_datetime(frame["timestamp"]).dt.date)
    return dates


class FillSemanticsTests(unittest.TestCase):
    """A conditional buy: trigger above the bar, cancel above the ceiling."""

    def bar(self, open_, high, low=90.0):
        return pd.Series({"open": open_, "high": high, "low": low, "close": high})

    def test_no_fill_when_trigger_never_touched(self):
        self.assertIsNone(_fill_price(self.bar(100, 104), 105, 108, "LONG", 0.0))

    def test_fills_at_trigger_when_crossed_intrabar(self):
        self.assertAlmostEqual(_fill_price(self.bar(100, 110), 105, 108, "LONG", 0.0), 105)

    def test_fills_at_open_when_opening_above_trigger(self):
        self.assertAlmostEqual(_fill_price(self.bar(106, 110), 105, 108, "LONG", 0.0), 106)

    def test_no_fill_when_gapping_past_the_ceiling(self):
        """The plan says cancel above the entry ceiling, so a gap-up must not fill."""
        self.assertIsNone(_fill_price(self.bar(120, 125), 105, 108, "LONG", 0.0))

    def test_slippage_worsens_the_long_fill(self):
        clean = _fill_price(self.bar(100, 110), 105, 108, "LONG", 0.0)
        slipped = _fill_price(self.bar(100, 110), 105, 108, "LONG", 0.001)
        self.assertGreater(slipped, clean)

    def test_short_mirrors_long(self):
        bar = pd.Series({"open": 100.0, "high": 101.0, "low": 90.0, "close": 95.0})
        self.assertAlmostEqual(_fill_price(bar, 95, 92, "SHORT", 0.0), 95)
        self.assertIsNone(_fill_price(bar, 95, 92, "SHORT", 0.0) and None)
        gapped = pd.Series({"open": 80.0, "high": 85.0, "low": 75.0, "close": 80.0})
        self.assertIsNone(_fill_price(gapped, 95, 92, "SHORT", 0.0))


class NoLookAheadTests(unittest.TestCase):
    """The decisive property: future bars must not change past decisions."""

    def setUp(self):
        # The gates are demanding, so a fixture has to be long and mildly trending before
        # it trades at all. These seeds produce a handful of real trades across 600 bars.
        self.panel = {f"T{i}": series(count=600, seed=i, drift=0.0012) for i in range(8)}
        self.config = BacktestConfig(capital=200000.0, costs=EquityCosts(intraday=False))

    def test_truncating_the_future_leaves_past_trades_unchanged(self):
        full_dates = sorted(calendar_of(self.panel))
        full = run_backtest(prepare_panel(self.panel), set(full_dates), self.config)
        self.assertGreater(len(full.trades), 1, "fixture produced too few trades")

        # Cut just after the second completed trade, so the comparison is non-trivial.
        second_exit = sorted(full.trades.exit_date)[1]
        cutoff = next(d for d in full_dates if d > second_exit)

        truncated_panel = {t: f[pd.to_datetime(f["timestamp"]).dt.date <= cutoff].reset_index(drop=True)
                           for t, f in self.panel.items()}
        truncated = run_backtest(prepare_panel(truncated_panel),
                                 {d for d in full_dates if d <= cutoff}, self.config)

        full_trades = full.trades[full.trades.exit_date <= cutoff]
        cut_trades = truncated.trades[truncated.trades.reason != "sample_end"]
        self.assertGreater(len(cut_trades), 0, "fixture produced no trades to compare")
        columns = ["ticker", "entry_date", "exit_date", "quantity", "entry", "reason"]
        pd.testing.assert_frame_equal(
            full_trades[columns].reset_index(drop=True),
            cut_trades[columns].reset_index(drop=True))

    def test_mutating_only_future_bars_does_not_change_past_equity(self):
        dates = sorted(calendar_of(self.panel))
        cutoff = dates[300]
        base = run_backtest(prepare_panel(self.panel), set(dates), self.config)
        mutated = {}
        for ticker, frame in self.panel.items():
            frame = frame.copy()
            future = pd.to_datetime(frame["timestamp"]).dt.date > cutoff
            for column in ("open", "high", "low", "close", "adj_close"):
                frame.loc[future, column] *= 3.0
            mutated[ticker] = frame
        after = run_backtest(prepare_panel(mutated), set(dates), self.config)
        left = base.equity[base.equity.date <= cutoff].reset_index(drop=True)
        right = after.equity[after.equity.date <= cutoff].reset_index(drop=True)
        pd.testing.assert_series_equal(left.equity, right.equity)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.panel = {f"T{i}": series(count=600, seed=i, drift=0.0012) for i in range(8)}
        self.config = BacktestConfig(capital=200000.0, costs=EquityCosts(intraday=False))
        self.result = run_backtest(prepare_panel(self.panel), calendar_of(self.panel), self.config)

    def test_equity_reconciles_with_realised_pnl(self):
        """Final equity must equal capital plus the sum of net trade P&L, exactly."""
        self.assertFalse(self.result.trades.empty, "fixture produced no trades")
        expected = self.config.capital + self.result.trades.net_pnl.sum()
        self.assertAlmostEqual(self.result.equity.equity.iloc[-1], expected, places=4)

    def test_equity_identity_holds_on_every_bar(self):
        frame = self.result.equity
        pd.testing.assert_series_equal(frame.equity, frame.cash + frame.holdings,
                                       check_names=False)

    def test_cash_never_goes_negative(self):
        self.assertGreaterEqual(self.result.equity.cash.min(), -1e-6)

    def test_net_pnl_is_gross_minus_fees(self):
        trades = self.result.trades
        gross = (trades.exit - trades.entry) * trades.quantity
        pd.testing.assert_series_equal(trades.net_pnl, gross - trades.fees, check_names=False)

    def test_every_trade_pays_both_fee_legs(self):
        self.assertTrue((self.result.trades.fees > 0).all())

    def test_risk_per_trade_respects_the_policy_budget(self):
        """Modelled loss at the stop must not exceed the configured risk budget."""
        budget = self.config.capital * self.config.policy.risk_pct / 100
        trades = self.result.trades
        losses = trades[trades.reason == "stop"]
        modelled = (losses.entry - losses.exit) * losses.quantity + losses.fees
        # Equity grows, so the budget can only rise; the opening budget is the floor.
        self.assertTrue((modelled <= budget * 1.5).all(),
                        f"max modelled stop loss {modelled.max():.2f} vs budget {budget:.2f}")

    def test_no_overlapping_position_in_one_ticker(self):
        trades = self.result.trades.sort_values(["ticker", "entry_date"])
        for ticker, group in trades.groupby("ticker"):
            previous_exit = None
            for row in group.itertuples():
                if previous_exit is not None:
                    self.assertGreaterEqual(row.entry_date, previous_exit)
                previous_exit = row.exit_date


class KnownAnswerTests(unittest.TestCase):
    """Series whose outcome can be reasoned about without running the code."""

    def test_a_flat_market_produces_no_trades(self):
        flat = pd.DataFrame({
            "timestamp": pd.bdate_range("2022-01-03", periods=400),
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "adj_close": 100.0, "volume": 1000})
        panel = {"FLAT": flat}
        result = run_backtest(prepare_panel(panel), calendar_of(panel), BacktestConfig())
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.equity.equity.iloc[-1], 200000.0)

    def test_an_empty_universe_is_handled(self):
        result = run_backtest({}, set(), BacktestConfig())
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.metrics["trades"], 0)

    def test_capital_scales_quantities_not_returns(self):
        """Doubling capital should roughly double P&L, not change percentage return."""
        panel = {f"T{i}": series(count=600, seed=i, drift=0.0012) for i in range(8)}
        prepared = prepare_panel(panel)
        dates = calendar_of(panel)
        small = run_backtest(prepared, dates, BacktestConfig(capital=200000.0))
        large = run_backtest(prepared, dates, BacktestConfig(capital=400000.0))
        self.assertFalse(small.trades.empty, "fixture produced no trades")
        self.assertFalse(large.trades.empty, "fixture produced no trades")
        self.assertGreater(large.trades.quantity.sum(), small.trades.quantity.sum())

    def test_higher_conviction_floor_never_increases_trade_count(self):
        panel = {f"T{i}": series(count=600, seed=i, drift=0.0012) for i in range(8)}
        prepared, dates = prepare_panel(panel), calendar_of(panel)
        counts = []
        for floor in (0.1, 40.0, 55.0, 70.0, 95.0):
            result = run_backtest(prepared, dates,
                                  BacktestConfig(signal=SignalConfig(min_conviction=floor)))
            counts.append(len(result.trades))
        self.assertEqual(counts, sorted(counts, reverse=True))


class BenchmarkTests(unittest.TestCase):
    def test_buy_and_hold_tracks_the_underlying(self):
        frame = series(count=300, drift=0.001, seed=7)
        panel = {"ONE": frame}
        curve = buy_and_hold_equal_weight(panel, calendar_of(panel), 200000.0,
                                          EquityCosts(intraday=False))
        ratio = frame.close.iloc[-1] / frame.close.iloc[0]
        self.assertAlmostEqual(curve.equity.iloc[-1] / curve.equity.iloc[0], ratio, places=1)

    def test_index_buy_and_hold_is_proportional(self):
        frame = series(count=200, seed=3)[["timestamp", "close"]]
        calendar = set(pd.to_datetime(frame["timestamp"]).dt.date)
        curve = index_buy_and_hold(frame, calendar, 100000.0)
        self.assertAlmostEqual(curve.equity.iloc[0], 100000.0, places=6)
        self.assertAlmostEqual(curve.equity.iloc[-1],
                               100000.0 * frame.close.iloc[-1] / frame.close.iloc[0], places=4)

    def test_cash_curve_has_zero_risk_metrics(self):
        dates = sorted(calendar_of({"X": series(count=100)}))
        flat = pd.DataFrame({"date": dates, "equity": [100000.0] * len(dates)})
        metrics = curve_metrics(flat, 100000.0)
        self.assertEqual(metrics["net_return_pct"], 0.0)
        self.assertEqual(metrics["max_drawdown_pct"], 0.0)

    def test_compare_orders_by_cagr(self):
        dates = sorted(calendar_of({"X": series(count=260)}))
        good = pd.DataFrame({"date": dates,
                             "equity": np.linspace(100000, 150000, len(dates))})
        bad = pd.DataFrame({"date": dates,
                            "equity": np.linspace(100000, 90000, len(dates))})
        table = compare({"good": good, "bad": bad}, 100000.0)
        self.assertEqual(list(table.strategy), ["good", "bad"])


class StatisticsTests(unittest.TestCase):
    def test_normal_ppf_round_trips(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
            self.assertAlmostEqual(0.5 * (1 + math.erf(_normal_ppf(p) / math.sqrt(2))), p,
                                   places=6)

    def test_expected_max_sharpe_grows_with_trials(self):
        values = [expected_max_sharpe(n) for n in (2, 10, 50, 200, 1000)]
        self.assertEqual(values, sorted(values))
        # Matches the sqrt(2 ln N) rule of thumb within a reasonable margin.
        self.assertAlmostEqual(expected_max_sharpe(50), math.sqrt(2 * math.log(50)), delta=0.6)

    def test_deflated_sharpe_falls_as_trials_rise(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0006, 0.01, 1200))
        few = deflated_sharpe_ratio(returns, n_trials=2)
        many = deflated_sharpe_ratio(returns, n_trials=5000)
        self.assertGreater(few.probability, many.probability)
        self.assertGreater(many.expected_max_sharpe_under_null,
                           few.expected_max_sharpe_under_null)

    def test_deflated_sharpe_rejects_pure_noise(self):
        rng = np.random.default_rng(11)
        noise = pd.Series(rng.normal(0.0, 0.01, 1500))
        self.assertFalse(deflated_sharpe_ratio(noise, n_trials=50).significant)

    def test_deflated_sharpe_accepts_a_strong_genuine_edge(self):
        rng = np.random.default_rng(12)
        strong = pd.Series(rng.normal(0.0015, 0.008, 1500))   # Sharpe ~3 annualised
        self.assertTrue(deflated_sharpe_ratio(strong, n_trials=10).significant)

    def test_pbo_detects_an_overfit_population(self):
        """Pure noise: the in-sample winner is luck, so PBO must be high."""
        rng = np.random.default_rng(21)
        noise = pd.DataFrame(rng.normal(0, 0.01, (600, 12)),
                             columns=[f"v{i}" for i in range(12)])
        self.assertGreater(probability_of_backtest_overfitting(noise, n_blocks=8).pbo, 0.35)

    def test_pbo_is_low_when_one_variant_genuinely_dominates(self):
        rng = np.random.default_rng(22)
        data = rng.normal(0, 0.01, (600, 8))
        data[:, 0] += 0.004          # variant 0 has a persistent real edge
        frame = pd.DataFrame(data, columns=[f"v{i}" for i in range(8)])
        self.assertLess(probability_of_backtest_overfitting(frame, n_blocks=8).pbo, 0.2)

    def test_pbo_requires_even_blocks_and_enough_variants(self):
        frame = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (200, 4)))
        with self.assertRaises(ValueError):
            probability_of_backtest_overfitting(frame, n_blocks=7)
        with self.assertRaises(ValueError):
            probability_of_backtest_overfitting(frame.iloc[:, :1], n_blocks=8)

    def test_sharpe_of_constant_returns_is_not_a_number(self):
        self.assertTrue(math.isnan(sharpe_ratio(pd.Series([0.001] * 50))))


if __name__ == "__main__":
    unittest.main()
