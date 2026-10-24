"""Multi-market universe: key resolution, alias safety, and per-market fee schedules.

The fee tests matter most. Applying the NSE schedule to a US trade would overstate costs by
roughly an order of magnitude — 0.34% against 0.003% — which would make a US strategy look
unprofitable for an entirely fictional reason. These pin both schedules down numerically.
"""
import unittest

from src.data_pipeline.universe import (MARKETS, NSE, US, US_SEED, _us_aliases,
                                        build_us_instruments, costs_for, load_market_instruments,
                                        load_universe, market_for, split_key, yahoo_symbol_for)
from src.quant_engine.research import EquityCosts


class MarketTests(unittest.TestCase):
    def test_markets_declare_what_differs(self):
        self.assertEqual(NSE.currency, "INR")
        self.assertEqual(US.currency, "USD")
        self.assertTrue(NSE.intraday_short_only)
        self.assertFalse(US.intraday_short_only)
        self.assertNotEqual(NSE.default_tick, US.default_tick)
        self.assertNotEqual(NSE.timezone, US.timezone)

    def test_symbol_suffix_differs(self):
        self.assertEqual(NSE.yahoo_symbol("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(US.yahoo_symbol("AAPL"), "AAPL")

    def test_registry_is_complete(self):
        self.assertEqual(set(MARKETS), {"NSE", "US"})


class KeyTests(unittest.TestCase):
    def test_qualified_keys_round_trip(self):
        self.assertEqual(split_key("US:AAPL"), ("US", "AAPL"))
        self.assertEqual(split_key("NSE:RELIANCE"), ("NSE", "RELIANCE"))

    def test_bare_symbol_defaults_to_nse(self):
        """Back-compatibility: existing callers pass bare NSE tickers."""
        self.assertEqual(split_key("RELIANCE"), ("NSE", "RELIANCE"))

    def test_unknown_prefix_is_not_treated_as_a_market(self):
        self.assertEqual(split_key("BSE:RELIANCE"), ("NSE", "BSE:RELIANCE"))

    def test_market_for_resolves(self):
        self.assertEqual(market_for("US:AAPL").code, "US")
        self.assertEqual(market_for("RELIANCE").code, "NSE")

    def test_universe_keys_are_market_qualified(self):
        """Bare symbols collide across markets; qualification prevents silent shadowing."""
        universe = load_universe(("NSE", "US"))
        self.assertTrue(all(":" in key for key in universe))
        self.assertIn("US:AAPL", universe)
        self.assertEqual(universe["US:AAPL"]["currency"], "USD")

    def test_each_entry_carries_its_market_and_ticker(self):
        universe = load_universe(("US",))
        entry = universe["US:MSFT"]
        self.assertEqual(entry["market"], "US")
        self.assertEqual(entry["ticker"], "MSFT")

    def test_yahoo_symbol_prefers_an_explicit_override(self):
        instruments = {"NSE:ZOMATO": {"yahoo_symbol": "ETERNAL.NS"}}
        self.assertEqual(yahoo_symbol_for("NSE:ZOMATO", instruments), "ETERNAL.NS")
        self.assertEqual(yahoo_symbol_for("US:AAPL", {}), "AAPL")
        self.assertEqual(yahoo_symbol_for("NSE:TCS", {}), "TCS.NS")


class AliasTests(unittest.TestCase):
    def test_corporate_suffixes_are_stripped(self):
        self.assertIn("Advanced Micro Devices", _us_aliases("AMD", "Advanced Micro Devices, Inc."))
        self.assertIn("JPMorgan Chase", _us_aliases("JPM", "JPMorgan Chase & Co."))

    def test_apple_keeps_only_its_full_name(self):
        """The stripped form is the bare word "Apple", which would match fruit, retail and
        half the technology press, so only the full legal name survives."""
        self.assertEqual(_us_aliases("AAPL", "Apple Inc."), ["Apple Inc."])

    def test_common_english_single_tokens_are_excluded(self):
        """Same rule as the NSE builder: a false match corrupts a reading, a miss is free."""
        for ticker, name in (("META", "Meta Platforms, Inc."), ("V", "Visa Inc."),
                             ("ORCL", "Oracle Corporation")):
            aliases = _us_aliases(ticker, name)
            singles = [a for a in aliases if len(a.split()) == 1]
            for single in singles:
                self.assertNotIn(single.lower(), {"meta", "visa", "oracle"},
                                 f"{ticker} kept the unsafe alias {single!r}")

    def test_distinctive_single_tokens_are_kept(self):
        self.assertIn("Netflix", _us_aliases("NFLX", "Netflix, Inc."))
        self.assertIn("Microsoft", _us_aliases("MSFT", "Microsoft Corporation"))

    def test_every_seeded_instrument_has_at_least_one_alias(self):
        for ticker, meta in build_us_instruments().items():
            self.assertTrue(meta["aliases"], f"{ticker} has no alias")

    def test_seed_is_well_formed(self):
        instruments = build_us_instruments()
        self.assertEqual(len(instruments), len(US_SEED))
        for meta in instruments.values():
            self.assertEqual(meta["market"], "US")
            self.assertEqual(meta["currency"], "USD")
            self.assertEqual(meta["tick_size"], US.default_tick)


class CostTests(unittest.TestCase):
    """The schedules must not be interchangeable."""

    def test_nse_schedule_is_unchanged_by_parameterisation(self):
        costs = costs_for("NSE", intraday=True)
        self.assertAlmostEqual(costs.fee(1000, 100, "SELL"), 52.34, places=2)
        self.assertAlmostEqual(costs.fee(1000, 100, "BUY"), 30.34, places=2)
        delivery = costs_for("NSE", intraday=False)
        self.assertAlmostEqual(delivery.fee(1000, 100, "SELL"), 119.08, places=2)

    def test_nse_delivery_stt_is_charged_on_both_legs(self):
        breakdown = costs_for("NSE", intraday=False).breakdown(1000, 100, "BUY")
        self.assertGreater(breakdown["stt"], 0)

    def test_us_sec_fee_is_sell_side_only(self):
        costs = costs_for("US", intraday=False)
        self.assertEqual(costs.fee(1000, 100, "BUY"), 0.0)
        self.assertGreater(costs.fee(1000, 100, "SELL"), 0.0)

    def test_us_has_no_stamp_gst_or_dp(self):
        breakdown = costs_for("US", intraday=False).breakdown(1000, 100, "SELL")
        for component in ("stamp", "gst", "dp", "brokerage"):
            self.assertEqual(breakdown[component], 0.0, f"{component} should be zero in the US")

    def test_us_costs_are_an_order_of_magnitude_below_nse(self):
        """The reason this had to be parameterised at all."""
        notional = 100000.0
        us_round_trip = (costs_for("US", False).fee(1000, 100, "BUY")
                         + costs_for("US", False).fee(1000, 100, "SELL")) / notional
        nse_round_trip = (costs_for("NSE", False).fee(1000, 100, "BUY")
                          + costs_for("NSE", False).fee(1000, 100, "SELL")) / notional
        self.assertLess(us_round_trip * 10, nse_round_trip)

    def test_slippage_is_honoured_per_market(self):
        self.assertEqual(costs_for("US", slippage_bps=12.0).slippage_bps, 12.0)
        self.assertEqual(costs_for("NSE", slippage_bps=12.0).slippage_bps, 12.0)

    def test_unknown_market_raises(self):
        with self.assertRaises(ValueError):
            costs_for("LSE")
        with self.assertRaises(ValueError):
            load_market_instruments("LSE")

    def test_default_equity_costs_still_match_the_nse_schedule(self):
        """Nothing outside this module should notice the new fields exist."""
        self.assertAlmostEqual(EquityCosts(intraday=True).fee(1000, 100, "SELL"), 52.34,
                               places=2)
        self.assertFalse(EquityCosts().stt_delivery_sell_only)


class LoadTests(unittest.TestCase):
    def test_nse_instruments_load_with_market_tags(self):
        instruments = load_market_instruments("NSE")
        self.assertTrue(instruments)
        sample = next(iter(instruments.values()))
        self.assertEqual(sample["market"], "NSE")
        self.assertEqual(sample["currency"], "INR")

    def test_default_universe_is_nse_only(self):
        universe = load_universe()
        self.assertTrue(all(key.startswith("NSE:") for key in universe))


if __name__ == "__main__":
    unittest.main()
