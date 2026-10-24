"""Offline Streamlit integration tests: no Yahoo, model downloads or database writes.

The default workspace changed from the research workbench to the trading-signals
workspace, so the workbench flows below select their workspace explicitly. The original
intent of the old default test - that opening the app must not drag in torch, langgraph or
chroma - is preserved by test_default_workspace_imports_no_heavy_optional_libraries.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

RESEARCH_WORKSPACE = "Research & historical paper replay"
HEAVY_MODULES = ("torch", "langgraph", "chromadb", "langchain_ollama", "transformers")


class SignalsWorkspaceTests(unittest.TestCase):
    def app(self):
        path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        return AppTest.from_file(str(path), default_timeout=60).run()

    def test_signals_workspace_is_the_default(self):
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("trading signals", app.title[0].value.lower())

    def test_default_workspace_states_it_is_not_executable(self):
        app = self.app()
        self.assertTrue(any("not executable orders" in warning.value
                            or "no broker is connected" in warning.value.lower()
                            for warning in app.warning))

    def test_default_workspace_imports_no_heavy_optional_libraries(self):
        """Opening the app must not load torch, langgraph, chroma or transformers.

        FinBERT is loaded lazily inside load_finbert, and historical_outlook is imported
        inside analyze_market, specifically so that starting the app stays cheap.
        """
        for name in HEAVY_MODULES:
            for loaded in list(sys.modules):
                if loaded == name or loaded.startswith(f"{name}."):
                    del sys.modules[loaded]
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        still_absent = [name for name in HEAVY_MODULES if name not in sys.modules]
        self.assertEqual(sorted(still_absent), sorted(HEAVY_MODULES),
                         f"startup imported: {set(HEAVY_MODULES) - set(still_absent)}")

    def test_scanner_does_not_run_on_load(self):
        """A scan is button-triggered; a page load must not fetch the whole watchlist."""
        app = self.app()
        self.assertTrue(any("Run scan" in button.label for button in app.button))
        self.assertTrue(any("Press" in info.value for info in app.info))


class WorkbenchTests(unittest.TestCase):
    def app(self):
        path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(path), default_timeout=30).run()
        app.radio[0].set_value(RESEARCH_WORKSPACE).run()
        return app

    def test_research_workspace_starts_without_optional_ai(self):
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Historical Paper Replay", app.title[0].value)
        self.assertIn("No broker", app.warning[0].value)
        self.assertTrue(any("Cash- and risk-limited" in metric.label for metric in app.metric))

    def test_fetch_and_replay_with_offline_candles(self):
        dates = pd.date_range("2025-06-02 09:15", periods=40, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame({"timestamp": dates, "open": 100.0, "high": 101.0,
                              "low": 99.0, "close": 100.0, "volume": 1000})
        app = self.app()
        with patch("src.data_pipeline.broker_client.MarketDataFetcher.fetch_intraday_data", return_value=frame):
            next(button for button in app.button if button.label == "Fetch research candles").click().run()
        self.assertEqual(len(app.exception), 0)
        next(button for button in app.button if button.label == "Run historical paper replay").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("No eligible trades" in info.value for info in app.info))

    def test_fetch_failure_is_visible(self):
        app = self.app()
        with patch("src.data_pipeline.broker_client.MarketDataFetcher.fetch_intraday_data", return_value=pd.DataFrame()):
            next(button for button in app.button if button.label == "Fetch research candles").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("unavailable" in error.value for error in app.error))


class LegacyDashboardTests(unittest.TestCase):
    def test_legacy_workspace_is_marked_deprecated(self):
        path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(path), default_timeout=60).run()
        app.radio[0].set_value("Experimental AI dashboard (deprecated)").run()
        self.assertTrue(any("Deprecated" in error.value for error in app.error),
                        "the legacy dashboard must say it cannot emit a signal")


if __name__ == "__main__":
    unittest.main()
