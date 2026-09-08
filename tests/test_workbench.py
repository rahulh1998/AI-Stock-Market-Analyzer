"""Offline Streamlit integration tests: no Yahoo, model downloads or database writes."""
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class WorkbenchTests(unittest.TestCase):
    def app(self):
        path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        return AppTest.from_file(str(path), default_timeout=30).run()

    def test_default_workspace_starts_without_optional_ai(self):
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


if __name__ == "__main__":
    unittest.main()