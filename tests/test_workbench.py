"""Offline Streamlit integration tests: no Yahoo, model downloads or database writes.

The default workspace changed from the research workbench to the trading-signals
workspace, so the workbench flows below select their workspace explicitly. The original
intent of the old default test - that opening the app must not drag in torch, langgraph or
chroma - is preserved by test_default_workspace_imports_no_heavy_optional_libraries.
"""
import json
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


class CrewRenderingTests(unittest.TestCase):
    """The agent-evidence panel renders, and says plainly what was scored.

    Driven through AppTest with a generated script, so this exercises the real Streamlit
    render path without touching Ollama, ChromaDB or the network.
    """

    CREW = {
        "direction": "LONG",
        "penalties": {"event_risk": 0.0},
        "conviction_multiplier": 1.0,
        "vetoes": [],
        "scored_agents": ["event_risk"],
        "rule": "Direction comes from the rule engine alone.",
        "evidence": [
            {"agent": "technical", "stance": "bullish", "strength": 1.0,
             "headline": "Rule engine: CONDITIONAL_BUY", "detail": "All gates passed.",
             "citations": [], "available": True, "reason": None, "data": {}},
            {"agent": "rag", "stance": "neutral", "strength": 0.0,
             "headline": "2 passage(s) from the trading literature",
             "detail": "Confirm a breakout with volume.",
             "citations": ["murphy.pdf"], "available": True, "reason": None, "data": {}},
            {"agent": "event_risk", "stance": "neutral", "strength": 0.0,
             "headline": "Event risk: no_known_event", "detail": "No pending event.",
             "citations": [], "available": True, "reason": None, "data": {}},
            {"agent": "algo", "stance": "neutral", "strength": 0.0, "headline": "",
             "detail": "", "citations": [], "available": False,
             "reason": "Model abstains", "data": {}},
        ],
    }

    SCRIPT = """
import json, sys
sys.path.insert(0, {root!r})
from app.components.signals_workspace import _render_crew
_render_crew(json.loads({crew!r}), json.loads({errors!r}))
"""

    def render(self, crew, errors=None):
        script = self.SCRIPT.format(
            root=str(Path(__file__).resolve().parents[1]),
            crew=json.dumps(crew), errors=json.dumps(errors or []))
        return AppTest.from_string(script, default_timeout=30).run()

    def body(self, app):
        parts = [str(getattr(m, "value", "")) for m in app.markdown]
        parts += [str(getattr(c, "value", "")) for c in app.caption]
        return " ".join(parts)

    def test_crew_panel_renders_with_scored_and_explanatory_agents(self):
        app = self.render(self.CREW, errors=["RAG retriever unavailable"])
        self.assertEqual(len(app.exception), 0)
        body = self.body(app)
        self.assertIn("technical", body)
        self.assertIn("scored", body)
        self.assertIn("explanation only", body)
        self.assertIn("murphy.pdf", body)
        self.assertIn("not available", body)
        self.assertIn("RAG retriever unavailable", body)

    def test_crew_panel_reports_a_veto(self):
        vetoed = {**self.CREW, "vetoes": ["event_risk: binary event pending"],
                  "conviction_multiplier": 0.6}
        app = self.render(vetoed)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("binary event pending" in e.value for e in app.error))
        self.assertTrue(any("0.600" in w.value for w in app.warning))

    def test_absent_crew_is_explained_not_blank(self):
        app = self.render(None)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("did not run" in i.value for i in app.info))
