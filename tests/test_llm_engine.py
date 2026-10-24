"""Decision-model layer: output coercion, provider selection, caching, failure containment.

No network and no Ollama: every provider is faked. A model's output is treated as hostile
input throughout, because during development a model asked for a 0-1 score returned 100,
and another invented its own label when the schema did not pin one.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_engine.providers import (Decision, JevProvider, NullProvider, OllamaProvider,
                                      _clamp, build_provider, provider_status)
from src.llm_engine.tasks import (DecisionCache, DecisionEngine, EVENT_LABELS,
                                  RELEVANCE_LABELS, RISK_LABELS, _schema)

LABELS = ("about_company", "mentions_only", "different_company", "market_wide", "other")


class FakeProvider:
    """Returns a scripted payload, or raises, without touching a network."""

    name = "fake"
    model = "fake-1"

    def __init__(self, payload=None, available=True, error=None, boom=False):
        self.payload, self._available, self.error, self.boom = payload, available, error, boom
        self.calls = 0

    def available(self):
        return self._available

    def decide(self, prompt, labels, schema, timeout=60.0):
        self.calls += 1
        if self.boom:
            raise RuntimeError("provider exploded")
        if self.error:
            return Decision.unavailable(self.name, self.error)
        return Decision.parse(self.payload, labels, self.name, self.model, 5)

    def describe(self):
        return {"provider": self.name, "available": self._available}


class ClampTests(unittest.TestCase):
    def test_in_range_values_pass_through(self):
        for value in (0.0, 0.25, 0.5, 1.0):
            self.assertAlmostEqual(_clamp(value), value)

    def test_percent_scale_is_rescaled(self):
        """An observed failure: a model answered 100 when asked for 0-1."""
        self.assertAlmostEqual(_clamp(100), 1.0)
        self.assertAlmostEqual(_clamp(95), 0.95)
        self.assertAlmostEqual(_clamp(50), 0.5)

    def test_slight_overshoot_is_clamped_not_rescaled(self):
        """1.5 is an overshoot of a 0-1 scale; dividing by 100 would destroy it."""
        self.assertAlmostEqual(_clamp(1.5), 1.0)
        self.assertAlmostEqual(_clamp(1.01), 1.0)

    def test_garbage_becomes_the_default(self):
        for value in ("abc", None, [], {}, float("nan")):
            self.assertEqual(_clamp(value), 0.0)

    def test_negative_is_floored(self):
        self.assertEqual(_clamp(-7), 0.0)


class DecisionParseTests(unittest.TestCase):
    def test_unknown_label_becomes_other(self):
        """Observed: a model answered 'Stock Price Movement' with no enum in the schema."""
        decision = Decision.parse({"label": "Stock Price Movement", "score": 0.8,
                                   "confidence": 0.9}, LABELS, "p", "m", 1)
        self.assertEqual(decision.label, "other")

    def test_label_is_normalised(self):
        decision = Decision.parse({"label": "About Company", "score": 1, "confidence": 1},
                                  LABELS, "p", "m", 1)
        self.assertEqual(decision.label, "about_company")

    def test_non_object_is_unavailable(self):
        for payload in ("a string", 42, None, ["list"]):
            self.assertFalse(Decision.parse(payload, LABELS, "p", "m", 1).available)

    def test_missing_keys_default_to_zero(self):
        decision = Decision.parse({"label": "other"}, LABELS, "p", "m", 1)
        self.assertTrue(decision.available)
        self.assertEqual((decision.score, decision.confidence), (0.0, 0.0))

    def test_unavailable_carries_a_reason(self):
        decision = Decision.unavailable("p", "no key")
        self.assertFalse(decision.available)
        self.assertEqual(decision.error, "no key")
        self.assertEqual(decision.label, "unknown")

    def test_to_dict_is_json_safe(self):
        json.dumps(Decision.parse({"label": "other", "score": 0.5, "confidence": 0.5},
                                  LABELS, "p", "m", 1).to_dict(), allow_nan=False)


class ProviderSelectionTests(unittest.TestCase):
    def test_null_provider_is_always_unavailable(self):
        provider = NullProvider()
        self.assertFalse(provider.available())
        self.assertFalse(provider.decide("p", LABELS, {}).available)

    def test_jev_without_a_key_is_unavailable_not_silent(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = JevProvider()
            self.assertFalse(provider.available())
            decision = provider.decide("p", LABELS, {})
            self.assertFalse(decision.available)
            self.assertIn("API_KEY", decision.error)

    def test_jev_is_available_with_a_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            self.assertTrue(JevProvider().available())

    def test_explicit_preference_is_honoured(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsInstance(build_provider("none"), NullProvider)
            self.assertIsInstance(build_provider("jev"), JevProvider)
            self.assertIsInstance(build_provider("ollama"), OllamaProvider)

    def test_auto_prefers_jev_then_ollama_then_null(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}, clear=True):
            self.assertEqual(build_provider("auto").name, "jev")
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(OllamaProvider, "available", return_value=True):
            self.assertEqual(build_provider("auto").name, "ollama")
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(OllamaProvider, "available", return_value=False):
            self.assertEqual(build_provider("auto").name, "none")

    def test_status_reports_every_provider(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(OllamaProvider, "available", return_value=False):
            status = provider_status()
        self.assertFalse(status["jev"]["available"])
        self.assertIn("API key", status["jev"]["reason"])
        self.assertEqual(status["active"], "none")

    def test_ollama_network_failure_is_contained(self):
        provider = OllamaProvider()
        with patch.object(OllamaProvider, "available", return_value=True), \
             patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            decision = provider.decide("p", LABELS, {})
        self.assertFalse(decision.available)
        self.assertIn("OSError", decision.error)

    def test_jev_request_shapes_differ_by_endpoint(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}, clear=True):
            url, body = JevProvider()._request_body("p", LABELS, {"type": "object"})
        self.assertIn("openrouter", url)
        self.assertIn("messages", body)
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "k"}, clear=True):
            url, body = JevProvider(native=True)._request_body("p", LABELS, {"type": "object"})
        self.assertIn("typesafe", url)
        self.assertIn("categories", body)

    def test_jev_extracts_both_response_shapes(self):
        wrapped = {"choices": [{"message": {"content": '{"label": "other", "score": 1}'}}]}
        self.assertEqual(JevProvider._extract(wrapped)["label"], "other")
        native = {"label": "other", "score": 1}
        self.assertEqual(JevProvider._extract(native)["label"], "other")


class SchemaTests(unittest.TestCase):
    def test_schema_pins_the_label_enum(self):
        """Without an enum, models invent labels. This is the fix for that."""
        schema = _schema(RELEVANCE_LABELS)
        self.assertEqual(schema["properties"]["label"]["enum"], list(RELEVANCE_LABELS))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"label", "score", "confidence"})

    def test_every_task_has_disjoint_meaningful_labels(self):
        for labels in (RELEVANCE_LABELS, EVENT_LABELS, RISK_LABELS):
            self.assertEqual(len(labels), len(set(labels)))
            self.assertIn("other", labels)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.db"

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip(self):
        cache = DecisionCache(self.path)
        decision = Decision(label="about_company", score=0.8, confidence=0.9,
                            provider="p", model="m")
        key = DecisionCache.key("relevance", "p", "m", "prompt")
        cache.put(key, "relevance", decision)
        hit = cache.get(key)
        self.assertEqual(hit.label, "about_company")
        self.assertAlmostEqual(hit.score, 0.8)
        cache.close()

    def test_key_is_sensitive_to_every_component(self):
        base = DecisionCache.key("t", "p", "m", "prompt")
        self.assertNotEqual(base, DecisionCache.key("t2", "p", "m", "prompt"))
        self.assertNotEqual(base, DecisionCache.key("t", "p2", "m", "prompt"))
        self.assertNotEqual(base, DecisionCache.key("t", "p", "m2", "prompt"))
        self.assertNotEqual(base, DecisionCache.key("t", "p", "m", "other"))

    def test_unavailable_decisions_are_not_cached(self):
        """A transient failure must not be remembered as an answer."""
        cache = DecisionCache(self.path)
        key = DecisionCache.key("t", "p", "m", "prompt")
        cache.put(key, "t", Decision.unavailable("p", "timeout"))
        self.assertIsNone(cache.get(key))
        cache.close()

    def test_disabled_cache_is_inert(self):
        cache = DecisionCache(enabled=False)
        key = DecisionCache.key("t", "p", "m", "prompt")
        cache.put(key, "t", Decision(label="other", provider="p"))
        self.assertIsNone(cache.get(key))


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = DecisionCache(Path(self.directory.name) / "c.db")

    def tearDown(self):
        self.cache.close()
        self.directory.cleanup()

    def engine(self, provider):
        return DecisionEngine(provider, self.cache)

    def test_second_identical_call_is_served_from_cache(self):
        provider = FakeProvider({"label": "about_company", "score": 1, "confidence": 1})
        engine = self.engine(provider)
        engine.classify_relevance("Headline", "Company", "T")
        engine.classify_relevance("Headline", "Company", "T")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(engine.stats["cache_hits"], 1)

    def test_different_input_misses_the_cache(self):
        provider = FakeProvider({"label": "about_company", "score": 1, "confidence": 1})
        engine = self.engine(provider)
        engine.classify_relevance("A", "Company")
        engine.classify_relevance("B", "Company")
        self.assertEqual(provider.calls, 2)

    def test_provider_failure_is_reported_not_raised(self):
        engine = self.engine(FakeProvider(error="model down"))
        decision = engine.classify_event("Headline", "Company")
        self.assertFalse(decision.available)
        self.assertEqual(engine.stats["failures"], 1)

    def test_provider_exception_propagates_as_an_exception(self):
        """The engine does not swallow a programming error into a silent abstention."""
        engine = self.engine(FakeProvider(boom=True))
        with self.assertRaises(RuntimeError):
            engine.classify_event("Headline", "Company")

    def test_unavailable_engine_reports_so(self):
        self.assertFalse(self.engine(FakeProvider(available=False)).available)

    def test_event_risk_without_headlines_still_answers(self):
        engine = self.engine(FakeProvider({"label": "no_known_event", "score": 0,
                                           "confidence": 1}))
        self.assertTrue(engine.assess_event_risk("Co", []).available)

    def test_prompts_name_the_company_and_the_labels(self):
        captured = {}

        class Capturing(FakeProvider):
            def decide(inner, prompt, labels, schema, timeout=60.0):
                captured["prompt"], captured["labels"] = prompt, labels
                return Decision.parse({"label": "other", "score": 0, "confidence": 0},
                                      labels, "fake", "m", 1)

        engine = self.engine(Capturing())
        engine.classify_relevance("Tata Steel slumps", "Tata Steel", "TATASTEEL")
        self.assertIn("Tata Steel", captured["prompt"])
        self.assertIn("TATASTEEL", captured["prompt"])
        self.assertEqual(captured["labels"], RELEVANCE_LABELS)

    def test_describe_exposes_stats(self):
        engine = self.engine(FakeProvider({"label": "other", "score": 0, "confidence": 0}))
        engine.classify_event("H", "C")
        self.assertEqual(engine.describe()["stats"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
