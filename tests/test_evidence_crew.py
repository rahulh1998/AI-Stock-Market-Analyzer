"""Multi-agent crew: the arbiter's invariants, and every agent's unavailable path.

The point of these tests is that an agent cannot argue its way to a trade. The previous
LangGraph layer in this repo could produce a confident bullish narrative with nothing
behind it; here an agent emits a typed record and a deterministic arbiter decides what, if
anything, it is allowed to do with it. These tests pin that down.
"""
import json
import unittest

from src.agents.evidence_crew import (BEARISH, BLOCKING, BULLISH, EVENT_VETO_CONFIDENCE,
                                      Evidence, MAX_PENALTY, NEUTRAL, algo_agent, arbiter,
                                      event_risk_agent, gather_evidence, news_agent,
                                      pattern_agent, rag_agent, technical_agent)
from src.llm_engine.providers import Decision


class FakeEngine:
    available = True

    def __init__(self, label="no_known_event", score=0.0, confidence=0.9, error=None):
        self.label, self.score, self.confidence, self.error = label, score, confidence, error

    def assess_event_risk(self, company, headlines):
        if self.error:
            return Decision.unavailable("fake", self.error)
        return Decision(label=self.label, score=self.score, confidence=self.confidence,
                        provider="fake", model="m")


class FakeRetriever:
    def __init__(self, passages=None, boom=False):
        self.passages, self.boom = passages, boom

    def retrieve(self, query, k=3):
        if self.boom:
            raise RuntimeError("chroma down")
        return self.passages if self.passages is not None else [
            {"text": "A breakout should be confirmed by volume.", "source": "murphy.pdf"}]


def report(status="CONDITIONAL_BUY", patterns=None, model=None, rejections=()):
    return {"status": status, "patterns": patterns or [], "recent_patterns": [],
            "rejections": list(rejections), "model": model or {"available": False,
                                                               "reason": "no model"}}


def sentiment(aggregate=0.0, count=5, divergence="CONGRUENT"):
    return {"sentiment_1h": aggregate, "sentiment_1d": aggregate, "sentiment_1w": aggregate,
            "articles_count": count, "divergence_flag": divergence, "scorer": "test",
            "latest_headlines": [{"title": "Something happened"}]}


BEATEN = {"available": True, "validation_status": "EXPERIMENTAL_BASELINE_BEATEN",
          "median_return_pct": -1.5, "directional_accuracy_pct": 58, "evaluation_windows": 30,
          "horizon_bars": 5}


class ArbiterInvariantTests(unittest.TestCase):
    """The rules that stop an agent becoming an oracle."""

    def test_no_agent_can_create_a_direction(self):
        """With no rule setup there is no direction, however loud the other agents are."""
        evidence = [technical_agent(report("WAIT"), "NONE"),
                    news_agent(sentiment(1.0)),
                    algo_agent(BEATEN, "LONG")]
        verdict = arbiter(evidence, "NONE")
        self.assertEqual(verdict.direction, "NONE")
        # A penalty on a non-existent trade is meaningless; nothing can make one exist.
        self.assertLessEqual(verdict.multiplier, 1.0)

    def test_agreeing_agents_never_raise_conviction(self):
        """The multiplier is capped at 1: supportive evidence adds exactly nothing."""
        evidence = [news_agent(sentiment(1.0)), algo_agent(
            {**BEATEN, "median_return_pct": 5.0}, "LONG")]
        verdict = arbiter(evidence, "LONG")
        self.assertLessEqual(verdict.multiplier, 1.0)
        self.assertAlmostEqual(verdict.multiplier, 1.0)

    def test_opposing_agents_only_reduce(self):
        against = arbiter([news_agent(sentiment(-1.0))], "LONG")
        neutral = arbiter([news_agent(sentiment(0.0))], "LONG")
        self.assertLess(against.multiplier, neutral.multiplier)
        self.assertGreaterEqual(against.multiplier, 0.0)

    def test_each_penalty_is_capped(self):
        evidence = [news_agent(sentiment(-1.0)), algo_agent(
            {**BEATEN, "median_return_pct": -50.0}, "LONG")]
        verdict = arbiter(evidence, "LONG")
        for agent, penalty in verdict.penalties.items():
            self.assertLessEqual(penalty, MAX_PENALTY[agent] + 1e-9)

    def test_a_blocking_agent_raises_a_veto(self):
        evidence = [event_risk_agent(
            FakeEngine("binary_event_pending", score=0.9, confidence=0.9),
            "Co", sentiment())]
        verdict = arbiter(evidence, "LONG")
        self.assertTrue(verdict.vetoes)
        self.assertIn("event_risk", verdict.vetoes[0])

    def test_scored_agents_restricts_what_counts(self):
        """composite passes {event_risk} so news is not charged twice."""
        evidence = [news_agent(sentiment(-1.0)),
                    event_risk_agent(FakeEngine(), "Co", sentiment())]
        scoped = arbiter(evidence, "LONG", scored_agents={"event_risk"})
        self.assertNotIn("news", scoped.penalties)
        self.assertAlmostEqual(scoped.multiplier, 1.0)
        unscoped = arbiter(evidence, "LONG")
        self.assertIn("news", unscoped.penalties)
        self.assertLess(unscoped.multiplier, 1.0)

    def test_rag_and_pattern_never_score(self):
        """Literature agreeing with a setup is not evidence the setup works."""
        self.assertEqual(MAX_PENALTY["rag"], 0.0)
        self.assertEqual(MAX_PENALTY["pattern"], 0.0)
        evidence = [rag_agent(FakeRetriever(), "q"),
                    pattern_agent(report(patterns=[{"name": "range_breakout",
                                                    "direction": "bullish", "rule": "r"}]))]
        self.assertAlmostEqual(arbiter(evidence, "LONG").multiplier, 1.0)

    def test_unavailable_agents_contribute_nothing(self):
        evidence = [news_agent(None), algo_agent(None, "LONG"),
                    rag_agent(None, "q"), event_risk_agent(None, "Co", None)]
        verdict = arbiter(evidence, "LONG")
        self.assertAlmostEqual(verdict.multiplier, 1.0)
        self.assertEqual(verdict.vetoes, ())

    def test_verdict_is_json_safe(self):
        evidence = [technical_agent(report(), "LONG"), news_agent(sentiment(-0.5)),
                    rag_agent(FakeRetriever(), "q")]
        json.dumps(arbiter(evidence, "LONG").to_dict(), allow_nan=False)

    def test_direction_flips_which_side_is_penalised(self):
        bearish_news = sentiment(-1.0)
        long_penalty = arbiter([news_agent(bearish_news)], "LONG").penalties["news"]
        short_penalty = arbiter([news_agent(bearish_news)], "SHORT").penalties["news"]
        self.assertGreater(long_penalty, 0.0)
        self.assertEqual(short_penalty, 0.0)


class AgentTests(unittest.TestCase):
    def test_technical_agent_is_the_only_direction_setter(self):
        self.assertEqual(technical_agent(report(), "LONG").stance, BULLISH)
        self.assertEqual(technical_agent(report(), "SHORT").stance, BEARISH)
        self.assertEqual(technical_agent(report("WAIT"), "NONE").stance, NEUTRAL)

    def test_technical_agent_surfaces_rejections_when_waiting(self):
        evidence = technical_agent(report("WAIT", rejections=["ADX too low"]), "NONE")
        self.assertIn("ADX too low", evidence.detail)

    def test_pattern_agent_is_neutral_on_conflict(self):
        conflicting = report(patterns=[{"name": "a", "direction": "bullish", "rule": ""},
                                       {"name": "b", "direction": "bearish", "rule": ""}])
        self.assertEqual(pattern_agent(conflicting).stance, NEUTRAL)

    def test_pattern_agent_handles_no_patterns(self):
        evidence = pattern_agent(report())
        self.assertEqual(evidence.stance, NEUTRAL)
        self.assertIn("valid state", evidence.detail)

    def test_rag_agent_is_always_neutral_and_cites(self):
        evidence = rag_agent(FakeRetriever(), "breakout")
        self.assertEqual(evidence.stance, NEUTRAL)
        self.assertEqual(evidence.strength, 0.0)
        self.assertEqual(evidence.citations, ("murphy.pdf",))

    def test_rag_agent_contains_a_backend_failure(self):
        evidence = rag_agent(FakeRetriever(boom=True), "q")
        self.assertFalse(evidence.available)
        self.assertIn("RuntimeError", evidence.reason)

    def test_rag_agent_handles_no_passages(self):
        self.assertFalse(rag_agent(FakeRetriever(passages=[]), "q").available)

    def test_news_agent_needs_at_least_one_article(self):
        self.assertFalse(news_agent(sentiment(0.9, count=0)).available)
        self.assertTrue(news_agent(sentiment(0.9, count=3)).available)

    def test_news_agent_stance_follows_aggregate(self):
        self.assertEqual(news_agent(sentiment(0.8)).stance, BULLISH)
        self.assertEqual(news_agent(sentiment(-0.8)).stance, BEARISH)
        self.assertEqual(news_agent(sentiment(0.0)).stance, NEUTRAL)

    def test_algo_agent_abstains_unless_it_beat_the_baseline(self):
        abstaining = {"available": True,
                      "validation_status": "ABSTAIN_NO_BASELINE_IMPROVEMENT",
                      "median_return_pct": -9.0}
        self.assertFalse(algo_agent(abstaining, "LONG").available)
        self.assertTrue(algo_agent(BEATEN, "LONG").available)

    def test_algo_stance_is_market_direction_not_trade_relative(self):
        """Analogues that fell are bearish whichever way the trade points; the arbiter,
        not the agent, decides whether that leans against the position."""
        self.assertEqual(algo_agent(BEATEN, "LONG").stance, BEARISH)
        self.assertEqual(algo_agent(BEATEN, "SHORT").stance, BEARISH)
        rising = {**BEATEN, "median_return_pct": 1.5}
        self.assertEqual(algo_agent(rising, "LONG").stance, BULLISH)
        self.assertEqual(algo_agent(rising, "SHORT").stance, BULLISH)

    def test_algo_penalises_only_the_side_it_opposes(self):
        rising = {**BEATEN, "median_return_pct": 1.5}
        self.assertEqual(arbiter([algo_agent(BEATEN, "LONG")], "LONG").penalties["algo"] > 0, True)
        self.assertEqual(arbiter([algo_agent(BEATEN, "SHORT")], "SHORT").penalties["algo"], 0.0)
        self.assertEqual(arbiter([algo_agent(rising, "SHORT")], "SHORT").penalties["algo"] > 0, True)
        self.assertEqual(arbiter([algo_agent(rising, "LONG")], "LONG").penalties["algo"], 0.0)

    def test_event_risk_blocks_only_above_the_confidence_floor(self):
        confident = event_risk_agent(
            FakeEngine("binary_event_pending", 0.9, EVENT_VETO_CONFIDENCE + 0.1),
            "Co", sentiment())
        self.assertEqual(confident.stance, BLOCKING)
        unsure = event_risk_agent(
            FakeEngine("binary_event_pending", 0.9, EVENT_VETO_CONFIDENCE - 0.2),
            "Co", sentiment())
        self.assertEqual(unsure.stance, NEUTRAL)

    def test_event_risk_needs_severity_too(self):
        low_severity = event_risk_agent(
            FakeEngine("binary_event_pending", 0.1, 0.99), "Co", sentiment())
        self.assertEqual(low_severity.stance, NEUTRAL)

    def test_event_risk_without_a_model_is_unavailable(self):
        self.assertFalse(event_risk_agent(None, "Co", sentiment()).available)

    def test_event_risk_model_failure_is_contained(self):
        evidence = event_risk_agent(FakeEngine(error="timeout"), "Co", sentiment())
        self.assertFalse(evidence.available)


class GatherTests(unittest.TestCase):
    def test_gather_runs_every_agent(self):
        evidence = gather_evidence(
            report=report(patterns=[{"name": "range_breakout", "direction": "bullish",
                                     "rule": "r"}], model=BEATEN),
            direction="LONG", sentiment=sentiment(0.3), retriever=FakeRetriever(),
            decision_engine=FakeEngine(), company="Test Co")
        agents = {e.agent for e in evidence}
        self.assertEqual(agents, {"technical", "pattern", "algo", "news", "rag", "event_risk"})

    def test_gather_without_optional_services_still_works(self):
        evidence = gather_evidence(report=report(), direction="LONG", sentiment=None)
        self.assertEqual({e.agent for e in evidence},
                         {"technical", "pattern", "algo", "news"})

    def test_one_failing_agent_does_not_break_the_crew(self):
        evidence = gather_evidence(report=report(), direction="LONG",
                                   sentiment=sentiment(0.1),
                                   retriever=FakeRetriever(boom=True))
        rag = next(e for e in evidence if e.agent == "rag")
        self.assertFalse(rag.available)
        self.assertTrue(any(e.available for e in evidence))

    def test_every_evidence_record_serialises(self):
        evidence = gather_evidence(report=report(), direction="LONG",
                                   sentiment=sentiment(0.1), retriever=FakeRetriever(),
                                   decision_engine=FakeEngine(), company="C")
        json.dumps([e.to_dict() for e in evidence], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
