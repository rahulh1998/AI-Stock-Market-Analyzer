"""Multi-agent evidence gathering with a deterministic arbiter.

This repository has been here before. An earlier version ran a LangGraph debate — a
Technical Analyst, a Bear Advocate, a Lead Portfolio Manager — that produced confident
prose and was found on audit to be decorative, wrapping a strategy later measured as having
no edge. The lesson was not "use a better model". It was that **an agent that can argue its
way to a trade will**, and prose is not evidence.

So the agents here cannot decide anything. Each one produces a typed `Evidence` record with
a stance, a strength and a citation. A deterministic `arbiter` then applies the same rule
the rest of the system follows:

    Direction comes from the rule engine alone. Every other agent may only reduce
    conviction or veto. Nothing can create a signal the rules do not support.

Concretely, an agent's stance can scale conviction *down* through a penalty, or raise a
veto that clears the plan. No agent output ever reaches a price, a stop or a position size.

The agents:
  - `technical_agent`  — the rule engine. The only one with a vote on direction.
  - `pattern_agent`    — candlestick structure on the deciding bar, with its own rule text.
  - `rag_agent`        — retrieves what the trading literature says about this setup, as
                         citations. It is explanatory; its stance is always neutral,
                         because a book agreeing with you is not evidence that a trade works.
  - `news_agent`       — multi-horizon sentiment over entity-matched headlines.
  - `event_risk_agent` — the one model-driven agent allowed to touch the decision, and only
                         ever to veto a setup facing a pending binary event.
  - `algo_agent`       — the causal nearest-neighbour model, which abstains honestly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging

from src.llm_engine.tasks import DecisionEngine

logger = logging.getLogger(__name__)

NEUTRAL, BULLISH, BEARISH, BLOCKING = "neutral", "bullish", "bearish", "blocking"

# An agent may only ever reduce conviction by at most this much, per agent.
MAX_PENALTY = {"news": 0.15, "algo": 0.35, "event_risk": 0.40, "pattern": 0.0, "rag": 0.0}
# Confidence a model must reach before its veto is allowed to clear a plan.
EVENT_VETO_CONFIDENCE = 0.65
EVENT_VETO_SEVERITY = 0.60


@dataclass(frozen=True)
class Evidence:
    """One agent's finding. `strength` is 0..1; `stance` never implies an action."""
    agent: str
    stance: str = NEUTRAL
    strength: float = 0.0
    headline: str = ""
    detail: str = ""
    citations: tuple = ()
    available: bool = True
    reason: str = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"agent": self.agent, "stance": self.stance,
                "strength": round(float(self.strength), 4), "headline": self.headline,
                "detail": self.detail, "citations": list(self.citations),
                "available": self.available, "reason": self.reason, "data": self.data}


def _unavailable(agent: str, reason: str) -> Evidence:
    return Evidence(agent=agent, available=False, reason=reason,
                    headline=f"{agent}: unavailable", detail=reason)


# --------------------------------------------------------------------- agents

def technical_agent(report: dict, direction: str) -> Evidence:
    """The rule engine's own verdict. The only agent that decides direction."""
    status = report.get("status", "WAIT")
    rejections = report.get("rejections", [])
    if direction in ("LONG", "SHORT"):
        return Evidence(
            agent="technical", stance=BULLISH if direction == "LONG" else BEARISH,
            strength=1.0,
            headline=f"Rule engine: {status}",
            detail="All hard gates passed: trend, ADX, RSI band, relative volume, "
                   "and the horizon's own location filter.",
            data={"status": status})
    return Evidence(agent="technical", stance=NEUTRAL, strength=0.0,
                    headline=f"Rule engine: {status} — no qualifying setup",
                    detail="; ".join(rejections[:4]) or "No directional setup.",
                    data={"status": status, "rejections": rejections[:8]})


def pattern_agent(report: dict) -> Evidence:
    """Candlestick structure on the deciding bar, quoting the engine's own rule text."""
    patterns = report.get("patterns", [])
    if not patterns:
        return Evidence(agent="pattern", stance=NEUTRAL,
                        headline="No pattern on the deciding bar",
                        detail="Absence of a pattern is a valid state, not a weak signal.")
    bulls = [p for p in patterns if p.get("direction") == "bullish"]
    bears = [p for p in patterns if p.get("direction") == "bearish"]
    stance = BULLISH if bulls and not bears else BEARISH if bears and not bulls else NEUTRAL
    names = ", ".join(p["name"] for p in patterns)
    return Evidence(
        agent="pattern", stance=stance, strength=min(1.0, len(patterns) / 3.0),
        headline=f"Patterns: {names}",
        detail=" ".join(p.get("rule", "") for p in patterns[:2]),
        data={"patterns": [p["name"] for p in patterns],
              "recent": report.get("recent_patterns", [])[-6:]})


def rag_agent(retriever, query: str, k: int = 2) -> Evidence:
    """What the trading literature says about this setup — as citations, not a verdict.

    Deliberately always neutral. A book agreeing with a setup is not evidence the setup
    makes money, and letting retrieval shade conviction would smuggle authority in as alpha.
    """
    if retriever is None:
        return _unavailable("rag", "No book index configured")
    try:
        passages = retriever.retrieve(query, k=k)
    except Exception as exc:
        return _unavailable("rag", f"{type(exc).__name__}: {exc}")
    if not passages:
        return _unavailable("rag", "No relevant passage found")
    return Evidence(
        agent="rag", stance=NEUTRAL, strength=0.0,
        headline=f"{len(passages)} passage(s) from the trading literature",
        detail=passages[0]["text"][:400],
        citations=tuple(p.get("source", "unknown") for p in passages),
        data={"passages": passages})


def news_agent(sentiment: dict) -> Evidence:
    """Multi-horizon sentiment over entity-matched, deduplicated headlines."""
    if not sentiment:
        return _unavailable("news", "No sentiment reading")
    count = int(sentiment.get("articles_count") or 0)
    if count < 1:
        return _unavailable("news", "No articles matched this instrument")
    from src.signal_engine.composite import aggregate_sentiment
    aggregate, _ = aggregate_sentiment(sentiment)
    if aggregate is None:
        return _unavailable("news", "No scored horizon available")
    stance = BULLISH if aggregate > 0.2 else BEARISH if aggregate < -0.2 else NEUTRAL
    horizons = ", ".join(
        f"{h}:{sentiment.get(f'sentiment_{h}'):+.2f}" if sentiment.get(f"sentiment_{h}") is not None
        else f"{h}:n/a" for h in ("1h", "1d", "1w"))
    return Evidence(
        agent="news", stance=stance, strength=min(1.0, abs(aggregate)),
        headline=f"News {aggregate:+.2f} over {count} articles ({sentiment.get('scorer')})",
        detail=f"Horizons {horizons}. Divergence: {sentiment.get('divergence_flag')}",
        citations=tuple(h.get("title", "")[:90]
                        for h in (sentiment.get("latest_headlines") or [])[:3]),
        data={"aggregate": aggregate, "articles": count,
              "scorer": sentiment.get("scorer"),
              "divergence": sentiment.get("divergence_flag")})


def event_risk_agent(engine: DecisionEngine, company: str, sentiment: dict) -> Evidence:
    """The only model-driven agent that can touch the trade — and only to veto.

    Asks whether a pending binary event (results, a verdict, a regulatory decision) makes a
    breakout entry unusually risky. A technical setup cannot know this; headlines can.
    """
    if engine is None or not engine.available:
        return _unavailable("event_risk", "No decision model configured")
    headlines = [h.get("title", "") for h in (sentiment or {}).get("latest_headlines", [])]
    if not headlines:
        return _unavailable("event_risk", "No headlines to assess")
    decision = engine.assess_event_risk(company, headlines)
    if not decision.available:
        return _unavailable("event_risk", decision.error)
    blocking = (decision.label in ("binary_event_pending", "elevated_uncertainty")
                and decision.confidence >= EVENT_VETO_CONFIDENCE
                and decision.score >= EVENT_VETO_SEVERITY)
    return Evidence(
        agent="event_risk", stance=BLOCKING if blocking else NEUTRAL,
        strength=decision.score if blocking else 0.0,
        headline=f"Event risk: {decision.label} "
                 f"(severity {decision.score:.2f}, confidence {decision.confidence:.2f})",
        detail=("A pending binary event can gap through a stop, so this setup is blocked."
                if blocking else "No pending binary event evident in recent headlines."),
        data={**decision.to_dict(), "blocking": blocking})


def algo_agent(model: dict, direction: str) -> Evidence:
    """Causal nearest-neighbour outlook. Abstains unless it beat a flat baseline."""
    if not model or not model.get("available"):
        return _unavailable("algo", (model or {}).get("reason") or "No model output")
    status = model.get("validation_status")
    if status != "EXPERIMENTAL_BASELINE_BEATEN":
        return _unavailable("algo", f"Model abstains ({status})")
    median = model.get("median_return_pct")
    if median is None:
        return _unavailable("algo", "No median analogue return")
    # Stance is always stated in MARKET terms — analogues that rose are bullish, whichever
    # way the trade points. The arbiter is what decides whether that leans against the
    # trade, so expressing stance trade-relative here would make it penalise twice.
    stance = BULLISH if float(median) > 0 else BEARISH if float(median) < 0 else NEUTRAL
    signed = float(median) * (1.0 if direction == "LONG" else -1.0)
    return Evidence(
        agent="algo", stance=stance, strength=min(1.0, abs(float(median)) / 2.0),
        headline=f"Historical analogues: median {float(median):+.2f}% over "
                 f"{model.get('horizon_bars')} bars",
        detail=f"Directional accuracy {model.get('directional_accuracy_pct', 0):.0f}% across "
               f"{model.get('evaluation_windows')} chronological windows. "
               "Empirical, uncalibrated, before costs.",
        data={"median_return_pct": median, "signed_return_pct": signed,
              "validation_status": status})


# -------------------------------------------------------------------- arbiter

@dataclass(frozen=True)
class Verdict:
    direction: str
    penalties: dict
    multiplier: float
    vetoes: tuple
    evidence: tuple

    def to_dict(self) -> dict:
        return {"direction": self.direction,
                "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
                "conviction_multiplier": round(self.multiplier, 4),
                "vetoes": list(self.vetoes),
                "evidence": [e.to_dict() for e in self.evidence],
                "scored_agents": sorted(self.penalties),
                "rule": "Direction comes from the rule engine alone. Other agents may only "
                        "reduce conviction or veto. Agents absent from scored_agents are "
                        "explanation only - their stance does not change the score."}


def arbiter(evidence, direction: str, scored_agents=None) -> Verdict:
    """Combine evidence deterministically. No agent can create or flip a direction.

    `scored_agents` restricts which agents contribute a penalty. The composite engine
    passes {"event_risk"} because news and the model are already priced upstream, and
    counting them again here would be double jeopardy. Agents outside the set still
    produce evidence; it is explanation rather than score.
    """
    penalties, vetoes = {}, []
    for item in evidence:
        cap = MAX_PENALTY.get(item.agent, 0.0)
        if scored_agents is not None and item.agent not in scored_agents:
            continue
        if not item.available or cap <= 0:
            continue
        if item.stance == BLOCKING:
            vetoes.append(f"{item.agent}: {item.headline}")
            penalties[item.agent] = cap
            continue
        # An agent only penalises when it leans AGAINST the rule engine's direction.
        against = ((direction == "LONG" and item.stance == BEARISH)
                   or (direction == "SHORT" and item.stance == BULLISH))
        penalties[item.agent] = cap * min(1.0, max(0.0, item.strength)) if against else 0.0

    multiplier = 1.0
    for value in penalties.values():
        multiplier *= (1.0 - value)
    return Verdict(direction=direction, penalties=penalties, multiplier=multiplier,
                   vetoes=tuple(vetoes), evidence=tuple(evidence))


def gather_evidence(*, report: dict, direction: str, sentiment: dict = None,
                    retriever=None, decision_engine: DecisionEngine = None,
                    company: str = "", rag_query: str = None,
                    max_workers: int = 4) -> list:
    """Run the agents, the slow ones concurrently. An agent failure is never fatal."""
    evidence = [technical_agent(report, direction), pattern_agent(report),
                algo_agent(report.get("model"), direction), news_agent(sentiment)]

    jobs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        if retriever is not None:
            patterns = ", ".join(p["name"] for p in report.get("patterns", [])) or "breakout"
            query = rag_query or (f"{patterns} with volume confirmation: entry, "
                                  "stop placement and failure modes")
            jobs["rag"] = pool.submit(rag_agent, retriever, query)
        if decision_engine is not None and company:
            jobs["event_risk"] = pool.submit(event_risk_agent, decision_engine, company,
                                             sentiment)
        for name, future in jobs.items():
            try:
                evidence.append(future.result())
            except Exception as exc:
                evidence.append(_unavailable(name, f"{type(exc).__name__}: {exc}"))
    return evidence
