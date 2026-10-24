"""The jobs a decision model is actually allowed to do here.

Every task in this module shares one property: **its answer can be checked against ground
truth.** "Is this headline about Reliance Industries?" has a right answer. "Will Reliance
go up tomorrow?" does not, in any checkable sense, and no task here asks it.

That line is the whole design. This repository has already been through one round of
LLM-as-oracle — a LangGraph debate that produced confident prose and was found, on audit,
to be decorative — and the strategy it wrapped was later measured as having no edge. Adding
a better model to the same architecture would produce better prose and the same returns.

So the model works upstream, on information quality, where it is genuinely better than the
regex it replaces:

- `classify_relevance` — is this headline about this company? Replaces alias matching,
  which cannot tell "Reliance Industries" from "Reliance Power" without a curated list, and
  cannot read context at all.
- `classify_event` — what kind of event, and how material? Lets the system distinguish an
  earnings miss from a routine board-meeting notice.
- `assess_event_risk` — is a binary event pending that should suppress a technical setup?
  This is the one place a model touches the trade decision, and it may only ever **veto**.

Each returns a `Decision` with a confidence, and every caller is required to behave
sensibly when the answer is `unavailable`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import sqlite3
import threading

from src.llm_engine.providers import Decision, LLMProvider, build_provider
from src.paths import DB_PATH

logger = logging.getLogger(__name__)

RELEVANCE_LABELS = ("about_company", "mentions_only", "different_company", "market_wide", "other")
EVENT_LABELS = ("earnings", "guidance", "mna", "regulatory", "legal", "credit",
                "management", "product", "ownership", "macro", "price_action", "other")
RISK_LABELS = ("binary_event_pending", "elevated_uncertainty", "no_known_event", "other")

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_decisions (
    cache_key  TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    label      TEXT NOT NULL,
    score      REAL NOT NULL,
    confidence REAL NOT NULL,
    raw_json   TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_decisions_task ON llm_decisions (task);
"""


def _schema(labels) -> dict:
    """Constrained schema. The enum matters: without it models invent their own labels
    (an observed `event_type: "Stock Price Movement"`)."""
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "score", "confidence"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class TaskSpec:
    name: str
    labels: tuple
    score_meaning: str

    @property
    def schema(self) -> dict:
        return _schema(self.labels)


RELEVANCE = TaskSpec("relevance", RELEVANCE_LABELS,
                     "how directly the headline concerns this company (0 not at all, 1 entirely)")
EVENT = TaskSpec("event", EVENT_LABELS,
                 "materiality: how much this could move the share price (0 none, 1 major)")
EVENT_RISK = TaskSpec("event_risk", RISK_LABELS,
                      "severity of pending event risk (0 none, 1 severe)")


class DecisionCache:
    """SQLite cache. Headlines repeat constantly across sources and refreshes."""

    def __init__(self, db_path=None, enabled: bool = True):
        self.enabled = enabled
        self.conn = None
        self._lock = threading.Lock()
        if not enabled:
            return
        try:
            self.conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False,
                                        timeout=30.0)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(CACHE_SCHEMA)
            self.conn.commit()
        except Exception as exc:
            logger.warning("LLM decision cache unavailable: %s", exc)
            self.conn = None

    @staticmethod
    def key(task: str, provider: str, model: str, prompt: str) -> str:
        return hashlib.sha1(f"{task}|{provider}|{model}|{prompt}".encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Decision:
        if self.conn is None:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT label, score, confidence, provider, model, raw_json "
                "FROM llm_decisions WHERE cache_key = ?", (cache_key,)).fetchone()
        if not row:
            return None
        return Decision(label=row[0], score=row[1], confidence=row[2], provider=row[3],
                        model=row[4], raw=json.loads(row[5]) if row[5] else {})

    def put(self, cache_key: str, task: str, decision: Decision):
        if self.conn is None or not decision.available:
            return
        from datetime import datetime, timezone
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO llm_decisions "
                "(cache_key, task, provider, model, label, score, confidence, raw_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cache_key, task, decision.provider, decision.model, decision.label,
                 decision.score, decision.confidence, json.dumps(decision.raw),
                 datetime.now(timezone.utc).isoformat()))
            self.conn.commit()

    def close(self):
        if self.conn is not None:
            self.conn.close()


class DecisionEngine:
    """Runs the tasks through a provider, with caching and hard failure containment."""

    def __init__(self, provider: LLMProvider = None, cache: DecisionCache = None,
                 timeout: float = 60.0):
        self.provider = provider or build_provider()
        self.cache = cache if cache is not None else DecisionCache()
        self.timeout = timeout
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}

    @property
    def available(self) -> bool:
        return self.provider.available()

    def close(self):
        if self.cache is not None:
            self.cache.close()

    def _run(self, spec: TaskSpec, prompt: str) -> Decision:
        model = getattr(self.provider, "model", "")
        cache_key = DecisionCache.key(spec.name, self.provider.name, model, prompt)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                return hit
        decision = self.provider.decide(prompt, spec.labels, spec.schema, self.timeout)
        self.stats["calls"] += 1
        if not decision.available:
            self.stats["failures"] += 1
            return decision
        if self.cache is not None:
            self.cache.put(cache_key, spec.name, decision)
        return decision

    # ------------------------------------------------------------------ tasks

    def classify_relevance(self, headline: str, company: str, ticker: str = "") -> Decision:
        prompt = (
            f'Headline: "{headline}"\n'
            f'Company of interest: {company}{f" (ticker {ticker})" if ticker else ""}\n\n'
            "Decide whether this headline is reporting news ABOUT that specific company.\n"
            "  about_company    - the headline is about this company\n"
            "  mentions_only    - the company is named in passing; the story is about something else\n"
            "  different_company- it is about a differently-named company, including a group affiliate\n"
            "  market_wide      - an index, sector or market round-up naming several companies\n"
            f"score = {RELEVANCE.score_meaning}\n"
            "confidence = how certain you are.\n"
            "Be strict: a group affiliate with a different legal name is different_company.")
        return self._run(RELEVANCE, prompt)

    def classify_event(self, headline: str, company: str) -> Decision:
        prompt = (
            f'Headline: "{headline}"\nCompany: {company}\n\n'
            "Classify the event this headline reports.\n"
            "  earnings/guidance/mna/regulatory/legal/credit/management/product/ownership/\n"
            "  macro/price_action/other\n"
            "price_action means the headline only describes a price move with no new fact.\n"
            f"score = {EVENT.score_meaning}\n"
            "confidence = how certain you are.")
        return self._run(EVENT, prompt)

    def assess_event_risk(self, company: str, headlines) -> Decision:
        """Is a binary event pending that should suppress a technical setup?

        The only task whose output can touch a trade, and it may only veto.
        """
        joined = "\n".join(f"- {h}" for h in list(headlines)[:12]) or "- (no recent headlines)"
        prompt = (
            f"Company: {company}\nRecent headlines:\n{joined}\n\n"
            "Is there a scheduled or pending BINARY event that could gap this share price,\n"
            "making a technical breakout entry unusually risky right now?\n"
            "  binary_event_pending  - results, a verdict, a regulatory decision or similar is imminent\n"
            "  elevated_uncertainty  - unresolved investigation, credit stress, management turmoil\n"
            "  no_known_event        - nothing in these headlines suggests pending binary risk\n"
            f"score = {EVENT_RISK.score_meaning}\n"
            "confidence = how certain you are.\n"
            "Answer no_known_event unless the headlines give real evidence.")
        return self._run(EVENT_RISK, prompt)

    def describe(self) -> dict:
        return {**self.provider.describe(), "stats": dict(self.stats),
                "model": getattr(self.provider, "model", "")}
