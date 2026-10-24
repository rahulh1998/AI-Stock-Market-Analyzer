"""Decision-model providers: Jev, local Ollama, and a deterministic fallback.

The whole point of this layer is that a model's answer is **evidence**, never a decision.
Every provider returns the same `Decision` shape — a label, a score in [0,1], and a
confidence — and the caller is free to ignore it. Nothing here is allowed to move a price,
a stop, or a position size.

Three providers, same interface:

- `JevProvider` — TypeSafe's System One model, which is built for exactly this: it returns
  structured categories and confidences rather than prose, in one non-autoregressive pass.
  Reached through OpenRouter (`typesafe/jev-1.14`) or TypeSafe's own endpoint. Needs an API
  key; without one it reports unavailable rather than pretending.
- `OllamaProvider` — a local model on the user's GPU. Measured on an RTX 3070 Ti:
  qwen3.5:9b answers a constrained-schema classification in ~3s warm. Thinking mode is
  disabled, because for a classification task the thinking tokens consume the budget and
  the structured answer never arrives.
- `NullProvider` — always unavailable. The system must work with no model at all, so this
  is what the tests and the offline path use.

Model output is never trusted numerically: `Decision.parse` clamps every score into range
and forces unknown labels to `other`, because a model returning `materiality: 100` when
asked for 0-1 is a thing that actually happened during development.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_JEV_MODEL = "typesafe/jev-1.14"


def _clamp(value, low=0.0, high=1.0, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:          # NaN
        return default
    # A model asked for 0-1 that answers on a 0-100 scale is a known failure mode (an
    # observed `materiality: 100`), so rescale rather than clamping 95 to 1.0 and calling
    # it maximum confidence. But only above 2.0: a 1.5 is an overshoot of a 0-1 scale, not
    # a percentage, and dividing it by 100 would silently turn high confidence into noise.
    if high == 1.0 and 2.0 <= number <= 100.0:
        number = number / 100.0
    return float(min(max(number, low), high))


@dataclass(frozen=True)
class Decision:
    """One structured answer. `label` is from a caller-supplied set; scores are [0,1]."""
    label: str
    score: float = 0.0
    confidence: float = 0.0
    provider: str = "none"
    model: str = ""
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)
    error: str = None

    @property
    def available(self) -> bool:
        return self.error is None

    @classmethod
    def unavailable(cls, provider: str, reason: str) -> "Decision":
        return cls(label="unknown", provider=provider, error=reason)

    @classmethod
    def parse(cls, payload: dict, labels, provider: str, model: str,
              latency_ms: int, label_key="label", score_key="score",
              confidence_key="confidence") -> "Decision":
        """Coerce a model's JSON into the contract. Never raises on bad model output."""
        if not isinstance(payload, dict):
            return cls.unavailable(provider, f"non-object response: {type(payload).__name__}")
        raw_label = str(payload.get(label_key, "")).strip().lower().replace(" ", "_")
        label = raw_label if raw_label in set(labels) else "other"
        return cls(label=label, score=_clamp(payload.get(score_key)),
                   confidence=_clamp(payload.get(confidence_key)), provider=provider,
                   model=model, latency_ms=latency_ms, raw=payload)

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(self.score, 4),
                "confidence": round(self.confidence, 4), "provider": self.provider,
                "model": self.model, "latency_ms": self.latency_ms,
                "available": self.available, "error": self.error}


class LLMProvider:
    """Interface. `decide` must never raise — a failure is an unavailable Decision."""

    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def decide(self, prompt: str, labels, schema: dict, timeout: float = 60.0) -> Decision:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"provider": self.name, "available": self.available()}


class NullProvider(LLMProvider):
    """No model. The system is required to work in this state."""

    name = "none"

    def available(self) -> bool:
        return False

    def decide(self, prompt, labels, schema, timeout=60.0) -> Decision:
        return Decision.unavailable(self.name, "No decision model configured")


class OllamaProvider(LLMProvider):
    """Local model via Ollama. Uses the GPU even though torch here is a CPU build."""

    name = "ollama"

    def __init__(self, model: str = None, url: str = OLLAMA_URL, num_ctx: int = 2048):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.url = url
        self.num_ctx = num_ctx
        self._available = None

    def available(self) -> bool:
        if self._available is None:
            try:
                request = urllib.request.Request(self.url.replace("/api/generate", "/api/tags"))
                with urllib.request.urlopen(request, timeout=3) as response:
                    tags = json.loads(response.read())
                names = {m.get("name", "") for m in tags.get("models", [])}
                self._available = self.model in names
                if not self._available:
                    logger.warning("Ollama is up but %s is not pulled (have: %s)",
                                   self.model, sorted(names))
            except Exception as exc:
                logger.info("Ollama unavailable: %s", exc)
                self._available = False
        return self._available

    def decide(self, prompt, labels, schema, timeout=60.0) -> Decision:
        if not self.available():
            return Decision.unavailable(self.name, f"Ollama or model {self.model} unavailable")
        body = {
            "model": self.model, "prompt": prompt, "stream": False, "format": schema,
            # Thinking tokens eat the budget and the structured answer never arrives.
            "think": False,
            "options": {"temperature": 0, "num_predict": 256, "num_ctx": self.num_ctx},
        }
        started = time.time()
        try:
            request = urllib.request.Request(
                self.url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            text = (payload.get("response") or "").strip()
            if not text:
                return Decision.unavailable(self.name, "empty response from model")
            return Decision.parse(json.loads(text), labels, self.name, self.model,
                                  int((time.time() - started) * 1000))
        except json.JSONDecodeError as exc:
            return Decision.unavailable(self.name, f"model returned non-JSON: {exc}")
        except Exception as exc:
            return Decision.unavailable(self.name, f"{type(exc).__name__}: {exc}")


class JevProvider(LLMProvider):
    """TypeSafe Jev, a System One decision model, via OpenRouter or the native endpoint.

    Jev is a better architectural fit than a chat LLM for this job: it is built to return
    categories, ratings and confidences in one pass rather than generating prose that then
    has to be parsed. It is also cheap enough to run over every headline.

    Requires `OPENROUTER_API_KEY` (or `TYPESAFE_API_KEY` for the native endpoint). Without
    one this reports unavailable; it does not fall back silently to something else.
    """

    name = "jev"

    def __init__(self, model: str = None, api_key: str = None, native: bool = False):
        self.model = model or os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL)
        self.native = native or bool(os.environ.get("TYPESAFE_API_KEY"))
        self.api_key = (api_key or os.environ.get("TYPESAFE_API_KEY")
                        or os.environ.get("OPENROUTER_API_KEY")
                        or os.environ.get("JEV_API_KEY"))

    def available(self) -> bool:
        return bool(self.api_key)

    def decide(self, prompt, labels, schema, timeout=30.0) -> Decision:
        if not self.available():
            return Decision.unavailable(
                self.name, "No OPENROUTER_API_KEY / TYPESAFE_API_KEY configured")
        started = time.time()
        try:
            url, body = self._request_body(prompt, labels, schema)
            request = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            return Decision.parse(self._extract(payload), labels, self.name, self.model,
                                  int((time.time() - started) * 1000))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200] if hasattr(exc, "read") else ""
            return Decision.unavailable(self.name, f"HTTP {exc.code}: {detail}")
        except Exception as exc:
            return Decision.unavailable(self.name, f"{type(exc).__name__}: {exc}")

    def _request_body(self, prompt, labels, schema):
        if self.native:
            return TYPESAFE_URL, {"model": self.model, "input": prompt,
                                  "schema": schema, "categories": list(labels)}
        return OPENROUTER_URL, {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "decision", "schema": schema,
                                                "strict": True}},
            "temperature": 0,
        }

    @staticmethod
    def _extract(payload: dict) -> dict:
        """Jev's native shape returns the object directly; OpenRouter wraps it in choices."""
        if "choices" in payload:
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content) if isinstance(content, str) else content
        for key in ("output", "result", "decision", "data"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload


def build_provider(preference: str = "auto", model: str = None) -> LLMProvider:
    """Pick a provider. 'auto' prefers Jev when a key exists, then a local model.

    Jev first because it is purpose-built for structured decisions and costs a fraction of
    a cent per thousand headlines; local second because it is free and private; Null last
    so the pipeline always has something to call.
    """
    preference = (preference or "auto").lower()
    if preference == "jev":
        return JevProvider(model=model)
    if preference == "ollama":
        return OllamaProvider(model=model)
    if preference == "none":
        return NullProvider()
    jev = JevProvider()
    if jev.available():
        return jev
    ollama = OllamaProvider(model=model)
    if ollama.available():
        return ollama
    return NullProvider()


def provider_status() -> dict:
    """What is reachable right now — surfaced in the UI so the state is never a mystery."""
    jev, ollama = JevProvider(), OllamaProvider()
    return {
        "jev": {"available": jev.available(), "model": jev.model,
                "reason": None if jev.available() else "no API key configured"},
        "ollama": {"available": ollama.available(), "model": ollama.model,
                   "reason": None if ollama.available() else "Ollama not reachable or model not pulled"},
        "active": build_provider().name,
    }
