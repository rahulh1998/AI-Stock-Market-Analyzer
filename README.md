# NSE Stock Market Analyzer

An end-to-end **trade-signal workbench** for Indian cash equities (NSE). It combines
rule/pattern analysis, an algorithmic model and news sentiment into one recommendation —
`BUY`, `SHORT_ENTRY`, `EXIT_LONG` or `HOLD` — with exact entry, stop, target and position
size, a **conviction** score, a **risk** score, and a persistent **paper book** that scores
those signals over time.

> **Status: research tool. No broker is connected and no real orders can be placed.**
> Market data comes from Yahoo Finance and carries no live-feed guarantee. Nothing here is
> investment advice. The paper book is a local SQLite simulation, not a broker sandbox.

---

## What the numbers mean

This is the most important section in this file. Three numbers are reported, and they mean
three different things.

| Number | Range | What it is | What it is **not** |
|---|---|---|---|
| **Conviction** | 0–100 | A *ranking statistic*: how strong the rule evidence is, after penalties. Use it to sort a watchlist and to scale size **down**. | Not a probability. Never read it as a percent chance, and it is never rendered with a `%`. |
| **Risk score** | 0–100 | Severity, from volatility, stop width, liquidity, participation, extension, net reward:risk and gap history. 0 benign, 100 severe. | Not derived from any model output, so it stays independent of conviction. |
| **Confidence** | a measured rate + 95% interval | The realised hit rate of *this paper book's own closed trades* in the same (direction, interval, grade) bucket, with a Wilson interval. | Absent until a bucket has 30 closed trades. Until then the UI says "n of 30 needed" rather than showing a made-up number. |

### The core invariant

> **Sentiment and the model may only reduce conviction or veto. They can never create a
> directional signal the rules do not support, and they never alter entry, stop, target or
> position size. Deterministic account-risk vetoes resolve last.**

This is structural, not a convention. Conviction is

```
conviction = setup_grade × (1 − penalty_algo) × (1 − penalty_sentiment)
```

where `setup_grade` comes only from the rule layer and each penalty is clamped to
`[0, max]`. Positive news maps to a penalty of exactly **zero**, so it cannot add
conviction; an unavailable component also maps to zero, so absence is never a silent 50%.
`setup_grade = 0` forces `conviction = 0`.

`tests/test_composite.py` exists to prove this: it sweeps sentiment across 21 values with
no rule setup and asserts the action never leaves `HOLD`, asserts a maximally bullish
model cannot create a signal, and asserts that a plan is either removed entirely or left
byte-identical — never repriced or resized.

---

## Quick start

Python 3.12. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.update_market_data
python -m streamlit run app/main.py --server.address 127.0.0.1
```

`update_market_data` fills each ticker's gap incrementally and prints what it did,
including any ticker it could not fetch. No API keys are needed. Ollama is optional and
only used by the deprecated agent/RAG panels.

FinBERT (`ProsusAI/finbert`, ~440MB) downloads on first sentiment use. If it cannot load,
scoring falls back to a keyword lexicon and every reading is tagged `lexicon_fallback`.

---

## The three workspaces

### 1. Trading signals (default)

- **Scanner** — concurrent watchlist scan, ranked by action then conviction. Button-
  triggered, not re-run on every widget change. *Fast mode* skips the nearest-neighbour
  model and uses only cached sentiment, and each row lists which components were deferred,
  so a conviction number is never quietly built from fewer inputs than it claims.
- **Deep dive** — full evaluation of one ticker: the plan, the setup-grade axes, the risk
  terms, every blocking reason, both components with their reasons, the multi-horizon news
  breakdown, and a "record in paper book" button.
- **Paper book** — open positions, closed trades, equity curve, win rate, profit factor,
  expectancy, drawdown, and the log of every signal generated whether taken or not.

**Evaluation clock.** The engine marks intraday data stale two bars after its close, which
is correct but means every intraday signal reads `HISTORICAL_ONLY` outside 09:15–15:30 IST.
*Last completed bar (review)* evaluates as at the last finished bar instead, and is
labelled as a review of a completed session rather than a live signal. (Exchange holidays
are **not** modelled — only weekday and clock time.)

### 2. Research & historical paper replay

The earlier cost-aware research workbench: fetch candles, inspect provenance, run a
deterministic long-only replay of the EMA/VWAP baseline, export a CSV trade ledger, and use
a standalone position calculator. Unchanged.

### 3. Experimental AI dashboard (deprecated)

Kept for reference and **marked deprecated in the UI**. Its signal path is permanently
vetoed to `HOLD` by design (it passes `research_only=True` into `verify_signal_guardrails`)
and its ML gauge is not wired to a validated model. That behaviour is pinned by
`tests/test_research.py`, so it was left exactly as it was rather than rewired.

---

## How a signal is built

```
candles ─► validate_bars ─► pattern_features (Wilder RSI/ATR/ADX, 14 patterns, session VWAP)
                                   │
                                   ├─► analyze_market        (long gates)   ─┐
                                   └─► analyze_market_short   (short gates)  ├─► direction + plan
                                                                             │
   setup_grade (5 continuous axes) ─────────────────────────────────────────┘
                 │
                 ├── × (1 − algo penalty)        ← historical neighbours, abstains honestly
                 └── × (1 − sentiment penalty)   ← FinBERT over entity-matched headlines
                              │
                              ├─► conviction floor
                              └─► risk score + hard vetoes (liquidity, participation)
                                          │
                                          └─► TradeSignal ─► paper book
```

**Rule layer.** The six long gates stay hard booleans. They are near-collinear by
construction (`trend_atr` is literally `(ema_fast − ema_slow)/atr`), so weighting them
would count one piece of evidence five times. Grading instead happens on five continuous
axes — trend, ADX, relative volume, net reward:risk and pattern strength — measuring margin
*above* the pass mark, and is shown as a letter grade so it cannot be misread as a percent.

**Shorts** are `SHORT_ENTRY`, intraday only, because NSE cash delivery cannot be sold
short. That is enforced in three places: `build_short_plan` (the only producer of short
prices), `analyze_market_short` (rejects `1d`), and the paper book at insert. The cost
model needs no change for shorts — `EquityCosts.breakdown` keys off `side`, so a short's
SELL entry correctly pays intraday STT and its BUY exit pays stamp duty.

**Risk** aggregates as a saturating soft-OR, not a mean, so one severe term dominates
rather than being diluted by six benign ones. Term breakpoints saturate exactly where the
engine's own vetoes fire, so the score and the vetoes agree at the boundary.

**Sentiment** is entity-matched before it is scored. The hard part is not the classifier:
a bare-ticker query returns the wrong company (searching `RELIANCE` returns "Reliance
Power"), short aliases overlap (`SBI` is a substring of `SBI Cards` and `SBI Life`, both
separate instruments here), and index round-ups mention many companies and are about none.
`config/instruments.json` carries curated aliases; matching prefers the longest alias,
matches the bare ticker **case-sensitively** as a symbol, drops quote pages and round-ups,
dedupes on a normalised title, and parses real publication timestamps so the 1h/1d/7d
buckets mean something.

**Paper book** is a single append-only `paper_events` table; positions, cash and P&L are
derived by query. Writes are idempotent on a content-hashed event id, so a Streamlit rerun
cannot double-fill, and state reconciles after a restart. Barrier resolution is delegated
to `research.resolve_exit`, the same function the historical replay uses, so the two cannot
drift: a gap through the stop fills at the worse open, and if both barriers are touched in
one bar the stop wins.

---

## Layout

```
app/
  main.py                          workspace switch
  components/
    signals_workspace.py           scanner, deep dive, paper book
    research_workbench.py          cost-aware replay workbench
    decision_panel.py              pattern/risk panel
    chart_view.py  agent_cards.py
config/
  watchlist.json                   82 NSE tickers
  instruments.json                 company names, curated aliases, tick size, feed status
src/
  paths.py                         project locations (no cwd dependence)
  signal_engine/
    composite.py                   TradeSignal, generate_signal, penalties  [CLI]
    scoring.py                     setup_grade, risk_score
    scanner.py                     concurrent scan, market-hours clock policy
  quant_engine/
    decision_engine.py             patterns, gates, build_long_plan, build_short_plan
    research.py                    EquityCosts, validate_bars, sizing, replay, resolve_exit
  sentiment_engine/
    news_sources.py                entity matching, dedup, timestamps  [CLI]
    finbert_scorer.py              FinBERT + lexicon fallback, cached  [CLI]
    multi_horizon.py               1h/1d/7d buckets, divergence  [CLI]
  paper_trading/ledger.py          append-only paper book
  ml_engine/
    historical_neighbors.py        the algo component (causal, abstains)
    model_comparator.py            NOT wired in - see "Known limitations"
    deep_learning_predictor.py     NOT wired in - experimental
  data_pipeline/                   Yahoo fetcher + SQLite
scripts/
  update_market_data.py            incremental gap-filling updater
  check_setup.py                   read-only diagnostics
```

### Command-line entry points

```bash
python -m src.signal_engine.composite --ticker RELIANCE --interval 15m
python -m src.signal_engine.composite --ticker TITAN --interval 1d --as-of 2024-09-05
python -m src.sentiment_engine.news_sources --ticker RELIANCE
python -m src.sentiment_engine.multi_horizon --ticker RELIANCE
python -m scripts.update_market_data --seed-years 5
python -m unittest discover -s tests -t .
```

---

## Known limitations

These are real and deliberate; none is silently papered over.

- **No broker, no orders, no live tick feed.** Yahoo data is research-grade; freshness is
  unverified and every quote carries `execution_eligible: False`.
- **No exchange holiday calendar.** Session detection is weekday plus clock time only.
- **Confidence is usually absent.** It appears only after 30 closed paper trades in a
  bucket. That is the point: it is the one measured number here.
- **The tabular ML path is not wired in and not validated.** `model_comparator` trains on
  11 lagged features while the old dashboard fed it 15 differently-named ones, and it
  concatenates every ticker before a row-index `TimeSeriesSplit`, so folds interleave
  tickers and dates. Worse, it trains on absolute price levels in a cross-sectional panel,
  which teaches a tree ticker identity rather than market structure. Rebuilding it needs
  scale-free features and date-based purged splits; until then the algo component is the
  nearest-neighbour model, which abstains when it cannot beat a flat baseline.
- **`LTIM` has no Yahoo feed** under any symbol variant tried, and is marked
  `feed_available: false` in `instruments.json` so it is skipped with a stated reason
  instead of showing zero placeholders.
- **Survivorship bias.** The watchlist is today's list applied to five years of history.
- **A stop is not a guaranteed maximum loss.** Gaps, circuits, illiquidity and execution
  failures can all exceed it.

See [docs/REALISTIC_TRADING_ROADMAP.md](docs/REALISTIC_TRADING_ROADMAP.md) for the prior
audit and the phased plan, and [docs/PATTERN_RISK_ANALYSIS.md](docs/PATTERN_RISK_ANALYSIS.md)
for pattern definitions and risk rules.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

| Suite | Covers |
|---|---|
| `test_composite.py` | the invariants: sentiment/model cannot create or reprice a trade, veto precedence, JSON safety on every branch |
| `test_decision_engine.py` | Wilder maths, prefix causality including pivots, plan geometry, account locks |
| `test_research.py` | cost model, cash/risk sizing, replay fill rules |
| `test_paper_ledger.py` | idempotent writes, restart reconciliation, stop-before-target, gap fills, Wilson intervals |
| `test_sentiment.py` | entity matching (Reliance Power, the SBI family), dedup, timestamps, scorer fallback |
| `test_workbench.py` | offline Streamlit flows; the default workspace loads no torch/langgraph/transformers |
