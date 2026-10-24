# Implementation log — composite signal engine

What was changed, why, and what was measured. Commit `7d38153` on branch
`feat/composite-signal-engine`. Session date 2026-10-23/24.

---

## 1. Starting state

The repository held **two overlapping systems**, and the one that was wanted did not exist.

An earlier pass built an "institutional AI" dashboard (pricing engine, Bi-LSTM, LangGraph
agent debate, RAG over trading books, sentiment lexicon). A later audit pass
(`6a0e31c`, `docs/REALISTIC_TRADING_ROADMAP.md`) judged that layer unvalidated, demoted it
to "experimental", and built a disciplined research workspace beside it — `decision_engine.py`,
`research.py`, `historical_neighbors.py` — with 41 passing tests and a real NSE cost model.

### The app could not emit a BUY or a SELL

Three verified causes:

1. `app/main.py:310` hardcoded `"research_only": True` into the guardrail snapshot.
   `risk_math.py:114` returns an unconditional veto on that flag, and `main.py:318-321`
   then forced `HOLD` and zeroed the levels and quantity.
2. `main.py:308` read `df_daily['SMA_200']`, but `database_manager.get_stock_data` only
   appends RSI and ATR. The column never existed, so the guardrail also failed closed on
   `NaN` — a second, independent always-HOLD path.
3. The working engine (`analyze_market`) was long-only and emitted
   `CONDITIONAL_BUY`/`WAIT`/`EXIT_REVIEW` with **no confidence score and no risk score**.

### Silently degraded layers

| Area | Finding |
|---|---|
| Tabular ML | `model_comparator.py:116-119` trained on 11 `*_lag1` features; `main.py:252-257` fed 15 differently-named un-lagged ones. The exception was swallowed at `:261`, so the gauge reported a constant 50%. |
| ML validation | `pd.concat` of 80 tickers followed by row-index `TimeSeriesSplit` — folds interleave tickers and dates. |
| News | `yfinance.news` returns **0 items** for `.NS` tickers (verified live). Google RSS was the only real source, and a `RELIANCE` query returns "Reliance Power" and index round-ups. |
| Lexicon | `'high': 0.6` bullish and `'low': 0.6` bearish both fire on "52-week low"; `'fine': 0.9` matched the adjective. |
| Timestamps | RSS `pubDate` was discarded and replaced with `now - 4h`, destroying the 1h/1d/7d bucketing. |
| Data | SQLite daily history ended `2026-09-01`, ~7 weeks stale; 80 of 82 tickers. |
| Performance | `sentiment_analyzer.analyze_ticker` ran on every Streamlit rerun — a network fetch plus two SQLite writes per widget interaction. |
| Packaging | No `__init__.py` anywhere; every path built from `os.getcwd()`, so the app only worked when launched from the project root. |

### Baseline measured before changing anything

- Yahoo live data works: `RELIANCE.NS` LTP ₹1190, 125×15m bars, 1236 daily bars.
- `analyze_market` runs clean and detects patterns (`range_breakout` RELIANCE,
  `bullish_engulfing` SBIN).
- `historical_neighbors` is live and abstains correctly
  (`ABSTAIN_NO_BASELINE_IMPROVEMENT` on RELIANCE, `EXPERIMENTAL_BASELINE_BEATEN` on TCS).
- All 41 tests pass.
- Cost per ticker: `analyze_market` 0.49s on 499 intraday bars, 0.82s on 1236 daily bars,
  split roughly evenly between `pattern_features` and `historical_outlook`. A naive
  sequential scan of the watchlist is 40–70s.
- The daily long gate passes **84 of 1236** RELIANCE bars (~6.8%) — selective, not broken.
- Outside 09:15–15:30 IST every intraday call returns `HISTORICAL_ONLY`, because
  `decision_engine.py:295` marks data stale two bars after close.

---

## 2. Design decisions and why

### The plan was revised after review, before implementation

The approved plan proposed `conviction = 0.55·rule + 0.30·algo + 0.15·sentiment`. A design
review found three parts unsound; all claims were verified against the code before adopting
the corrections.

**R1 — the linear sum contradicted its own invariant.** A positive-weighted term in
`[-100,+100]` *necessarily* raises the composite when sentiment is bullish. "Sentiment may
only reduce" and "sentiment has a positive weight" are mutually exclusive. Replaced with
multiplicative penalties:

```
conviction = setup_grade × (1 − π_algo) × (1 − π_sentiment)
```

Each penalty is clamped to `[0, max]`, so positive news maps to exactly zero and cannot add
conviction; an unavailable component also maps to zero, so absence is never a silent 50%.
Monotonicity and "`setup_grade = 0` ⟹ `conviction = 0`" now hold by construction.

**R2 — the gates must not be re-weighted.** The six long gates are near-collinear:
`trend_atr` is literally `(ema_fast − ema_slow)/atr` (`decision_engine.py:80`), so in a
trending bar five of six fire together. Weighting them would count one piece of evidence
five times. They stay boolean; grading moved to five continuous axes that measure margin
*above* the pass mark, rendered as a letter so it cannot be misread as a percentage.

**R3 — `algo_score = (p−0.5)·200` is numerically inert.** On a 3-bar label on NSE large
caps, `p ∈ [0.46, 0.54]`, i.e. ±8 points, contributing ±2.4 of a 0–100 composite. The 0.30
weight was decorative.

**R4 — the daily GBDT was deferred behind the paper book.** A further defect was found that
the original plan missed: `model_comparator` trains on `close_lag1`, `macd_lag1` — absolute
price levels in a panel concatenated across 80 tickers. A tree splitting on
`close_lag1 < 1500` learns ticker identity, not market structure. Even with perfect splits
that model is worthless. Deferred rather than patched.

**R5 — `analyze_market` must never emit a short.** Verified:
`tests/test_decision_engine.py:199-208` asserts a bearish setup yields `EXIT_REVIEW` with
no plan. That is an accurate statement of the long-only contract, so shorts went into a
sibling `analyze_market_short` and the test was left untouched.

**R11 — the legacy dashboard is deprecated, not revived.** Verified:
`tests/test_research.py` asserts `verify_signal_guardrails({..., "research_only": True})`
is not approved. The test pins the always-HOLD behaviour as *intended*. Rather than rewrite
a passing test to resurrect the weaker path, the dashboard was labelled deprecated in the UI.

### Three numbers, three meanings

A weighted sum of patterns, a model and news is not a probability, so it is not labelled as
one.

| Number | What it is |
|---|---|
| **Conviction** 0–100 | Ranking statistic for sorting and for scaling size *down*. Carries `conviction_units: "ranking_score"`. Never rendered with `%`. |
| **Risk score** 0–100 | Severity from price, volume and cost inputs only — never a model output — so a confident signal on a thin, gappy name still reads as risky. |
| **Confidence** | The paper book's own realised hit rate with a Wilson 95% interval. `available: False` with the shortfall until a bucket has 30 closed trades. The only measured number in the system. |

---

## 3. What was built

### Phase 0 — packaging and data

- `__init__.py` across `src`, `app`, `tests`, `scripts`; new `src/signal_engine`,
  `src/paper_trading`.
- `src/paths.py` resolves every location from the file, not the working directory.
  Rewired 9 modules off `os.getcwd()`. **Verified by importing and querying from `C:\`.**
- `config/instruments.json` — 82 instruments with company names, curated aliases, tick
  size and feed status. Names fetched from Yahoo rather than hand-typed.
  - Alias rule: generated aliases keep **≥2 tokens**; single-token aliases are curated only.
    This is what stops `RELIANCE` → "Reliance". A first attempt produced bare "Reliance"
    because regex alternation matched `Industries Limited` before `Limited`; the 2-token
    guard caught it before it shipped, and the suffix list was narrowed.
  - `"Eternal"` dropped as a common English word.
  - ZOMATO → `ETERNAL.NS` (renamed listing). LTIM marked `feed_available: false` — no Yahoo
    feed under any of `LTIM.NS`, `LTIMINDTREE.NS`, `MINDTREE.NS`, `LTIM.BO`, `540005.BO`.
- `scripts/update_market_data.py` — fills each ticker's *actual* gap concurrently.
  `update_latest_data` only ever requested 5 days and slept 1s per ticker, so it could not
  close a 7-week gap. **Result: 2026-09-01 → 2026-10-23, 102,991 rows, 81 tickers, all current.**

### Phase 1 — helper extraction (gate: 41/41 green, zero test edits)

Extracted `summarize_patterns`, `classify_bias`, `evaluate_data_quality`,
`evaluate_long_gates`, `evaluate_short_gates` from `analyze_market`, which composes them in
the identical order.

Constraints observed, each verified against the test that enforces it:
- `pattern_features` stays a module global in `decision_engine` —
  `test_decision_engine.py:181,194,204` patch it by that path.
- `build_long_plan`'s early returns were left alone — `:130` asserts `reasons[0]` contains
  "reward-to-risk", which only holds because they return a fresh single-element list.
- `analyze_market` stays JSON-safe — `:157` does `json.dumps(report, allow_nan=False)`.

**Verification beyond the tests:** ran the refactored engine against a copy of the
pre-refactor file on 8 live cases across both horizons. All 8 reports byte-identical.

### Phase 2 — short side

- `research.size_short` — mirror of `size_long` with inverted price order
  (`0 < entry < stop`), an **upward** adverse exit fill (`stop × (1 + slip)`), SELL-side
  entry fees, and `margin_pct` defaulting to 100 (no modelled leverage).
- `decision_engine.build_short_plan` — mirror of `build_long_plan` with two deliberate
  asymmetries: delivery is rejected outright, and the stop cap is 3% rather than 4%
  because the loss tail is unbounded.
- `analyze_market_short` — statuses `SHORT_ENTRY` / `WAIT` / `EXIT_REVIEW_SHORT` /
  `HISTORICAL_ONLY` / `INVALID_DATA`.
- Added `support_60` and `last_swing_low`, both causally shifted.

**Cost model needed no change — verified:** a short's SELL entry at ₹1000×100 intraday
charges ₹25 STT and no stamp; the BUY exit charges ₹2.97 stamp and no STT.
`EquityCosts.breakdown` keys off `side`, not leg order. A *delivery* short is **not**
blocked by costs (it merely costs ₹119), which is why enforcement is explicit.

**Intraday-only enforced in three places, all three verified firing:** `build_short_plan`,
`analyze_market_short`, and the paper book at insert.

**Exercised on real data:** across 127 historical bars that passed the short gates, 62
plans were built and 65 correctly rejected by risk. All geometry, tick-alignment, net-R:R
and stop-cap assertions held.

### Phase 3 — scoring core

- `src/signal_engine/scoring.py` — `setup_grade` (5 axes, letter grade) and `risk_score`.
  - Risk aggregates as a **saturating soft-OR**, `100·(1 − Π(1 − wᵢ·rᵢ/100))`, not a mean,
    so one severe term dominates instead of being diluted by six benign ones.
  - Breakpoints saturate where the engine's own vetoes already fire (4%/12% stop cap,
    `policy.min_net_rr`), so score and vetoes agree at the boundary.
  - `atr_fraction` cut as an additive term — it correlates ~0.9 with stop distance, since
    the stop *is* ~1.5 ATR by construction.
  - `pattern conflict` cut — `bias == "CONFLICTING"` is already an unconditional veto, so
    the term is identically zero whenever `risk_score` is computable.
  - `r_participation` added (order size vs median bar turnover); nothing in `size_long`
    caps quantity by traded volume.
  - Liquidity and participation at 100 are promoted to **hard vetoes** that clear the plan.
- `src/signal_engine/composite.py` — `TradeSignal`, `generate_signal`, the two penalties.
  Order of operations *is* the invariant: rules decide direction → penalties scale down →
  conviction floor → risk vetoes resolve last.
- `src/signal_engine/scanner.py` — concurrent scan, two-speed, plus the market-hours clock
  policy (`Live` vs `Last completed bar (review)`).

**Bug found and fixed during browser testing:** the no-setup path returned blank components,
so the UI rendered "not available — no reason given". Replaced with an explicit reason that
states the invariant: *"Not consulted: the rules produced no directional setup, so no
penalty could apply."*

### Phase 4 — paper book

`src/paper_trading/ledger.py`: one append-only `paper_events` table; positions, cash and
P&L derived by query. Chosen over a positions/orders state machine because it removes the
whole class of "position row disagrees with its fills" bugs and needs no migrations.

- Writes idempotent on a content-hashed `event_id` — a Streamlit rerun cannot double-fill.
- Barrier resolution extracted to `research.resolve_exit` and **shared with the historical
  replay**, so the two cannot drift. Gap through the stop fills at the worse open; if both
  barriers are touched in one bar the stop wins.
- `hit_rate` returns a Wilson interval, or `available: False` with the shortfall.

### Phase 5 — sentiment

- `news_sources.py` — company-name queries, `EntityMatcher` with **longest-alias-wins**,
  round-up and quote-page filters, title dedup, real `pubDate` parsing.
  - The bare ticker is matched **case-sensitively**, because "RELIANCE" as a symbol is a
    different thing from "Reliance" in "Reliance Power". A missed mention costs at most a
    zero penalty; a misattributed one corrupts the reading.
  - `normalise_title` deletes apostrophes rather than spacing them, so "Steel's" and
    "Steels" collide for dedup. (Found by a failing test; the code was wrong.)
- `finbert_scorer.py` — lazy singleton, one batched forward pass, SQLite cache, lexicon
  fallback tagged `lexicon_fallback`.
  - Cache keyed on **title alone**, with the scorer stored in the row. Keying on scorer too
    forced a 440MB model load just to compute the lookup key. After the fix the all-cached
    path is **0ms with no model load** (was 9s).
- `multi_horizon.py` — cumulative 1h/1d/7d buckets, TTL cache, and divergence recomputed
  from the **last completed bar's own return** rather than a live quote, which would have
  imported non-causal data into a bar-close score.

### Phase 6 — app

New default workspace with Scanner / Deep dive / Paper book. The legacy dashboard is
reachable but labelled deprecated. Deprecated `use_container_width` replaced with `width`.

---

## 4. Verification performed

| Check | Result |
|---|---|
| Test suite | **121 passing**, the original 41 untouched |
| Refactor safety | 8 live cases, reports byte-identical to pre-refactor |
| Data freshness | 102,991 rows, 81 tickers, all current to 2026-10-23 |
| Portability | imports and queries succeed with cwd `C:\` |
| BUY path | replayed historically: TITAN 2024-09-05 → BUY, conviction 67.1, grade B |
| Sizing | 5 shares risks ₹979.80 against a ₹1000 budget; 6 would breach at ₹1172.69 |
| Short enforcement | all 3 layers verified rejecting a delivery short |
| Short plans | 127 gate-passing bars → 62 plans, 65 risk-rejected, all assertions held |
| Entity matching | "Reliance Power", "Adani Power", "Bajaj Housing Finance" all correctly not attributed to their sibling tickers |
| News retrieval | 117 fetched → 25 relevant, every drop categorised |
| FinBERT | loads and scores; visibly better than the lexicon on headlines with no keywords |
| Paper book | 4 signals recorded, replay idempotent (4 ENTRY events), all closed by barriers |
| Browser | scanner, deep dive and paper book all exercised live; paper book survived reload |

### The paper book's own first result

4 trades, 1 target and 3 stops, **−₹303.56 net after ₹219.60 fees**, 25% win rate, max
drawdown 1.43%. Recorded as-is. This is a 4-trade sample and means nothing statistically —
it is included because the system reporting its own losses honestly is the point.

---

## 5. Two findings that matter more than the code

**No BUY signals fire on current data.** Most of the watchlist trades below its 200-day
average, and the daily gate passes ~7% of bars by design. The engine is selective, not
broken — which is why the scanner, not a single-ticker view, is the primary screen.

**Outside market hours every intraday signal reads `HISTORICAL_ONLY`.** Correct behaviour,
but it made the app look broken after 15:30 IST. Hence the review-clock mode.

---

## 6. Deliberately not done

- The tabular ML model is **not wired in**. It needs scale-free features and date-based
  purged splits first (see §2, R4).
- The Bi-LSTM, Monte-Carlo pricing engine and LangGraph agents are **not connected** to the
  signal path. They remain behind the deprecated dashboard.
- `ModelComparator.evaluate_models` was left in place rather than deleted, to keep the diff
  reviewable — but nothing calls it.
- No broker, no order placement, no live tick feed, no exchange holiday calendar.

---

## 7. Files

**New:** `src/paths.py`, `src/signal_engine/{composite,scoring,scanner}.py`,
`src/paper_trading/ledger.py`,
`src/sentiment_engine/{news_sources,finbert_scorer,multi_horizon}.py`,
`app/components/signals_workspace.py`, `scripts/update_market_data.py`,
`config/instruments.json`, `tests/{test_composite,test_paper_ledger,test_sentiment}.py`.

**Modified:** `decision_engine.py` (helpers + short side), `research.py`
(`size_short`, `resolve_exit`), `app/main.py` (workspace switch), `database_manager.py`
and 6 others (paths), `tests/test_workbench.py` (default workspace changed), `README.md`,
`requirements.txt`, `.gitignore`.

45 files, 4,931 insertions, 433 deletions.
