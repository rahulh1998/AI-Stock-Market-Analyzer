# Research-first trading platform: audit, plan and setup

Reviewed 2026-09-08. Scope: Indian cash equities; no broker chosen. This document
supersedes earlier claims of institutional quality, guaranteed feed latency,
leakage-free ML, and predictive accuracy. No performance claim has been verified.

## 1. What the existing application actually does

Python/Streamlit UI -> Yahoo snapshots and candles / SQLite daily history ->
technical indicators, candlestick patterns, sentiment lexicon, saved tabular ML
and BiLSTM -> heuristic pricing scenarios -> optional Ollama/RAG commentary.
There is no exchange streaming connection, order-management system, broker
reconciliation, or persistent paper account. News scoring is a keyword lexicon,
not FinBERT. Book retrieval does not validate the profitability of a strategy.

### Audit findings and first-phase changes

| Finding | Status |
|---|---|
| Retrieval time presented as live quote time | Added source, unknown exchange timestamp, quality and execution-ineligible metadata; warning in both workspaces |
| Indicator backfill leaks future values | Removed from market fetcher; new features preserve warm-up NaNs |
| Rolling VWAP called session VWAP | Replaced with per-IST-date cumulative price-volume / volume |
| Sizing can exceed cash; HOLD can receive shares | Legacy cash cap and invalid-input rejection; new cost-aware sizing |
| Risk veto only displayed, not applied | Veto clears levels and quantity in experimental dashboard |
| SMA-200 check reads SMA-50 | Corrected; missing indicators fail closed |
| Untrained BiLSTM emits random-weight output | Inference unavailable without trained weights and sufficient history |
| Intraday data substituted for daily model input | Removed from dashboard |
| Shared mutable cached account settings | Experimental services stored per Streamlit session |
| Incomplete dependency manifest | Added existing imports; separate lightweight research manifest |
| No strategy replay | Added deterministic long-only historical replay and CSV ledger |

### Important unresolved issues

- Model comparator concatenates ticker histories before row-based TimeSeriesSplit;
  this does not enforce chronological separation across the whole stock universe.
- Legacy inference features differ from training names/order/lags. Silent 50%
  fallback is not a valid forecast. Legacy ML remains experimental and cannot
  authorize execution. Retraining alone does not fix this schema mismatch.
- BiLSTM has no demonstrated chronological holdout/calibration; its confidence
  is a heuristic, not a measured probability. Existing artifacts are not certified.
- Legacy pricing's blended technical/Monte-Carlo value is not fundamental fair
  value. Simulation parameters and percentile coverage are unvalidated.
- Other legacy feature/backfill and dataset-label paths need a dedicated ML audit.
- Legacy intraday ADX is a range proxy, not Wilder ADX; those markers are exploratory.
- Yahoo failures may still display zero placeholders in the legacy table; its
  quality column and warning identify these as unavailable, not tradable prices.
- Holidays, session gaps, halts, circuit limits, corporate actions, symbol changes,
  tick sizes, adjusted OHLC consistency and point-in-time fundamentals are not
  fully integrated. Existing hardcoded ticker aliases require instrument-master review.

## 2. Implemented first-phase workflow

Launch -> Research & historical paper replay (default) -> choose ticker/horizon
-> fetch research candles -> inspect provenance and last observed bar -> run
replay -> inspect equity, fees, trades, drawdown -> export CSV.

The default workspace does not initialize LLM, Chroma, Torch, or a market database.
It excludes currently forming bars conservatively and refuses malformed OHLCV.
Data retrieval is button-triggered, not repeatedly performed on every widget change.
Replay results invalidate when parameters or source data change.

Intraday and daily workflows are separate. The latter is a **daily technical
strategy experiment**, not a complete long-term investing/valuation product.
The independent position calculator accepts assumptions, not order instructions.

## 3. Mathematical specification

Implemented in the research module (rates below are illustrative, not law):

- True range: `TR_t = max(H_t-L_t, abs(H_t-C_(t-1)), abs(L_t-C_(t-1)))`.
- ATR: arithmetic mean of the last 14 true ranges, with 14-bar warm-up. This
  implementation deliberately says SMA ATR, not Wilder's recursive smoothing.
- EMA: recursive `EMA_t = alpha*C_t + (1-alpha)*EMA_(t-1)`, `alpha=2/(n+1)`;
  require n observations for a displayed feature.
- Session VWAP: `sum(((H+L+C)/3)*V) / sum(V)` reset on IST date. Zero cumulative
  volume is unknown, not a fabricated price. Exchange sessions still need validation.
- Relative volume: current volume / mean of previous 20 bar volumes. A same-time-
  of-day volume baseline is a future improvement; intraday volume is seasonal.
- Intraday candidate: EMA9 > EMA21, close > VWAP, relative volume >= 1.25.
- Daily candidate: EMA9 > EMA21 and close > SMA200 with 200 completed bars.
- Observe signal on bar t; enter at bar t+1 open with adverse slippage. Long only.
- Stop distance: 1.5 * previous completed-bar ATR. Target: entry + 2 * distance.
- Quantity is the largest integer satisfying both:
  `q*entry + buy_fees <= min(cash, cash*allocation_fraction)`;
  `q*(entry - slipped_stop) + buy_fees + stop_sell_fees <= cash*risk_fraction`.
  Binary search handles fee caps and fixed DP charges. No assumed leverage.
- Default risk 0.5%, allocation 25%; these are conservative starting parameters,
  not universally optimal rules. Fee/slippage changes affect share count.
- Modeled entry loss lock: when realized cash falls below the session-start
  equity threshold, no new entries. It is not a real-time account kill switch.
- Stops crossed at the open fill at the worse open; if stop and target both touch
  in a bar, stop wins. Last sample bar closes remaining positions.
- Intraday exits use last observed bar per date. This is a retrospective session-
  end proxy: missing feed tails can make the result unrealistic. Not a live scheduler.
- Net P&L: `q*(sell_fill-buy_fill)-buy_fees-sell_fees`.
- Drawdown: `(running_peak-equity)/running_peak`, including initial capital in peak.
- Profit factor: positive net P&L / absolute negative net P&L; undefined without
  losses. Expectancy: average net P&L/trade; win rate undefined without trades.

Costs: illustrative intraday brokerage min(0.03% turnover, INR20/order), NSE
transaction 0.00307%, SEBI INR10/crore, GST 18% on brokerage + transaction + SEBI,
intraday STT 0.025% sell only, stamp 0.003% buy only. Delivery: zero brokerage,
STT 0.1% each side, stamp 0.015% buy, configurable DP estimate per exit. Defaults
are not a universal broker tariff. Contract-note rounding, extra levies, DP
aggregation by ISIN/day, funding and other charges remain outside this model.

### Mathematical features planned, not yet implemented

- Wilder RSI/ATR/ADX with verified initialization and reference test vectors.
- Opening-range breakout with a completed, calendar-aware opening window;
  VWAP pullback and volatility breakout with spread/liquidity gates.
- Net reward/risk gate, tick rounding, circuit checks, market impact and volume
  participation limits; trade risk cannot assume guaranteed stop fills.
- Portfolio volatility `sqrt(w' Sigma w)`, sector/issuer concentration, gross/net
  exposure, correlation-aware open risk, drawdown-triggered entry freeze.
- Historical VaR at specified confidence/horizon and expected shortfall as mean
  tail loss; stress scenarios and sample-size warnings. VaR is not worst-case loss.
- Daily Sharpe/Sortino with documented risk-free rate, sampling convention and
  minimum history; CAGR using actual elapsed years; avoid minute-bar sqrt(252).
- Corporate-action-aware total returns, drawdowns vs a total-return benchmark,
  rolling beta, turnover, exposure, and out-of-sample net expectancy.
- Long-term statements: ROE, ROCE with definitions, margins, cash conversion,
  leverage, interest coverage, earnings quality and sector-aware comparison.
- DCF scenario valuation: discounted explicit free cash flows plus terminal value
  `FCF_(n+1)/(discount_rate-g)`, require discount_rate > g; bridge enterprise to
  equity value using net debt, dilution and dated statements. Sensitivity ranges,
  not a single certain target. Avoid generic industrial ratios for banks.

## 4. Integration roadmap and acceptance gates

### Phase A — reproducible research (implemented baseline; validation ongoing)
Current workbench, causality tests, fee-aware replay and setup diagnostics.
Remaining: locked clean-environment dependency matrix, broader regression/UI tests,
reference indicator vectors and fee reconciliation with sample contract notes.

### Phase B — trustworthy market-data service
Choose broker and subscription. Implement adapter capabilities for instruments,
full quotes, candles and streaming; research fallback must never satisfy live gates.
Store source/exchange/receive timestamps separately, timezone, currency, tick/lot,
segment, adjustment policy and quality status. Maintain instrument master daily.
Background streaming worker: bounded queues, heartbeat, reconnect/backoff,
resubscription, duplicate/out-of-order tick handling and restart recovery.
Exchange/broker session-status adapter plus official holiday/special-session data;
never assume every weekday or a single closing time applies to all instruments.
Gate: stale/disconnected/missing timestamps -> no new execution candidates;
reconnect and missing-symbol tests; clock-skew monitoring; no credential logging.

### Phase C — durable paper trading and order lifecycle
SQLite transaction ledger: cash, positions, realized/unrealized P&L, fees, fills,
strategy version, provenance and rejection reasons. Idempotent client order IDs,
accepted/partial/filled/cancelled/rejected states, limit/stop semantics, spread and
participation limits, scheduled product-aware flattening and restart reconciliation.
Gate: no duplicate orders on UI reruns; accurate cash/holdings after restart;
stale feed blocks entries but does not hide existing risk. Paper != broker sandbox.

### Phase D — scientific strategy and ML evaluation
One versioned feature builder shared by training and inference. Split on unique
calendar dates across all tickers; purge overlapping label horizons, embargo,
fit scalers inside training folds, reserve untouched final test period. Remove
unobservable terminal labels and universe/survivorship leakage. Persist features,
horizon, adjustments, train end, split metrics, model/hash and calibration metadata.
Compare no-trade, buy/hold and simple strategies after costs. Report precision,
recall, Brier/log loss, reliability diagrams and uncertainty, not just accuracy.
Test model drift and missing features; abstain when invalid. No automatic model
promotion based on best historical score alone. Deep learning is optional.

### Phase E — investing and portfolio research
Point-in-time statements, release timestamps, corporate actions, filings, events,
sector classifications, valuation scenarios, watchlist filters, allocation and
rebalancing suggestions, benchmark-relative performance and portfolio stress tests.
News deduplication, entity matching, publication vs ingestion times, source links,
coverage/no-news states and stale-story rejection. AI explains cited observations;
it cannot infer proven institutional buying from a sentiment/price divergence.

### Phase F — operational controls and optional live execution
Alerts with deduplication, audit trail, feed/model health dashboard, backups,
encrypted secrets/OS credential store, local-only default UI, authentication before
remote deployment, rate limits, error metrics and incident/recovery runbooks.
Only after explicit broker selection and separate live authorization: read-only
account sync -> sandbox where available -> manual order preview -> broker risk
checks -> confirmed order submission -> reconciliation. LLM cannot place orders,
change risk limits or override deterministic vetoes. Verify current API/algo rules,
broker IP/auth requirements, data entitlements and terms before enabling.

## 5. Setup (Windows; run commands in cmd.exe)

```bat
cd /d "c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer"
python --version
```

The inspected environment reports Python 3.12.7. Use Python 3.12 and an isolated
environment; do not replace an existing working environment without a backup.
For a new checkout, create `.venv` with `py -3.12 -m venv .venv`, then:

```bat
"c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\.venv\Scripts\python.exe" -m pip install -r "c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\requirements-research.txt"
"c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\.venv\Scripts\python.exe" "c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\scripts\check_setup.py"
"c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\.venv\Scripts\python.exe" -m unittest discover -s "c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\tests" -v
"c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\.venv\Scripts\python.exe" -m streamlit run "c:\Users\rahul\OneDrive\Desktop\AI Stock Market Analyzer\app\main.py" --server.address 127.0.0.1
```

No API keys, database seeding or Ollama are required for the default workspace.
Internet/Yahoo availability is needed for fetching; failures must remain visible.
Requirements are manifests, not a verified cross-platform lockfile. Validate a
fresh environment with `pip check` before freezing a tested dependency snapshot.

For the experimental dashboard, install the full absolute `requirements.txt` path
instead. Ollama is optional external software, not installed by pip: install from
https://ollama.com/ and fetch `llama3.2` and `nomic-embed-text` only if desired.
Use lawfully obtained documents for book indexing. Do not treat old saved models
as validated or unpickle untrusted artifacts. Existing database/model scripts use
the working directory, so run from the project root. Do not share `.env` or tokens.

## 6. Research sources consulted

1. https://ranaroussi.github.io/yfinance/ — research/educational intent, no Yahoo
   affiliation, personal-use data terms; not an execution-grade feed guarantee.
2. https://kite.trade/docs/connect/v3/market-quotes/ — authenticated snapshots,
   timestamps, daily instrument dump (its last_price is not live), missing keys;
   full quote limit 500 instruments/request.
3. https://kite.trade/docs/connect/v3/websocket/ — separate authenticated stream,
   full-mode exchange timestamps/depth, heartbeats, 3000 instruments/connection,
   up to 3 connections/API key at the time reviewed. Verify current entitlements.
4. https://upstox.com/developer/api-documentation/announcements/ — current API
   changes, sandbox availability and evolving API/algo rules. Notices include
   August/September 2026 closing-auction/session changes: calendars cannot safely
   be hard-coded from older assumptions. These are broker notices, not independent
   legal/exchange verification.
5. https://zerodha.com/charges/ — illustrative NSE equity fee components; verify
   live tariff and contract notes before real use.
6. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
   — chronological splits, gaps and equal-spacing assumptions. A ticker-concatenated
   panel is not automatically a valid time-ordered dataset.
7. https://www.nseindia.com/resources/exchange-communication-holidays — attempted
   official calendar retrieval timed out; no authoritative holiday schedule was
   imported or claimed to be verified.

## 7. Explicit non-goals of this delivery

No live tick feed, broker authentication, actual order placement, persistent paper
account, licensed fundamentals, exchange calendar, profit guarantee or fully
validated predictive model. The historical ledger is session state plus CSV, not
a durable OMS. Those require subsequent phases above; none is silently simulated
as an already integrated production service.

## 8. Validation performed in the existing environment

- 18 tests passed: data validation, causal feature prefixes, warm-up, session VWAP,
  fee arithmetic, cash/risk sizing, next-bar entries, no current-volume lookahead,
  stop gaps, same-bar stop priority, loss lock and ledger cash reconciliation.
- Includes three offline Streamlit AppTest flows: default startup, fetch/replay
  with a fixture, and visible fetch failure. These do not exercise optional AI services.
- `pip check`: no broken requirements found in the existing environment.
- Python compilation check produced no errors. Git whitespace check showed only
  Windows line-ending warnings. pandas-ta emits a pandas copy-on-write deprecation
  warning; a future dependency upgrade needs regression testing.
- Actual Yahoo connectivity check returned 124 RELIANCE.NS 15-minute bars;
  validation passed, with last bar start 2026-09-07 15:15 IST. This proves only
  historical fetch connectivity, not live-feed timeliness or exchange correctness.
- No broker orders, new training runs, fresh-environment install, exchange calendar
  verification, profitability study, or end-to-end optional AI validation performed.