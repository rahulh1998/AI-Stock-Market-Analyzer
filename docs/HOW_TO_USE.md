# How to use this

A practical guide to running the NSE Stock Market Analyzer.

---

## Read this first

**This is a research tool, not a trading system you should put money behind.**

The strategy it ships with has been measured, and it does not work:

| | Result |
|---|---|
| Strategy CAGR (5 years, after costs) | **1.14%** |
| Buy & hold the same 81 stocks | **11.31%** |
| Probability the result reflects skill (Deflated Sharpe) | **0.62** — needs > 0.95 |
| Probability of Backtest Overfitting | **0.72** — needs < 0.5 |

Full detail in [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md). Two specific consequences
for how you read the screen:

- **Don't trade these signals with real money.** You would have done better buying the
  same stocks and doing nothing.
- **Don't trust the conviction ranking.** Grade A setups *lost* money while grade B made
  it, so sorting the scanner by conviction may be putting the worst ideas at the top.

What the tool *is* good for: seeing the market through a consistent, cost-aware,
causally-correct lens; recording signals and scoring them honestly over time; and — most
useful of all — **testing a new strategy idea in minutes** instead of months.

---

## 1. First-time setup

From the project root. Windows paths shown; adjust for other platforms.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you only want the lightweight research workspace (no FinBERT, no ML):

```bash
pip install -r requirements-research.txt
```

Check it worked:

```bash
python scripts/check_setup.py
python -m unittest discover -s tests -t .
```

You should see **174 tests OK**. If not, stop and fix that before trusting any output.

No API keys are needed. Ollama is optional and only used by the deprecated agent panels.

---

## 2. Get the data

```bash
python -m scripts.update_market_data
```

Fills each ticker's gap from Yahoo, concurrently, and prints what it did — including
anything it *couldn't* fetch. Run it before each session; it takes about a minute.

First run on an empty database, or to rebuild history:

```bash
python -m scripts.update_market_data --seed-years 5
```

One known gap: **LTIM has no Yahoo feed** under any symbol variant, so it is skipped with a
stated reason rather than silently showing zeros.

FinBERT (~440MB) downloads automatically the first time you use sentiment. If it can't
load, scoring silently falls back to a keyword lexicon — the UI tells you which was used.

---

## 3. Run the app

```bash
python -m streamlit run app/main.py --server.address 127.0.0.1
```

Then open http://localhost:8501.

### Workspace 1 — Trading signals (the default)

**Sidebar first.** Set your interval, capital and risk per trade. One control matters more
than the rest:

> **Evaluation clock.** Intraday data is treated as stale two bars after its close, so
> outside 09:15–15:30 IST, *Live* correctly reports `HISTORICAL_ONLY` and nothing else.
> Switch to **"Last completed bar (review)"** to evaluate as at the last finished bar. The
> app picks this for you when the market is closed.

**Scanner tab.** Pick tickers, press **Run scan**. It does not re-scan when you fiddle with
widgets — that is deliberate.

- *Fast mode* (default) skips the nearest-neighbour model and uses cached sentiment only.
  Each row lists what was deferred, so a conviction number never silently rests on fewer
  inputs than it claims.
- Most rows reading `HOLD` is **normal**. The daily gate passes on roughly 7% of bars.
  Nothing is wrong.

**Deep dive tab.** Pick one ticker, press **Analyse**. Full evaluation with the model and
news included. You get the plan (trigger, stop, target, quantity), the setup-grade axes,
the risk terms, every blocking reason, and the headlines that were actually matched to that
company. Press **Record in paper book** to log it.

**Paper book tab.** Open positions, closed trades, equity curve, win rate, profit factor,
drawdown. **Mark open positions against latest candles** walks each position forward and
closes it at whichever barrier it hit.

### Workspace 2 — Research & historical paper replay

The older cost-aware workbench: fetch candles, inspect provenance, replay the EMA/VWAP
baseline, export a CSV ledger, use a standalone position-size calculator.

### Workspace 3 — Experimental AI dashboard (deprecated)

Kept for reference only. **It cannot emit a BUY or SELL** — its signal path is vetoed to
`HOLD` by design and its ML gauge is not wired to a validated model. The UI says so.

---

## 4. Understanding what you see

Three numbers, three different meanings. This distinction is the whole point.

| Number | What it is | What it is NOT |
|---|---|---|
| **Conviction** 0–100 | A *ranking* score for sorting and for scaling size **down**. | Not a probability. Never read it as a percent chance. (And see the warning above — its ranking validity is unproven.) |
| **Risk score** 0–100 | Severity from volatility, stop width, liquidity, participation, extension and gap history. 0 benign, 100 severe. | Never derived from a model, so a confident signal on a thin, gappy stock still reads as risky. |
| **Confidence** | Your paper book's own realised hit rate with a 95% Wilson interval. | Absent until a bucket has 30 closed trades — it shows "n of 30 needed" rather than inventing a number. |

### Actions

| Action | Meaning |
|---|---|
| `BUY` | Conditional long entry: only if price breaks above the trigger, cancel above the entry ceiling |
| `SHORT_ENTRY` | Intraday only — NSE cash delivery cannot be sold short |
| `EXIT_LONG` | A bearish pattern against an existing long. Advisory; not an instruction to short |
| `HOLD` / `WAIT` | No qualifying setup. A valid and common outcome |

### The core rule the engine follows

> Sentiment and the model may only **reduce** conviction or veto. They can never create a
> signal the rules don't support, and they never change entry, stop, target or size.

So positive news adds exactly nothing, and a missing component is never treated as neutral —
it's reported as unavailable with a reason.

---

## 5. Command line

Everything the UI does is scriptable.

```bash
# One signal as JSON
python -m src.signal_engine.composite --ticker RELIANCE --interval 15m
python -m src.signal_engine.composite --ticker TITAN --interval 1d --as-of 2024-09-05

# What news is actually being matched to a company
python -m src.sentiment_engine.news_sources --ticker RELIANCE
python -m src.sentiment_engine.multi_horizon --ticker RELIANCE --force

# Refresh prices
python -m scripts.update_market_data

# Tests
python -m unittest discover -s tests -t .
```

`--as-of` truncates history so you can replay what the system would have said on a past
date. That is the quickest way to sanity-check a signal against what happened next.

---

## 6. Testing a strategy idea (the most valuable thing here)

```bash
python -m scripts.run_backtest           # backtest + benchmarks
python -m scripts.run_backtest --grid    # + variant grid, Deflated Sharpe, PBO
```

This runs the real strategy across all tickers and years, then judges it:

1. **Benchmarks after identical costs** — cash, buy & hold Nifty 50, buy & hold the
   watchlist equal-weighted. If you don't beat buy-and-hold, you've built an expensive
   index tracker.
2. **Deflated Sharpe Ratio** — penalises your Sharpe by how many variants you tried.
   Below ~0.95 means your result is indistinguishable from the luckiest attempt.
3. **Probability of Backtest Overfitting** — above 0.5 means selecting on this backtest is
   worse than picking at random.
4. **Robustness** — performance by year and by setup grade.

A JSON report lands in `data/backtest_report.json`.

### Three rules for using this honestly

- **Count every variant you try as a trial.** The expected best Sharpe of 50 zero-edge
  strategies is about 2.8 — a number most people would call excellent.
- **If it fails, do not tune until it passes.** That is exactly what PBO detects, and each
  attempt makes the next result less trustworthy.
- **A "no" is a successful result.** It cost you an afternoon instead of a year of capital.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Everything says `HISTORICAL_ONLY` | Market is closed. Switch the evaluation clock to "Last completed bar (review)". |
| Everything says `HOLD` | Usually correct — the gate passes ~7% of bars, and it stays quiet when most names are below their 200-day average. Check the Deep dive's blocking reasons to confirm. |
| Scanner is slow | First run fetches from Yahoo. Use Fast mode and fewer tickers; results cache for the session. |
| Sentiment says `lexicon_fallback` | FinBERT didn't load. Check the ~440MB download completed and `transformers` is installed. The number is still shown, just cruder. |
| A ticker shows UNAVAILABLE | Yahoo returned no candles. LTIM is permanently in this state. |
| "No module named src" | Run from the project root, or use `python -m ...` form. |
| Paper book looks wrong | It is derived from an append-only log, so it cannot drift. Reload the page; state is rebuilt from the database. |

---

## 8. Maintenance

| When | Do |
|---|---|
| Each session | `python -m scripts.update_market_data` |
| After changing any strategy code | `python -m unittest discover -s tests -t .` |
| After changing strategy *logic* | `python -m scripts.run_backtest --grid` — and record how many variants you tried |
| Occasionally | Review `config/instruments.json` for renamed listings (ZOMATO → ETERNAL was one) |

---

## 9. Where things live

| Path | What |
|---|---|
| `app/components/signals_workspace.py` | The main UI |
| `src/signal_engine/composite.py` | `generate_signal` — the whole decision |
| `src/signal_engine/scoring.py` | Setup grade and risk score |
| `src/quant_engine/decision_engine.py` | Patterns, gates, long/short plans |
| `src/quant_engine/research.py` | Fee model, sizing, fill rules |
| `src/backtesting/` | Portfolio backtest and the validation statistics |
| `src/paper_trading/ledger.py` | The paper book |
| `config/instruments.json` | Company names, aliases, tick sizes, feed status |
| `data/market_data.db` | SQLite: prices, sentiment cache, paper book |

Further reading: [README](../README.md) · [UPGRADE_ROADMAP](UPGRADE_ROADMAP.md) ·
[VALIDATION_RESULTS](VALIDATION_RESULTS.md) · [IMPLEMENTATION_LOG](IMPLEMENTATION_LOG.md)

---

## 10. The honest summary

The software works and is tested. The strategy it implements does not make money, and you
now have proof rather than a hunch. Use it to look at markets, to record and score ideas,
and above all to kill bad strategies cheaply — which is the only part of trading software
that reliably pays for itself.

Nothing here is investment advice.
