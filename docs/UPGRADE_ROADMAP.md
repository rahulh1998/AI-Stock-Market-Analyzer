# Upgrade roadmap: from "produces signals" to "trustworthy enough to risk money on"

Written 2026-10-24, after building the composite signal engine (`docs/IMPLEMENTATION_LOG.md`).

---

## 0. Read this part first

You asked for a plan to get this to where you can rely on it for real investment. Here is
the honest version, and then the engineering.

**The code is not your bottleneck. Evidence is.** You now have a system that produces
clean, cost-aware, causally-correct signals. What you do *not* have — what nobody has
until they do the work in Part 2 — is any evidence that those signals make money after
costs. Those are completely different claims, and the second one is the hard one.

Three facts worth internalising before spending another weekend on this:

1. **The expected maximum Sharpe ratio of `N` strategies with *zero* true edge grows
   roughly as `sqrt(2 ln N)`.** Computed: 10 trials → 2.15, 50 → 2.80, 200 → 3.26,
   1000 → 3.72. So try 50 parameter combinations on a strategy that is pure noise, and the
   best one will show a Sharpe near 2.8 — a number most people would call excellent. This
   is why "I backtested it and it worked" is close to meaningless unless you counted and
   corrected for how many things you tried.
2. **SEBI's own published studies show the large majority of retail F&O traders lose
   money.** Cash equities are kinder than F&O, but the base rate for retail systematic
   trading is poor, and it is poor *mostly because of this exact gap* between a backtest
   and a verified edge.
3. **Your costs are already measured, and they set a hard bar.** See §1.2 — your strategy
   needs a materially better win rate than naive 2R arithmetic suggests, just to break even.

So the roadmap below is **not** "add features until it makes money." It is a sequence of
**stage gates designed to kill the project cheaply if there is no edge** — because the
expensive mistake is not failing, it is failing slowly with real money on the line.

If an edge exists, this plan finds it and lets you trade it safely. If it does not, this
plan tells you within a few months instead of a few years. Both are wins. Please read the
second outcome as the *likely* one, and treat anything better as upside.

**I am not able to advise you on whether or what to invest** — that is a licensed
activity and genuinely depends on your circumstances. What follows is purely how to build
and validate the software.

---

## 1. Where you actually stand

### 1.1 What is solid

| Component | State |
|---|---|
| Causal feature engineering | **Good.** Prefix-causality is proven by test, not asserted. |
| NSE cost model | **Good.** Side-aware, verified against both legs of long and short. |
| Position sizing | **Good.** Cost-and-cash-aware binary search; verified exact at the budget boundary. |
| Risk vetoes | **Good.** Account-level, resolve last, clear the plan rather than colour a badge. |
| Signal structure | **Good.** Three numbers with distinct meanings; no fabricated probabilities. |
| Paper book | **Good.** Append-only, idempotent, reconciles after restart, shares fill rules with the backtest. |

### 1.2 The cost bar you must clear

Computed from your own `EquityCosts` on a ₹100,000 notional round trip, plus 5bps
slippage per leg:

| Product | Fees | Slippage | Round-trip cost | Risk unit (1.5×ATR) | Cost in R | Breakeven win rate at 2R |
|---|---|---|---|---|---|---|
| Delivery (swing) | 0.238% | 0.100% | **0.338%** | ~2.25% | 0.150R | **38.3%** (vs 33.3% gross) |
| Intraday | 0.083% | 0.100% | **0.183%** | ~1.00% | 0.183R | **39.4%** (vs 33.3% gross) |

Delivery fees are dominated by STT at 0.1% on the sell leg, which is why swing costs are
roughly 3× intraday fees despite zero brokerage.

**This is the single most useful number in this document.** Two consequences:

- **Intraday is not the easier path.** Its fees are lower, but they land on a risk unit
  less than half the size, so the required win rate is *higher* than swing — while tighter
  stops relative to intraday noise push realised win rates *lower*. The popular intuition
  is backwards.
- Your paper book's first 4 trades came in at 25%. That is a meaningless sample size, but
  it points at the right question: **can this strategy sustain >38% at 2R, out of sample,
  after costs?** Everything in Part 2 exists to answer that one question.

### 1.3 The statistical sample you need

To distinguish a real 45% win rate from the ~38% breakeven with 80% power at 5%
significance needs roughly **590 trades**. To detect a marginal 40% needs roughly **4,100**.

Your gate passes ~6.8% of daily bars. Across 82 tickers that is ~5.6 signals/day, but
portfolio limits (0.5% risk each, 2% max open risk) cap you near 4 concurrent positions —
realistically **200–400 trades/year**.

**So live trading alone would take 2–10 years to answer the question.** That is why the
backtest has to be rigorous: 5 years × 82 tickers is the only sample you can afford, and it
is only worth anything if it is honest. This reframes the whole project — the backtest is
not a formality before the fun part, it *is* the project.

### 1.4 What is missing or wrong

- **No multi-ticker portfolio backtest.** `research.replay` is single-instrument, long-only,
  and tests the old EMA/VWAP baseline — *not* the composite engine you now have. **You have
  never backtested the actual strategy.** This is the largest gap by far.
- **No overfitting control.** No trial counter, no deflated Sharpe, no PBO.
- **The ML component is unwired** and needs rebuilding on scale-free features.
- **Survivorship bias** — today's watchlist applied to five years of history.
- **No corporate-action adjustment** in the signal path; raw Yahoo closes jump on splits
  and bonuses, which both fabricates patterns and breaks stops.
- **No exchange holiday calendar.**
- **No regime awareness.** Current data is a downtrend and the long gate is near-silent;
  you cannot yet tell "correctly waiting" from "broken".

---

## 2. The stage gates

Work them in order. **Do not start a gate until the previous one passes.** Each has an
explicit kill condition — honour it.

### Gate A — Can you measure truthfully? (3–4 weekends)

Build the apparatus that can tell you you're wrong.

**A1. Corporate-action-correct price history.**
Yahoo's `adj_close` back-adjusts for splits and dividends but you currently sign signals off
raw `close`. Build a split/bonus detector (flag any overnight gap >20% not matched by index
movement), reconcile against `adj_close` ratios, and store an explicit adjustment factor
series. *Nothing downstream is trustworthy until this is done* — a 1:5 split looks exactly
like a −80% crash to every pattern in `decision_engine`.

**A2. Portfolio backtest for the composite engine.**
A new `src/backtesting/portfolio.py`:
- Walk one calendar date at a time across all tickers simultaneously.
- At each date, generate signals, rank by conviction, and allocate under the *real*
  portfolio constraints (`max_open_risk_pct`, `max_gross_exposure_pct`, cash).
- Reuse `research.resolve_exit` and `EquityCosts` — never re-derive fills or fees.
- Track the full equity curve, not just closed trades.
- **This must share the exact code path `generate_signal` uses.** A backtest that
  re-implements the strategy is testing a different strategy.

**A3. Point-in-time universe.** Record index membership by date if you can obtain it;
if not, state the survivorship bias as a known, quantified caveat and stop claiming
otherwise. Honest beats impossible.

**A4. Exchange calendar.** NSE trading holidays, properly sourced. Affects staleness
detection, label maturity, and every "next bar" assumption.

> **Gate A passes when** you can run the actual composite strategy across 82 tickers × 5
> years, with corporate actions handled, and get an equity curve you believe is arithmetically
> correct. **No performance claim yet — just correctness.**

### Gate B — Does the edge survive scrutiny? (2–3 weekends)

This is where most projects should die, and where yours either earns the rest of the
roadmap or doesn't.

**B1. Benchmarks, after costs.** Compare against: no-trade (cash), buy-and-hold Nifty 50,
equal-weight buy-and-hold of your watchlist, and the rule gate with random entry timing.
**If you do not beat buy-and-hold Nifty after costs, you have built an expensive index
tracker.** That comparison alone kills a majority of retail strategies and it costs you one
afternoon.

**B2. Count your trials. All of them.** Every threshold in `SignalConfig`, every breakpoint
in `scoring.py`, every gate constant you have *ever* adjusted, is a trial. Log them.

**B3. Deflated Sharpe Ratio** (Bailey & López de Prado). Penalises your Sharpe by the number
of trials, the backtest length, and return skew/kurtosis. Report DSR, never raw Sharpe.

**B4. Probability of Backtest Overfitting** via Combinatorially Symmetric Cross-Validation.
Splits performance into blocks, picks the in-sample best, records its out-of-sample rank.
**PBO > 0.5 means your selection process is worse than random** — stop.

**B5. Combinatorial Purged Cross-Validation** with purging and embargo around label
horizons, giving many test paths rather than one.

**B6. Regime and robustness slices.** Performance by year, by market regime (above/below
Nifty 200DMA), by sector, by volatility tercile, by liquidity decile. **An edge that exists
in exactly one regime or one sector is a story about that period, not an edge.**

**B7. Parameter sensitivity.** Vary each constant ±20%. A real edge degrades gracefully; an
overfit one falls off a cliff. Plot it.

> **Gate B passes when** DSR is positive and meaningful, PBO < 0.5, the strategy beats
> buy-and-hold Nifty after costs, and performance is not confined to one regime or sector.
>
> **KILL CONDITION:** if it fails, do *not* start tuning parameters to make it pass. That is
> precisely the behaviour PBO is designed to catch, and each attempt makes the next result
> less trustworthy. Go back to hypothesis generation (§3) or stop.

### Gate C — Does it work on time you've never seen? (3–6 months, mostly waiting)

Backtests are retrospective no matter how careful. Forward testing is not.

**C1. Lock everything.** Freeze the config, tag the commit, write the expected performance
envelope down *before* you start. This is pre-registration and it is the single highest-value
discipline in the whole document.

**C2. Automated daily paper runs.** Scheduled job → scan → record every signal to the paper
book → mark positions against the next day's candles. Your ledger already supports this; it
needs a scheduler and a job script.

**C3. Live-vs-backtest divergence monitor.** Track realised fills against modelled fills,
realised win rate against the backtest's confidence interval. **Divergence is the alarm that
your backtest was lying**, and it is far cheaper to learn this on paper.

**C4. Let the Wilson interval fill.** Your `hit_rate` needs 30 trades per bucket. At 200–400
trades/year you get meaningful buckets after ~6 months. There is no shortcut; this is the
price of knowing.

> **Gate C passes when** 3–6 months of forward paper trading sits inside the envelope you
> wrote down in advance, with no systematic fill divergence.

### Gate D — Can you actually execute it? (2–4 weekends)

**Only now** is broker integration worth building.

**D1. Understand the regulatory position.** SEBI's retail algo framework (circular
`SEBI/HO/MIRSD/MIRSD-PoD/P/2025/0000013`, Feb 2025) reached full broker compliance around
**April 2026** after several deferrals, so it governs you today. What the secondary sources
consistently report:
- Algos must run through a **registered broker's API**; providers cannot connect directly
  to exchanges, and the broker is accountable for algos on its platform.
- **Static IP whitelisting**, client-specific and vendor-specific API keys, and 2FA.
- A **~10 orders/second** threshold separates a "regular API user" from a registered algo.
  Your strategy is far below this — a meaningful advantage, as staying under it avoids the
  heavier registration burden.
- Black-box strategies may attract a Research Analyst licensing requirement.

**Verify all of this against the primary SEBI circular and your broker's current
implementation note before writing integration code.** The secondary sources disagree on
details, and this changed repeatedly through 2025–26.

**D2. Broker adapter behind an interface.** `BaseBrokerClient` with instruments, quotes,
candles, order placement, order status, positions. Implement one broker; keep Yahoo as a
*research-only* source that can never satisfy a live gate. Kite Connect is the most mature
(~₹500–2,000/month, verify current pricing); Dhan and Upstox are cheaper alternatives.
Median API latency ~99–124ms — irrelevant at your frequency, which is itself a sign you've
picked a sane frequency.

**D3. Order lifecycle.** Idempotent client order IDs, accepted/partial/filled/cancelled/
rejected states, reconciliation on restart, and a **hard daily order cap** as a circuit
breaker. Your paper book's append-only design extends naturally here.

**D4. Manual-confirm mode first.** The system proposes; you click. Run this for at least a
month before considering anything automatic. **An LLM must never be in the order path** —
keep the deterministic separation you already have.

### Gate E — Can you operate it safely? (ongoing)

**E1. Kill switches** at three levels: per-strategy, per-day loss, and global manual.
`AccountRisk.kill_switch` exists — wire it to something real.
**E2. Feed health monitoring.** Stale quotes, missing bars, clock skew → block new entries
while leaving exit management alive.
**E3. Alerting** with deduplication; an alert you learn to ignore is worse than none.
**E4. Secrets** in the OS credential store, never `.env`. Never log credentials.
**E5. Reconciliation** against the broker's own positions every morning. Trust their book
over yours.
**E6. Audit trail** sufficient to reconstruct any decision months later.

### Gate F — Capital, very slowly

Smallest viable size for 3 months. Then only scale if live matches paper, which it usually
won't at first because of fills. Scale in steps with a written rule, not by feel. Never
size up after a win streak.

---

## 3. If Gate B fails — better hypotheses

The current strategy is a momentum/breakout gate with volume confirmation. It is reasonable
and widely known, which is also the problem: **widely-known edges are usually arbitraged
thin.** If Gate B kills it, these are better-motivated directions than parameter tuning,
because each has a structural reason to exist:

- **Cross-sectional rather than time-series.** Rank all 82 names daily and trade the
  spread (long top decile, flat/short bottom) rather than each name against itself.
  Cross-sectional momentum has survived out-of-sample better than absolute momentum, and it
  neutralises market beta — which also fixes your "everything is below SMA200" silence.
- **Volatility targeting.** Size by inverse realised volatility rather than fixed fractional
  risk. Often improves risk-adjusted returns more than any entry-signal change.
- **Earnings/event drift**, using announcement dates. A documented structural effect with a
  behavioural cause, not a chart pattern.
- **Deliberately slower horizons.** Weekly rebalancing cuts cost drag by ~5× and competes
  with fewer sophisticated participants.
- **Regime filter as a first-class input** rather than a veto, e.g. trade momentum only when
  market breadth confirms.

Whatever you pick: **form the hypothesis first, state why the edge should exist and who is
on the other side of the trade, then test it once.** Mining the data for what worked is how
PBO goes to 1.

---

## 4. Hard ceilings you should know about now

**Point-in-time fundamentals are not available at retail prices in India.** NSE's Corporate
Data product runs ~₹10 lakh/year over a leased line; even the EOD announcements feed is
~₹5 lakh/year. Open-source scrapers exist but none handle filing-date-aware data correctly,
and point-in-time is exactly the part that matters — using today's financials to backtest
2022 is a look-ahead bias that will manufacture a beautiful, entirely fake edge.

**Consequence: genuine fundamental or long-term value investing is off the table for this
project at a retail budget.** That is not a gap to fill later; it is a boundary. Point this
system at technical/price-based strategies, where your data is adequate, and do long-term
investing through a different process entirely.

**Other ceilings:** no order-book depth (so no microstructure strategies), no tick data, no
reliable borrow availability for shorts, and retail latency/queue priority that rules out
anything speed-sensitive.

---

## 5. What to delete

Carrying dead weight costs you clarity on every future change.

| Component | Verdict |
|---|---|
| `ModelComparator.evaluate_models` | **Delete.** Three structural defects; a replacement shares no code. Leaving it importable risks someone overwriting `best_model.pkl`. |
| `deep_learning_predictor.py` (Bi-LSTM) | **Delete or archive.** Trained with no holdout; its "confidence" is a formula. Sequence models need far more data and discipline than this has. |
| `pricing_engine.py` Monte-Carlo "fair value" | **Delete.** Hardcoded unvalidated jump parameters; blends incommensurable quantities into a number presented as fair value. |
| LangGraph agents + RAG | **Keep as commentary, never as input.** An LLM paraphrasing Murphy and Nison cannot validate a strategy. Useful for explaining a signal to yourself; harmful if it touches the decision. |
| `model_trainer.py` | **Delete.** Superseded. |

Rebuilding the tabular ML (Gate B+) is only worth it after the rule strategy has a verified
edge — it should improve a working strategy, never rescue a broken one. When you do: scale-
free features only (never absolute price in a cross-sectional panel), date-based purged
splits, calibration fitted inside folds, an untouched final test period, and honest abstention.

---

## 6. Suggested order and effort

| Phase | Work | Effort | Kills the project? |
|---|---|---|---|
| 1 | Corporate actions + exchange calendar | 1–2 weekends | No, but blocks everything |
| 2 | Portfolio backtest of the real strategy | 2 weekends | No |
| 3 | **Benchmarks + DSR + PBO** | 2–3 weekends | **Yes — most likely here** |
| 4 | Robustness/regime slices | 1 weekend | **Yes** |
| 5 | Forward paper trading, pre-registered | 3–6 months elapsed, low effort | **Yes** |
| 6 | Broker adapter, manual-confirm | 2–4 weekends | No |
| 7 | Ops: kill switches, monitoring, reconciliation | 2 weekends | No |
| 8 | Small capital, slow scaling | ongoing | — |

**Roughly 3 weekends to the decisive answer** (phases 1–3), then mostly waiting. Running
costs are modest: ₹500–2,000/month for a broker API, and your own time.

Phase 3 is the fork in the road. Everything before it is cheap; everything after it is only
worth doing if phase 3 says yes.

---

## 7. How to know it's working — and when to stop

**Working:** DSR positive after honest trial counting; PBO < 0.5; beats buy-and-hold Nifty
after costs; performance distributed across years, regimes and sectors; forward paper
results inside a pre-registered envelope; live fills matching modelled fills.

**Stop, or go back to hypotheses:**
- You find yourself adjusting a threshold to make a test pass.
- Performance depends on one year, one sector, or a handful of trades.
- You cannot explain *why* the edge should exist or who is losing money to you.
- Live fills diverge from modelled fills and you "adjust slippage" to reconcile them.
- You want to skip Gate C because the backtest looks good.

That last one is the most dangerous, and it is the most common.

---

## 8. Immediate next three things

1. **Corporate-action adjustment** (Gate A1). Everything else is unreliable until this
   exists, and it is the cheapest high-impact fix available.
2. **Portfolio backtest of the composite strategy** (Gate A2) — you have genuinely never
   tested the thing you built.
3. **Buy-and-hold Nifty benchmark after costs** (Gate B1). One afternoon, and it reframes
   every subsequent decision.

---

## Sources

Regulation and brokers:
- [SEBI algo trading framework overview, QuantInsti](https://www.quantinsti.com/articles/algorithmic-trading-india/)
- [Sebi extends retail algo trading rollout, full compliance by April 2026, Business Standard](https://www.business-standard.com/markets/news/sebi-extends-retail-algo-trading-framework-rollout-to-2026-125093000956_1.html)
- [SEBI extends retail algo trading deadline, Outlook Business](https://www.outlookbusiness.com/markets/sebi-extends-deadline-to-implement-retail-algo-trading-by-april-2026)
- [SEBI algo trading framework broker compliance guide, Mutwo Labs](https://mutwolabs.com/blog/sebi-algo-trading-framework-2026-broker-guide)
- [Best brokers for algo trading in India, AlgoTest](https://algotest.in/blog/best-brokers-for-algo-trading-in-india/)
- [Kite Connect vs Dhan API comparison, Multibagg](https://www.multibagg.ai/market-pulse/articles/zerodha-nubra-dhan-api-comparison-cmp5l0ayn000ct70j0frlhjln)

Validation methodology:
- [Backtest overfitting and the deflated Sharpe ratio](https://www.tradinginterview.com/courses/machine-learning/lessons/backtest-overfitting-and-the-deflated-sharpe-ratio/)
- [Backtesting and research methodology, Trading Strategy](https://tradingstrategy.ai/docs/learn/backtesting.html)
- [Why a backtest Sharpe of 4 becomes 0.5 live](https://www.techinterview.org/post/3233477314/why-backtest-sharpe-collapses-live/?format=md)

Costs, slippage and failure modes:
- [Why algorithmic trading fails: the math behind losses](https://alexbobes.com/tech/the-brutal-truth-about-algorithmic-trading/)
- [Algorithmic trading strategy: why backtests often fail](https://linetrades.com/articles/algorithmic-trading-strategy-why-live/)
- [Common algorithmic trading mistakes, TrueData](https://www.truedata.in/blog/algorithmic-trading-mistakes-to-avoid)

Data availability:
- [NSE corporate data subscription](https://www.nseindia.com/market-data/corporate-data-subscription)
- [Databento corporate actions](https://databento.com/corporate-actions)

Most regulatory and cost figures above come from secondary sources that disagree on
details. Verify against the primary SEBI circular and your broker's current documentation
before relying on any of them.
