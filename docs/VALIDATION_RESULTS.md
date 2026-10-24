# Validation results — Gate A and Gate B

Run 2026-10-24 on 81 NSE tickers, 1,268 trading days (2021-09-02 → 2026-10-23).
Reproduce with `python -m scripts.run_backtest --grid`.

---

## Verdict

**Gate B fails. The strategy has no demonstrated edge and should not be traded with money.**

Three independent tests agree, and none of them is a close call:

| Test | Result | Threshold | Outcome |
|---|---|---|---|
| Beats buy-and-hold after costs | CAGR **1.14%** vs **11.31%** (watchlist) and **5.53%** (Nifty 50) | must beat both | ❌ |
| Deflated Sharpe Ratio | probability of skill **0.62** | > 0.95 | ❌ |
| Probability of Backtest Overfitting | **0.72** | < 0.5 | ❌ |

Per the roadmap's own kill condition, the correct response is **not** to tune parameters
until these pass. That is precisely the behaviour PBO exists to detect, and every attempt
makes the next result less trustworthy.

---

## 1. Gate A — data correctness

Two defects were found and fixed before any performance number was computed. Both would
have corrupted the result.

| Defect | Scale | Fix |
|---|---|---|
| Fabricated bars on non-trading days | **413** flat zero-volume bars across 5 shared dates (2026-05-01 is Maharashtra Day), plus **81** weekend rows | dropped; calendar derived by quorum |
| Unadjusted prices in the signal path | **77 of 81** tickers needed adjustment; 102 corporate actions detected | OHLC back-adjusted, volume inverted |

**The corporate-action detector discriminates rather than just detecting.** Of nine moves
above 20% in this history, the Adani/Hindenburg crash (2023-02-01), the IndusInd
derivatives disclosure (2025-03-11) and the 2024 election-day drop are all *real*. A naive
"large move = corporate action" rule would have destroyed genuine data. The detector keys
off divergence between the raw close ratio and the adjusted close ratio, which only a
share-count or distribution change produces — all three crashes were correctly left alone.

**The adjustment materially changed the answer**, which is why it had to come first:

| | Trades | CAGR | Sharpe |
|---|---|---|---|
| Adjusted | 153 | 1.14% | 0.289 |
| Unadjusted | 143 | 0.72% | 0.195 |

---

## 2. Gate B — the strategy, measured

First portfolio backtest of the strategy the app actually emits. `research.replay` only
ever tested the old EMA/VWAP baseline on a single instrument.

```
trades              153          win rate        27.45%
CAGR                1.14%        profit factor   1.154
max drawdown       10.10%        expectancy      +₹77/trade
Sharpe              0.289        fees paid       ₹7,051
Sortino             0.348        avg hold        46 days
exits: stop 101 · target 41 · gap_stop 10 · sample_end 1
```

Profit factor above 1 with a 27% win rate is internally consistent — the plan targets 3R
capped by resistance, so winners are larger than losers. The strategy is *marginally*
positive. It is simply not competitive.

### Benchmarks, after identical costs

| Strategy | Final equity | CAGR | Max DD | Sharpe |
|---|---|---|---|---|
| Buy & hold watchlist (equal weight) | ₹342,918 | **11.31%** | 17.15% | **0.98** |
| Buy & hold Nifty 50 | ₹262,270 | 5.53% | 17.23% | 0.46 |
| **Strategy** | ₹211,783 | **1.14%** | 10.10% | **0.29** |
| Cash | ₹200,000 | 0.00% | 0.00% | — |

**Anticipating the obvious objection:** the strategy only deploys 22.1% of capital on
average (3.74 concurrent positions; idle 17% of days), so of course its absolute return is
low. That defence does not survive, because **Sharpe is invariant to position scaling** —
levering the strategy up multiplies mean and standard deviation alike. At 0.29 versus 0.98,
it loses on the one metric that deployment cannot explain away. Scaling it up would mostly
scale up the drawdown.

### Overfitting controls

```
Deflated Sharpe Ratio                 Probability of Backtest Overfitting
  annual Sharpe          0.289          PBO                        0.722
  E[max Sharpe | null]   0.151          splits evaluated             252
  probability of skill   0.621          variants                      12
  significant at 95%     NO             median OOS rank of IS best  0.25
  trials                    12          -> worse than chance
  skew                  -0.554
  excess kurtosis        5.850
```

PBO of 0.72 says the variant that looks best in-sample typically lands in the **bottom
quartile** out of sample. Selecting on this backtest is worse than picking at random.

The return distribution is also negatively skewed (−0.55) with heavy tails (excess kurtosis
5.85) — small frequent gains against occasional large losses, which is the shape that makes
a raw Sharpe most misleading.

### The variant grid (all 12 counted as trials)

| min conviction | risk 0.25% | risk 0.5% | risk 1.0% |
|---|---|---|---|
| 45 | 1.19% | 1.55% | **3.04%** |
| 50 | 1.06% | 2.16% | 2.73% |
| 55 | 1.04% | 1.14% | 1.53% |
| 60 | 1.05% | 1.32% | 1.96% |

The best cell (3.04% CAGR) still loses to both benchmarks. Note the temptation it
represents: it is 2.7× the shipped configuration, and selecting it is exactly the move PBO
says will not generalise.

---

## 3. Robustness — where it actually breaks

### By year

| Year | Trades | Net P&L | Win rate |
|---|---|---|---|
| 2022 | 11 | −₹565 | 27.3% |
| 2023 | 40 | +₹12,297 | 35.0% |
| 2024 | 40 | +₹12,521 | 32.5% |
| 2025 | 29 | +₹8,507 | 31.0% |
| **2026** | **33** | **−₹20,977** | **9.1%** |

2026 gave back essentially all of three good years. A strategy whose profit lives in one
regime is a description of that regime, not an edge.

### By setup grade — the most damaging finding

| Grade | Trades | Net P&L | Win rate |
|---|---|---|---|
| **A** | 11 | **−₹5,233** | **9.1%** |
| B | 142 | +₹17,016 | 28.9% |

**The grading is inverted.** Setups graded A — the ones the UI presents most confidently and
the ones a conviction-ranked scanner surfaces first — lost money, while the B setups carried
the entire result. Eleven trades is a small sample and this may be noise, but there is
certainly no evidence that `setup_grade` ranks anything correctly, and the burden of proof
runs the other way.

**This directly undermines the conviction score as a ranking statistic**, which was its only
claimed property. The scanner's default sort may be actively misleading.

---

## 4. Is the backtest itself correct?

A negative result from a broken backtest is worthless, so the machinery was attacked
directly. **53 new tests** (174 total, all passing).

| Property | How it is tested |
|---|---|
| **No look-ahead** | Multiplying every bar after a cutoff by 3× leaves the equity curve before that cutoff bit-identical. Truncating the panel reproduces the earlier trades exactly, field by field. |
| Accounting identity | `equity == cash + holdings` on all 1,268 bars; final equity equals capital plus summed net P&L to 4 decimals. |
| Cash | never negative on any bar |
| Fees | every trade pays both legs; `net_pnl == gross − fees` exactly |
| Risk budget | modelled stop loss stays within the configured budget |
| Position integrity | no overlapping positions in one ticker |
| Fill semantics | fills at the trigger when crossed intrabar, at the open when gapping into the zone, and **not at all** when gapping past the entry ceiling — matching what the plan actually promises |
| Barriers | stop wins when both touch in one bar; gap-stops fill at the worse open |
| Monotonicity | raising the conviction floor never increases trade count |
| Known answers | a flat market produces zero trades; an empty universe is handled |
| **PBO estimator** | returns high PBO on pure-noise variants and **low** PBO when one variant has a genuine edge — it discriminates, it does not merely run |
| **DSR estimator** | rejects pure noise, accepts a real Sharpe-3 series, and falls as trial count rises |

The two statistical estimators are themselves tested against known-answer populations,
because an overfitting detector that always says "overfit" would be useless.

---

## 5. What this does and does not say

**Does say:** this specific strategy — EMA/ADX/volume/SMA200 breakout gating with
1.5×ATR stops and 3R resistance-capped targets, on these 81 NSE names over these 5 years —
did not beat buying and holding the same names, and the evidence that it has any edge is
indistinguishable from the luckiest of 12 tries.

**Does not say:** that no strategy works, that the codebase is wrong, or that the signal
engine is broken. The engine is correct; it faithfully implements a strategy that does not
make money. Those are different statements, and distinguishing them is the entire point of
having built this.

**Also worth stating plainly:** five years is one market cycle, 153 trades is below the ~590
needed to resolve a modest edge statistically, and the universe carries survivorship bias
(today's watchlist applied to history). These limitations cut *against* the strategy here —
a result this weak is not going to be rescued by more data.

---

## 6. What to do next

Per the roadmap, from §3 "If Gate B fails — better hypotheses":

1. **Do not tune the current strategy.** PBO 0.72 means the backtest cannot distinguish a
   good parameter set from a lucky one. More searching makes this worse, not better.
2. **Fix or remove the conviction score.** Grade A losing to grade B says the ranking has no
   demonstrated validity. Until it does, the scanner should not sort by it, and the UI
   should not present it as a confidence-like quantity.
3. **Test a structurally different hypothesis**, with a stated reason the edge should exist
   and who is on the other side. Cross-sectional momentum (rank all 81 names, trade the
   spread) is the strongest candidate: it neutralises market beta, which is also why the
   current absolute-momentum gate falls silent in a downtrend.
4. **Keep the apparatus.** The backtest, benchmarks, DSR and PBO are now reusable. Any
   future hypothesis can be judged in minutes rather than months, and that — not the
   strategy — is the durable asset this work produced.

The honest summary: **three weekends of work replaced a plausible-looking system with a
measured answer.** The answer is no. That is a successful outcome, and a far cheaper one
than discovering it with capital.
