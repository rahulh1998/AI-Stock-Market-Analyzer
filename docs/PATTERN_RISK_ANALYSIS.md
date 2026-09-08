# Automatic pattern, price-zone and risk analysis

## Scope and use

Open the default research workspace, select an instrument and candle interval,
and fetch candles. Analysis runs automatically on completed observations on each
Streamlit rerun. Fetching is user-triggered, not background streaming. Reports
carry an observation time, review deadline, reasons, and JSON export. Always
refresh before interpreting a scenario; an idle browser does not expire itself.

The supported intervals are 5 minutes, 15 minutes, 1 hour and daily. Intraday
requires 60 completed bars; daily requires 200. This is a conditional research
decision engine, not an optimizer that knows today's high/low or best execution
price in advance. Every report has `execution_eligible=False`.

## Pattern knowledge implemented

Fourteen explicit definitions, not a promise to recognize every chart formation:

- Bullish/bearish engulfing: opposite bodies, full body engulfment, larger current
  body and preceding directional context. Intraday pairs must share a date.
- Hammer/shooting star: directional context, long rejection wick >= 2 times body,
  short opposing wick <= 0.3 times body, nontrivial body > 5% of range.
- Morning/evening three-bar reversals: small middle body and recovery beyond the
  first body midpoint. Gap-optional variants, not strict classical gapped stars.
- Doji (body <= 10% of nonzero range) and inside bar: neutral observations only.
- Range breakout/breakdown: close beyond the PREVIOUS 20 highs/lows.
- Session VWAP reclaim/rejection: same-date cross of cumulative VWAP; intraday only.
- Double-bottom/top neckline breaks: last two strictly confirmed swing extrema,
  5–60 bars apart, within 0.5 ATR of one another, age <= 100 bars, neckline cross.
  Pivot confirmation requires two subsequent bars. No backdating the signal.

Multi-bar context, rolling levels and swing structures can span dates except
where expressly constrained above. They are bar-based heuristics, not exchange-
session-aware pattern research. Recent pattern history shows the last 20 bars.

## Indicator and entry logic

The pattern engine computes correctly seeded Wilder averages for ATR, RSI and
ADX, preserving warm-up NaNs. Wilder update is `(previous*(n-1)+value)/n` after
an n-observation arithmetic seed. Flat RSI is 50; all gains/no losses is 100.
EMA9/21, SMA200, prior-volume relative volume and cumulative session VWAP reuse
causal research features. The separate baseline replay still uses SMA ATR.

A conditional long candidate requires:

1. At least one supported bullish observation and no conflicting bearish one.
2. EMA9 > EMA21, ADX >= 20, RSI in [40,75], relative volume >= 1.25.
3. Intraday close > session VWAP; daily close > warmed-up SMA200.
4. Data and account risk checks passing, and net reward/risk >= 1.5 after costs.

These thresholds are disclosed conventions, not optimized or proven edges. A
reversal in a downtrend may deliberately fail confirmation. Pattern agreement
is not converted into a fabricated probability.

## Conditional prices and sizing

- Buy trigger: current completed high + one instrument tick, rounded up.
- Entry ceiling: trigger + 0.25 Wilder ATR, rounded up. Do not chase above it.
- Stop: smaller of candle low - 0.1 ATR and trigger - 1.5 ATR, rounded down.
- Reject nonpositive stops or stop distances > 4% intraday / 12% daily.
- Initial target: worst modeled entry + 3 times price risk. Cap below available
  prior 20/60-bar resistance and most recent confirmed swing high above trigger.
- Model worst entry at the ceiling plus adverse slippage; model stop/target sells
  with adverse slippage. Include illustrative product-specific fees on both legs.
- Quantity: whole shares constrained by cash, allocation, remaining portfolio
  gross exposure and remaining modeled open risk. Cost-aware binary-search sizing.
- Reject if target proceeds net of fees do not meet minimum reward/risk.

Targets/stops are scenarios, not orders. A trigger can fail, gap through the
ceiling or become obsolete. Recompute each completed bar, never widen a stop
automatically or average down. Gap/circuit/liquidity losses can exceed modeled risk.

## Account and data controls

Manual account snapshot: current equity (research capital), uncommitted cash,
day-start equity, peak equity, gross exposure including commitments, open stop
risk including commitments, losing streak, and kill switch. Defaults assume no
positions; the app cannot independently verify them.

Default limits: 0.5% new-trade risk, 25% allocation, 2% combined open risk, 80%
gross exposure, 2% daily equity loss, 10% drawdown, three consecutive losses.
Daily loss includes unrealized P&L via supplied equity. Locks only block new
research buy plans; they do not liquidate positions or persist an account ledger.

Data checks reject malformed/duplicate/out-of-order/future observations; exclude
incomplete bars; block stale observations, irregular intervals in the latest
observed intraday date, recent zero volume, and >20% recent close discontinuities.
Intraday freshness expires at the next expected completed-bar boundary; daily
freshness uses a coarse seven-day allowance after completion, NOT a verified
exchange calendar. A daily report's deadline is a data-review bound, not a
seven-day guaranteed entry opportunity.

States: CONDITIONAL_BUY, WAIT, EXIT_REVIEW, HISTORICAL_ONLY, INVALID_DATA.
EXIT_REVIEW flags bearish observations for existing longs; it never opens shorts.
Data problems suppress price plans. Account kill switches do not suppress warnings
to review existing holdings.

## Historical-neighbor ML

Uses existing numpy/pandas, with six shared causal features: EMA spread/ATR, RSI,
ADX, relative volume, three-bar return and ATR/close. Standardization is fitted
only on preceding examples; outcomes must mature strictly before each query.
Neighbor outcome windows do not overlap. Up to 30 nearest examples are selected
from up to 1000 eligible non-overlapping history examples.

Targets: terminal return, maximum high and minimum low measured from NEXT bar
open over 3 intraday bars or 5 daily bars. Intraday labels cannot cross date or
interval gaps; interval estimation uses only the prefix known at the query.

Reports require >=40 eligible examples and >=10 chronological evaluation windows
(up to 30). Report median and empirical 10th/90th percentiles, mean absolute
error, zero-return baseline MAE and directional accuracy. A failure to improve
baseline is labeled ABSTAIN_NO_BASELINE_IMPROVEMENT; diagnostic ranges remain
visible. Even a baseline improvement is experimental, not statistical proof.

Ranges are BEFORE costs, not calibrated confidence/prediction intervals. This
is neither an untouched final test nor a net-profitability study. The model never
changes the rule-based signal, levels or risk gates. No LLM/DL is needed to run
this workspace; legacy AI forecasts cannot override it. No unvalidated random
weights or arbitrary confidence score is substituted for missing model evidence.

## Validation and remaining work

Tests cover prefix causality, Wilder seed/recursion, flat/uptrend indicators,
selected pattern definitions, delayed pivots, next-open labels, label maturity,
future-frequency invariance, baseline reporting, cash/portfolio limits, costs,
tick rounding, all account locks, conflict rejection, bearish exit-only status,
incomplete/stale/malformed candles and offline Streamlit integration.

Still required before any live use: backtest THIS pattern strategy with a shared
execution specification (the UI's existing replay is a DIFFERENT EMA baseline),
external indicator vectors, broader pattern fixtures, out-of-sample validation,
broker quotes/spreads/circuits/lot sizes, exchange calendar and square-off logic,
corporate-action adjustment, liquidity/impact limits, durable broker-synced risk,
and operational failure testing. This implementation does not claim all trading
logic is complete, every pattern is detected, or that profitability is established.