"""Benchmarks and overfitting controls — the tests that decide whether a backtest means anything.

A backtest on its own is close to uninformative. Two things make it informative:

**Benchmarks.** A strategy must beat the alternatives you could have had for free, after
costs: holding cash, buying and holding the index, and buying and holding the same
universe equal-weighted. A strategy that loses to buy-and-hold is an expensive index
tracker with extra risk.

**Overfitting controls.** The expected maximum Sharpe of N strategies with *zero* edge
grows about as sqrt(2 ln N) — 2.80 at 50 trials. So a raw Sharpe is meaningless without
knowing how many variants were tried.

- `deflated_sharpe_ratio` (Bailey & Lopez de Prado) asks whether the observed Sharpe beats
  what the luckiest of N skill-free variants would produce, and penalises negative skew
  and fat tails.
- `probability_of_backtest_overfitting` implements Combinatorially Symmetric
  Cross-Validation: it splits performance into blocks, and for every way of halving them
  picks the in-sample best and records that variant's out-of-sample rank. PBO above 0.5
  means the selection procedure does worse than chance.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS = 252.0


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation; ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe_ratio(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))


def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """Expected maximum Sharpe across `n_trials` strategies with zero true edge."""
    if n_trials < 2:
        return 0.0
    scale = math.sqrt(max(sharpe_variance, 0.0))
    left = _normal_ppf(1.0 - 1.0 / n_trials)
    right = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return scale * ((1 - EULER_MASCHERONI) * left + EULER_MASCHERONI * right)


@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe: float
    deflated_sharpe: float
    probability: float
    expected_max_sharpe_under_null: float
    n_trials: int
    observations: int
    skew: float
    kurtosis: float

    @property
    def significant(self) -> bool:
        return self.probability > 0.95

    def to_dict(self) -> dict:
        return {"sharpe_annual": round(self.sharpe, 4),
                "expected_max_sharpe_under_null": round(self.expected_max_sharpe_under_null, 4),
                "deflated_sharpe_statistic": round(self.deflated_sharpe, 4),
                "probability_skill": round(self.probability, 4),
                "significant_at_95pct": self.significant,
                "n_trials": self.n_trials, "observations": self.observations,
                "skew": round(self.skew, 4), "excess_kurtosis": round(self.kurtosis, 4)}


def deflated_sharpe_ratio(returns, n_trials: int, periods_per_year: float = TRADING_DAYS,
                          trial_sharpes=None) -> DeflatedSharpe:
    """Probability the observed Sharpe reflects skill rather than the best of N tries.

    `probability_skill` below ~0.95 means the result is indistinguishable from the luckiest
    of your trials. Count the whole parameter grid as trials — that is the conservative and
    correct choice.
    """
    series = pd.Series(returns).dropna()
    observations = len(series)
    if observations < 10:
        raise ValueError("Need at least 10 return observations")
    periodic_sharpe = series.mean() / series.std(ddof=1) if series.std(ddof=1) > 0 else 0.0
    annual_sharpe = periodic_sharpe * math.sqrt(periods_per_year)
    skew = float(series.skew())
    kurtosis = float(series.kurtosis())      # pandas returns excess kurtosis

    variance = float(np.var(trial_sharpes, ddof=1)) if trial_sharpes is not None and len(trial_sharpes) > 1 else 1.0
    threshold = expected_max_sharpe(n_trials, variance) / math.sqrt(periods_per_year)

    denominator = 1.0 - skew * periodic_sharpe + ((kurtosis + 3.0) - 1.0) / 4.0 * periodic_sharpe ** 2
    if denominator <= 0:
        denominator = 1e-12
    statistic = (periodic_sharpe - threshold) * math.sqrt(observations - 1) / math.sqrt(denominator)
    return DeflatedSharpe(
        sharpe=annual_sharpe, deflated_sharpe=statistic, probability=_normal_cdf(statistic),
        expected_max_sharpe_under_null=threshold * math.sqrt(periods_per_year),
        n_trials=n_trials, observations=observations, skew=skew, kurtosis=kurtosis)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_variants: int
    n_blocks: int
    median_oos_rank: float
    logits: list

    @property
    def overfit(self) -> bool:
        return self.pbo > 0.5

    def to_dict(self) -> dict:
        return {"pbo": round(self.pbo, 4), "overfit_flag": self.overfit,
                "n_splits_evaluated": self.n_splits, "n_variants": self.n_variants,
                "n_blocks": self.n_blocks,
                "median_oos_rank_of_is_best": round(self.median_oos_rank, 4),
                "interpretation": ("Selection is worse than chance; the in-sample winner "
                                   "tends to underperform out of sample."
                                   if self.overfit else
                                   "In-sample winners tend to hold up out of sample.")}


def probability_of_backtest_overfitting(performance: pd.DataFrame, n_blocks: int = 10,
                                        max_splits: int = 2000) -> PBOResult:
    """PBO via Combinatorially Symmetric Cross-Validation.

    `performance` is a (time x variant) frame of per-period returns, one column per
    strategy variant tried. Needs an even `n_blocks`.
    """
    if performance.shape[1] < 2:
        raise ValueError("PBO needs at least two strategy variants")
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be even and at least 4")
    frame = performance.dropna(how="all").fillna(0.0)
    if len(frame) < n_blocks * 2:
        raise ValueError(f"Need at least {n_blocks * 2} observations for {n_blocks} blocks")

    blocks = np.array_split(np.arange(len(frame)), n_blocks)
    half = n_blocks // 2
    logits, ranks = [], []
    for count, chosen in enumerate(combinations(range(n_blocks), half)):
        if count >= max_splits:
            break
        in_sample_rows = np.concatenate([blocks[b] for b in chosen])
        out_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in chosen])
        in_sample, out_sample = frame.iloc[in_sample_rows], frame.iloc[out_rows]

        in_scores = in_sample.mean() / in_sample.std(ddof=1).replace(0, np.nan)
        out_scores = out_sample.mean() / out_sample.std(ddof=1).replace(0, np.nan)
        in_scores, out_scores = in_scores.fillna(-np.inf), out_scores.fillna(-np.inf)
        best = in_scores.idxmax()
        # Relative rank of the IS winner within the OOS distribution.
        relative = out_scores.rank(pct=True)[best]
        relative = min(max(float(relative), 1e-6), 1 - 1e-6)
        ranks.append(relative)
        logits.append(math.log(relative / (1 - relative)))

    if not logits:
        raise ValueError("No splits evaluated")
    return PBOResult(pbo=float(np.mean([l <= 0 for l in logits])), n_splits=len(logits),
                     n_variants=performance.shape[1], n_blocks=n_blocks,
                     median_oos_rank=float(np.median(ranks)), logits=logits)


def buy_and_hold_equal_weight(panel: dict, calendar, capital: float, costs=None) -> pd.DataFrame:
    """Equal-weight buy-and-hold of the universe, bought on the first common date.

    Deliberately charges the same entry/exit fees as the strategy, so the comparison is
    like for like.
    """
    dates = sorted(calendar)
    if not dates:
        return pd.DataFrame(columns=["date", "equity"])
    usable = {t: f for t, f in panel.items() if f is not None and not f.empty}
    if not usable:
        return pd.DataFrame(columns=["date", "equity"])
    per_name = capital / len(usable)
    holdings, spent = {}, 0.0
    for ticker, frame in usable.items():
        series = frame.set_index(frame.timestamp.dt.date)["close"]
        first = next((d for d in dates if d in series.index), None)
        if first is None:
            continue
        price = float(series.loc[first])
        quantity = int(per_name // price) if price > 0 else 0
        if quantity <= 0:
            continue
        fee = costs.fee(price, quantity, "BUY") if costs else 0.0
        holdings[ticker] = (quantity, series)
        spent += price * quantity + fee
    cash = capital - spent
    rows = []
    for date in dates:
        value = cash
        for quantity, series in holdings.values():
            if date in series.index:
                value += float(series.loc[date]) * quantity
            else:
                previous = series.loc[:date]
                if len(previous):
                    value += float(previous.iloc[-1]) * quantity
        rows.append({"date": date, "equity": value})
    return pd.DataFrame(rows)


def index_buy_and_hold(frame: pd.DataFrame, calendar, capital: float) -> pd.DataFrame:
    """Buy-and-hold an index series (fractional units; an index is not directly tradable)."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "equity"])
    series = frame.set_index(pd.to_datetime(frame["timestamp"]).dt.date)["close"]
    dates = [d for d in sorted(calendar) if d in series.index]
    if not dates:
        return pd.DataFrame(columns=["date", "equity"])
    base = float(series.loc[dates[0]])
    return pd.DataFrame([{"date": d, "equity": capital * float(series.loc[d]) / base}
                         for d in dates])


def curve_metrics(curve: pd.DataFrame, capital: float,
                  periods_per_year: float = TRADING_DAYS) -> dict:
    """Return/risk summary for any equity curve, computed identically across benchmarks."""
    if curve is None or curve.empty:
        return {"note": "empty curve"}
    equity = curve.set_index("date")["equity"].astype(float)
    returns = equity.pct_change().dropna()
    peak = equity.cummax().clip(lower=capital)
    years = max(len(equity) / periods_per_year, 1e-9)
    total = equity.iloc[-1] / capital
    cagr = (total ** (1 / years) - 1) * 100 if total > 0 else None
    drawdown = float(((peak - equity) / peak).max() * 100)
    return {
        "final_equity": float(equity.iloc[-1]),
        "net_return_pct": float((total - 1) * 100),
        "cagr_pct": float(cagr) if cagr is not None else None,
        "max_drawdown_pct": drawdown,
        "volatility_annual_pct": float(returns.std(ddof=1) * math.sqrt(periods_per_year) * 100),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "calmar": float(cagr / drawdown) if cagr and drawdown > 0 else None,
        "observations": int(len(equity)),
    }


def compare(curves: dict, capital: float, periods_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """One row per curve, ordered by CAGR. `curves` maps a label to an equity frame."""
    rows = []
    for label, curve in curves.items():
        metrics = curve_metrics(curve, capital, periods_per_year)
        metrics["strategy"] = label
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    columns = ["strategy", "final_equity", "net_return_pct", "cagr_pct",
               "max_drawdown_pct", "volatility_annual_pct", "sharpe", "calmar", "observations"]
    frame = frame[[c for c in columns if c in frame.columns]]
    return frame.sort_values("cagr_pct", ascending=False, na_position="last").reset_index(drop=True)
