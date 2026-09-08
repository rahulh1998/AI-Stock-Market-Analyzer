"""Small deterministic nearest-neighbor research model using numpy only.

Predicts horizon returns relative to the NEXT bar open, not today's optimal price.
Scaling and neighbors are fit strictly before each query. Labels must mature
before the query; intraday labels cannot cross dates or missing-bar gaps.
Quantiles are empirical neighbor outcomes, NOT calibrated prediction intervals.
"""
import numpy as np
import pandas as pd


FEATURES = ["trend_atr", "rsi", "adx", "relative_volume", "return_3", "atr_fraction"]


def outcome_labels(df: pd.DataFrame, horizon: int, intraday: bool) -> np.ndarray:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Positive integer horizon required")
    values = np.full((len(df), 3), np.nan)
    dates = df.timestamp.dt.date
    # Infer interval only from the prefix available at each query. Future changes
    # in feed frequency must not change historical labels/evaluation eligibility.
    diffs = df.timestamp.diff()
    positive = diffs[(dates == dates.shift()) & (diffs > pd.Timedelta(0))]
    for i in range(len(df) - horizon):
        end = i + horizon
        observed = positive.loc[positive.index <= df.index[i]]
        duration = observed.mode().iloc[0] if len(observed) else None
        if intraday and (duration is None or dates.iloc[i] != dates.iloc[end]
                         or not (diffs.iloc[i + 1:end + 1] == duration).all()):
            continue
        entry = float(df.open.iloc[i + 1])
        future = df.iloc[i + 1:end + 1]
        values[i] = [future.close.iloc[-1] / entry - 1,
                     future.high.max() / entry - 1, future.low.min() / entry - 1]
    return values


def _neighbors(x: np.ndarray, y: np.ndarray, query: int, horizon: int, k: int):
    indices = np.arange(len(x))
    eligible = indices[(indices + horizon < query) & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)]
    # Use non-overlapping outcome windows to reduce repeated evidence counting.
    chosen, last = [], query
    for index in eligible[::-1]:
        if last - index > horizon:
            chosen.append(index)
            last = index
        if len(chosen) == 1000:
            break
    train = np.asarray(chosen[::-1], dtype=int)
    if len(train) < 40 or not np.isfinite(x[query]).all():
        return None
    center = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-9] = 1.0
    distance = np.sum(((x[train] - center) / scale - (x[query] - center) / scale) ** 2, axis=1)
    selected = train[np.argsort(distance, kind="stable")[:k]]
    return selected


def historical_outlook(df: pd.DataFrame, horizon: int = 3, intraday: bool = True) -> dict:
    result = {"available": False, "method": "standardized historical nearest neighbors",
              "horizon_bars": horizon, "reason": "Insufficient matured, non-overlapping history",
              "warning": "Empirical ranges are uncalibrated, before costs, and not optimal buy/sell prices."}
    if len(df) < 200 or not set(FEATURES).issubset(df.columns):
        return result
    x = df[FEATURES].to_numpy(dtype=float)
    y = outcome_labels(df, horizon, intraday)
    latest = len(df) - 1
    neighbors = _neighbors(x, y, latest, horizon, 30)
    if neighbors is None:
        return result
    errors, baseline, directions = [], [], []
    evaluation_indices = []
    # Up to 30 distinct, matured evaluation windows. Every fit is expanding-window.
    for query in range(latest - horizon - 1, 100, -(horizon + 1)):
        if not np.isfinite(y[query]).all():
            continue
        selected = _neighbors(x, y, query, horizon, 30)
        if selected is None:
            continue
        prediction = float(np.median(y[selected, 0]))
        actual = float(y[query, 0])
        errors.append(abs(prediction - actual))
        baseline.append(abs(actual))
        directions.append((prediction > 0) == (actual > 0))
        evaluation_indices.append(query)
        if len(errors) == 30:
            break
    if len(errors) < 10:
        result["reason"] = "Fewer than 10 chronological evaluation windows; model abstains"
        return result
    quantiles = np.quantile(y[neighbors], [0.1, 0.5, 0.9], axis=0) * 100
    mae, baseline_mae = float(np.mean(errors)), float(np.mean(baseline))
    beats_baseline = mae < baseline_mae
    result.update(available=True, reason=None, neighbors=len(neighbors),
                  validation_status="EXPERIMENTAL_BASELINE_BEATEN" if beats_baseline else "ABSTAIN_NO_BASELINE_IMPROVEMENT",
                  median_return_pct=float(quantiles[1, 0]),
                  terminal_return_p10_p90_pct=[float(quantiles[0, 0]), float(quantiles[2, 0])],
                  median_maximum_upside_pct=float(quantiles[1, 1]),
                  median_maximum_downside_pct=float(quantiles[1, 2]),
                  evaluation_windows=len(errors), mae_pct=mae * 100, flat_baseline_mae_pct=baseline_mae * 100,
                  directional_accuracy_pct=float(np.mean(directions) * 100),
                  latest_neighbor_label_end=df.timestamp.iloc[int(neighbors.max()) + horizon].isoformat(),
                  evaluation_start=df.timestamp.iloc[min(evaluation_indices)].isoformat(),
                  evaluation_end=df.timestamp.iloc[max(evaluation_indices)].isoformat(),
                  note="Diagnostic walk-forward evaluation, not an untouched final test or evidence of net profitability. "
                       "ML never overrides deterministic signal/risk gates.")
    return result