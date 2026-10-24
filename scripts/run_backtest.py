"""Gate A + Gate B in one run: clean, backtest, benchmark, and test for overfitting.

    python -m scripts.run_backtest                 # backtest + benchmarks
    python -m scripts.run_backtest --grid          # also run the variant grid, DSR and PBO
    python -m scripts.run_backtest --no-adjust     # compare against unadjusted prices

Writes a JSON report to data/backtest_report.json.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

import pandas as pd
import sqlite3

from src.backtesting.portfolio import BacktestConfig, prepare_panel, run_backtest
from src.backtesting.validation import (buy_and_hold_equal_weight, compare,
                                        deflated_sharpe_ratio, index_buy_and_hold,
                                        probability_of_backtest_overfitting, sharpe_ratio)
from src.data_pipeline.data_quality import clean_panel
from src.paths import DATA_DIR, DB_PATH
from src.quant_engine.decision_engine import SignalPolicy
from src.quant_engine.research import EquityCosts
from src.signal_engine.composite import SignalConfig

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_panel() -> dict:
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT ticker, timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv",
        connection)
    connection.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return {ticker: group.drop(columns=["ticker"]).sort_values("timestamp").reset_index(drop=True)
            for ticker, group in frame.groupby("ticker")}


def fetch_index(symbol: str = "^NSEI") -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(symbol, period="6y", interval="1d", progress=False,
                          auto_adjust=False, timeout=30)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        stamp = "date" if "date" in raw.columns else "datetime"
        return raw.rename(columns={stamp: "timestamp"})[["timestamp", "close"]]
    except Exception as exc:
        print(f"  index fetch failed ({symbol}): {exc}")
        return None


def headline(title: str):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=200000.0)
    parser.add_argument("--grid", action="store_true", help="Run the variant grid, DSR and PBO")
    parser.add_argument("--no-adjust", action="store_true",
                        help="Skip corporate-action adjustment (to measure its effect)")
    parser.add_argument("--blocks", type=int, default=10, help="CSCV blocks (even)")
    args = parser.parse_args()
    report = {"generated": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
              "capital": args.capital, "adjusted": not args.no_adjust}

    headline("1. DATA QUALITY")
    panel = load_panel()
    cleaned, calendar, summary, _ = clean_panel(panel, adjust=not args.no_adjust)
    print(json.dumps(summary, indent=2))
    report["data_quality"] = summary

    headline("2. PORTFOLIO BACKTEST — the strategy the app actually emits")
    started = time.time()
    prepared = prepare_panel(cleaned, intraday=False)
    config = BacktestConfig(capital=args.capital, costs=EquityCosts(intraday=False),
                            intraday=False)
    result = run_backtest(prepared, calendar, config)
    print(f"  {len(prepared)} tickers, {len(calendar)} trading days, "
          f"{time.time() - started:.0f}s")
    print(json.dumps(result.metrics, indent=2, default=str))
    report["strategy"] = result.metrics
    report["strategy_config"] = result.config

    headline("3. BENCHMARKS — after the same costs")
    costs = EquityCosts(intraday=False)
    curves = {"Strategy": result.equity[["date", "equity"]]}
    curves["Buy & hold watchlist (equal weight)"] = buy_and_hold_equal_weight(
        cleaned, calendar, args.capital, costs)
    index = fetch_index("^NSEI")
    if index is not None:
        curves["Buy & hold Nifty 50"] = index_buy_and_hold(index, calendar, args.capital)
    dates = sorted(calendar)
    curves["Cash (no trades)"] = pd.DataFrame({"date": dates,
                                               "equity": [args.capital] * len(dates)})
    table = compare(curves, args.capital)
    print(table.to_string(index=False))
    report["benchmarks"] = json.loads(table.to_json(orient="records"))

    strategy_cagr = table.loc[table.strategy == "Strategy", "cagr_pct"].iloc[0]
    beaten = [row.strategy for row in table.itertuples()
              if row.strategy != "Strategy" and (row.cagr_pct or -99) > strategy_cagr]
    report["beaten_by"] = beaten
    print(f"\n  VERDICT: the strategy is beaten by {len(beaten)} benchmark(s): "
          f"{beaten if beaten else 'none'}")

    if args.grid:
        headline("4. VARIANT GRID — every variant counts as a trial")
        variants, returns = {}, {}
        grid = [(mc, rp) for mc in (45.0, 50.0, 55.0, 60.0) for rp in (0.25, 0.5, 1.0)]
        for index_, (min_conviction, risk_pct) in enumerate(grid, start=1):
            variant_config = BacktestConfig(
                capital=args.capital, costs=costs, intraday=False,
                policy=SignalPolicy(risk_pct=risk_pct),
                signal=SignalConfig(min_conviction=min_conviction))
            variant = run_backtest(prepared, calendar, variant_config)
            label = f"mc{min_conviction:.0f}_r{risk_pct}"
            variants[label] = variant.metrics
            if not variant.equity.empty:
                series = variant.equity.set_index("date")["equity"].pct_change().dropna()
                returns[label] = series
            print(f"  [{index_:2d}/{len(grid)}] {label:14s} "
                  f"CAGR {variant.metrics.get('cagr_pct', 0):6.2f}%  "
                  f"Sharpe {variant.metrics.get('sharpe') or 0:6.3f}  "
                  f"trades {variant.metrics.get('trades', 0):4d}")
        report["variants"] = variants

        headline("5. DEFLATED SHARPE RATIO")
        matrix = pd.DataFrame(returns).dropna(how="all")
        trial_sharpes = [sharpe_ratio(matrix[c]) for c in matrix.columns]
        base = result.equity.set_index("date")["equity"].pct_change().dropna()
        deflated = deflated_sharpe_ratio(base, n_trials=len(grid), trial_sharpes=trial_sharpes)
        print(json.dumps(deflated.to_dict(), indent=2))
        report["deflated_sharpe"] = deflated.to_dict()

        headline("6. PROBABILITY OF BACKTEST OVERFITTING (CSCV)")
        try:
            pbo = probability_of_backtest_overfitting(matrix, n_blocks=args.blocks)
            print(json.dumps(pbo.to_dict(), indent=2))
            report["pbo"] = pbo.to_dict()
        except ValueError as exc:
            print(f"  PBO unavailable: {exc}")
            report["pbo"] = {"error": str(exc)}

    headline("7. ROBUSTNESS — performance by year")
    if not result.trades.empty:
        trades = result.trades.copy()
        trades["year"] = pd.to_datetime(trades["exit_date"]).dt.year
        yearly = trades.groupby("year").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda s: (s > 0).mean() * 100)).round(2)
        print(yearly.to_string())
        report["by_year"] = json.loads(yearly.to_json(orient="index"))
        by_grade = trades.groupby("grade").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda s: (s > 0).mean() * 100)).round(2)
        print("\nBy setup grade:")
        print(by_grade.to_string())
        report["by_grade"] = json.loads(by_grade.to_json(orient="index"))
    else:
        print("  no trades")

    path = DATA_DIR / "backtest_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
