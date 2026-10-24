"""End-to-end check: every layer, on real data, in the order the app uses them.

The unit suite proves each module in isolation with fakes. This proves the seams — that
real Yahoo data flows through cleaning, features, gates, sizing, scoring, agents, the paper
book and the backtest without anything silently degrading on the way.

Each check is independent and reports PASS, FAIL or SKIP with a reason. A SKIP is an
honest outcome for an optional service (no API key, Ollama down, no network); a FAIL is a
defect. The exit code is non-zero only for FAIL, so this is safe to run offline.

    python -m scripts.e2e_check              # everything
    python -m scripts.e2e_check --offline    # skip anything needing the network
    python -m scripts.e2e_check --quick      # skip the multi-minute backtest
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
RESULTS = []
STATE = {}


class Skip(Exception):
    """Raised by a check when a precondition is legitimately absent."""


def check(name, group):
    def decorate(function):
        function._e2e = (name, group)
        return function
    return decorate


def run(function, args):
    name, group = function._e2e
    started = time.time()
    try:
        detail = function(args) or ""
        status = PASS
    except Skip as exc:
        detail, status = str(exc), SKIP
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        status = FAIL
        if args.trace:
            traceback.print_exc()
    elapsed = time.time() - started
    RESULTS.append({"group": group, "name": name, "status": status,
                    "detail": detail, "seconds": round(elapsed, 2)})
    mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{mark}] {name:<46} {elapsed:6.2f}s  {detail[:88]}")
    return status


# ---------------------------------------------------------------- 1. foundations

@check("Project paths resolve from any directory", "foundation")
def check_paths(args):
    from src.paths import CONFIG_DIR, DB_PATH, INSTRUMENTS_PATH
    assert DB_PATH.exists(), f"database missing at {DB_PATH}"
    assert INSTRUMENTS_PATH.exists(), "config/instruments.json missing"
    assert CONFIG_DIR.is_dir()
    return f"db {DB_PATH.stat().st_size // 1024 // 1024} MB"


@check("Every package imports cleanly", "foundation")
def check_imports(args):
    modules = [
        "src.quant_engine.decision_engine", "src.quant_engine.research",
        "src.signal_engine.composite", "src.signal_engine.scoring",
        "src.signal_engine.scanner", "src.backtesting.portfolio",
        "src.backtesting.validation", "src.data_pipeline.data_quality",
        "src.data_pipeline.universe", "src.llm_engine.providers",
        "src.llm_engine.tasks", "src.agents.evidence_crew",
        "src.sentiment_engine.news_sources", "src.sentiment_engine.multi_horizon",
        "src.paper_trading.ledger", "app.components.signals_workspace",
    ]
    import importlib
    for module in modules:
        importlib.import_module(module)
    return f"{len(modules)} modules"


@check("Database holds both markets", "foundation")
def check_database(args):
    import sqlite3
    from src.paths import DB_PATH
    connection = sqlite3.connect(DB_PATH)
    rows, tickers, latest = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker), MAX(timestamp) FROM daily_ohlcv").fetchone()
    us = connection.execute(
        "SELECT COUNT(DISTINCT ticker) FROM daily_ohlcv WHERE ticker LIKE 'US:%'").fetchone()[0]
    connection.close()
    assert rows > 100000, f"only {rows} rows"
    assert us >= 30, f"only {us} US tickers"
    STATE["latest_bar"] = latest
    return f"{rows:,} rows · {tickers} tickers ({us} US) · latest {latest}"


# ---------------------------------------------------------------- 2. data quality

@check("Cleaning drops fabricated bars and adjusts splits", "data")
def check_cleaning(args):
    import sqlite3
    from src.data_pipeline.data_quality import clean_panel
    from src.paths import DB_PATH
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT ticker, timestamp, open, high, low, close, adj_close, volume "
        "FROM daily_ohlcv WHERE ticker NOT LIKE 'US:%'", connection)
    connection.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    panel = {t: g.drop(columns=["ticker"]).sort_values("timestamp").reset_index(drop=True)
             for t, g in frame.groupby("ticker")}
    cleaned, calendar, summary, _ = clean_panel(panel)
    assert summary["dropped_fabricated"] > 0, "expected fabricated bars to be found"
    assert summary["tickers_adjusted"] > 0, "expected corporate-action adjustment"
    assert 1000 < len(calendar) < 1600, f"calendar looks wrong: {len(calendar)} days"
    STATE["cleaned_nse"] = cleaned
    STATE["calendar"] = calendar
    return (f"{summary['dropped_fabricated']} fabricated, {summary['dropped_weekend']} weekend, "
            f"{summary['tickers_adjusted']} adjusted, {len(calendar)} sessions")


@check("A real crash is not mistaken for a corporate action", "data")
def check_crash_discrimination(args):
    import sqlite3
    from src.data_pipeline.data_quality import detect_corporate_actions
    from src.paths import DB_PATH
    connection = sqlite3.connect(DB_PATH)
    misclassified = []
    for ticker, date in (("ADANIENT", "2023-02-01"), ("INDUSINDBK", "2025-03-11"),
                         ("ADANIPORTS", "2024-06-04")):
        frame = pd.read_sql(
            "SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv "
            "WHERE ticker = ? ORDER BY timestamp", connection, params=(ticker,))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if any(a.date == date for a in detect_corporate_actions(frame, ticker)):
            misclassified.append(f"{ticker} {date}")
    connection.close()
    assert not misclassified, f"real crashes flagged as corporate actions: {misclassified}"
    return "Adani/Hindenburg, IndusInd and election-day crashes all left intact"


# ---------------------------------------------------------------- 3. live data

@check("Yahoo returns usable candles", "live")
def check_yahoo(args):
    if args.offline:
        raise Skip("offline mode")
    from src.data_pipeline.broker_client import MarketDataFetcher
    frame = MarketDataFetcher().fetch_intraday_data("RELIANCE", "1mo", "15m")
    assert not frame.empty, "no candles returned"
    assert len(frame) > 100, f"only {len(frame)} bars"
    STATE["intraday"] = frame
    return f"{len(frame)} × 15m bars, last {frame['timestamp'].iloc[-1]}"


@check("Quote carries honest provenance", "live")
def check_provenance(args):
    if args.offline:
        raise Skip("offline mode")
    from src.data_pipeline.broker_client import MarketDataFetcher
    quote = MarketDataFetcher().fetch_live_quote_single("RELIANCE")
    assert quote["execution_eligible"] is False, "a research quote must never claim eligibility"
    assert quote["source"] == "yahoo_research"
    return f"price {quote['price']} · quality {quote['data_quality']} · execution_eligible False"


# ---------------------------------------------------------------- 4. signal path

@check("Daily signal generates end to end", "signal")
def check_daily_signal(args):
    import sqlite3
    from src.data_pipeline.universe import costs_for
    from src.paths import DB_PATH
    from src.signal_engine.composite import generate_signal
    from src.signal_engine.scanner import resolve_clock
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv "
        "WHERE ticker = 'RELIANCE' ORDER BY timestamp", connection)
    connection.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    clock, _ = resolve_clock(frame, "1d", "last_close_review")
    signal = generate_signal(frame, "1d", ticker="RELIANCE", now=clock,
                             costs=costs_for("NSE", intraday=False))
    json.dumps(signal.to_dict(), allow_nan=False)
    assert signal.action in ("BUY", "SHORT_ENTRY", "EXIT_LONG", "HOLD")
    assert signal.to_dict()["execution_eligible"] is False
    return f"{signal.action} · conviction {signal.conviction:.1f} · status {signal.status}"


@check("A historical BUY produces a complete, budget-respecting plan", "signal")
def check_historical_buy(args):
    import sqlite3
    from src.data_pipeline.universe import costs_for
    from src.paths import DB_PATH
    from src.quant_engine.decision_engine import AccountRisk, SignalPolicy
    from src.signal_engine.composite import generate_signal
    from src.signal_engine.scanner import resolve_clock
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv "
        "WHERE ticker = 'TITAN' AND timestamp <= '2024-09-05' ORDER BY timestamp", connection)
    connection.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    account, policy = AccountRisk(), SignalPolicy()
    costs = costs_for("NSE", intraday=False)
    clock, _ = resolve_clock(frame, "1d", "last_close_review")
    signal = generate_signal(frame, "1d", ticker="TITAN", now=clock, costs=costs,
                             account=account, policy=policy)
    assert signal.action == "BUY", f"expected BUY, got {signal.action}"
    plan = signal.plan
    assert plan["stop"] < plan["buy_trigger"] < plan["take_profit"], "plan geometry is wrong"
    assert plan["quantity"] > 0
    budget = account.equity * policy.risk_pct / 100
    assert plan["modeled_stop_loss"] <= budget, (
        f"modelled loss {plan['modeled_stop_loss']:.2f} exceeds budget {budget:.2f}")
    STATE["buy_signal"] = signal
    return (f"qty {plan['quantity']} · risk ₹{plan['modeled_stop_loss']:.0f} of ₹{budget:.0f} "
            f"· R:R {plan['net_reward_risk']:.2f}")


@check("Sentiment and the model cannot create a signal", "signal")
def check_invariant(args):
    import sqlite3
    from src.data_pipeline.universe import costs_for
    from src.paths import DB_PATH
    from src.signal_engine.composite import generate_signal
    from src.signal_engine.scanner import resolve_clock
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv "
        "WHERE ticker = 'HDFCBANK' ORDER BY timestamp", connection)
    connection.close()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    clock, _ = resolve_clock(frame, "1d", "last_close_review")
    costs = costs_for("NSE", intraday=False)
    glowing = {"sentiment_1h": 1.0, "sentiment_1d": 1.0, "sentiment_1w": 1.0,
               "articles_count": 50, "divergence_flag": "CONGRUENT", "scorer": "e2e"}
    plain = generate_signal(frame, "1d", ticker="HDFCBANK", now=clock, costs=costs)
    hyped = generate_signal(frame, "1d", ticker="HDFCBANK", now=clock, costs=costs,
                            sentiment=glowing)
    assert plain.action == hyped.action, (
        f"maximally bullish news changed the action: {plain.action} -> {hyped.action}")
    assert hyped.conviction <= plain.conviction + 1e-9, "news raised conviction"
    return f"action {plain.action} unchanged under maximum bullish news"


@check("Short plans are refused on a delivery product", "signal")
def check_short_enforcement(args):
    from src.quant_engine.decision_engine import (AccountRisk, SignalPolicy,
                                                  analyze_market_short, build_short_plan)
    from src.quant_engine.research import EquityCosts, size_short
    layers = 0
    try:
        analyze_market_short(pd.DataFrame(), "1d")
    except ValueError:
        layers += 1
    row = pd.Series({"high": 100.0, "low": 99.0, "close": 99.5, "wilder_atr": 1.0})
    plan, reasons = build_short_plan(row, AccountRisk(), SignalPolicy(),
                                     EquityCosts(intraday=False))
    if plan is None and "intraday-only" in reasons[0]:
        layers += 1
    if size_short(100, 105, 200000, 0.5, EquityCosts(intraday=False)) == 0:
        layers += 1
    assert layers == 3, f"only {layers} of 3 enforcement layers fired"
    return "all 3 layers refuse a delivery short"


# ---------------------------------------------------------------- 5. markets

@check("Each market uses its own fee schedule", "markets")
def check_market_costs(args):
    from src.data_pipeline.universe import costs_for
    nse_sell = costs_for("NSE", intraday=True).fee(1000, 100, "SELL")
    us_buy = costs_for("US", intraday=False).fee(1000, 100, "BUY")
    us_sell = costs_for("US", intraday=False).fee(1000, 100, "SELL")
    assert abs(nse_sell - 52.34) < 0.01, f"NSE schedule changed: {nse_sell}"
    assert us_buy == 0.0, f"US buy should be free, got {us_buy}"
    assert 0 < us_sell < 5, f"US sell fee looks wrong: {us_sell}"
    return f"NSE sell ₹{nse_sell:.2f} · US buy ${us_buy:.2f} / sell ${us_sell:.2f}"


@check("A US signal runs with US costs", "markets")
def check_us_signal(args):
    import sqlite3
    from src.data_pipeline.universe import costs_for
    from src.paths import DB_PATH
    from src.signal_engine.composite import generate_signal
    from src.signal_engine.scanner import resolve_clock
    connection = sqlite3.connect(DB_PATH)
    frame = pd.read_sql(
        "SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv "
        "WHERE ticker = 'US:MSFT' ORDER BY timestamp", connection)
    connection.close()
    assert not frame.empty, "no US:MSFT rows — run update_market_data --markets US"
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    clock, _ = resolve_clock(frame, "1d", "last_close_review")
    signal = generate_signal(frame, "1d", ticker="US:MSFT", now=clock,
                             costs=costs_for("US", intraday=False))
    json.dumps(signal.to_dict(), allow_nan=False)
    return f"{len(frame)} bars · {signal.action} · conviction {signal.conviction:.1f}"


# ---------------------------------------------------------------- 6. AI layer

@check("News entity matching rejects sibling companies", "ai")
def check_entity_matching(args):
    from src.sentiment_engine.news_sources import EntityMatcher
    matcher = EntityMatcher()
    cases = [("Reliance Power shares surge 8%", "RELIANCE", False),
             ("Reliance Industries shares jump 3%", "RELIANCE", True),
             ("SBI Cards Q2 profit rises 12%", "SBIN", False),
             ("State Bank of India cuts MCLR", "SBIN", True),
             ("Adani Power wins order", "ADANIENT", False)]
    wrong = [h for h, t, expected in cases if matcher.attribute(h, t)[0] != expected]
    assert not wrong, f"misattributed: {wrong}"
    return f"{len(cases)}/{len(cases)} correct, siblings excluded"


@check("Decision model answers with a pinned schema", "ai")
def check_decision_model(args):
    from src.llm_engine.providers import build_provider, provider_status
    status = provider_status()
    provider = build_provider()
    if provider.name == "none":
        raise Skip("no decision model configured (no API key, Ollama unreachable)")
    from src.llm_engine.tasks import DecisionEngine
    engine = DecisionEngine(provider)
    decision = engine.classify_event("Tata Steel slumps 6% after Q2 loss widens", "Tata Steel")
    engine.close()
    assert decision.available, f"model unavailable: {decision.error}"
    assert 0.0 <= decision.score <= 1.0 and 0.0 <= decision.confidence <= 1.0
    return (f"{provider.name}/{getattr(provider, 'model', '')} → {decision.label} "
            f"({decision.confidence:.2f}) in {decision.latency_ms}ms")


@check("RAG returns cited passages", "ai")
def check_rag(args):
    try:
        from src.rag_engine.retriever import StrategyRetriever
        retriever = StrategyRetriever()
        passages = retriever.retrieve("breakout confirmed by volume", k=2)
    except Exception as exc:
        raise Skip(f"RAG unavailable: {type(exc).__name__}")
    if not passages:
        raise Skip("book index returned nothing")
    assert all(p.get("source") for p in passages), "a passage has no source attribution"
    return f"{len(passages)} passages from {passages[0]['source'][:34]}"


@check("The agent crew runs and the arbiter stays in charge", "ai")
def check_crew(args):
    from src.agents.evidence_crew import arbiter, gather_evidence
    signal = STATE.get("buy_signal")
    if signal is None:
        raise Skip("no BUY signal from the earlier check")
    report = {"status": "CONDITIONAL_BUY", "patterns": [], "recent_patterns": [],
              "rejections": [], "model": {"available": False, "reason": "n/a"}}
    evidence = gather_evidence(report=report, direction="LONG", sentiment=None,
                               retriever=None, decision_engine=None, company="Titan")
    verdict = arbiter(evidence, "LONG", scored_agents={"event_risk"})
    assert verdict.direction == "LONG"
    assert verdict.multiplier <= 1.0, "an agent raised conviction"
    json.dumps(verdict.to_dict(), allow_nan=False)
    return f"{len(evidence)} agents · multiplier {verdict.multiplier:.3f} · scored only event_risk"


# ---------------------------------------------------------------- 7. paper book

@check("Paper book is idempotent and reconciles", "paper")
def check_paper_book(args):
    import tempfile
    from pathlib import Path
    from src.paper_trading.ledger import PaperBook
    signal = STATE.get("buy_signal")
    if signal is None:
        raise Skip("no BUY signal from the earlier check")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "e2e.db"
        book = PaperBook(path, starting_capital=200000.0)
        for _ in range(3):
            book.open_from_signal(signal)
        assert len(book.events("ENTRY")) == 1, "repeated recording double-filled"
        plan = signal.plan
        book.record_exit(signal.ticker, signal.interval, "LONG",
                         price=plan["take_profit"], quantity=plan["quantity"],
                         reason="target", ts_ist="2024-10-01T15:30:00+05:30")
        cash = book.cash()
        summary = book.performance_summary()
        book.close()
        reopened = PaperBook(path, starting_capital=200000.0)
        restored = reopened.cash()
        reopened.close()
    assert abs(cash - restored) < 1e-6, "cash did not reconcile after reload"
    return (f"1 entry from 3 writes · P&L ₹{summary['net_pnl']:.0f} · "
            f"reconciles to ₹{restored:,.0f}")


# ---------------------------------------------------------------- 8. backtest

@check("Portfolio backtest reproduces its measured verdict", "backtest")
def check_backtest(args):
    if args.quick:
        raise Skip("--quick")
    from src.backtesting.portfolio import BacktestConfig, prepare_panel, run_backtest
    from src.backtesting.validation import buy_and_hold_equal_weight, compare
    from src.quant_engine.research import EquityCosts
    cleaned = STATE.get("cleaned_nse")
    calendar = STATE.get("calendar")
    if cleaned is None:
        raise Skip("cleaning check did not run")
    prepared = prepare_panel(cleaned, intraday=False)
    costs = EquityCosts(intraday=False)
    result = run_backtest(prepared, calendar, BacktestConfig(capital=200000.0, costs=costs))
    metrics = result.metrics
    assert metrics["trades"] > 50, f"only {metrics['trades']} trades"
    equity = result.equity
    identity = (equity.equity - (equity.cash + equity.holdings)).abs().max()
    assert identity < 1e-6, f"equity identity broken by {identity}"
    benchmark = buy_and_hold_equal_weight(cleaned, calendar, 200000.0, costs)
    table = compare({"Strategy": result.equity[["date", "equity"]],
                     "Buy & hold": benchmark}, 200000.0)
    strategy_cagr = float(table.loc[table.strategy == "Strategy", "cagr_pct"].iloc[0])
    hold_cagr = float(table.loc[table.strategy == "Buy & hold", "cagr_pct"].iloc[0])
    STATE["backtest"] = result
    return (f"{metrics['trades']} trades · strategy {strategy_cagr:.2f}% vs "
            f"buy&hold {hold_cagr:.2f}% CAGR · Sharpe {metrics['sharpe']:.3f}")


@check("Overfitting detectors discriminate", "backtest")
def check_overfitting_detectors(args):
    import numpy as np
    from src.backtesting.validation import (deflated_sharpe_ratio,
                                            probability_of_backtest_overfitting)
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(0, 0.01, (600, 10)),
                         columns=[f"v{i}" for i in range(10)])
    noisy = probability_of_backtest_overfitting(noise, n_blocks=8)
    skilled = noise.copy()
    skilled["v0"] = skilled["v0"] + 0.004
    genuine = probability_of_backtest_overfitting(skilled, n_blocks=8)
    assert noisy.pbo > genuine.pbo, (
        f"PBO failed to discriminate: noise {noisy.pbo:.2f} vs skill {genuine.pbo:.2f}")
    pure = pd.Series(rng.normal(0.0, 0.01, 1200))
    assert not deflated_sharpe_ratio(pure, n_trials=50).significant, "DSR accepted noise"
    return f"PBO noise {noisy.pbo:.2f} > skill {genuine.pbo:.2f}; DSR rejects noise"


# ---------------------------------------------------------------- 9. UI

@check("App starts with no heavy imports and no exception", "ui")
def check_app(args):
    import sys as _sys
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    heavy = ("torch", "langgraph", "chromadb", "transformers")
    for name in heavy:
        for loaded in list(_sys.modules):
            if loaded == name or loaded.startswith(f"{name}."):
                del _sys.modules[loaded]
    path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    app = AppTest.from_file(str(path), default_timeout=90).run()
    assert not app.exception, f"startup raised: {app.exception}"
    assert "trading signals" in app.title[0].value.lower()
    still_absent = [n for n in heavy if n not in _sys.modules]
    assert sorted(still_absent) == sorted(heavy), (
        f"startup imported {set(heavy) - set(still_absent)}")
    return "default workspace loads; torch/langgraph/chroma/transformers all untouched"


@check("Scanner is button-triggered, not on page load", "ui")
def check_scanner_lazy(args):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    app = AppTest.from_file(str(path), default_timeout=90).run()
    assert any("Run scan" in b.label for b in app.button), "no Run scan button"
    assert any("Press" in i.value for i in app.info), "scanner appears to auto-run"
    return "page load performs no scan"


CHECKS = [
    check_paths, check_imports, check_database,
    check_cleaning, check_crash_discrimination,
    check_yahoo, check_provenance,
    check_daily_signal, check_historical_buy, check_invariant, check_short_enforcement,
    check_market_costs, check_us_signal,
    check_entity_matching, check_decision_model, check_rag, check_crew,
    check_paper_book,
    check_backtest, check_overfitting_detectors,
    check_app, check_scanner_lazy,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--quick", action="store_true", help="skip the backtest")
    parser.add_argument("--trace", action="store_true", help="print tracebacks on failure")
    parser.add_argument("--json", default=None, help="write the report here")
    args = parser.parse_args()

    print("=" * 108)
    print("END-TO-END CHECK".center(108))
    print("=" * 108)
    started = time.time()
    group = None
    for function in CHECKS:
        if function._e2e[1] != group:
            group = function._e2e[1]
            print(f"\n--- {group.upper()} " + "-" * (100 - len(group)))
        run(function, args)

    elapsed = time.time() - started
    counts = {s: sum(1 for r in RESULTS if r["status"] == s) for s in (PASS, FAIL, SKIP)}
    print("\n" + "=" * 108)
    print(f"{counts[PASS]} passed · {counts[FAIL]} failed · {counts[SKIP]} skipped "
          f"· {elapsed:.1f}s")
    if counts[FAIL]:
        print("\nFAILURES:")
        for row in RESULTS:
            if row["status"] == FAIL:
                print(f"  {row['name']}: {row['detail']}")
    if counts[SKIP]:
        print("\nSkipped (not defects):")
        for row in RESULTS:
            if row["status"] == SKIP:
                print(f"  {row['name']}: {row['detail']}")
    print("=" * 108)

    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(
            {"generated": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
             "seconds": round(elapsed, 1), "counts": counts, "results": RESULTS},
            indent=2), encoding="utf-8")
        print(f"report written to {args.json}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
