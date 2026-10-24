"""Trading-signals workspace: watchlist scanner, single-stock deep dive, paper book.

Everything shown here comes from src/signal_engine. The UI deliberately holds no scoring
logic of its own, and it never invents a number: an unavailable component renders as
"not available" with the reason, not as a neutral default.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.data_pipeline.broker_client import MarketDataFetcher
from src.data_pipeline.universe import MARKETS, costs_for, load_universe, split_key
from src.paper_trading.ledger import PaperBook
from src.paths import WATCHLIST_PATH
from src.quant_engine.decision_engine import AccountRisk, SignalPolicy
from src.quant_engine.research import EquityCosts
from src.sentiment_engine.multi_horizon import SentimentService
from src.signal_engine.composite import SignalConfig, generate_signal
from src.signal_engine.scanner import (LAST_CLOSE_REVIEW, LIVE, PERIOD_FOR_INTERVAL,
                                       market_is_open, resolve_clock, scan, to_frame)

ACTION_STYLE = {
    "BUY": ("🟢", "success"), "SHORT_ENTRY": ("🔻", "warning"),
    "EXIT_LONG": ("🟠", "warning"), "HOLD": ("⚪", "info"), "UNAVAILABLE": ("⚫", "info"),
}


@st.cache_data(show_spinner=False)
def load_watchlist(markets=("NSE",)) -> list:
    """Instrument keys for the selected markets.

    NSE keys stay bare for continuity with five years of stored history; other markets are
    qualified ("US:AAPL") because bare symbols collide across exchanges.
    """
    try:
        universe = load_universe(tuple(markets))
    except Exception:
        universe = {}
    if not universe:
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    keys = []
    for key, meta in universe.items():
        code, ticker = split_key(key)
        keys.append(ticker if code == "NSE" else key)
    return sorted(keys, key=lambda k: (split_key(k)[0] != "NSE", k))


def services():
    """One set of services per Streamlit session."""
    if "signal_services" not in st.session_state:
        st.session_state.signal_services = {
            "fetcher": MarketDataFetcher(cache_ttl_seconds=120),
            "sentiment": SentimentService(),
            "book": PaperBook(),
        }
    return st.session_state.signal_services


def ai_services() -> dict:
    """RAG retriever and decision model, built on first use.

    Kept out of `services()` because constructing them touches Ollama and ChromaDB, and the
    app must start cleanly with neither available. Failures are recorded and surfaced,
    never raised — the whole pipeline is designed to run without a model.
    """
    if "ai_services" not in st.session_state:
        retriever, decision_engine, errors = None, None, []
        try:
            from src.rag_engine.retriever import StrategyRetriever

            retriever = StrategyRetriever()
        except Exception as exc:
            errors.append(f"RAG retriever unavailable: {type(exc).__name__}: {exc}")
        try:
            from src.llm_engine.tasks import DecisionEngine

            engine = DecisionEngine()
            decision_engine = engine if engine.available else None
            if decision_engine is None:
                errors.append("No decision model reachable (no API key, and Ollama is down "
                              "or the model is not pulled).")
        except Exception as exc:
            errors.append(f"Decision model unavailable: {type(exc).__name__}: {exc}")
        st.session_state.ai_services = {"retriever": retriever,
                                        "decision_engine": decision_engine,
                                        "errors": errors}
    return st.session_state.ai_services


@st.cache_data(show_spinner=False, ttl=120)
def cached_provider_status() -> dict:
    from src.llm_engine.providers import provider_status

    try:
        return provider_status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def sidebar_settings() -> dict:
    st.sidebar.title("Signal settings")
    markets = st.sidebar.multiselect(
        "Markets", sorted(MARKETS), default=["NSE"],
        help="Each market carries its own fee schedule, currency and shorting rules. "
             "Mixing currencies in one portfolio is not supported.")
    markets = tuple(markets or ["NSE"])
    if len(markets) > 1:
        st.sidebar.warning("Multiple markets selected: position sizes are computed in each "
                           "market's own currency and are not comparable.")
    interval = st.sidebar.selectbox("Interval", ["15m", "5m", "1h", "1d"], index=0)
    intraday = interval != "1d"

    open_now = market_is_open()
    st.sidebar.caption(
        f"NSE session: {'open' if open_now else 'closed'} "
        f"({datetime.now().astimezone().strftime('%H:%M %Z')}). Holidays are not modelled.")
    mode_label = st.sidebar.radio(
        "Evaluation clock",
        ["Live", "Last completed bar (review)"],
        index=0 if open_now else 1,
        help="Intraday data is treated as stale two bars after its close, so outside "
             "market hours Live correctly reports HISTORICAL_ONLY. Review mode evaluates "
             "as at the last completed bar instead.")
    mode = LIVE if mode_label == "Live" else LAST_CLOSE_REVIEW
    if mode == LIVE and not open_now and intraday:
        st.sidebar.warning("Market is closed: Live will report HISTORICAL_ONLY.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Account")
    capital = st.sidebar.number_input("Equity (INR)", min_value=10000.0, value=200000.0,
                                      step=10000.0)
    cash = st.sidebar.number_input("Uncommitted cash (INR)", min_value=0.0, value=capital,
                                   step=10000.0)
    risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 0.5, 0.25)
    allocation_pct = st.sidebar.slider("Max allocation (%)", 5, 100, 25, 5)
    enable_short = st.sidebar.checkbox(
        "Offer intraday shorts", value=intraday, disabled=not intraday,
        help="NSE cash delivery cannot be sold short, so shorts are intraday only.")
    sentiment_on = st.sidebar.checkbox("Use news sentiment", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Costs are illustrative NSE rates. Verify against your contract note.")
    slippage = st.sidebar.number_input("Slippage per leg (bps)", 0.0, 100.0, 5.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI evidence")
    use_crew = st.sidebar.checkbox(
        "Run the agent crew on deep dive", value=True,
        help="Retrieves cited passages from the trading literature and asks a decision "
             "model whether a pending binary event makes this entry risky. Adds a few "
             "seconds. It can only reduce conviction or veto, never create a signal.")
    status = cached_provider_status()
    if "error" not in status:
        active = status.get("active", "none")
        label = {"jev": "Jev (TypeSafe)", "ollama": f"local {status['ollama']['model']}",
                 "none": "none"}.get(active, active)
        (st.sidebar.success if active != "none" else st.sidebar.info)(
            f"Decision model: {label}")
        if not status["jev"]["available"]:
            st.sidebar.caption("Jev: set OPENROUTER_API_KEY to enable.")

    return {
        "markets": markets, "use_crew": use_crew,
        "interval": interval, "intraday": intraday, "mode": mode,
        "account": AccountRisk(equity=capital, cash=min(cash, capital),
                               day_start_equity=capital, peak_equity=capital),
        "policy": SignalPolicy(risk_pct=risk_pct, allocation_pct=allocation_pct),
        "costs": costs_for(markets[0], intraday=intraday, slippage_bps=slippage),
        "slippage": slippage,
        "config": SignalConfig(sentiment_enabled=sentiment_on),
        "enable_short": enable_short,
    }


def render_scanner(settings: dict):
    st.subheader("Watchlist scanner")
    st.caption("Ranked by action, then conviction. The daily long gate passes on roughly 7% "
               "of bars, so most names reading HOLD is the normal state, not a fault.")
    watchlist = load_watchlist(settings["markets"])
    service = services()

    left, middle, right = st.columns([2, 1, 1])
    with left:
        universe = st.multiselect("Tickers to scan", watchlist, default=watchlist[:25])
    with middle:
        fast = st.checkbox("Fast mode", value=True,
                           help="Skips the nearest-neighbour model and uses only cached "
                                "sentiment. Deferred components are listed per row.")
    with right:
        st.write("")
        run = st.button("Run scan", type="primary")

    key = (tuple(universe), settings["interval"], settings["mode"], fast)
    if run and universe:
        with st.spinner(f"Scanning {len(universe)} tickers..."):
            rows = scan(universe, service["fetcher"], settings["interval"], max_workers=8,
                        account=settings["account"], policy=settings["policy"],
                        costs=settings["costs"], config=settings["config"],
                        sentiment_service=service["sentiment"] if settings["config"].sentiment_enabled else None,
                        fast=fast, mode=settings["mode"],
                        confidence_lookup=service["book"].confidence_lookup())
        st.session_state["scan_result"] = (key, rows, datetime.now().isoformat(timespec="seconds"))

    stored = st.session_state.get("scan_result")
    if not stored or stored[0] != key:
        st.info("Press **Run scan**. Results are not refreshed on every widget change.")
        return
    _, rows, ran_at = stored
    frame = to_frame(rows)
    st.caption(f"Scanned {len(frame)} tickers at {ran_at}. "
               f"{'Fast mode: some components deferred.' if fast else 'Full mode.'}")

    counts = frame["Action"].value_counts().to_dict()
    columns = st.columns(5)
    for column, action in zip(columns, ["BUY", "SHORT_ENTRY", "EXIT_LONG", "HOLD", "UNAVAILABLE"]):
        column.metric(f"{ACTION_STYLE[action][0]} {action}", counts.get(action, 0))

    actionable = frame[frame["Action"].isin(["BUY", "SHORT_ENTRY"])]
    if actionable.empty:
        st.info("No entry candidates in this scan. WAIT and HOLD are valid outcomes; "
                "no signal is forced.")
    st.dataframe(frame, width="stretch", height=460)
    st.download_button("Export scan (CSV)", frame.to_csv(index=False),
                       f"scan_{settings['interval']}.csv", "text/csv")


STANCE_ICON = {"bullish": "🟢", "bearish": "🔴", "blocking": "⛔", "neutral": "⚪"}


def _render_crew(crew: dict, errors=None):
    """Agent evidence. Deliberately shows which agents were scored and which were not."""
    st.markdown("#### Agent evidence")
    for message in errors or []:
        st.caption(f"⚠️ {message}")
    if not crew:
        st.info("The agent crew did not run. Enable it in the sidebar, or no AI service "
                "was reachable.")
        return
    scored = set(crew.get("scored_agents") or [])
    st.caption(crew.get("rule", ""))
    for item in crew.get("evidence", []):
        icon = STANCE_ICON.get(item.get("stance"), "⚪")
        agent = item.get("agent", "?")
        tag = "scored" if agent in scored else "explanation only"
        if not item.get("available"):
            st.markdown(f"⚫ **{agent}** — not available · {item.get('reason', '')}")
            continue
        st.markdown(f"{icon} **{agent}** · _{tag}_ — {item.get('headline', '')}")
        if item.get("detail"):
            st.caption(item["detail"])
        for citation in item.get("citations", [])[:3]:
            st.caption(f"  ↳ {citation}")
    multiplier = crew.get("conviction_multiplier", 1.0)
    if multiplier < 1.0:
        st.warning(f"Agent penalties scaled conviction by ×{multiplier:.3f}")
    for veto in crew.get("vetoes", []):
        st.error(f"VETO — {veto}")


def _component_badge(label: str, detail: dict):
    if detail.get("available"):
        st.success(f"**{label}** · penalty {detail.get('penalty', 0):.0%}")
    else:
        st.info(f"**{label}** · not available — {detail.get('reason', 'no reason given')}")
    with st.expander(f"{label} detail", expanded=False):
        st.json(detail)


def render_deep_dive(settings: dict):
    st.subheader("Single-stock deep dive")
    service = services()
    watchlist = load_watchlist(settings["markets"])
    left, right = st.columns([2, 1])
    with left:
        ticker = st.selectbox("Ticker", watchlist, index=0)
    with right:
        st.write("")
        go = st.button("Analyse", type="primary")

    key = (ticker, settings["interval"], settings["mode"])
    if go:
        with st.spinner(f"Analysing {ticker} (full mode, model and news included)..."):
            bars = service["fetcher"].fetch_intraday_data(
                ticker, PERIOD_FOR_INTERVAL[settings["interval"]], settings["interval"])
            if bars is None or bars.empty:
                st.error(f"No candles returned for {ticker}. Yahoo may not serve this symbol.")
                return
            sentiment = None
            if settings["config"].sentiment_enabled:
                sentiment = service["sentiment"].reading(ticker, bars=bars)
            clock, clock_label = resolve_clock(bars, settings["interval"], settings["mode"])
            market_code, _ = split_key(ticker)
            market_costs = costs_for(market_code, intraday=settings["intraday"],
                                     slippage_bps=settings["slippage"])
            ai = ai_services() if settings["use_crew"] else {"retriever": None,
                                                             "decision_engine": None,
                                                             "errors": []}
            company = (load_universe(settings["markets"]).get(
                f"{market_code}:{split_key(ticker)[1]}", {}).get("company_name") or ticker)
            signal = generate_signal(
                bars, settings["interval"], ticker=ticker, account=settings["account"],
                policy=settings["policy"], costs=market_costs, sentiment=sentiment,
                now=clock, config=settings["config"],
                enable_short=(settings["enable_short"] and settings["intraday"]
                              and MARKETS[market_code].intraday_short_only is not None),
                confidence_lookup=service["book"].confidence_lookup(),
                retriever=ai["retriever"], decision_engine=ai["decision_engine"],
                company=company)
            st.session_state["deep_dive"] = (key, signal, bars, sentiment, clock_label,
                                             ai.get("errors", []), market_code)

    stored = st.session_state.get("deep_dive")
    if not stored or stored[0] != key:
        st.info("Press **Analyse** to evaluate this ticker.")
        return
    _, signal, bars, sentiment, clock_label, ai_errors, market_code = stored

    icon, _ = ACTION_STYLE.get(signal.action, ("⚪", "info"))
    head = st.columns(4)
    head[0].metric("Action", f"{icon} {signal.action}")
    head[1].metric("Conviction", f"{signal.conviction:.1f}",
                   help="Ranking score, NOT a probability.")
    head[2].metric("Setup grade", (signal.setup or {}).get("letter") or "-")
    head[3].metric("Risk score", f"{(signal.risk or {}).get('score', 0):.1f}",
                   help="0 benign, 100 severe. Independent of any model output.")
    st.caption(f"{clock_label} · bar close {signal.as_of_bar_close} · "
               f"valid until {signal.valid_until} · status **{signal.status}** · "
               f"execution eligible: NO")

    confidence = signal.confidence
    if confidence and confidence.get("available"):
        low, high = confidence["ci"]
        st.success(f"Measured hit rate for {confidence['direction']}/{confidence['interval']}/"
                   f"grade {confidence['grade']}: **{confidence['p']:.0%}** "
                   f"(95% CI {low:.0%}–{high:.0%}, n={confidence['n']}) — "
                   f"event: {confidence['event']}")
    elif confidence:
        st.info(f"Confidence not measured yet — {confidence.get('reason')}. "
                "This number only ever comes from this paper book's own closed trades.")

    if signal.plan:
        st.markdown("#### Execution plan")
        plan = signal.plan
        entry_key = "buy_trigger" if signal.direction == "LONG" else "sell_trigger"
        cells = st.columns(5)
        cells[0].metric("Trigger", f"₹{plan[entry_key]:,.2f}")
        cells[1].metric("Stop", f"₹{plan['stop']:,.2f}")
        cells[2].metric("Target", f"₹{plan['take_profit']:,.2f}")
        cells[3].metric("Quantity", plan["quantity"])
        cells[4].metric("Net R:R", f"{plan['net_reward_risk']:.2f}")
        st.caption(plan["rule"])
        money = st.columns(3)
        money[0].metric("Modelled risk", f"₹{plan['modeled_stop_loss']:,.2f}")
        money[1].metric("Modelled reward", f"₹{plan['modeled_target_profit']:,.2f}")
        money[2].metric("Capital needed",
                        f"₹{plan.get('cash_required', plan.get('margin_required', 0)):,.2f}")

        if st.button(f"Record {signal.action} in paper book"):
            try:
                event_id = service["book"].open_from_signal(signal, settings["costs"])
                st.success(f"Recorded. Event {event_id[:12]} — repeat presses are ignored.")
            except ValueError as exc:
                st.error(f"Rejected: {exc}")
    else:
        st.warning("No execution plan. Reasons below.")

    st.markdown("#### Why")
    grade_col, risk_col = st.columns(2)
    with grade_col:
        st.markdown("**Setup grade axes**")
        axes = (signal.setup or {}).get("axes", {})
        if axes:
            st.dataframe(pd.DataFrame({"axis": list(axes), "score": list(axes.values())}),
                         width="stretch", hide_index=True)
        else:
            st.caption("Not graded: no qualifying setup.")
    with risk_col:
        st.markdown("**Risk terms** (0 benign … 100 severe)")
        terms = (signal.risk or {}).get("terms", {})
        if terms:
            st.dataframe(pd.DataFrame({"term": list(terms), "score": list(terms.values())}),
                         width="stretch", hide_index=True)
        else:
            st.caption("Not scored: no plan to score.")

    if signal.rejections:
        st.markdown("**Blocking reasons**")
        for reason in signal.rejections:
            st.write(f"• {reason}")
    if signal.vetoes:
        st.error("Hard vetoes: " + "; ".join(signal.vetoes))

    st.markdown("#### Components")
    algo_col, sent_col = st.columns(2)
    with algo_col:
        _component_badge("Algo (historical neighbours)", signal.components.get("algo", {}))
    with sent_col:
        _component_badge("News sentiment", signal.components.get("sentiment", {}))

    _render_crew(signal.components.get("crew"), ai_errors)

    if sentiment:
        st.markdown("#### News")
        horizon_cols = st.columns(3)
        for column, horizon in zip(horizon_cols, ("1h", "1d", "1w")):
            value = sentiment.get(f"sentiment_{horizon}")
            column.metric(f"{horizon} · {sentiment.get(f'label_{horizon}')}",
                          f"{value:+.2f}" if value is not None else "no data",
                          help=f"{sentiment.get('counts', {}).get(horizon, 0)} articles")
        st.caption(f"Scorer: **{sentiment.get('scorer')}** · "
                   f"{sentiment.get('articles_count', 0)} relevant articles · "
                   f"divergence: {sentiment.get('divergence_flag')}")
        with st.expander("Headlines and retrieval diagnostics", expanded=False):
            for headline in sentiment.get("latest_headlines", []):
                st.write(f"• ({headline['matched_alias']}) {headline['title']}")
            st.json(sentiment.get("diagnostics", {}))

    with st.expander("Observed levels", expanded=False):
        st.json(signal.levels)
    st.download_button("Export signal (JSON)",
                       json.dumps(signal.to_dict(), indent=2, allow_nan=False),
                       f"{ticker}_{settings['interval']}_signal.json", "application/json")
    for warning in signal.warnings:
        st.caption(warning)


def render_paper_book(settings: dict):
    st.subheader("Paper book")
    st.caption("A local SQLite simulation. No broker is connected and no real orders exist. "
               "Positions and cash are derived from an append-only event log, so they "
               "reconcile after a restart.")
    book = services()["book"]

    summary = book.performance_summary()
    if summary["trades"] == 0:
        st.info(summary.get("note", "No closed paper trades yet."))
    else:
        cells = st.columns(5)
        cells[0].metric("Closed trades", summary["trades"])
        cells[1].metric("Net P&L", f"₹{summary['net_pnl']:,.0f}",
                        f"{summary['net_return_pct']:+.2f}%")
        cells[2].metric("Win rate", f"{summary['win_rate_pct']:.0f}%")
        cells[3].metric("Profit factor",
                        f"{summary['profit_factor']:.2f}" if summary["profit_factor"] else "n/a")
        cells[4].metric("Max drawdown", f"{summary['max_drawdown_pct']:.1f}%")
        st.caption(f"Fees paid: ₹{summary['total_fees']:,.2f} · "
                   f"expectancy ₹{summary['expectancy_rupees']:,.2f}/trade")

    open_positions = book.open_positions()
    st.markdown("#### Open positions")
    if open_positions.empty:
        st.write("None.")
    else:
        st.dataframe(open_positions, width="stretch")
        if st.button("Mark open positions against latest candles"):
            fetcher = services()["fetcher"]
            candles = {}
            for ticker in open_positions["ticker"].unique():
                interval = open_positions[open_positions.ticker == ticker]["interval"].iloc[0]
                candles[ticker] = fetcher.fetch_intraday_data(
                    ticker, PERIOD_FOR_INTERVAL.get(interval, "1mo"), interval)
            closures = book.evaluate_open_positions(candles)
            if closures:
                st.success(f"Closed {len(closures)}: " +
                           ", ".join(f"{c['ticker']} at {c['price']} ({c['reason']})"
                                     for c in closures))
            else:
                st.info("No stop or target touched since entry.")

    closed = book.closed_trades()
    st.markdown("#### Closed trades")
    if closed.empty:
        st.write("None.")
    else:
        st.dataframe(closed, width="stretch")
        curve = book.equity_curve()
        if not curve.empty:
            st.line_chart(curve.set_index("exit_ts"))
        st.download_button("Export trades (CSV)", closed.to_csv(index=False),
                           "paper_trades.csv", "text/csv")

    with st.expander("Signal log (every signal, taken or not)", expanded=False):
        signals = book.events("SIGNAL")
        st.dataframe(signals.drop(columns=["report_json"], errors="ignore")
                     if not signals.empty else signals, width="stretch")


def render_signals_workspace():
    st.title("NSE trading signals")
    st.warning("Research output, not investment advice and not executable orders. Yahoo data "
               "has no verified live-feed guarantee, no broker is connected, and the paper "
               "book is a local simulation.")
    settings = sidebar_settings()
    scanner_tab, deep_tab, book_tab = st.tabs(
        ["Scanner", "Deep dive", "Paper book"])
    with scanner_tab:
        render_scanner(settings)
    with deep_tab:
        render_deep_dive(settings)
    with book_tab:
        render_paper_book(settings)
