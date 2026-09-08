"""Research UI intentionally independent of optional LLM and model services."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_pipeline.broker_client import MarketDataFetcher
from src.quant_engine.research import EquityCosts, replay, size_long, validate_bars
from app.components.decision_panel import render_decision_panel


def render_research_workbench():
    root = Path(__file__).resolve().parents[2]
    st.title("NSE Research & Historical Paper Replay")
    st.warning("Research only. Yahoo data has no verified live-feed guarantee here. "
               "No broker is connected and no real orders can be placed.")
    st.caption("Historical replay is not a persistent paper account. Results are hypothetical, "
               "long-only, and not evidence of future profitability.")
    path = root / "config" / "watchlist.json"
    watchlist = json.loads(path.read_text(encoding="utf-8")).get("watchlist", []) if path.exists() else []
    watchlist = watchlist or ["RELIANCE", "TCS", "INFY"]
    ticker = st.sidebar.selectbox("Research ticker", watchlist)
    horizon = st.sidebar.selectbox("Research horizon", ["Intraday replay", "Daily / long-horizon research"])
    intraday = horizon == "Intraday replay"
    interval = st.sidebar.selectbox("Candle interval", ["5m", "15m", "1h"] if intraday else ["1d"])
    period = st.sidebar.selectbox("History window", ["5d", "1mo"] if intraday else ["2y", "5y"])
    capital = st.sidebar.number_input("Research capital (INR)", min_value=1000.0, value=200000.0, step=1000.0)
    risk = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 0.5, 0.25)
    allocation = st.sidebar.slider("Maximum cash allocation (%)", 5, 100, 25, 5)
    daily_loss = st.sidebar.slider("Realized-loss entry lock (%)", 0.5, 5.0, 2.0, 0.5)
    with st.expander("Transaction-cost assumptions — verify with your broker", expanded=False):
        st.caption("Illustrative NSE equity rates from Zerodha's published schedule, reviewed 2026-09-08. "
                   "Not a universal tariff or exact contract-note calculator; excludes extra levies, "
                   "rounding differences, funding, and broker-specific charges.")
        slip = st.number_input("Adverse slippage per leg (basis points)", 0.0, 100.0, 5.0)
        dp = st.number_input("Delivery exit DP estimate including tax (INR)", 0.0, 100.0, 15.34)
        exchange_pct = st.number_input("Exchange transaction rate (%)", 0.0, 1.0, 0.00307, format="%.5f")
    costs = EquityCosts(intraday=intraday, slippage_bps=slip, delivery_dp=dp, exchange_rate=exchange_pct / 100)
    key = (ticker, interval, period)
    if st.button("Fetch research candles"):
        try:
            with st.spinner("Fetching Yahoo research history..."):
                raw = MarketDataFetcher().fetch_intraday_data(ticker, period, interval)
                bars = validate_bars(raw)
                # Drop a potentially forming last bar. Daily data is only final after
                # its calendar date, deliberately conservative without a session feed.
                now = pd.Timestamp.now(tz="Asia/Kolkata")
                if intraday:
                    duration = pd.Timedelta({"5m": "5min", "15m": "15min", "1h": "1h"}[interval])
                    bars = bars[bars.timestamp + duration <= now]
                else:
                    bars = bars[bars.timestamp.dt.date < now.date()]
                if bars.empty:
                    raise ValueError("No completed candles available")
                st.session_state["research_bars"] = (key, bars, now.isoformat())
                st.session_state.pop("research_result", None)
        except Exception as exc:
            st.session_state.pop("research_bars", None)
            st.session_state.pop("research_result", None)
            st.error(f"Research data unavailable: {exc}")
    stored = st.session_state.get("research_bars")
    if not stored or stored[0] != key:
        st.info("Fetch data for the selected ticker and horizon. No synthetic prices will be substituted.")
    else:
        _, bars, received = stored
        st.caption(f"Source: Yahoo research | Retrieved: {received} | Last bar start: {bars.timestamp.iloc[-1]} "
                   f"| Bars: {len(bars)} | Exchange quote freshness: unverified")
        st.line_chart(bars.set_index("timestamp")[["close"]])
        st.download_button("Export source candles (CSV)", bars.to_csv(index=False), f"{ticker}_{interval}.csv", "text/csv")
        render_decision_panel(bars, interval, capital, risk, allocation, daily_loss, costs)
        st.subheader("Separate baseline strategy replay")
        st.caption("The replay below evaluates the original EMA/VWAP baseline, not the new pattern-risk decision engine. "
                   "Its results must not be used as validation of the pattern signals or ML model.")
        st.markdown("**Strategy:** EMA(9) above EMA(21); intraday requires session VWAP and "
                    "relative volume ≥ 1.25; daily requires 200 completed bars and price above SMA(200). "
                    "Entry next bar open, 1.5 × 14-bar SMA ATR stop, 2R target. No leverage or shorting.")
        st.info("Replay assumptions: adverse slippage both ways; stop before target if both touch; "
                "stop gaps fill at the worse open. Intraday flattening uses the last observed bar "
                "of each date, not an exchange-verified square-off schedule. Missing bars, liquidity, "
                "halts and corporate actions can invalidate results. Raw-price daily history is not "
                "a corporate-action-correct investment return series.")
        settings = (key, received, capital, risk, allocation, intraday, costs, daily_loss)
        if st.button("Run historical paper replay"):
            try:
                result = replay(bars, capital, risk, allocation, intraday, costs, daily_loss_pct=daily_loss)
                st.session_state["research_result"] = (settings, result)
            except ValueError as exc:
                st.error(str(exc))
        saved = st.session_state.get("research_result")
        if saved and saved[0] == settings:
            result = saved[1]
            st.subheader("Hypothetical results — after modeled fees and slippage")
            st.json(result["metrics"])
            st.line_chart(result["equity"].set_index("timestamp"))
            if result["trades"].empty:
                st.info("No eligible trades. This is a valid result; do not force a signal.")
            else:
                st.dataframe(result["trades"], use_container_width=True)
                st.download_button("Export hypothetical trade ledger", result["trades"].to_csv(index=False),
                                   f"{ticker}_replay.csv", "text/csv")
    st.subheader("Independent cost-aware long-position calculator")
    entry = st.number_input("Assumed entry fill (INR)", min_value=0.01, value=1000.0)
    stop = st.number_input("Planned stop (INR)", min_value=0.01, value=980.0)
    quantity = size_long(entry, stop, capital, risk, costs, allocation)
    st.metric("Cash- and risk-limited shares", quantity)
    if quantity:
        exit_fill = stop * (1 - costs.slippage_bps / 10000)
        entry_fees, exit_fees = costs.breakdown(entry, quantity, "BUY"), costs.breakdown(exit_fill, quantity, "SELL")
        st.json({"entry_costs": entry_fees, "stop_exit_costs": exit_fees,
                 "cash_required": entry * quantity + entry_fees["total"],
                 "modeled_stop_loss_rupees": (entry - exit_fill) * quantity + entry_fees["total"] + exit_fees["total"]})
    st.caption("A stop is not a guaranteed maximum loss: gaps, circuits, lack of liquidity and execution failures can exceed it.")