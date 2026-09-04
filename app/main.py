import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import time
from datetime import datetime

from src.data_pipeline.broker_client import MarketDataFetcher
from src.data_pipeline.database_manager import DatabaseManager
from src.sentiment_engine.sentiment_analyzer import MultiHorizonSentimentAnalyzer
from src.ml_engine.deep_learning_predictor import DeepLearningPredictor
from src.ml_engine.candle_patterns import detect_candlestick_patterns
from src.quant_engine.risk_math import RiskEngine
from src.quant_engine.pricing_engine import InstitutionalPricingEngine
from src.agents.state import AgentTradingState
from src.agents.agent_nodes import technical_agent, rag_agent, sentiment_agent, bear_advocate, lead_synthesizer
from src.rag_engine.retriever import StrategyRetriever
from app.components.chart_view import render_candlestick
from app.components.agent_cards import render_agent_debate

st.set_page_config(
    page_title="AI Stock Market Analyzer — NSE Institutional Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
CONFIG_PATH = os.path.join(os.getcwd(), "config", "watchlist.json")
MODEL_PATH = os.path.join(os.getcwd(), "data", "models", "best_model.pkl")

# Initialize services
@st.cache_resource
def get_services():
    fetcher = MarketDataFetcher(cache_ttl_seconds=45)
    db_mgr = DatabaseManager()
    sentiment_eng = MultiHorizonSentimentAnalyzer()
    dl_predictor = DeepLearningPredictor()
    risk_engine = RiskEngine()
    pricing_engine = InstitutionalPricingEngine(num_simulations=1000, forecast_steps=5)
    rag_retriever = StrategyRetriever()
    return fetcher, db_mgr, sentiment_eng, dl_predictor, risk_engine, pricing_engine, rag_retriever

fetcher, db_mgr, sentiment_eng, dl_predictor, risk_engine, pricing_engine, rag_retriever = get_services()

@st.cache_data
def load_watchlist() -> list:
    if not os.path.exists(CONFIG_PATH):
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    with open(CONFIG_PATH, "r") as f:
        return json.load(f).get("watchlist", [])

watchlist = load_watchlist()

# --- Sidebar Configuration ---
st.sidebar.title("🎛️ Terminal Controls")
st.sidebar.markdown("**NSE Watchlist:** 84 Tracked Equities")

selected_ticker = st.sidebar.selectbox("🎯 Select Stock", watchlist, index=0)
timeframe = st.sidebar.selectbox("⏱️ Timeframe Resolution", ["15m", "5m", "1m", "1h", "1d"], index=0)
chart_period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "6mo"}
selected_period = chart_period_map.get(timeframe, "5d")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk & Portfolio Sizing")
user_capital = st.sidebar.number_input("Account Balance (₹)", min_value=10000.0, value=200000.0, step=10000.0)
user_risk_pct = st.sidebar.slider("Max Capital Risk per Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
risk_engine.default_capital = user_capital
risk_engine.max_risk_per_trade_pct = user_risk_pct

# --- Header Section ---
st.title("📈 AI-Powered NSE Stock Market Analyzer")
st.caption("Institutional Intelligence Terminal | Microstructure VWAP • Merton Jump-Diffusion Monte Carlo • Bi-LSTM Sequences • Multi-Horizon Sentiment • Multi-Agent Debate")

tab1, tab2 = st.tabs(["📊 Watchlist Scanner & Live Heatmap", "🔍 Single-Stock Deep Dive"])

# ==============================================================================
# TAB 1: WATCHLIST SCANNER & HEATMAP
# ==============================================================================
with tab1:
    st.subheader("⚡ Live Market Overview & Watchlist Scanner")
    col_scan1, col_scan2, col_scan3 = st.columns([2, 1, 1])

    with col_scan1:
        search_filter = st.text_input("🔍 Filter stocks by ticker name", "").upper().strip()
    with col_scan2:
        sort_by = st.selectbox("Sort By", ["Pct Change (High to Low)", "Pct Change (Low to High)", "Price", "Alphabetical"])
    with col_scan3:
        st.write("")
        st.write("")
        btn_refresh = st.button("🔄 Refresh Live Quotes")

    # Filter tickers
    filtered_tickers = [t for t in watchlist if search_filter in t] if search_filter else watchlist[:40]

    with st.spinner("Fetching high-speed concurrent live quotes for watchlist..."):
        quotes_dict = fetcher.fetch_live_quotes_batch(filtered_tickers)

    rows = []
    for t in filtered_tickers:
        q = quotes_dict.get(t, {})
        price = q.get("price", 0.0)
        pct = q.get("pct_change", 0.0)
        high = q.get("day_high", 0.0)
        low = q.get("day_low", 0.0)
        vol = q.get("volume", 0)

        # Retrieve cached sentiment
        sent = db_mgr.get_latest_multi_horizon_sentiment(t)
        s_1h = sent.get("sentiment_1h", 0.0)
        s_1d = sent.get("sentiment_1d", 0.0)
        s_1w = sent.get("sentiment_1w", 0.0)
        div = sent.get("divergence_flag", "CONGRUENT")

        # Quick Signal classification
        if pct > 0.8 and s_1d >= 0.0:
            quick_sig = "🟢 STRONG BUY"
        elif pct < -0.8 or s_1d < -0.2:
            quick_sig = "🔴 SELL / SHORT"
        else:
            quick_sig = "🟡 HOLD"

        rows.append({
            "Ticker": t,
            "Price (₹)": price,
            "Change (%)": pct,
            "Day Low (₹)": low,
            "Day High (₹)": high,
            "Volume": f"{vol:,}",
            "1h Sent": f"{s_1h:+.2f}",
            "1d Sent": f"{s_1d:+.2f}",
            "1w Sent": f"{s_1w:+.2f}",
            "Divergence Alert": "⚠️ Divergence" if "DIVERGENCE" in div else "—",
            "Action Bias": quick_sig
        })

    df_table = pd.DataFrame(rows)

    if not df_table.empty:
        if sort_by == "Pct Change (High to Low)":
            df_table = df_table.sort_values(by="Change (%)", ascending=False)
        elif sort_by == "Pct Change (Low to High)":
            df_table = df_table.sort_values(by="Change (%)", ascending=True)
        elif sort_by == "Price":
            df_table = df_table.sort_values(by="Price (₹)", ascending=False)
        else:
            df_table = df_table.sort_values(by="Ticker", ascending=True)

        # Summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        top_gainer = df_table.iloc[0]
        top_loser = df_table.iloc[-1]
        advances = len(df_table[df_table["Change (%)"] > 0])
        declines = len(df_table[df_table["Change (%)"] < 0])

        kpi1.metric("Market Breadth (Advances / Declines)", f"{advances} / {declines}")
        kpi2.metric("Top Gainer", f"{top_gainer['Ticker']}", f"{top_gainer['Change (%)']:+.2f}%")
        kpi3.metric("Top Loser", f"{top_loser['Ticker']}", f"{top_loser['Change (%)']:+.2f}%")
        kpi4.metric("Monitored Universe", f"{len(df_table)} Equities")

        st.dataframe(
            df_table.reset_index(drop=True),
            use_container_width=True,
            height=480
        )
    else:
        st.info("No matching stocks found for the filter.")

# ==============================================================================
# TAB 2: SINGLE STOCK DEEP DIVE
# ==============================================================================
with tab2:
    st.subheader(f"🔍 Deep Dive Analysis: **{selected_ticker}**")

    # Load Candles & Historical Data
    with st.spinner(f"Loading {timeframe} market data and running Institutional Pricing Engine for {selected_ticker}..."):
        df_candles = fetcher.fetch_intraday_data(selected_ticker, period=selected_period, interval=timeframe)
        live_quote = fetcher.fetch_live_quote_single(selected_ticker)
        
        df_daily = db_mgr.get_stock_data(selected_ticker)
        if df_daily.empty and not df_candles.empty:
            df_daily = df_candles

    # Latest Sentiment & Price Divergence
    sentiment_data = sentiment_eng.analyze_ticker(selected_ticker, live_pct_change=live_quote.get("pct_change", 0.0))
    s_1h = sentiment_data.get("sentiment_1h", 0.0)
    s_1d = sentiment_data.get("sentiment_1d", 0.0)
    s_1w = sentiment_data.get("sentiment_1w", 0.0)
    div_flag = sentiment_data.get("divergence_flag", "CONGRUENT")

    # Deep Learning Sequence Prediction
    dl_bounds = dl_predictor.predict_trajectory(df_daily if not df_daily.empty else df_candles)

    # Tabular ML Directional Probability
    ml_prob = 50.0
    if os.path.exists(MODEL_PATH) and not df_daily.empty and len(df_daily) > 30:
        try:
            model = joblib.load(MODEL_PATH)
            temp = df_daily.copy()
            temp['return_1d'] = temp['adj_close'].pct_change(1)
            temp['return_3d'] = temp['adj_close'].pct_change(3)
            temp['return_5d'] = temp['adj_close'].pct_change(5)
            temp['volatility_14'] = temp['return_1d'].rolling(window=14, min_periods=1).std()
            temp['intraday_spread'] = (temp['high'] - temp['low']) / (temp['close'] + 1e-5)
            temp['ema_9'] = temp['close'].ewm(span=9, adjust=False).mean()
            temp['ema_21'] = temp['close'].ewm(span=21, adjust=False).mean()
            temp['sma_50'] = temp['close'].rolling(window=50, min_periods=1).mean()
            temp['sma_200'] = temp['close'].rolling(window=200, min_periods=1).mean()

            delta = temp['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-5)
            temp['rsi_14'] = 100 - (100 / (1 + rs))

            exp1 = temp['close'].ewm(span=12, adjust=False).mean()
            exp2 = temp['close'].ewm(span=26, adjust=False).mean()
            temp['macd'] = exp1 - exp2
            temp['macd_signal'] = temp['macd'].ewm(span=9, adjust=False).mean()
            temp['macd_hist'] = temp['macd'] - temp['macd_signal']
            temp['vol_ma_20'] = temp['volume'].rolling(window=20, min_periods=1).mean()
            temp['vol_ratio'] = temp['volume'] / (temp['vol_ma_20'] + 1e-5)

            feature_cols = [
                'close', 'return_1d', 'return_3d', 'return_5d',
                'volatility_14', 'intraday_spread',
                'ema_9', 'ema_21', 'sma_50', 'sma_200',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_ratio'
            ]
            latest_feat = temp[feature_cols].iloc[[-1]].dropna(axis=1)
            if not latest_feat.empty and hasattr(model, "predict_proba"):
                ml_prob = float(model.predict_proba(latest_feat)[0][1] * 100)
        except Exception:
            ml_prob = 50.0

    # Institutional Pricing Engine Envelope
    pricing_info = pricing_engine.calculate_fair_value_envelope(
        df=df_daily if not df_daily.empty else df_candles,
        dl_forecast=dl_bounds,
        ml_prob=ml_prob
    )

    current_price = live_quote.get("price", 0.0)
    if current_price <= 0 and not df_candles.empty:
        current_price = float(df_candles['close'].iloc[-1])

    atr_val = current_price * 0.015
    if not df_candles.empty and 'ATRr_14' in df_candles.columns:
        atr_val = float(df_candles['ATRr_14'].iloc[-1])
    elif not df_daily.empty and 'ATRr_14' in df_daily.columns:
        atr_val = float(df_daily['ATRr_14'].iloc[-1])

    # Action Determination incorporating Pricing Engine Alpha Edge
    dl_bias = dl_bounds.get("trajectory_bias", "NEUTRAL")
    mispricing_edge = pricing_info.get("mispricing_edge_pct", 0.0)
    regime = pricing_info.get("market_regime", "MEAN_REVERTING")

    if (mispricing_edge >= 0.5 or ml_prob > 54.0 or dl_bias == "BULLISH") and s_1d >= -0.1 and regime != "HIGH_VOLATILITY_SHOCK":
        action_decision = "BUY"
    elif mispricing_edge <= -0.7 or ml_prob < 46.0 or dl_bias == "BEARISH" or s_1d < -0.3:
        action_decision = "SELL"
    else:
        action_decision = "HOLD"

    levels = risk_engine.calculate_levels(
        current_price=current_price,
        atr_14=atr_val,
        action=action_decision,
        dl_bounds=dl_bounds
    )
    sizing = risk_engine.calculate_position_sizing(
        current_price=current_price,
        stop_loss=levels["stop_loss"]
    )

    # Technical Guardrails Check
    tech_snapshot = {
        "close": current_price,
        "SMA_200": float(df_daily['SMA_50'].iloc[-1]) if 'SMA_50' in df_daily.columns else current_price * 0.95,
        "RSI_14": float(df_daily['RSI_14'].iloc[-1]) if 'RSI_14' in df_daily.columns else 50.0
    }
    guardrail_res = risk_engine.verify_signal_guardrails(
        signal={"action": action_decision, "confidence_score": int(ml_prob)},
        technical_snapshot=tech_snapshot
    )

    # --- Top Metrics Bar ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Live Market Price", f"₹{current_price:.2f}", f"{live_quote.get('pct_change', 0.0):+.2f}%")
    m2.metric("Institutional Fair Value", f"₹{pricing_info.get('institutional_fair_value', current_price):.2f}", f"Edge: {mispricing_edge:+.2f}%")
    m3.metric("Deep Learning Target", f"₹{dl_bounds.get('predicted_close', current_price):.2f}", f"Envelope: ₹{dl_bounds.get('predicted_low', 0):.0f} – ₹{dl_bounds.get('predicted_high', 0):.0f}")
    m4.metric("Market Regime", f"{regime.replace('_', ' ')}", f"Conf: {pricing_info.get('regime_confidence', 75)}%")
    m5.metric("System Signal", f"{action_decision}", f"Guardrail: {'PASS' if guardrail_res['is_approved'] else 'VETO'}")

    # Divergence Warning Banner
    if "DIVERGENCE" in div_flag:
        st.warning(f"⚠️ **Sentiment-Price Divergence Detected**: {div_flag}")

    # ==============================================================================
    # 🏛️ INSTITUTIONAL PRICING ENGINE SCORECARD
    # ==============================================================================
    with st.container():
        st.markdown("### 🏛️ Institutional Pricing Engine Analysis")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        with p_col1:
            st.markdown("#### 🎯 Valuation Anchor")
            st.markdown(f"**Theoretical Fair Value:** ₹{pricing_info.get('institutional_fair_value', current_price):.2f}")
            st.markdown(f"**Current Market Price:** ₹{current_price:.2f}")
            st.markdown(f"**Mispricing Edge:** **{mispricing_edge:+.2f}%**")
            st.caption(f"Status: **{pricing_info.get('valuation_status', 'FAIRLY_VALUED')}**")

        with p_col2:
            st.markdown("#### 🌊 Microstructure & VWAP")
            vwap_meta = pricing_info.get("vwap_microstructure", {})
            st.markdown(f"**Anchored VWAP:** ₹{vwap_meta.get('vwap', current_price):.2f}")
            st.markdown(f"**VWAP +2σ Band:** ₹{vwap_meta.get('vwap_upper_2s', 0):.2f}")
            st.markdown(f"**VWAP -2σ Band:** ₹{vwap_meta.get('vwap_lower_2s', 0):.2f}")
            st.caption(f"Intraday Dispersion: ±{vwap_meta.get('vwap_dispersion', 1.0)}%")

        with p_col3:
            st.markdown("#### 🎲 Jump-Diffusion Simulation")
            mc = pricing_info.get("monte_carlo_simulation", {})
            st.markdown(f"**Expected Price (5-day):** ₹{mc.get('sim_expected_price', current_price):.2f}")
            st.markdown(f"**Upside 95th Percentile:** ₹{mc.get('sim_p95_upside', current_price):.2f}")
            st.markdown(f"**5-Day 95% VaR:** **{mc.get('var_95_pct', 0.0)}%** (₹{mc.get('var_95_rupees', 0.0)})")
            st.caption(f"Monte Carlo Probability of Gain: **{mc.get('monte_carlo_prob_up', 50)}%**")

        with p_col4:
            st.markdown("#### ⚡ Multi-Factor Scorecard")
            f_score = pricing_info.get("factor_scores", {})
            st.markdown(f"**Composite Alpha:** **{f_score.get('composite_alpha', 0.0):+0.1f}** / 100")
            st.markdown(f"**Momentum Factor:** {f_score.get('momentum_factor', 0.0):+0.1f}")
            st.markdown(f"**Volume Force Factor:** {f_score.get('volume_force_factor', 0.0):+0.1f}")
            st.caption(f"Mean Reversion Factor: {f_score.get('mean_reversion_factor', 0.0):+0.1f}")

    # --- Candlestick Chart ---
    st.subheader(f"📊 {selected_ticker} ({timeframe}) Institutional Chart & Execution Envelope")
    fig = render_candlestick(
        df_candles, selected_ticker,
        levels=levels,
        dl_trajectory=dl_bounds,
        pricing_envelope=pricing_info
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Quantitative Execution & Guardrails ---
    col_exec1, col_exec2, col_exec3 = st.columns(3)

    with col_exec1:
        st.markdown("### 🎯 Trade Execution Levels")
        st.markdown(f"* **Action:** **{action_decision}**")
        st.markdown(f"* **Entry Price:** ₹{levels['entry_price']:.2f}")
        st.markdown(f"* **Stop-Loss (Risk Bound):** ₹{levels['stop_loss']:.2f}")
        st.markdown(f"* **Target 1 (1:2 RR):** ₹{levels['target_1']:.2f}")
        if levels.get("target_2", 0) > 0:
            st.markdown(f"* **Target 2 (1:3 RR):** ₹{levels['target_2']:.2f}")
        st.markdown(f"* **Risk-to-Reward:** **1 : {levels['risk_reward_ratio']}**")

    with col_exec2:
        st.markdown("### 🛡️ 1% Capital Risk Sizing")
        st.markdown(f"* **Account Capital:** ₹{user_capital:,.2f}")
        st.markdown(f"* **Max Rupee Risk (1%):** ₹{sizing['max_rupee_risk']:.2f}")
        st.markdown(f"* **Recommended Shares:** **{sizing['quantity']} units**")
        st.markdown(f"* **Total Capital Exposure:** ₹{sizing['total_exposure']:,.2f}")
        if current_price > 0:
            st.markdown(f"* **Portfolio Allocation:** {((sizing['total_exposure'] / user_capital) * 100):.1f}%")

    with col_exec3:
        st.markdown("### ⚖️ Deterministic Guardrails")
        if guardrail_res["is_approved"]:
            st.success("✅ **ALL GUARDRAILS PASSED**")
            st.write("• Price regime above 200 SMA threshold.")
            st.write("• RSI momentum not overbought (< 75).")
            st.write("• Confidence meets minimum execution criteria.")
        else:
            st.error(f"🚫 **TRADE VETOED**: {guardrail_res['veto_reason']}")
            st.write(f"• System enforces **{guardrail_res['revised_action']}** override.")

    # --- Multi-Horizon Sentiment Breakdown ---
    st.markdown("---")
    st.subheader("📰 Multi-Horizon News Sentiment Dynamics")
    sent_c1, sent_c2, sent_c3 = st.columns(3)

    with sent_c1:
        st.metric("1-Hour Flash Sentiment", f"{sentiment_data.get('label_1h', 'NEUTRAL')}", f"{s_1h:+.2f}")
        st.caption("Captures breaking news, sudden announcements, and flash catalysts.")
    with sent_c2:
        st.metric("24-Hour Daily Sentiment", f"{sentiment_data.get('label_1d', 'NEUTRAL')}", f"{s_1d:+.2f}")
        st.caption("Captures daily session digest, overnight global cues, and earnings.")
    with sent_c3:
        st.metric("7-Day Trend Sentiment", f"{sentiment_data.get('label_1w', 'NEUTRAL')}", f"{s_1w:+.2f}")
        st.caption("Captures macro drift, institutional upgrades, and sector trends.")

    with st.expander("📄 Recent Financial Headlines Analyzed", expanded=False):
        headlines = sentiment_data.get("latest_headlines", [])
        if headlines:
            for h in headlines:
                st.write(f"• {h}")
        else:
            st.write("No direct headlines retrieved in the current query window.")

    # --- 15 Candlestick Pattern Rules ---
    st.markdown("---")
    pattern_df = detect_candlestick_patterns(df_candles.copy())
    with st.expander("🕯️ View 15 Candlestick Pattern Rules (Latest Session Status)", expanded=False):
        pattern_cols = [
            'Doji', 'Bullish Marubozu', 'Bearish Marubozu', 'Hammer', 'Shooting Star',
            'Bullish Engulfing', 'Bearish Engulfing', 'Piercing Line', 'Dark Cloud Cover',
            'Morning Star', 'Evening Star', 'Three White Soldiers', 'Three Black Crows',
            'Spinning Top', 'Hanging Man'
        ]
        if not pattern_df.empty:
            latest_patterns = pattern_df.iloc[-1]
            cols = st.columns(3)
            for idx, pat in enumerate(pattern_cols):
                is_detected = bool(latest_patterns.get(pat, False))
                with cols[idx % 3]:
                    if is_detected:
                        st.success(f"✅ **{pat}**: DETECTED")
                    else:
                        st.text(f"⚪ {pat}: Inactive")

    # --- LangGraph Multi-Agent Assembly Line ---
    st.markdown("---")
    st.subheader("🤖 LangGraph Multi-Agent Assembly Line")
    st.markdown("Orchestrates **Technical Analyst → RAG Strategy Expert → Multi-Horizon Sentiment Analyst → Bear Advocate → Lead Portfolio Manager**.")

    if st.button("🚀 Trigger Multi-Agent Debate for " + selected_ticker):
        with st.spinner("Assembling agent panel and running debate workflow..."):
            tech_desc = f"LTP: ₹{current_price}, 1-Day Change: {live_quote.get('pct_change', 0.0)}%, ML Prob: {ml_prob:.1f}%, ATR: ₹{atr_val:.2f}"
            rag_rules = rag_retriever.get_trading_rules(f"Trading setup and momentum rules for {selected_ticker}", k=2)

            state: AgentTradingState = {
                "ticker": selected_ticker,
                "current_price": current_price,
                "technical_data": tech_desc,
                "rag_context": rag_rules,
                "sentiment_data": sentiment_data.get("summary", ""),
                "sentiment_1h": s_1h,
                "sentiment_1d": s_1d,
                "sentiment_1w": s_1w,
                "divergence_flag": div_flag,
                "dl_trajectory": dl_bounds,
                "institutional_fair_value": pricing_info.get("institutional_fair_value", current_price),
                "mispricing_edge_pct": mispricing_edge,
                "valuation_status": pricing_info.get("valuation_status", "FAIRLY_VALUED"),
                "market_regime": regime,
                "auction_corridor": pricing_info.get("auction_corridor", {}),
                "monte_carlo_var95": pricing_info.get("monte_carlo_simulation", {}).get("var_95_pct", 3.0)
            }

            # Run sequential assembly line
            state.update(technical_agent(state))
            state.update(rag_agent(state))
            state.update(sentiment_agent(state))
            state.update(bear_advocate(state))
            state.update(lead_synthesizer(state))

            render_agent_debate(state)