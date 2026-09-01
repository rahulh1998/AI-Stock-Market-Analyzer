import streamlit as st
import sys
import os

# Ensure python path is correct for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline.broker_client import MarketDataFetcher
from src.quant_engine.risk_math import RiskEngine
from src.agents.orchestrator import build_workflow
from app.components.chart_view import render_candlestick
from app.components.agent_cards import render_agent_debate

# Configure Streamlit page
st.set_page_config(page_title="AI Stock Analyzer", layout="wide", page_icon="📈")

st.title("Institutional AI Stock Analyzer")
st.markdown("Multi-Agent reasoning engine with local Llama 3.2 and deterministic risk guardrails.")

# Sidebar Inputs
with st.sidebar:
    st.header("Trade Parameters")
    ticker = st.text_input("NSE Ticker Symbol", value="TATAMOTORS.NS").upper()
    capital = st.number_input("Account Capital (₹)", min_value=10000, value=200000, step=10000)
    risk_pct = st.slider("Max Risk per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    run_analysis = st.button("Run Multi-Agent Analysis", type="primary", use_container_width=True)

if run_analysis:
    with st.spinner("Fetching market data..."):
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_historical_data(ticker, period="6mo", interval="1d")
        
    if df.empty:
        st.error("Failed to fetch data. Check the ticker symbol.")
        st.stop()
        
    with st.spinner("Calculating indicators..."):
        df = fetcher.enrich_with_indicators(df)
        latest_data = fetcher.get_latest_features(df)
        current_price = latest_data.get("close", 0)
        atr_14 = latest_data.get("ATRr_14", 10) # Fallback to 10 if missing

    with st.spinner("Agents are debating the setup (this takes 30-60 seconds)..."):
        # Initialize state for LangGraph
        initial_state = {
            "ticker": ticker,
            "current_price": current_price,
            "technical_data": str(latest_data),
            "rag_context": "Identify patterns like Engulfing, Hammer, or MA Crossovers.",
            "sentiment_data": "Assume neutral macroeconomic sentiment for now."
        }
        
        graph = build_workflow()
        final_state = graph.invoke(initial_state)
        
    # Parse output and apply risk math
    signal = final_state.get("final_trade_signal", {})
    action = signal.get("action", "HOLD")
    
    risk_engine = RiskEngine(default_capital=capital, max_risk_per_trade_pct=risk_pct)
    levels = risk_engine.calculate_levels(current_price, atr_14, action)
    sizing = risk_engine.calculate_position_sizing(current_price, levels.get("stop_loss", 0), capital)
    
    # --- UI Rendering ---
    
    # 1. Top Action Banner
    if action == "BUY":
        st.success(f"### 🟢 VERDICT: BUY | Confidence: {signal.get('confidence_score', 'N/A')}%")
    elif action == "SELL":
        st.error(f"### 🔴 VERDICT: SELL | Confidence: {signal.get('confidence_score', 'N/A')}%")
    else:
        st.warning(f"### 🟡 VERDICT: HOLD / NO TRADE")
        
    st.info(f"**Lead Synthesizer:** {signal.get('reasoning', 'No reasoning provided.')}")

    # 2. Main Chart
    st.plotly_chart(render_candlestick(df, ticker, levels), use_container_width=True)

    # 3. Position Sizing Metrics
    if action in ["BUY", "SELL"]:
        st.subheader("⚖️ Risk & Position Sizing")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Quantity", f"{sizing['quantity']} Shares")
        m2.metric("Total Capital Required", f"₹ {sizing['total_exposure']:,.2f}")
        m3.metric(f"Max Rupee Risk ({risk_pct}%)", f"₹ {sizing['max_rupee_risk']:,.2f}")
        m4.metric("Risk:Reward Ratio", f"1 : {levels['risk_reward_ratio']}")

    st.markdown("---")
    
    # 4. Agent Debate Logs
    render_agent_debate(final_state)