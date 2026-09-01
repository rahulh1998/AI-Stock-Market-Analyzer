import streamlit as st
import pandas as pd
import sqlite3
import os
import joblib
import json
import plotly.graph_objects as go

st.set_page_config(page_title="AI Stock Market Analyzer - NSE", layout="wide")

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
CONFIG_PATH = os.path.join(os.getcwd(), "config", "watchlist.json")
MODEL_PATH = os.path.join(os.getcwd(), "data", "models", "best_model.pkl")

@st.cache_data
def load_watchlist():
    if not os.path.exists(CONFIG_PATH):
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    with open(CONFIG_PATH, "r") as f:
        return json.load(f).get("watchlist", [])

def load_stock_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_sentiment(ticker: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT timestamp, sentiment_score, sentiment_label, summary FROM stock_sentiment WHERE ticker = '{ticker}' ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty:
        return {"sentiment_score": 0.0, "sentiment_label": "NEUTRAL", "summary": "No sentiment data available."}
    return df.iloc[0].to_dict()

# --- UI Layout ---
st.title("📈 AI-Powered NSE Stock Market Analyzer")
st.markdown("Institutional Quant Framework with Dynamic Price Targets, ML Probabilities, & Sentiment.")

watchlist = load_watchlist()
selected_ticker = st.sidebar.selectbox("Select Nifty Ticker", watchlist)

if selected_ticker:
    df = load_stock_data(selected_ticker)
    sentiment = get_sentiment(selected_ticker)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"📊 {selected_ticker} Overview")
        if not df.empty:
            latest_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            change = ((latest_close - prev_close) / prev_close) * 100
            st.metric(label="Latest Close Price", value=f"₹{latest_close:.2f}", delta=f"{change:.2f}%")
        else:
            st.warning("No historical data found in database.")

    with col2:
        st.subheader("📰 Daily News Sentiment")
        st.metric(label="Sentiment Rating", value=sentiment['sentiment_label'], delta=f"Score: {sentiment['sentiment_score']}")
        st.write(f"**Summary:** {sentiment['summary']}")

    with col3:
        st.subheader("🤖 AI Execution & Price Targets")
        
        # ML Model Inference
        ml_prob = 50.0
        if os.path.exists(MODEL_PATH) and not df.empty and len(df) > 50:
            try:
                model = joblib.load(MODEL_PATH)
                temp_df = df.copy()
                temp_df['return_1d'] = temp_df['adj_close'].pct_change(1)
                temp_df['return_3d'] = temp_df['adj_close'].pct_change(3)
                temp_df['return_5d'] = temp_df['adj_close'].pct_change(5)
                temp_df['volatility_14'] = temp_df['return_1d'].rolling(window=14).std()
                temp_df['intraday_spread'] = (temp_df['high'] - temp_df['low']) / temp_df['close']
                temp_df['ema_9'] = temp_df['close'].ewm(span=9, adjust=False).mean()
                temp_df['ema_21'] = temp_df['close'].ewm(span=21, adjust=False).mean()
                temp_df['sma_50'] = temp_df['close'].rolling(window=50).mean()
                temp_df['sma_200'] = temp_df['close'].rolling(window=200).mean()
                
                delta = temp_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                temp_df['rsi_14'] = 100 - (100 / (1 + rs))
                
                exp1 = temp_df['close'].ewm(span=12, adjust=False).mean()
                exp2 = temp_df['close'].ewm(span=26, adjust=False).mean()
                temp_df['macd'] = exp1 - exp2
                temp_df['macd_signal'] = temp_df['macd'].ewm(span=9, adjust=False).mean()
                temp_df['macd_hist'] = temp_df['macd'] - temp_df['macd_signal']
                temp_df['vol_ma_20'] = temp_df['volume'].rolling(window=20).mean()
                temp_df['vol_ratio'] = temp_df['volume'] / (temp_df['vol_ma_20'] + 1e-5)
                
                feature_cols = [
                    'close', 'return_1d', 'return_3d', 'return_5d',
                    'volatility_14', 'intraday_spread',
                    'ema_9', 'ema_21', 'sma_50', 'sma_200',
                    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_ratio'
                ]
                
                latest_features = temp_df[feature_cols].iloc[[-1]].dropna(axis=1)
                if not latest_features.empty and hasattr(model, "predict_proba"):
                    ml_prob = model.predict_proba(latest_features)[0][1] * 100
            except Exception as e:
                pass

        # Calculate ATR for dynamic price targets
        if not df.empty and len(df) > 15:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
        else:
            atr_14 = latest_close * 0.015 # Fallback 1.5% estimate

        sent_score = sentiment['sentiment_score']
        
        # Signal & Price Level Calculation
        if ml_prob > 55.0 and sent_score >= 0.0:
            signal = "🟢 STRONG BUY"
            entry_price = latest_close
            target_price = latest_close + (2.0 * atr_14)
            stop_loss = latest_close - (1.2 * atr_14)
        elif ml_prob < 45.0 or sent_score < -0.3:
            signal = "🔴 SELL / SHORT"
            entry_price = latest_close
            target_price = latest_close - (2.0 * atr_14)
            stop_loss = latest_close + (1.2 * atr_14)
        else:
            signal = "🟡 HOLD / NEUTRAL"
            entry_price = latest_close
            target_price = latest_close + (1.0 * atr_14)
            stop_loss = latest_close - (1.0 * atr_14)

        st.markdown(f"### Signal: **{signal}**")
        st.write(f"**ML Upward Prob:** {ml_prob:.1f}%")
        
        st.markdown("---")
        st.markdown(f"* **Recommended Entry:** ₹{entry_price:.2f}")
        st.markdown(f"* **Target Price (Take Profit):** ₹{target_price:.2f}")
        st.markdown(f"* **Stop-Loss Level:** ₹{stop_loss:.2f}")

    # --- Candlestick Chart ---
    if not df.empty:
        st.subheader("Price Chart & Technical Trend")
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)