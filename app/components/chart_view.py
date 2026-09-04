import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, Any

def render_candlestick(df: pd.DataFrame, ticker: str, levels: Optional[Dict[str, Any]] = None, 
                       dl_trajectory: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    Generates an institutional-grade dual-panel Plotly chart (OHLCV + Volume)
    with moving averages, ATR execution levels, and Deep Learning forecast trajectory.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data available for {ticker}", template="plotly_dark")
        return fig

    # Create subplots: Row 1 = Price + Indicators, Row 2 = Volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22]
    )

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price (OHLC)',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # 2. Moving Averages
    if 'EMA_9' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_9'], mode='lines', name='9 EMA', line=dict(color='#29b6f6', width=1.2)), row=1, col=1)
    elif 'ema_9' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_9'], mode='lines', name='9 EMA', line=dict(color='#29b6f6', width=1.2)), row=1, col=1)

    if 'EMA_21' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_21'], mode='lines', name='21 EMA', line=dict(color='#ab47bc', width=1.2)), row=1, col=1)
    elif 'ema_21' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_21'], mode='lines', name='21 EMA', line=dict(color='#ab47bc', width=1.2)), row=1, col=1)

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], mode='lines', name='50 SMA', line=dict(color='#ffa726', width=1.5)), row=1, col=1)
    elif 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], mode='lines', name='50 SMA', line=dict(color='#ffa726', width=1.5)), row=1, col=1)

    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_200'], mode='lines', name='200 SMA', line=dict(color='#ef5350', width=1.8, dash='dot')), row=1, col=1)
    elif 'sma_200' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_200'], mode='lines', name='200 SMA', line=dict(color='#ef5350', width=1.8, dash='dot')), row=1, col=1)

    # 3. Volume Bar Trace
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.65
    ), row=2, col=1)

    # 4. Deep Learning Trajectory Envelope (if available)
    if dl_trajectory and dl_trajectory.get('predicted_close', 0) > 0:
        latest_time = df['timestamp'].iloc[-1]
        latest_price = float(df['close'].iloc[-1])
        pred_close = dl_trajectory['predicted_close']
        pred_high = dl_trajectory.get('predicted_high', pred_close)
        pred_low = dl_trajectory.get('predicted_low', pred_close)

        fig.add_hline(y=pred_close, line_dash="dashdot", line_color="#00e5ff",
                      annotation_text=f"DL Target: ₹{pred_close}", row=1, col=1)
        fig.add_hrect(y0=pred_low, y1=pred_high, fillcolor="#00e5ff", opacity=0.08,
                      layer="below", line_width=0, row=1, col=1)

    # 5. Execution Levels & Risk/Reward Shading
    if levels and levels.get('stop_loss', 0) > 0:
        entry = levels.get('entry_price', 0.0)
        sl = levels.get('stop_loss', 0.0)
        t1 = levels.get('target_1', 0.0)
        t2 = levels.get('target_2', 0.0)

        fig.add_hline(y=entry, line_dash="dot", line_color="#ffffff", annotation_text=f"Entry ₹{entry}", row=1, col=1)
        fig.add_hline(y=sl, line_dash="solid", line_color="#ff5252", annotation_text=f"SL ₹{sl}", row=1, col=1)
        fig.add_hline(y=t1, line_dash="dash", line_color="#00e676", annotation_text=f"Target 1 ₹{t1}", row=1, col=1)
        if t2 > 0:
            fig.add_hline(y=t2, line_dash="dash", line_color="#69f0ae", annotation_text=f"Target 2 ₹{t2}", row=1, col=1)

        # Risk zone
        min_risk = min(sl, entry)
        max_risk = max(sl, entry)
        fig.add_hrect(y0=min_risk, y1=max_risk, fillcolor="#ff5252", opacity=0.12, layer="below", row=1, col=1)

        # Reward zone
        min_reward = min(entry, t1)
        max_reward = max(entry, t1)
        fig.add_hrect(y0=min_reward, y1=max_reward, fillcolor="#00e676", opacity=0.12, layer="below", row=1, col=1)

    fig.update_layout(
        title=f"<b>{ticker}</b> — Real-Time Technical Action & Execution Envelope",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig