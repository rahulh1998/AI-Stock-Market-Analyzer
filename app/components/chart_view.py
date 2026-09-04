import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, Any, List

def render_candlestick(df: pd.DataFrame, ticker: str, levels: Optional[Dict[str, Any]] = None, 
                       dl_trajectory: Optional[Dict[str, Any]] = None,
                       pricing_envelope: Optional[Dict[str, Any]] = None,
                       buy_markers: Optional[List[Dict[str, Any]]] = None,
                       sell_markers: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    """
    Institutional dual-panel chart (OHLCV + Volume) with:
    - Moving averages (9 EMA, 21 EMA, 50 SMA, 200 SMA)
    - Supertrend line overlay (Green for Bullish, Red for Bearish)
    - Real-Time BUY & SELL execution markers (Green/Red Triangles)
    - Institutional Fair Value & VWAP ±1σ, ±2σ bands
    - Deep Learning Volatility Corridor
    - ATR Execution zones (Entry, SL, Targets)
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data available for {ticker}", template="plotly_dark")
        return fig

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
    for col_name, color, label in [('EMA_9', '#29b6f6', '9 EMA'), ('ema_9', '#29b6f6', '9 EMA'),
                                    ('EMA_21', '#ab47bc', '21 EMA'), ('ema_21', '#ab47bc', '21 EMA'),
                                    ('SMA_50', '#ffa726', '50 SMA'), ('sma_50', '#ffa726', '50 SMA'),
                                    ('SMA_200', '#ef5350', '200 SMA'), ('sma_200', '#ef5350', '200 SMA')]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col_name], mode='lines', name=label,
                                     line=dict(color=color, width=1.3 if '50' not in label and '200' not in label else 1.8)), row=1, col=1)

    # 3. Supertrend Line (if calculated)
    if 'supertrend' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['supertrend'],
            mode='lines',
            name='Supertrend (10, 2.5)',
            line=dict(color='#ffeb3b', width=1.4, dash='dash')
        ), row=1, col=1)

    # 4. Institutional Fair Value & VWAP Bands
    if pricing_envelope:
        fv = pricing_envelope.get("institutional_fair_value")
        if fv and fv > 0:
            fig.add_hline(y=fv, line_dash="solid", line_color="#ffd700", line_width=2.0,
                          annotation_text=f"Fair Value ₹{fv}", row=1, col=1)

        vwap_data = pricing_envelope.get("vwap_microstructure", {})
        if vwap_data and "vwap" in vwap_data:
            vwap_val = vwap_data["vwap"]
            fig.add_hline(y=vwap_val, line_dash="dot", line_color="#00e5ff",
                          annotation_text=f"VWAP ₹{vwap_val}", row=1, col=1)
            
            u2 = vwap_data.get("vwap_upper_2s")
            l2 = vwap_data.get("vwap_lower_2s")
            if u2 and l2:
                fig.add_hline(y=u2, line_dash="dash", line_color="#00bcd4", line_width=1.0,
                              annotation_text=f"VWAP +2σ ₹{u2}", row=1, col=1)
                fig.add_hline(y=l2, line_dash="dash", line_color="#00bcd4", line_width=1.0,
                              annotation_text=f"VWAP -2σ ₹{l2}", row=1, col=1)
                fig.add_hrect(y0=l2, y1=u2, fillcolor="#00e5ff", opacity=0.04, layer="below", row=1, col=1)

    # 5. REAL-TIME INTRADAY BUY & SELL MARKERS
    if buy_markers:
        bx = [m['timestamp'] for m in buy_markers]
        by = [m['marker_y'] for m in buy_markers]
        b_text = [f"🟢 BUY TRIGGER<br>Price: ₹{m['price']}<br>SL: ₹{m['stop_loss']}<br>T1: ₹{m['target_1']}<br>Reason: {m['trigger_type']}" for m in buy_markers]

        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode='markers+text',
            name='BUY Signal',
            marker=dict(symbol='triangle-up', size=14, color='#00e676', line=dict(color='#ffffff', width=1.5)),
            text=["BUY" for _ in bx],
            textposition="bottom center",
            textfont=dict(color='#00e676', size=11, family="Arial Black"),
            hovertext=b_text,
            hoverinfo="text"
        ), row=1, col=1)

    if sell_markers:
        sx = [m['timestamp'] for m in sell_markers]
        sy = [m['marker_y'] for m in sell_markers]
        s_text = [f"🔴 SELL / SHORT<br>Price: ₹{m['price']}<br>SL: ₹{m['stop_loss']}<br>T1: ₹{m['target_1']}<br>Reason: {m['trigger_type']}" for m in sell_markers]

        fig.add_trace(go.Scatter(
            x=sx, y=sy,
            mode='markers+text',
            name='SELL Signal',
            marker=dict(symbol='triangle-down', size=14, color='#ff1744', line=dict(color='#ffffff', width=1.5)),
            text=["SELL" for _ in sx],
            textposition="top center",
            textfont=dict(color='#ff1744', size=11, family="Arial Black"),
            hovertext=s_text,
            hoverinfo="text"
        ), row=1, col=1)

    # 6. Deep Learning Trajectory Envelope
    if dl_trajectory and dl_trajectory.get('predicted_close', 0) > 0:
        pred_close = dl_trajectory['predicted_close']
        pred_high = dl_trajectory.get('predicted_high', pred_close)
        pred_low = dl_trajectory.get('predicted_low', pred_close)

        fig.add_hline(y=pred_close, line_dash="dashdot", line_color="#b388ff",
                      annotation_text=f"DL Target ₹{pred_close}", row=1, col=1)
        fig.add_hrect(y0=pred_low, y1=pred_high, fillcolor="#b388ff", opacity=0.07,
                      layer="below", line_width=0, row=1, col=1)

    # 7. Volume Bar Trace
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.65
    ), row=2, col=1)

    # 8. Execution Levels & Risk/Reward Shading
    if levels and levels.get('stop_loss', 0) > 0:
        entry = levels.get('entry_price', 0.0)
        sl = levels.get('stop_loss', 0.0)
        t1 = levels.get('target_1', 0.0)
        t2 = levels.get('target_2', 0.0)

        fig.add_hline(y=entry, line_dash="dot", line_color="#ffffff", annotation_text=f"Entry ₹{entry}", row=1, col=1)
        fig.add_hline(y=sl, line_dash="solid", line_color="#ff5252", line_width=1.8, annotation_text=f"SL ₹{sl}", row=1, col=1)
        fig.add_hline(y=t1, line_dash="dash", line_color="#00e676", line_width=1.8, annotation_text=f"Target 1 ₹{t1}", row=1, col=1)
        if t2 > 0:
            fig.add_hline(y=t2, line_dash="dash", line_color="#69f0ae", line_width=1.2, annotation_text=f"Target 2 ₹{t2}", row=1, col=1)

        min_risk, max_risk = min(sl, entry), max(sl, entry)
        fig.add_hrect(y0=min_risk, y1=max_risk, fillcolor="#ff5252", opacity=0.12, layer="below", row=1, col=1)

        min_reward, max_reward = min(entry, t1), max(entry, t1)
        fig.add_hrect(y0=min_reward, y1=max_reward, fillcolor="#00e676", opacity=0.12, layer="below", row=1, col=1)

    fig.update_layout(
        title=f"<b>{ticker}</b> — Live Intraday Execution Chart with Real-Time Buy/Sell Markers",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=660,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig