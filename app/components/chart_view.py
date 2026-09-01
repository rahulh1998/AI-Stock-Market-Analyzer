import plotly.graph_objects as go
import pandas as pd

def render_candlestick(df: pd.DataFrame, ticker: str, levels: dict = None):
    """
    Generates an interactive Plotly candlestick chart with risk/reward zones.
    """
    fig = go.Figure()

    # Add the Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # Add Moving Averages
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='blue', width=1)))
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], mode='lines', name='50 SMA', line=dict(color='orange', width=1.5)))

    # Overlay Trade Execution Levels if a signal exists
    if levels and levels.get('stop_loss', 0) > 0:
        fig.add_hline(y=levels['entry_price'], line_dash="dot", line_color="white", annotation_text="Entry")
        fig.add_hline(y=levels['stop_loss'], line_dash="solid", line_color="red", annotation_text="Stop Loss (1.5 ATR)")
        fig.add_hline(y=levels['target_1'], line_dash="dash", line_color="green", annotation_text="Target 1 (1:2 RR)")
        
        # Highlight the risk zone (Entry to Stop Loss)
        fig.add_hrect(y0=levels['stop_loss'], y1=levels['entry_price'], fillcolor="red", opacity=0.1, layer="below")
        # Highlight the reward zone (Entry to Target 1)
        fig.add_hrect(y0=levels['entry_price'], y1=levels['target_1'], fillcolor="green", opacity=0.1, layer="below")

    fig.update_layout(
        title=f"{ticker} - Market Action",
        yaxis_title="Price (INR)",
        xaxis_title="Date",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig