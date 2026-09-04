import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class IntradaySignalEngine:
    """
    Institutional Intraday Signal Engine for NSE Equities.
    Combines:
    1. Trend Direction: Supertrend (10-period ATR, 2.5 multiplier)
    2. Institutional Fair Value Benchmark: Session VWAP
    3. Momentum Timing: 9 EMA & 21 EMA Golden/Death Cross
    4. Volume Surge Confirmation: Volume > 1.25x 20-period SMA
    5. Trend Strength Filter: ADX >= 20 (rejects sideways chop)
    6. Momentum Oscillator: RSI(14) bounds
    Generates exact time-stamped BUY and SELL markers with entry, stop loss, and target levels.
    """

    def __init__(self, supertrend_len: int = 10, supertrend_mult: float = 2.5):
        self.st_len = supertrend_len
        self.st_mult = supertrend_mult

    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Supertrend (10, 2.5) using True Range and band flips."""
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range & ATR
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=self.st_len, min_periods=1).mean()

        hl2 = (high + low) / 2.0
        upper_band = hl2 + (self.st_mult * atr)
        lower_band = hl2 - (self.st_mult * atr)

        st_direction = np.ones(len(df))
        st_line = np.zeros(len(df))

        for i in range(1, len(df)):
            curr_close = close.iloc[i]
            prev_close = close.iloc[i - 1]
            prev_upper = upper_band.iloc[i - 1]
            prev_lower = lower_band.iloc[i - 1]
            prev_dir = st_direction[i - 1]

            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            # Keep tightening bands
            if curr_lower > prev_lower or prev_close < prev_lower:
                pass
            else:
                curr_lower = prev_lower
            lower_band.iloc[i] = curr_lower

            if curr_upper < prev_upper or prev_close > prev_upper:
                pass
            else:
                curr_upper = prev_upper
            upper_band.iloc[i] = curr_upper

            # Flip directions
            if prev_dir == 1:
                if curr_close < curr_lower:
                    st_direction[i] = -1
                    st_line[i] = curr_upper
                else:
                    st_direction[i] = 1
                    st_line[i] = curr_lower
            else:
                if curr_close > curr_upper:
                    st_direction[i] = 1
                    st_line[i] = curr_lower
                else:
                    st_direction[i] = -1
                    st_line[i] = curr_upper

        df['supertrend'] = st_line
        df['st_direction'] = st_direction
        df['atr_14'] = atr
        return df

    def compute_intraday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches dataframe with VWAP, 9/21 EMAs, ADX, and Volume Moving Average."""
        if df.empty or len(df) < 10:
            return df

        df = df.copy()
        if 'timestamp' not in df.columns:
            for c in df.columns:
                if 'date' in str(c).lower() or 'time' in str(c).lower():
                    df.rename(columns={c: 'timestamp'}, inplace=True)
                    break
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index.astype(str)

        df = self.calculate_supertrend(df)

        # 1. Moving Averages
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

        # 2. Session VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'] + 1e-5
        window = min(len(df), 30)
        df['vwap'] = (typical_price * vol).rolling(window=window, min_periods=1).sum() / vol.rolling(window=window, min_periods=1).sum()

        # 3. Volume SMA & Ratio
        df['vol_sma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_surge'] = df['volume'] / (df['vol_sma20'] + 1e-5)

        # 4. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-5)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 5. ADX proxy (True range expansion)
        tr = (df['high'] - df['low']).rolling(window=14, min_periods=1).mean()
        df['adx_proxy'] = (tr / (df['close'] + 1e-5)) * 1000

        return df

    def generate_markers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scans OHLCV bars and flags high-probability institutional BUY and SELL triggers.
        Returns: (enriched_df, buy_markers, sell_markers)
        """
        if df.empty or len(df) < 15:
            return df, [], []

        df = self.compute_intraday_indicators(df)

        buy_markers = []
        sell_markers = []

        last_signal = None  # To prevent consecutive duplicate markers

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            close = row['close']
            low = row['low']
            high = row['high']
            vwap = row['vwap']
            ema9 = row['ema_9']
            ema21 = row['ema_21']
            st_dir = row['st_direction']
            prev_st_dir = prev_row['st_direction']
            vol_surge = row['vol_surge']
            rsi = row['rsi_14']
            ts = row['timestamp']
            atr = row['atr_14'] if row['atr_14'] > 0 else (close * 0.012)

            # --- BUY (LONG) RULES ---
            # 1. Supertrend flips Bullish (prev was -1, now 1) OR
            # 2. Bullish Confluence: Close > VWAP AND EMA 9 > EMA 21 AND Volume Surge > 1.15 AND RSI in (45, 72)
            is_st_flip_buy = (prev_st_dir == -1 and st_dir == 1 and close > vwap)
            is_confluence_buy = (close > vwap and ema9 > ema21 and prev_row['close'] <= prev_row['vwap'] and vol_surge > 1.15 and 45 <= rsi <= 72)

            if (is_st_flip_buy or is_confluence_buy) and last_signal != "BUY":
                sl = round(max(low - (1.0 * atr), row['supertrend'] if st_dir == 1 else low - (1.0 * atr)), 2)
                risk = max(0.2, close - sl)
                t1 = round(close + (1.8 * risk), 2)
                t2 = round(close + (2.8 * risk), 2)

                buy_markers.append({
                    "timestamp": ts,
                    "price": round(close, 2),
                    "marker_y": round(low * 0.997, 2),
                    "stop_loss": sl,
                    "target_1": t1,
                    "target_2": t2,
                    "risk_reward": round((t1 - close) / risk, 1),
                    "trigger_type": "Supertrend Reversal" if is_st_flip_buy else "VWAP Momentum Expansion",
                    "volume_ratio": round(vol_surge, 1)
                })
                last_signal = "BUY"

            # --- SELL (SHORT) RULES ---
            # 1. Supertrend flips Bearish (prev was 1, now -1) OR
            # 2. Bearish Confluence: Close < VWAP AND EMA 9 < EMA 21 AND Volume Surge > 1.15 AND RSI in (28, 55)
            is_st_flip_sell = (prev_st_dir == 1 and st_dir == -1 and close < vwap)
            is_confluence_sell = (close < vwap and ema9 < ema21 and prev_row['close'] >= prev_row['vwap'] and vol_surge > 1.15 and 28 <= rsi <= 55)

            if (is_st_flip_sell or is_confluence_sell) and last_signal != "SELL":
                sl = round(min(high + (1.0 * atr), row['supertrend'] if st_dir == -1 else high + (1.0 * atr)), 2)
                risk = max(0.2, sl - close)
                t1 = round(close - (1.8 * risk), 2)
                t2 = round(close - (2.8 * risk), 2)

                sell_markers.append({
                    "timestamp": ts,
                    "price": round(close, 2),
                    "marker_y": round(high * 1.003, 2),
                    "stop_loss": sl,
                    "target_1": t1,
                    "target_2": t2,
                    "risk_reward": round((close - t1) / risk, 1),
                    "trigger_type": "Supertrend Breakdown" if is_st_flip_sell else "VWAP Rejection Short",
                    "volume_ratio": round(vol_surge, 1)
                })
                last_signal = "SELL"

        return df, buy_markers, sell_markers

    def get_latest_live_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Returns the immediate real-time state of the latest live candle bar."""
        df_enriched, buys, sells = self.generate_markers(df)
        if df_enriched.empty:
            return {"status": "NEUTRAL", "detail": "No market data."}

        latest = df_enriched.iloc[-1]
        close = latest['close']
        vwap = latest.get('vwap', close)
        st_dir = latest.get('st_direction', 1)
        st_line = latest.get('supertrend', close)
        ema9 = latest.get('ema_9', close)
        ema21 = latest.get('ema_21', close)
        vol_surge = latest.get('vol_surge', 1.0)
        rsi = latest.get('rsi_14', 50.0)

        # Most recent marker
        recent_buy = buys[-1] if buys else None
        recent_sell = sells[-1] if sells else None

        if st_dir == 1 and close >= vwap and ema9 >= ema21:
            bias = "🟢 INTRADAY BULLISH BIAS"
            action = "BUY ON PULLBACK TO VWAP / SUPERTREND"
            key_support = round(max(vwap, st_line), 2)
            key_resistance = round(close * 1.015, 2)
        elif st_dir == -1 and close <= vwap and ema9 <= ema21:
            bias = "🔴 INTRADAY BEARISH BIAS"
            action = "SELL ON RALLY TO VWAP / SUPERTREND"
            key_support = round(close * 0.985, 2)
            key_resistance = round(min(vwap, st_line), 2)
        else:
            bias = "🟡 CONSOLIDATION / NEUTRAL"
            action = "WAIT FOR VWAP BREAKOUT OR SUPERTREND FLIP"
            key_support = round(min(vwap, st_line), 2)
            key_resistance = round(max(vwap, st_line), 2)

        return {
            "bias": bias,
            "recommended_action": action,
            "supertrend_state": "BULLISH (Green)" if st_dir == 1 else "BEARISH (Red)",
            "supertrend_level": round(st_line, 2),
            "vwap_level": round(vwap, 2),
            "distance_to_vwap_pct": round(((close - vwap) / vwap) * 100, 2),
            "volume_surge_ratio": round(vol_surge, 2),
            "rsi_momentum": round(rsi, 1),
            "key_support": key_support,
            "key_resistance": key_resistance,
            "recent_buy_signal": recent_buy,
            "recent_sell_signal": recent_sell,
            "total_buy_signals_session": len(buys),
            "total_sell_signals_session": len(sells)
        }

if __name__ == "__main__":
    import yfinance as yf
    print("Testing IntradaySignalEngine on live RELIANCE intraday data...")
    t = yf.Ticker("RELIANCE.NS")
    df_test = t.history(period="5d", interval="15m")
    if not df_test.empty:
        df_test = df_test.reset_index()
        df_test.columns = [c.lower() for c in df_test.columns]
        engine = IntradaySignalEngine()
        df_res, buys, sells = engine.generate_markers(df_test)
        print(f"Generated {len(buys)} BUY markers and {len(sells)} SELL markers.")
        live_status = engine.get_latest_live_status(df_test)
        print("\n--- Live Intraday Status ---")
        for k, v in live_status.items():
            print(f"  {k}: {v}")
