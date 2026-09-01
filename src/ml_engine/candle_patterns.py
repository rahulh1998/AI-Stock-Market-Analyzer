import pandas as pd
import numpy as np

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans OHLCV data and evaluates 15 classic candlestick pattern rules 
    using pure pandas and numpy vectorization.
    """
    if df.empty or len(df) < 5:
        return df

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    
    body = (c - o).abs()
    range_hl = h - l
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    
    # 1. Doji: body is less than 10% of total range
    df['Doji'] = (body <= (0.1 * range_hl)) & (range_hl > 0)
    
    # 2. Bullish Marubozu: long green body with virtually no shadows
    df['Bullish Marubozu'] = (c > o) & (body >= 0.8 * range_hl) & (upper_shadow <= 0.05 * range_hl) & (lower_shadow <= 0.05 * range_hl)
    
    # 3. Bearish Marubozu: long red body with virtually no shadows
    df['Bearish Marubozu'] = (o > c) & (body >= 0.8 * range_hl) & (upper_shadow <= 0.05 * range_hl) & (lower_shadow <= 0.05 * range_hl)
    
    # 4. Hammer: small body at upper end, long lower shadow (>= 2x body)
    df['Hammer'] = (lower_shadow >= 2 * body) & (upper_shadow <= 0.3 * body) & (body > 0)
    
    # 5. Shooting Star: small body at lower end, long upper shadow (>= 2x body)
    df['Shooting Star'] = (upper_shadow >= 2 * body) & (lower_shadow <= 0.3 * body) & (body > 0)
    
    # Shift variables for multi-candle patterns
    o1, c1, h1, l1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    o2, c2, h2, l2 = o.shift(2), c.shift(2), h.shift(2), l.shift(2)
    
    # 6. Bullish Engulfing
    df['Bullish Engulfing'] = (c1 < o1) & (c > o) & (c >= o1) & (o <= c1) & ((c - o) > (o1 - c1))
    
    # 7. Bearish Engulfing
    df['Bearish Engulfing'] = (c1 > o1) & (c < o) & (c <= o1) & (o >= c1) & ((o - c) > (c1 - o1))
    
    # 8. Piercing Line
    df['Piercing Line'] = (c1 < o1) & (c > o) & (o < l1) & (c >= (o1 + c1) / 2) & (c < o1)
    
    # 9. Dark Cloud Cover
    df['Dark Cloud Cover'] = (c1 > o1) & (c < o) & (h > h1) & (c <= (o1 + c1) / 2) & (c > o1)
    
    # 10. Morning Star (3 candles)
    body1 = (c1 - o1).abs()
    body2 = (c2 - o2).abs()
    df['Morning Star'] = (c2 < o2) & (body1 < 0.3 * (h2 - l2)) & (c > o) & (c >= (o2 + c2) / 2) & (c1 < o1)
    
    # 11. Evening Star (3 candles)
    df['Evening Star'] = (c2 > o2) & (body1 < 0.3 * (h2 - l2)) & (c < o) & (c <= (o2 + c2) / 2) & (c1 > o1)
    
    # 12. Three White Soldiers
    df['Three White Soldiers'] = (c > o) & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2) & (o > o1) & (o1 > o2)
    
    # 13. Three Black Crows
    df['Three Black Crows'] = (o > c) & (o1 > c1) & (o2 > c2) & (c < c1) & (c1 < c2) & (o < o1) & (o1 < o2)
    
    # 14. Spinning Top: small real body centered between upper and lower shadows
    df['Spinning Top'] = (body <= 0.3 * range_hl) & (upper_shadow >= 0.3 * range_hl) & (lower_shadow >= 0.3 * range_hl)
    
    # 15. Hanging Man: small body near top with long lower shadow after an uptrend
    df['Hanging Man'] = (lower_shadow >= 2 * body) & (upper_shadow <= 0.2 * body) & (body > 0) & (c > c.rolling(20, min_periods=5).mean())
    
    return df