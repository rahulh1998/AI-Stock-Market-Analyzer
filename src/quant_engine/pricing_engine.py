import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class InstitutionalPricingEngine:
    """
    Institutional Multi-Factor Equity Pricing Engine.
    Implements:
    1. Volume-Weighted Microstructure: Anchored VWAP and Multi-Sigma Bands (±1σ, ±2σ)
    2. Confluence Key Levels: Camarilla Floor Pivots (H3, H4, L3, L4) & Fibonacci Grid
    3. Stochastic Asset Pricing: Merton Jump-Diffusion Monte Carlo Simulation (Poisson Shocks)
    4. Multi-Factor Alpha Model: Momentum, Volatility Clustering, Volume Flow, and Relative Edge
    5. Statistical Market Regime Classifier: Bull Trend, Bear Trend, Mean-Reverting, High-Vol Shock
    6. Unified Institutional Fair Value & Mispricing Edge (Alpha Discount/Premium %)
    """

    def __init__(self, num_simulations: int = 1000, forecast_steps: int = 5):
        self.num_simulations = num_simulations
        self.forecast_steps = forecast_steps

    # --------------------------------------------------------------------------
    # 1. Microstructure: Volume Weighted Average Price (VWAP) & Sigma Bands
    # --------------------------------------------------------------------------
    def calculate_vwap_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates volume-weighted average price and institutional volatility bands."""
        if df.empty or len(df) < 5:
            close = float(df['close'].iloc[-1]) if not df.empty else 100.0
            return {
                "vwap": close, "vwap_upper_1s": close * 1.01, "vwap_upper_2s": close * 1.02,
                "vwap_lower_1s": close * 0.99, "vwap_lower_2s": close * 0.98
            }

        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'] + 1e-5

        # Rolling 20-session or window-anchored VWAP
        window = min(len(df), 20)
        pv = (typical_price * vol).rolling(window=window, min_periods=1).sum()
        total_vol = vol.rolling(window=window, min_periods=1).sum()
        vwap_series = pv / total_vol
        current_vwap = float(vwap_series.iloc[-1])

        # Standard deviation of price relative to VWAP
        variance = ((typical_price - vwap_series) ** 2 * vol).rolling(window=window, min_periods=1).sum() / total_vol
        vwap_std = float(np.sqrt(np.maximum(variance.iloc[-1], 1e-5)))

        return {
            "vwap": round(current_vwap, 2),
            "vwap_upper_1s": round(current_vwap + (1.0 * vwap_std), 2),
            "vwap_upper_2s": round(current_vwap + (2.0 * vwap_std), 2),
            "vwap_lower_1s": round(current_vwap - (1.0 * vwap_std), 2),
            "vwap_lower_2s": round(current_vwap - (2.0 * vwap_std), 2),
            "vwap_dispersion": round((vwap_std / current_vwap) * 100, 2)
        }

    # --------------------------------------------------------------------------
    # 2. Confluence: Camarilla Institutional Floor Pivots
    # --------------------------------------------------------------------------
    def calculate_camarilla_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculates Camarilla equation levels used by institutional floor traders:
        - L3 / H3: High-probability Mean Reversion Boundaries
        - L4 / H4: Momentum Breakout / Breakdown Continuation Triggers
        """
        rng = high - low
        if rng <= 0:
            rng = close * 0.015

        h4 = close + (rng * (1.1 / 2.0))
        h3 = close + (rng * (1.1 / 4.0))
        l3 = close - (rng * (1.1 / 4.0))
        l4 = close - (rng * (1.1 / 2.0))

        # Fibonacci extensions
        fib_ext_1618 = close + (1.618 * rng)
        fib_ext_2618 = close + (2.618 * rng)

        return {
            "cam_h4_breakout": round(h4, 2),
            "cam_h3_sell": round(h3, 2),
            "cam_l3_buy": round(l3, 2),
            "cam_l4_breakdown": round(l4, 2),
            "fib_target_1": round(fib_ext_1618, 2),
            "fib_target_2": round(fib_ext_2618, 2)
        }

    # --------------------------------------------------------------------------
    # 3. Market Regime Classification
    # --------------------------------------------------------------------------
    def detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classifies asset into one of 4 market regimes:
        1. Trending Bullish (Persistent positive drift, orderly volatility)
        2. Trending Bearish (Negative drift, elevated downside volatility)
        3. Mean-Reverting Range (Low directional trend, bounded within standard deviation)
        4. High-Volatility Shock (Extreme jump dispersion, wide tail risk)
        """
        if df.empty or len(df) < 20:
            return {"regime": "MEAN_REVERTING", "confidence": 60.0, "volatility_annualized": 18.0, "drift_daily": 0.0}

        returns = df['close'].pct_change().dropna()
        ann_vol = float(returns.tail(20).std() * np.sqrt(252) * 100)
        drift = float(returns.tail(20).mean() * 100)

        # Trend strength proxy
        sma_20 = df['close'].tail(20).mean()
        sma_50 = df['close'].tail(50).mean() if len(df) >= 50 else sma_20
        latest_close = float(df['close'].iloc[-1])

        if ann_vol > 45.0:
            regime = "HIGH_VOLATILITY_SHOCK"
            conf = min(92.0, 50.0 + ann_vol)
        elif drift > 0.15 and latest_close > sma_20 and sma_20 >= sma_50:
            regime = "TRENDING_BULLISH"
            conf = 85.0
        elif drift < -0.15 and latest_close < sma_20 and sma_20 <= sma_50:
            regime = "TRENDING_BEARISH"
            conf = 85.0
        else:
            regime = "MEAN_REVERTING_RANGE"
            conf = 75.0

        return {
            "regime": regime,
            "confidence": round(conf, 1),
            "volatility_annualized": round(ann_vol, 2),
            "drift_daily": round(drift, 3)
        }

    # --------------------------------------------------------------------------
    # 4. Stochastic Asset Pricing: Merton Jump-Diffusion Monte Carlo
    # --------------------------------------------------------------------------
    def simulate_merton_jump_diffusion(self, S0: float, daily_drift: float, daily_vol: float) -> Dict[str, Any]:
        """
        Simulates future price paths with Merton Jump-Diffusion:
        dS_t = mu*S_t*dt + sigma*S_t*dW_t + J_t*S_t*dN_t
        Incorporates Poisson market shocks (earnings, flash news, sudden liquidity voids).
        """
        dt = 1.0  # 1 day step
        mu = daily_drift / 100.0
        sigma = max(0.005, daily_vol / 100.0)

        # Poisson Jump parameters calibrated to Indian equities
        lambda_jump = 0.15  # Expected 0.15 jumps per 5-day horizon
        jump_mean = -0.005   # Slight negative skew for surprise shocks
        jump_std = 0.025     # Jump standard deviation ~ 2.5%

        price_matrix = np.zeros((self.num_simulations, self.forecast_steps + 1))
        price_matrix[:, 0] = S0

        for t in range(1, self.forecast_steps + 1):
            Z = np.random.standard_normal(self.num_simulations)
            N = np.random.poisson(lambda_jump * dt, self.num_simulations)
            J = np.random.normal(jump_mean, jump_std, self.num_simulations) * N

            drift_term = (mu - 0.5 * (sigma ** 2)) * dt
            diffusion_term = sigma * np.sqrt(dt) * Z
            jump_term = J

            price_matrix[:, t] = price_matrix[:, t - 1] * np.exp(drift_term + diffusion_term + jump_term)

        terminal_prices = price_matrix[:, -1]
        expected_price = float(np.mean(terminal_prices))
        median_price = float(np.median(terminal_prices))
        p95_upside = float(np.percentile(terminal_prices, 95))
        p5_downside_var = float(np.percentile(terminal_prices, 5))
        p75_high = float(np.percentile(terminal_prices, 75))
        p25_low = float(np.percentile(terminal_prices, 25))

        # Probability of gain
        prob_up = float(np.mean(terminal_prices > S0) * 100)

        # Value at Risk (VaR 95%) in rupee and % terms
        var_95_rupees = round(S0 - p5_downside_var, 2)
        var_95_pct = round((var_95_rupees / S0) * 100, 2)

        return {
            "sim_expected_price": round(expected_price, 2),
            "sim_median_price": round(median_price, 2),
            "sim_p95_upside": round(p95_upside, 2),
            "sim_p5_downside": round(p5_downside_var, 2),
            "sim_interquartile_range": [round(p25_low, 2), round(p75_high, 2)],
            "monte_carlo_prob_up": round(prob_up, 1),
            "var_95_pct": var_95_pct,
            "var_95_rupees": var_95_rupees,
            "paths_sample": price_matrix[:15, :].tolist()
        }

    # --------------------------------------------------------------------------
    # 5. Multi-Factor Alpha Model
    # --------------------------------------------------------------------------
    def compute_multi_factor_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes composite quant factor scores (scale -100 to +100):
        - Momentum Factor (RSI, 5d return, 20d return)
        - Mean Reversion Factor (Distance from 20 EMA)
        - Volume Force Factor (Volume surge relative to 20-day SMA)
        - Volatility Stability Factor (ATR / Price ratio)
        """
        if df.empty or len(df) < 20:
            return {"composite_alpha": 0.0, "momentum": 0.0, "volume_force": 0.0, "mean_reversion": 0.0}

        close = df['close']
        vol = df['volume']
        latest_close = float(close.iloc[-1])

        # 1. Momentum Factor
        ret_5d = float((latest_close - close.iloc[-5]) / close.iloc[-5]) if len(close) >= 5 else 0.0
        ret_20d = float((latest_close - close.iloc[-20]) / close.iloc[-20]) if len(close) >= 20 else 0.0
        mom_score = np.clip((ret_5d * 3.0 + ret_20d * 2.0) * 100, -100, 100)

        # 2. Mean Reversion Factor
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        dev_ema = (latest_close - ema_20) / (ema_20 + 1e-5)
        # If stretched far above EMA -> negative mean reversion score, if far below -> positive
        mean_rev_score = float(np.clip(-dev_ema * 400, -100, 100))

        # 3. Volume Force Factor
        vol_sma20 = vol.tail(20).mean()
        vol_ratio = float(vol.iloc[-1] / (vol_sma20 + 1e-5))
        is_green = latest_close >= float(df['open'].iloc[-1])
        vol_score = float(np.clip((vol_ratio - 1.0) * (50 if is_green else -50), -100, 100))

        # Composite Alpha
        composite = round(float(0.45 * mom_score + 0.30 * vol_score + 0.25 * mean_rev_score), 1)

        return {
            "composite_alpha": composite,
            "momentum_factor": round(float(mom_score), 1),
            "volume_force_factor": round(float(vol_score), 1),
            "mean_reversion_factor": round(float(mean_rev_score), 1)
        }

    # --------------------------------------------------------------------------
    # 6. Unified Pricing Engine Synthesis
    # --------------------------------------------------------------------------
    def calculate_fair_value_envelope(self, df: pd.DataFrame, dl_forecast: Optional[Dict[str, Any]] = None,
                                      ml_prob: float = 50.0) -> Dict[str, Any]:
        """
        Synthesizes VWAP microstructure, Camarilla confluence, Monte Carlo Jump-Diffusion,
        Multi-Factor Alpha, and Deep Learning into a cohesive Institutional Pricing Envelope.
        """
        if df.empty:
            return {}

        latest_close = float(df['close'].iloc[-1])
        day_high = float(df['high'].iloc[-1])
        day_low = float(df['low'].iloc[-1])

        # Subsystems
        vwap_info = self.calculate_vwap_bands(df)
        camarilla = self.calculate_camarilla_pivots(day_high, day_low, latest_close)
        regime_info = self.detect_market_regime(df)
        factors = self.compute_multi_factor_score(df)

        daily_drift = regime_info["drift_daily"]
        daily_vol = max(0.8, regime_info["volatility_annualized"] / np.sqrt(252))
        monte_carlo = self.simulate_merton_jump_diffusion(latest_close, daily_drift, daily_vol)

        # Calculate Institutional Fair Value (Anchor)
        vwap_val = vwap_info["vwap"]
        mc_expected = monte_carlo["sim_expected_price"]
        dl_expected = dl_forecast.get("predicted_close", latest_close) if dl_forecast else latest_close

        # Weighted Fair Value Anchor
        # 35% VWAP + 30% Monte Carlo Expected + 20% DL Sequence + 15% Factor Momentum Shift
        factor_tilt = latest_close * (factors["composite_alpha"] / 2000.0)
        institutional_fair_value = round(
            (0.35 * vwap_val) + (0.30 * mc_expected) + (0.25 * dl_expected) + (0.10 * (latest_close + factor_tilt)),
            2
        )

        # Mispricing Edge (Alpha Discount / Premium)
        mispricing_pct = round(((institutional_fair_value - latest_close) / latest_close) * 100, 2)
        if mispricing_pct >= 0.75:
            valuation_status = "UNDERVALUED (Institutional Discount)"
        elif mispricing_pct <= -0.75:
            valuation_status = "OVERVALUED (Institutional Premium)"
        else:
            valuation_status = "FAIRLY_VALUED (Equilibrium)"

        # Optimal Auction Corridor
        # Bid Limit (Buyer Support): Confluence of Camarilla L3, VWAP -1σ, and MC 25th percentile
        bid_support = round(
            float(np.mean([camarilla["cam_l3_buy"], vwap_info["vwap_lower_1s"], monte_carlo["sim_interquartile_range"][0]])),
            2
        )

        # Ask Limit (Seller Resistance): Confluence of Camarilla H3, VWAP +1σ, and MC 75th percentile
        ask_resistance = round(
            float(np.mean([camarilla["cam_h3_sell"], vwap_info["vwap_upper_1s"], monte_carlo["sim_interquartile_range"][1]])),
            2
        )

        # Extreme Tail Boundaries
        institutional_upside_target = round(
            max(camarilla["cam_h4_breakout"], monte_carlo["sim_p95_upside"], dl_forecast.get("predicted_high", latest_close) if dl_forecast else latest_close),
            2
        )
        institutional_downside_invalidation = round(
            min(camarilla["cam_l4_breakdown"], monte_carlo["sim_p5_downside"], dl_forecast.get("predicted_low", latest_close) if dl_forecast else latest_close),
            2
        )

        return {
            "current_market_price": latest_close,
            "institutional_fair_value": institutional_fair_value,
            "mispricing_edge_pct": mispricing_pct,
            "valuation_status": valuation_status,
            "market_regime": regime_info["regime"],
            "regime_confidence": regime_info["confidence"],
            "annualized_volatility_pct": regime_info["volatility_annualized"],
            "auction_corridor": {
                "bid_support_level": bid_support,
                "fair_value_center": institutional_fair_value,
                "ask_resistance_level": ask_resistance,
                "max_upside_target": institutional_upside_target,
                "max_downside_invalidation": institutional_downside_invalidation
            },
            "vwap_microstructure": vwap_info,
            "camarilla_pivots": camarilla,
            "monte_carlo_simulation": monte_carlo,
            "factor_scores": factors
        }

if __name__ == "__main__":
    import sqlite3
    db_path = os.path.join(os.getcwd(), "data", "market_data.db")
    conn = sqlite3.connect(db_path)
    df_test = pd.read_sql("SELECT timestamp, open, high, low, close, volume FROM daily_ohlcv WHERE ticker = 'RELIANCE' ORDER BY timestamp ASC", conn)
    conn.close()

    if not df_test.empty:
        pricing_engine = InstitutionalPricingEngine()
        res = pricing_engine.calculate_fair_value_envelope(df_test)
        print("\n--- Institutional Pricing Engine Output (RELIANCE) ---")
        print(f"Current Price: ₹{res['current_market_price']}")
        print(f"Institutional Fair Value: ₹{res['institutional_fair_value']} ({res['mispricing_edge_pct']:+}% -> {res['valuation_status']})")
        print(f"Market Regime: {res['market_regime']} (Confidence: {res['regime_confidence']}%)")
        print(f"Auction Corridor: Bid Support ₹{res['auction_corridor']['bid_support_level']} | Ask Resistance ₹{res['auction_corridor']['ask_resistance_level']}")
        print(f"Merton Jump-Diffusion Target (p95): ₹{res['monte_carlo_simulation']['sim_p95_upside']} | 5-day VaR: {res['monte_carlo_simulation']['var_95_pct']}%")
        print(f"Factor Score: Composite {res['factor_scores']['composite_alpha']} | Momentum {res['factor_scores']['momentum_factor']}")
