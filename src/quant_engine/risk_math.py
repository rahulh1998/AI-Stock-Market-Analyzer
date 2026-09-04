import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self, default_capital: float = 100000.0, max_risk_per_trade_pct: float = 1.0):
        """
        :param default_capital: Total account portfolio size in INR (₹).
        :param max_risk_per_trade_pct: Maximum percentage of capital to risk on a single trade (e.g., 1.0%).
        """
        self.default_capital = default_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct

    def calculate_levels(self, current_price: float, atr_14: float, action: str, 
                         dl_bounds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculates ATR-based volatility Stop-Loss and Target levels,
        harmonized with deep learning sequential trajectory bounds.
        """
        action = action.upper()
        if action not in ["BUY", "SELL"]:
            return {
                "entry_price": current_price,
                "stop_loss": 0.0,
                "target_1": 0.0,
                "target_2": 0.0,
                "risk_per_share": 0.0,
                "risk_reward_ratio": 0.0
            }

        # 1.5x ATR buffer accommodates standard volatility swings
        sl_buffer = round(1.5 * (atr_14 if atr_14 > 0 else current_price * 0.015), 2)

        if action == "BUY":
            stop_loss = round(current_price - sl_buffer, 2)
            # If DL predicted a higher low bound above stop loss, calibrate for tighter protection
            if dl_bounds and "predicted_low" in dl_bounds and dl_bounds["predicted_low"] > 0:
                dl_low = dl_bounds["predicted_low"]
                if dl_low < current_price:
                    stop_loss = round(max(stop_loss, dl_low * 0.995), 2)

            risk_per_share = max(0.1, round(current_price - stop_loss, 2))
            target_1 = round(current_price + (2.0 * risk_per_share), 2)  # 1:2 R:R
            
            # Incorporate DL predicted high if it offers superior expansion
            if dl_bounds and "predicted_high" in dl_bounds and dl_bounds["predicted_high"] > current_price:
                target_1 = round(max(target_1, dl_bounds["predicted_high"]), 2)

            target_2 = round(current_price + (3.0 * risk_per_share), 2)  # 1:3 R:R
        else:  # SELL / Short
            stop_loss = round(current_price + sl_buffer, 2)
            if dl_bounds and "predicted_high" in dl_bounds and dl_bounds["predicted_high"] > current_price:
                stop_loss = round(max(stop_loss, dl_bounds["predicted_high"] * 1.005), 2)

            risk_per_share = max(0.1, round(stop_loss - current_price, 2))
            target_1 = round(current_price - (2.0 * risk_per_share), 2)
            if dl_bounds and "predicted_low" in dl_bounds and dl_bounds["predicted_low"] < current_price:
                target_1 = round(min(target_1, dl_bounds["predicted_low"]), 2)

            target_2 = round(current_price - (3.0 * risk_per_share), 2)

        risk_reward_ratio = round((abs(target_1 - current_price)) / (risk_per_share if risk_per_share > 0 else 1), 2)

        return {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": risk_per_share,
            "risk_reward_ratio": risk_reward_ratio
        }

    def calculate_position_sizing(self, current_price: float, stop_loss: float, capital: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates exact share count using the 1% Capital Risk Rule.
        """
        account_balance = capital if capital is not None else self.default_capital
        risk_amount = round(account_balance * (self.max_risk_per_trade_pct / 100.0), 2)
        
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share <= 0:
            return {"quantity": 0, "total_exposure": 0.0, "max_rupee_risk": 0.0}

        quantity = int(risk_amount // risk_per_share)
        total_exposure = round(quantity * current_price, 2)

        return {
            "quantity": quantity,
            "total_exposure": total_exposure,
            "max_rupee_risk": risk_amount,
            "capital_risk_pct": self.max_risk_per_trade_pct
        }

    def verify_signal_guardrails(self, signal: Dict[str, Any], technical_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies deterministic safety checks to prevent LLM hallucinations or counter-trend signals.
        """
        action = signal.get("action", "HOLD").upper()
        confidence = signal.get("confidence_score", 0)
        
        close = technical_snapshot.get("close", 0.0)
        sma_200 = technical_snapshot.get("SMA_200", 0.0)
        rsi = technical_snapshot.get("RSI_14", 50.0)

        # Guardrail 1: Minimum confidence threshold
        if confidence < 70 and action in ["BUY", "SELL"]:
            return {
                "is_approved": False,
                "revised_action": "HOLD",
                "veto_reason": f"Confidence score ({confidence}) is below the required 70 threshold."
            }

        # Guardrail 2: Do not BUY if price is below 200 SMA (Long-term downtrend)
        if action == "BUY" and sma_200 > 0 and close < sma_200:
            return {
                "is_approved": False,
                "revised_action": "HOLD",
                "veto_reason": "VETO: Price is trading below the 200 SMA (Major Downtrend Regime)."
            }

        # Guardrail 3: Overbought RSI Protection (> 75)
        if action == "BUY" and rsi > 75:
            return {
                "is_approved": False,
                "revised_action": "HOLD",
                "veto_reason": f"VETO: RSI is severely overbought ({rsi:.1f}). Wait for pullback."
            }

        return {
            "is_approved": True,
            "revised_action": action,
            "veto_reason": "None. All mathematical guardrails passed."
        }

if __name__ == "__main__":
    # Test simulation
    engine = RiskEngine(default_capital=200000.0, max_risk_per_trade_pct=1.0)
    
    test_price = 1000.00
    test_atr = 15.50
    
    levels = engine.calculate_levels(current_price=test_price, atr_14=test_atr, action="BUY")
    sizing = engine.calculate_position_sizing(current_price=test_price, stop_loss=levels["stop_loss"])
    
    print("\n--- Risk Math & Levels Test ---")
    print(f"Entry: ₹{levels['entry_price']} | SL: ₹{levels['stop_loss']} | T1: ₹{levels['target_1']} | R:R: {levels['risk_reward_ratio']}")
    print(f"Max Risk: ₹{sizing['max_rupee_risk']} | Recommended Shares: {sizing['quantity']} | Capital Exposure: ₹{sizing['total_exposure']}")