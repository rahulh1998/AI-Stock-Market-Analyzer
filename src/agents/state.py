from typing import TypedDict, Dict, Any, Optional

class AgentTradingState(TypedDict, total=False):
    # Initial Inputs
    ticker: str
    current_price: float
    technical_data: str
    rag_context: str
    sentiment_data: str
    
    # Multi-Horizon Sentiment & Deep Learning
    sentiment_1h: float
    sentiment_1d: float
    sentiment_1w: float
    divergence_flag: str
    dl_trajectory: Optional[Dict[str, Any]]
    
    # Institutional Pricing Engine Outputs
    institutional_fair_value: float
    mispricing_edge_pct: float
    valuation_status: str
    market_regime: str
    auction_corridor: Optional[Dict[str, Any]]
    monte_carlo_var95: float
    
    # Inter-Agent Communication Logs
    technical_analysis: str
    rag_analysis: str
    sentiment_analysis: str
    bear_objections: str
    
    # The Final Verdict
    final_trade_signal: Optional[Dict[str, Any]]