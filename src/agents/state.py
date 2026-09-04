from typing import TypedDict, Dict, Any, Optional

class AgentTradingState(TypedDict, total=False):
    # Initial Inputs
    ticker: str
    current_price: float
    technical_data: str
    rag_context: str
    sentiment_data: str
    
    # Multi-Horizon Sentiment & Deep Learning Extensions
    sentiment_1h: float
    sentiment_1d: float
    sentiment_1w: float
    divergence_flag: str
    dl_trajectory: Optional[Dict[str, Any]]
    
    # Inter-Agent Communication Logs
    technical_analysis: str
    rag_analysis: str
    sentiment_analysis: str
    bear_objections: str
    
    # The Final Verdict
    final_trade_signal: Optional[Dict[str, Any]]