from typing import TypedDict, Dict, Any, Optional

class AgentTradingState(TypedDict):
    # Initial Inputs
    ticker: str
    current_price: float
    technical_data: str
    rag_context: str
    sentiment_data: str
    
    # Inter-Agent Communication Logs
    technical_analysis: str
    rag_analysis: str
    sentiment_analysis: str
    bear_objections: str
    
    # The Final Verdict
    final_trade_signal: Optional[Dict[str, Any]]