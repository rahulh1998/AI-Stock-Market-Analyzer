import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentTradingState

logger = logging.getLogger(__name__)

# Attempt to initialize Ollama with fallback
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)
    llm_json = ChatOllama(model="llama3.2", temperature=0, format="json")
    OLLAMA_AVAILABLE = True
except Exception as e:
    logger.warning(f"Ollama not initialized: {e}. Fallback heuristics will be used.")
    llm = None
    llm_json = None
    OLLAMA_AVAILABLE = False

def technical_agent(state: AgentTradingState) -> Dict[str, Any]:
    logger.info("--- Technical Quantitative & Pricing Agent Analyzing ---")
    ticker = state.get('ticker', 'UNKNOWN')
    price = state.get('current_price', 0.0)
    tech = state.get('technical_data', '')
    fv = state.get('institutional_fair_value', price)
    edge = state.get('mispricing_edge_pct', 0.0)
    regime = state.get('market_regime', 'MEAN_REVERTING')

    pricing_summary = f"LTP: ₹{price} | Fair Value: ₹{fv} ({edge:+.2f}% Edge) | Regime: {regime}"

    if OLLAMA_AVAILABLE and llm is not None:
        try:
            sys_prompt = "You are an institutional quantitative equity analyst. Evaluate indicators, Fair Value, and market regime."
            user_prompt = f"Ticker: {ticker}\nPricing & Regime: {pricing_summary}\nIndicators: {tech}\nDiagnose the trend, value edge, and momentum confluence."
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return {"technical_analysis": response.content}
        except Exception as e:
            logger.warning(f"Ollama execution failed for technical_agent: {e}")

    return {
        "technical_analysis": (
            f"Quantitative & Pricing Analysis for {ticker}: {pricing_summary}. "
            f"Indicators: {tech}. Structure indicates asset trading within institutional auction corridor."
        )
    }

def rag_agent(state: AgentTradingState) -> Dict[str, Any]:
    logger.info("--- RAG Agent Validating ---")
    tech_analysis = state.get('technical_analysis', '')
    rag_context = state.get('rag_context', '')

    if OLLAMA_AVAILABLE and llm is not None:
        try:
            sys_prompt = "You are a trading strategy expert. Validate the setup against classical literature rules."
            user_prompt = f"Setup: {tech_analysis}\nRules: {rag_context}\nDoes the setup conform to textbook rules?"
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return {"rag_analysis": response.content}
        except Exception as e:
            logger.warning(f"Ollama execution failed for rag_agent: {e}")

    return {
        "rag_analysis": f"Conforms to established technical principles (Murphy/Nison/O'Neil). Context: {rag_context[:180]}..."
    }

def sentiment_agent(state: AgentTradingState) -> Dict[str, Any]:
    logger.info("--- Multi-Horizon Sentiment Agent Analyzing ---")
    ticker = state.get('ticker', '')
    s_1h = state.get('sentiment_1h', 0.0)
    s_1d = state.get('sentiment_1d', 0.0)
    s_1w = state.get('sentiment_1w', 0.0)
    div_flag = state.get('divergence_flag', 'CONGRUENT')
    raw_sent = state.get('sentiment_data', '')

    horizon_summary = (
        f"1-Hour Flash Sentiment: {s_1h:.2f} | "
        f"24-Hour Daily Sentiment: {s_1d:.2f} | "
        f"7-Day Trend Sentiment: {s_1w:.2f} | "
        f"Divergence Alert: {div_flag}"
    )

    if OLLAMA_AVAILABLE and llm is not None:
        try:
            sys_prompt = "You evaluate multi-horizon news sentiment and price divergences for Indian equities."
            user_prompt = f"Ticker: {ticker}\nMulti-Horizon Sentiment: {horizon_summary}\nHeadlines: {raw_sent}\nSynthesize sentiment velocity and divergence."
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return {"sentiment_analysis": response.content}
        except Exception as e:
            logger.warning(f"Ollama execution failed for sentiment_agent: {e}")

    return {
        "sentiment_analysis": f"Multi-horizon sentiment synthesis for {ticker}: {horizon_summary}."
    }

def bear_advocate(state: AgentTradingState) -> Dict[str, Any]:
    logger.info("--- Bear Advocate Challenging ---")
    tech = state.get('technical_analysis', '')
    sent = state.get('sentiment_analysis', '')
    div_flag = state.get('divergence_flag', 'CONGRUENT')
    edge = state.get('mispricing_edge_pct', 0.0)
    var95 = state.get('monte_carlo_var95', 3.0)
    regime = state.get('market_regime', '')
    dl = state.get('dl_trajectory', {})

    risk_context = (
        f"Mispricing: {edge:+.2f}% | "
        f"5-day 95% VaR: {var95:.2f}% | "
        f"Regime: {regime} | "
        f"Divergence: {div_flag} | "
        f"Expected Low: ₹{dl.get('predicted_low', 0)}"
    )

    if OLLAMA_AVAILABLE and llm is not None:
        try:
            sys_prompt = "You are a ruthless, risk-averse Chief Risk Officer / Bear Advocate. Find every reason to reject this trade."
            user_prompt = f"Tech: {tech}\nSentiment: {sent}\nRisk Profile: {risk_context}\nIdentify hidden traps, liquidation risks, and downside tail risks."
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return {"bear_objections": response.content}
        except Exception as e:
            logger.warning(f"Ollama execution failed for bear_advocate: {e}")

    objections = (
        f"Vulnerabilities flagged: 5-Day VaR risk is {var95:.2f}%. "
        f"Asset regime is {regime} with mispricing edge of {edge:+.2f}%. "
        f"Divergence status ({div_flag}) indicates potential liquidity trap if key support breaks."
    )
    return {"bear_objections": objections}

def lead_synthesizer(state: AgentTradingState) -> Dict[str, Any]:
    logger.info("--- Lead Synthesizer Formulating Final Institutional Trade ---")
    tech = state.get('technical_analysis', '')
    rag = state.get('rag_analysis', '')
    sent = state.get('sentiment_analysis', '')
    bear = state.get('bear_objections', '')
    edge = state.get('mispricing_edge_pct', 0.0)
    regime = state.get('market_regime', 'MEAN_REVERTING')
    s_1d = state.get('sentiment_1d', 0.0)
    dl = state.get('dl_trajectory', {})

    if OLLAMA_AVAILABLE and llm_json is not None:
        try:
            sys_prompt = """You are the Lead Portfolio Manager. Review technical indicators, pricing fair value, sentiment, and bear objections to issue a definitive trade decision.
            You MUST output valid JSON only, using this schema:
            {
              "action": "BUY" | "SELL" | "HOLD",
              "confidence_score": 0-100,
              "reasoning": "1-2 sentence institutional justification"
            }"""
            user_prompt = f"Tech: {tech}\nRAG: {rag}\nSentiment: {sent}\nBear Objections: {bear}\nFinal Allocation Decision?"
            response = llm_json.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            final_signal = json.loads(response.content)
            return {"final_trade_signal": final_signal}
        except Exception as e:
            logger.warning(f"Ollama execution failed for lead_synthesizer: {e}")

    # Quantitative Decision Logic
    ret_pct = dl.get('expected_return_pct', 0.0) if dl else 0.0

    # Institutional Edge: Undervalued + Positive Trajectory + Constructive Regime
    if edge >= 0.50 and ret_pct > 0.3 and s_1d >= -0.1 and regime != "HIGH_VOLATILITY_SHOCK":
        action = "BUY"
        conf = min(92, int(72 + edge * 4 + ret_pct * 3))
        reason = f"Institutional undervaluation ({edge:+.2f}%) aligns with positive DL trajectory (+{ret_pct}%) in {regime} regime."
    elif edge <= -0.75 or ret_pct < -0.4 or s_1d < -0.3:
        action = "SELL"
        conf = min(88, int(68 + abs(edge) * 3 + abs(ret_pct) * 4))
        reason = f"Asset trading at premium ({edge:+.2f}%) with negative drift and elevated risk."
    else:
        action = "HOLD"
        conf = 60
        reason = f"Asset trading near equilibrium ({edge:+.2f}% mispricing) within {regime}; await clear breakout."

    return {
        "final_trade_signal": {
            "action": action,
            "confidence_score": conf,
            "reasoning": reason
        }
    }