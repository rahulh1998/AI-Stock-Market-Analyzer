import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from src.agents.state import AgentTradingState

logger = logging.getLogger(__name__)

# Initialize the local LLMs
# Standard LLM for reasoning
llm = ChatOllama(model="llama3.2", temperature=0)
# JSON-forced LLM for the final structured output
llm_json = ChatOllama(model="llama3.2", temperature=0, format="json")

def technical_agent(state: AgentTradingState):
    logger.info("--- Technical Agent Analyzing ---")
    sys_prompt = "You are an expert quantitative technical analyst. Evaluate the provided indicators."
    user_prompt = f"Ticker: {state['ticker']} @ {state['current_price']}\nIndicators: {state['technical_data']}\nIdentify the trend and momentum."
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {"technical_analysis": response.content}

def rag_agent(state: AgentTradingState):
    logger.info("--- RAG Agent Validating ---")
    sys_prompt = "You are a trading strategy expert. Validate the technical setup using textbook rules."
    user_prompt = f"Setup: {state['technical_analysis']}\nRules: {state['rag_context']}\nDoes the setup match the rules?"
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {"rag_analysis": response.content}

def sentiment_agent(state: AgentTradingState):
    logger.info("--- Sentiment Agent Analyzing ---")
    sys_prompt = "You evaluate macroeconomic and news sentiment for Indian equities."
    user_prompt = f"Ticker: {state['ticker']}\nNews/Fundamentals: {state['sentiment_data']}\nIs the sentiment bullish, bearish, or neutral?"
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {"sentiment_analysis": response.content}

def bear_advocate(state: AgentTradingState):
    logger.info("--- Bear Advocate Challenging ---")
    sys_prompt = "You are a risk-averse Bear Advocate. Your job is to find reasons NOT to take this trade."
    user_prompt = f"Tech: {state['technical_analysis']}\nSentiment: {state['sentiment_analysis']}\nFind the hidden risks, overhead resistances, and macroeconomic traps."
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {"bear_objections": response.content}

def lead_synthesizer(state: AgentTradingState):
    logger.info("--- Lead Synthesizer Formulating Final Trade ---")
    sys_prompt = """You are the Lead Portfolio Manager. Review all agent reports and issue a final trade signal.
    You MUST output valid JSON only, using this schema:
    {
      "action": "BUY" | "SELL" | "HOLD",
      "confidence_score": 0-100,
      "reasoning": "1 sentence summary"
    }"""
    user_prompt = f"Tech: {state['technical_analysis']}\nRAG: {state['rag_analysis']}\nSentiment: {state['sentiment_analysis']}\nBear: {state['bear_objections']}\nDecision?"
    
    response = llm_json.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    
    try:
        # Parse the JSON string returned by Ollama into a Python dictionary
        final_signal = json.loads(response.content)
    except Exception:
        final_signal = {"action": "HOLD", "confidence_score": 0, "reasoning": "Failed to parse JSON."}
        
    return {"final_trade_signal": final_signal}