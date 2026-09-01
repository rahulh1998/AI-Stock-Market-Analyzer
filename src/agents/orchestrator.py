import logging
from langgraph.graph import StateGraph, END
from src.agents.state import AgentTradingState
from src.agents.agent_nodes import technical_agent, rag_agent, sentiment_agent, bear_advocate, lead_synthesizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_workflow():
    """Builds the cyclic LangGraph multi-agent state machine."""
    workflow = StateGraph(AgentTradingState)
    
    # Add all agent nodes to the graph
    workflow.add_node("Technical", technical_agent)
    workflow.add_node("RAG", rag_agent)
    workflow.add_node("Sentiment", sentiment_agent)
    workflow.add_node("Bear", bear_advocate)
    workflow.add_node("Lead", lead_synthesizer)
    
    # Define the execution flow (Assembly Line)
    workflow.set_entry_point("Technical")
    workflow.add_edge("Technical", "RAG")
    workflow.add_edge("RAG", "Sentiment")
    workflow.add_edge("Sentiment", "Bear")
    workflow.add_edge("Bear", "Lead")
    workflow.add_edge("Lead", END)
    
    # Compile into an executable application
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Sanity check to ensure the graph compiles successfully
    app = build_workflow()
    print("\n✅ LangGraph Agent Workflow compiled successfully!")