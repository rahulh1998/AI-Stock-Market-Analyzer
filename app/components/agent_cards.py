import streamlit as st

def render_agent_debate(state: dict):
    """Displays the reasoning of each specialized agent in expander cards."""
    st.subheader("🤖 Multi-Agent Debate Transcript")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📈 Technical Agent", expanded=True):
            st.write(state.get("technical_analysis", "No data."))
            
        with st.expander("📰 Sentiment Agent", expanded=True):
            st.write(state.get("sentiment_analysis", "No data."))
            
    with col2:
        with st.expander("📚 RAG Strategy Agent", expanded=True):
            st.write(state.get("rag_analysis", "No data."))
            
        with st.expander("🐻 Bear Advocate (Risk Finder)", expanded=True):
            st.error(state.get("bear_objections", "No data."))