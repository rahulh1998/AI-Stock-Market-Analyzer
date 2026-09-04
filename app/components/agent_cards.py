import streamlit as st

def render_agent_debate(state: dict):
    """Displays the reasoning of each specialized agent and the portfolio manager verdict."""
    st.subheader("🤖 LangGraph Multi-Agent Assembly Line Transcript")

    final_signal = state.get("final_trade_signal", {})
    if final_signal:
        action = final_signal.get("action", "HOLD")
        confidence = final_signal.get("confidence_score", 50)
        reasoning = final_signal.get("reasoning", "")

        action_color = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "🟡")
        st.info(f"**Lead Portfolio Manager Verdict:** {action_color} **{action}** | **Confidence:** {confidence}% | *{reasoning}*")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📈 Technical Quantitative Agent", expanded=True):
            st.write(state.get("technical_analysis", "No data available."))

        with st.expander("📰 Multi-Horizon Sentiment Agent", expanded=True):
            st.write(state.get("sentiment_analysis", "No data available."))

    with col2:
        with st.expander("📚 RAG Strategy Agent (Textbook Rules)", expanded=True):
            st.write(state.get("rag_analysis", "No data available."))

        with st.expander("🐻 Bear Advocate (Chief Risk Officer)", expanded=True):
            st.error(state.get("bear_objections", "No data available."))