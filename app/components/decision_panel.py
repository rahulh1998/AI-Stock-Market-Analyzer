"""Automatic research analysis UI; account state is explicitly user-supplied."""
import json

import streamlit as st

from src.quant_engine.decision_engine import AccountRisk, SignalPolicy, analyze_market


def render_decision_panel(bars, interval, capital, risk, allocation, daily_loss, costs):
    st.subheader("Automatic pattern, price-zone & risk analysis")
    st.caption("Runs on completed fetched candles and recomputes when settings change. "
               "Not a background live feed. Prices below are conditional research scenarios, not the day's best prices.")
    with st.expander("Account-wide risk controls — manual research snapshot", expanded=False):
        st.warning("Not broker-synced. Defaults assume a flat account. Include every existing position, "
                   "unrealized P&L and open-order commitment before interpreting capacity.")
        cash = st.number_input("Uncommitted cash (INR)", min_value=0.0, value=float(capital))
        day_start = st.number_input("Session-start equity (INR)", min_value=1.0, value=float(capital))
        peak = st.number_input("Peak account equity (INR)", min_value=1.0, value=float(capital))
        exposure = st.number_input("Existing + committed gross exposure (INR)", min_value=0.0, value=0.0)
        open_risk = st.number_input("Existing + committed modeled stop risk (INR)", min_value=0.0, value=0.0)
        losses = st.number_input("Consecutive losing trades", min_value=0, value=0, step=1)
        kill = st.checkbox("Kill switch — block new conditional buy setups", value=False)
        tick = st.number_input("Instrument tick size — verify with broker (INR)", min_value=0.0001, value=0.05, format="%.4f")
        max_open = st.slider("Maximum combined open risk (%)", 0.5, 5.0, 2.0, 0.5)
        max_gross = st.slider("Maximum combined gross exposure (%)", 10, 100, 80, 5)
        drawdown = st.slider("Drawdown entry lock (%)", 2.0, 20.0, 10.0, 1.0)
        st.caption("Research capital is treated as current total equity. These settings affect this "
                   "analysis only, not the independent historical baseline replay. Locks do not place exit orders.")
    try:
        account = AccountRisk(equity=capital, cash=cash, day_start_equity=day_start, peak_equity=peak,
                              gross_exposure=exposure, open_risk=open_risk, consecutive_losses=int(losses), kill_switch=kill)
        policy = SignalPolicy(risk_pct=risk, allocation_pct=allocation, daily_loss_pct=daily_loss,
                              tick_size=tick, max_open_risk_pct=max_open, max_gross_exposure_pct=max_gross,
                              max_drawdown_pct=drawdown)
        report = analyze_market(bars, interval, account=account, policy=policy, costs=costs)
    except ValueError as exc:
        st.error(f"Analysis blocked: {exc}")
        return
    st.metric("Research decision — never a live order", report["status"])
    st.caption(f"Known at: {report.get('as_of_bar_close', 'unavailable')} | "
               f"Scenario review deadline: {report.get('valid_until', 'unavailable')} | Execution eligible: NO")
    if report["patterns"]:
        st.dataframe(report["patterns"], width='stretch')
    else:
        st.info("No supported pattern on the latest completed candle. WAIT is a valid result.")
    if report["plan"]:
        plan = report["plan"]
        left, middle, right = st.columns(3)
        left.metric("Conditional buy trigger", f"INR {plan['buy_trigger']:.4f}")
        middle.metric("Protective stop", f"INR {plan['stop']:.4f}")
        right.metric("Conditional profit-taking price", f"INR {plan['take_profit']:.4f}")
        st.json(plan)
    for reason in report["rejections"]:
        st.write(f"• {reason}")
    st.info(report.get("exit_guidance", "No actionable price scenario available."))
    with st.expander("Observed levels & recent pattern timeline"):
        st.json(report.get("levels", {}))
        if report["recent_patterns"]:
            st.dataframe(report["recent_patterns"], width='stretch')
        st.caption("Swing patterns appear only when confirmed. Supported pattern definitions are in the analysis documentation; "
                   "not every chart pattern is detectable and a candle shape alone is not a trade.")
    with st.expander("Historical-neighbor ML evidence (experimental)"):
        st.json(report["model"])
        st.caption("Model targets begin at the next bar OPEN and span 3 intraday bars or 5 daily bars. "
                   "Ranges are empirical, before costs, and are not a calibrated probability or full-day forecast. "
                   "Legacy DL/LLM forecasts cannot override this engine.")
    for warning in report["warnings"]:
        st.caption(warning)
    st.download_button("Export analysis report (JSON)", json.dumps(report, indent=2, allow_nan=False),
                       "pattern_risk_report.json", "application/json")