"""
dashboard/pages/5_Recommendations.py
Rule-based recommendations, styled as an alerts/advisory panel.
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, TEXT, TEXT_DIM, AMBER, GREEN, YELLOW, RED, PANEL, BORDER
from src.recommendations.recommendation_engine import generate_recommendations

st.set_page_config(page_title="Recommendations | Drilling AI", page_icon="💡", layout="wide")
apply_theme()
page_header("ADVISORY PANEL", "💡 Operational Recommendations",
            "Transparent, rule-based advice — every suggestion traces to a simple, auditable condition.")

if "last_inputs" not in st.session_state:
    st.warning("Visit **Forecasting** first to auto-fill this page, or enter conditions manually below.")

manual = st.checkbox("Enter conditions manually", value=("last_inputs" not in st.session_state))

if manual:
    c1, c2, c3 = st.columns(3)
    weather = c1.slider("Weather Severity (1-10)", 1, 10, 5)
    failures = c2.slider("Equipment Failures", 0, 4, 0)
    hardness = c3.slider("Formation Hardness (1-10)", 1, 10, 5)
    phase = c1.selectbox("Phase", ["Drilling", "Casing", "Cementing", "Tripping", "Completion", "Logging", "Rig Move", "Setup"])
    npt_ratio = c2.slider("Predicted NPT Ratio", 0.0, 0.5, 0.15)
    spread_pct = c3.slider("P90-P10 Spread % of P50", 0, 150, 50)
    inputs = {
        "Weather_Severity": weather, "Equipment_Failures": failures, "Formation_Hardness": hardness,
        "Phase": phase, "NPT_Ratio_Predicted": npt_ratio, "Spread_Pct_of_P50": spread_pct,
    }
else:
    inputs = st.session_state["last_inputs"]
    if "mc_results" in st.session_state:
        dur_res = st.session_state["mc_results"]["Duration_Hours"]
        inputs["Spread_Pct_of_P50"] = dur_res["spread"] / dur_res["P50"] * 100 if dur_res["P50"] != 0 else 0

st.write("")
recs = generate_recommendations(inputs)

n_high = sum(1 for r in recs if r["severity"] == "High")
n_med = sum(1 for r in recs if r["severity"] == "Medium")
n_low = sum(1 for r in recs if r["severity"] == "Low")

c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="rig-card" style="text-align:center;border-color:{RED}55;">
<div style="color:{RED};font-size:2rem;font-weight:800;">{n_high}</div>
<div style="color:{TEXT_DIM};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">High Priority</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="rig-card" style="text-align:center;border-color:{YELLOW}55;">
<div style="color:{YELLOW};font-size:2rem;font-weight:800;">{n_med}</div>
<div style="color:{TEXT_DIM};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">Medium Priority</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="rig-card" style="text-align:center;border-color:{GREEN}55;">
<div style="color:{GREEN};font-size:2rem;font-weight:800;">{n_low}</div>
<div style="color:{TEXT_DIM};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">Nominal</div></div>""", unsafe_allow_html=True)

st.write("")
icon_map = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
border_map = {"High": RED, "Medium": YELLOW, "Low": GREEN}

for r in recs:
    icon = icon_map.get(r["severity"], "⚪")
    color = border_map.get(r["severity"], AMBER)
    st.markdown(f"""
    <div class="rig-card" style="border-left: 4px solid {color}; margin-bottom: 12px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:700; color:{TEXT}; font-size:1rem;">{icon} {r['message']}</span>
            <span class="rig-pill rig-pill-{'red' if r['severity']=='High' else 'amber' if r['severity']=='Medium' else 'green'}">{r['severity']}</span>
        </div>
        <div style="color:{TEXT_DIM}; font-size:0.82rem; margin-top:8px;"><b>Trigger:</b> {r['trigger']}</div>
        <div style="color:{TEXT}; font-size:0.88rem; margin-top:4px;"><b style="color:{AMBER};">→ Action:</b> {r['action']}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
with st.expander("Why rule-based (not ML) for recommendations?"):
    st.markdown("""
    - **Explainable**: every recommendation traces to a simple `if condition: action` rule a non-technical manager can read.
    - **Fast to build, safe to trust**: fits a tight timeline; a learned policy (Reinforcement Learning) would need
      months of campaign data to train and validate safely.
    - **Upgrade path**: once 5–10 campaigns of real outcomes exist, these rules can evolve into a learned policy
      that optimizes contingency days against actual cost/NPT outcomes.
    """)
