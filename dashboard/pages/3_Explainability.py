"""
dashboard/pages/3_Explainability.py
SHAP-driven explainability, styled as a diagnostics panel.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, kpi_strip, TEXT, TEXT_DIM, AMBER, STEEL, GREEN, RED, YELLOW, PANEL_ALT, BORDER

st.set_page_config(page_title="Explainability | Drilling AI", page_icon="🔍", layout="wide")
apply_theme()
page_header("DIAGNOSTICS", "🔍 Explainability — Why the Model Predicted This",
            "SHAP (Shapley Additive exPlanations) values quantify exactly how much each feature pushed a prediction up or down.")

REPORTS_DIR = "reports"
SHAP_PATH = os.path.join(REPORTS_DIR, "shap_summary.json")

if not os.path.exists(SHAP_PATH):
    st.error("Run `python src/explainability/shap_analysis.py` first to generate this page's data.")
    st.stop()

shap_summary = json.load(open(SHAP_PATH))
LABELS = {"Duration_Hours": "Duration", "Total_Cost_USD": "Cost", "NPT_Hours": "NPT"}

tabs = st.tabs([f"⚙️ {LABELS[t]}" for t in shap_summary.keys()])

interpretation_map = {
    "Meterage_Drilled_m": "More meters drilled in a phase directly extends duration — physically expected, sanity-checks the model.",
    "Depth_Interval_m": "Larger depth intervals per phase take longer — consistent with drilling physics.",
    "ROP_m_per_hr": "Rate of penetration is a direct driver of how fast a Drilling phase completes.",
    "Equipment_Failures": "Equipment failures are a major driver of cost and NPT — supports prioritizing preventive maintenance.",
    "Weather_Severity": "Weather severity meaningfully impacts NPT — supports building weather contingency into schedules.",
    "Productive_Hours": "More productive hours correlates with higher absolute cost (longer phases cost more in total).",
    "NPT_Ratio": "Higher NPT ratio is associated with higher cost per phase — inefficiency has a direct cost.",
    "Failure_Density": "Failure RATE (not just count) matters — frequent small failures add up.",
    "Phase_Encoded": "Different phase types have systematically different baseline costs.",
    "Drilling_Efficiency": "ROP normalized by formation hardness captures 'true' drilling performance.",
    "Cumulative_Duration": "Wells that have already taken longer tend to continue trending that way (compounding delays).",
    "Mud_Weight_ppg": "Mud weight reflects formation pressure complexity, which can affect time and cost.",
}

for tab, (target, info) in zip(tabs, shap_summary.items()):
    with tab:
        if "error" in info:
            st.warning(f"SHAP could not be computed for {target}: {info['error']}")
            continue

        fi_df = pd.DataFrame(info["feature_importance"], columns=["Feature", "Impact"])
        fi_df = fi_df.sort_values("Impact", ascending=True)
        max_impact = fi_df["Impact"].max()

        c1, c2 = st.columns([1.5, 1])

        with c1:
            st.markdown(f"###### Top Feature Drivers — Model: `{info['model']}`")
            colors = [AMBER if v == fi_df["Impact"].max() else STEEL for v in fi_df["Impact"]]
            fig = go.Figure(go.Bar(
                x=fi_df["Impact"], y=fi_df["Feature"], orientation="h",
                marker=dict(color=colors),
            ))
            fig.update_layout(height=420, xaxis_title="Mean |SHAP value| (impact on prediction)",
                               margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.markdown("###### Driver Strength")
            top5 = fi_df.sort_values("Impact", ascending=False).head(5)
            for _, r in top5.iterrows():
                pct = r["Impact"] / max_impact * 100
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="font-size:0.82rem;color:{TEXT};font-weight:600;">{r['Feature']}</div>
                    <div style="background:{PANEL_ALT};border-radius:6px;height:8px;margin-top:3px;border:1px solid {BORDER};">
                        <div style="background:linear-gradient(90deg,{AMBER},{STEEL});width:{pct}%;height:100%;border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("###### Operational Interpretation")
        top3 = fi_df.sort_values("Impact", ascending=False).head(3)["Feature"].tolist()
        for feat in top3:
            note = interpretation_map.get(feat, "A meaningful driver of this prediction.")
            st.markdown(f"""
            <div class="rig-card" style="margin-bottom:10px;">
                <b style="color:{AMBER};">{feat}</b><br>
                <span style="color:{TEXT_DIM};font-size:0.88rem;">{note}</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown("##### 🆚 All Targets — Top 5 Drivers Compared")
rows = []
for t, info in shap_summary.items():
    if "error" not in info:
        for feat, val in info["feature_importance"][:5]:
            rows.append({"Target": LABELS.get(t, t), "Feature": feat, "Impact": val})
if rows:
    comp_df = pd.DataFrame(rows)
    fig2 = px.bar(comp_df, x="Impact", y="Feature", color="Target", orientation="h", barmode="group",
                  color_discrete_sequence=[AMBER, STEEL, GREEN])
    fig2.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10))
    st.plotly_chart(fig2, width='stretch')
