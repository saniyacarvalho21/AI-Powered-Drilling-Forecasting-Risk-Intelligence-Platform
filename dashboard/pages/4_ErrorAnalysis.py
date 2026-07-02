"""
dashboard/pages/4_ErrorAnalysis.py
Accuracy + calibration, styled as a QA/diagnostics report.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, kpi_strip, TEXT, TEXT_DIM, AMBER, STEEL, GREEN, RED, YELLOW, PANEL_ALT, BORDER

st.set_page_config(page_title="Error Analysis | Drilling AI", page_icon="📊", layout="wide")
apply_theme()
page_header("QUALITY ASSURANCE", "📊 Error Analysis & Model Calibration",
            "How much should you trust these numbers, and exactly where does the model struggle?")

REPORTS_DIR = "reports"
COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.json")
ERROR_PATH = os.path.join(REPORTS_DIR, "error_analysis.json")

if not (os.path.exists(COMPARISON_PATH) and os.path.exists(ERROR_PATH)):
    st.error("Run `python src/models/train_models.py` and `python -m src.models.error_analysis` first.")
    st.stop()

comparison = json.load(open(COMPARISON_PATH))
error_analysis = json.load(open(ERROR_PATH))
LABELS = {"Duration_Hours": "Duration", "Total_Cost_USD": "Cost", "NPT_Hours": "NPT"}

# ---- Top-line calibration KPI strip ----
st.markdown("##### 🎯 Calibration Scorecard")
items = []
for target, info in error_analysis.items():
    cov = info["coverage_pct_actual_in_p10_p90"]
    level = "Low" if 75 <= cov <= 88 else "Medium" if 65 <= cov <= 92 else "High"
    items.append({"label": f"{LABELS[target]} Coverage", "value": f"{cov:.1f}%", "sub": "target: 80%", "level": level})
kpi_strip(items)
st.write("")

with st.expander("ℹ️ What is 'coverage' and why does it matter?", expanded=False):
    st.markdown("""
    If the model says *"P10 = 23 hrs, P90 = 35 hrs"*, then **80% of real outcomes**
    should land between 23 and 35 hours. **Coverage** is the % of actual test-set
    outcomes that actually fell inside that band. Too low → the model is overconfident
    (band too narrow). Too high → the model is overcautious (band too wide, wasting
    planning margin).
    """)

st.divider()

# ---- Metric definitions as cards ----
st.markdown("##### 📐 Metric Definitions")
metric_cards = [
    ("MAE", "Mean Absolute Error", "Average size of the miss, in the target's own units (hrs/$). The number you'd say out loud to a manager."),
    ("RMSE", "Root Mean Squared Error", "Like MAE but penalizes big misses harder. RMSE >> MAE means occasional large errors exist."),
    ("MAPE", "Mean Absolute % Error", "Error as a % of actual value. Lets you compare Duration vs Cost vs NPT on one scale."),
    ("R²", "R-Squared", "Fraction of the real pattern the model explains. 0 = no better than guessing the average. 1 = perfect."),
]
cols = st.columns(4)
for col, (abbr, name, desc) in zip(cols, metric_cards):
    col.markdown(f"""
    <div class="rig-card" style="height:100%;">
        <div class="rig-mono" style="color:{AMBER};font-size:1.3rem;font-weight:700;">{abbr}</div>
        <div style="color:{TEXT};font-size:0.85rem;font-weight:600;margin:4px 0;">{name}</div>
        <div style="color:{TEXT_DIM};font-size:0.78rem;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# ---- Model comparison bars per target ----
st.markdown("##### 🏁 Model Comparison — All 4 Algorithms, All Targets")
for target, info in comparison.items():
    st.markdown(f"###### {LABELS[target]}")
    c1, c2 = st.columns([2, 1])
    rows = [{"Model": m, **v} for m, v in info["metrics"].items()]
    mdf = pd.DataFrame(rows)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mdf["Model"], y=mdf["MAE"], name="MAE",
                              marker_color=[GREEN if m == info["best_model"] else STEEL for m in mdf["Model"]]))
        fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="MAE (lower = better)")
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.dataframe(
            mdf[["Model", "MAE", "RMSE", "MAPE", "R2"]].round(3).style.apply(
                lambda r: ['background-color: rgba(61,220,151,0.15)' if r["Model"] == info["best_model"] else '' for _ in r],
                axis=1
            ),
            width='stretch', hide_index=True, height=180
        )
    st.write("")

st.divider()

# ---- Calibration table ----
st.markdown("##### 🎯 Calibration / Coverage Detail")
cov_rows = []
for target, info in error_analysis.items():
    cov_rows.append({
        "Target": LABELS[target], "Model": info["model"],
        "Coverage %": info["coverage_pct_actual_in_p10_p90"],
        "Target %": info["target_coverage_pct"],
        "Avg P90-P10 Spread": info["avg_p90_minus_p10_spread"],
        "Spread % of Pred.": info["avg_spread_pct_of_prediction"],
        "Verdict": info["calibration_verdict"].split(":")[0],
    })
st.dataframe(pd.DataFrame(cov_rows), width='stretch', hide_index=True)

st.divider()
st.markdown("##### 💬 Honest Interpretation")
st.markdown(f"""
<div class="rig-card">
<ul style="color:{TEXT_DIM};line-height:1.8;margin:0;padding-left:18px;">
<li><b style="color:{TEXT};">R² around 0.25–0.75</b> means the model explains a meaningful but limited share of the variation —
realistic for operational drilling data, where weather, geology, and equipment introduce genuine randomness no model fully predicts.</li>
<li><b style="color:{TEXT};">MAPE on NPT looks huge (~115%)</b> because many NPT values are near zero, so small absolute errors
become large percentages. MAE (≈0.7 hrs) is the more honest metric for NPT.</li>
<li><b style="color:{TEXT};">Coverage near 80–87%</b> across targets means the P10–P90 bands are reasonably well-calibrated.</li>
<li><b style="color:{TEXT};">Takeaway:</b> this system gives a data-driven, explainable starting point — refined further by Bayesian-style
updating as real campaign data comes in (see the Retrain page), not a replacement for engineering judgment.</li>
</ul>
</div>
""", unsafe_allow_html=True)
