"""
dashboard/pages/7_Iteration_Tracker.py

THE CORE "improving and improving" VISUAL.

Shows the actual loop, as a loop:
  PLAN (model predicts) -> EXECUTE (real result happens) ->
  COMPARE (predicted vs actual, scatter + error) -> RETRAIN ->
  next campaign's PLAN is sharper.

Reads reports/plan_vs_actual_log.csv, which campaign_manager.py builds
by predicting on each new campaign's rows with the models AS THEY WERE
BEFORE that campaign's retrain -- so this is a genuine, not simulated,
before/after comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, kpi_strip, TEXT, TEXT_DIM, AMBER, STEEL, GREEN, RED, YELLOW, PANEL_ALT, BORDER

st.set_page_config(page_title="Iteration Tracker | Drilling AI", page_icon="🔄", layout="wide")
apply_theme()
page_header("THE IMPROVEMENT LOOP", "🔄 Plan → Execute → Compare → Retrain",
            "This is the literal evidence of the self-improving system: every campaign's PLAN was made with the "
            "model as it existed BEFORE that campaign, then compared to what ACTUALLY happened, then fed back in.")

LOG_PATH = os.path.join("reports", "plan_vs_actual_log.csv")
HISTORY_PATH = os.path.join("reports", "campaign_history.json")
LABELS = {"Duration_Hours": "Duration", "Total_Cost_USD": "Cost", "NPT_Hours": "NPT"}
TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]
UNIT = {"Duration_Hours": "hrs", "Total_Cost_USD": "$", "NPT_Hours": "hrs"}

# ---- Loop diagram (always shown, even with no data yet) ----
st.markdown("##### The Loop, Visually")
fig_loop = go.Figure()

stages = [
    {"x": 0.15, "y": 0.8, "label": "1. PLAN", "sub": "Model predicts\nDuration/Cost/NPT", "color": AMBER},
    {"x": 0.85, "y": 0.8, "label": "2. EXECUTE", "sub": "Real drilling\nhappens", "color": STEEL},
    {"x": 0.85, "y": 0.2, "label": "3. COMPARE", "sub": "Actual vs.\nPredicted", "color": YELLOW},
    {"x": 0.15, "y": 0.2, "label": "4. RETRAIN", "sub": "Models learn\nfrom the gap", "color": GREEN},
]
for s in stages:
    fig_loop.add_shape(type="circle", x0=s["x"]-0.10, x1=s["x"]+0.10, y0=s["y"]-0.13, y1=s["y"]+0.13,
                        fillcolor=s["color"], opacity=0.18, line=dict(color=s["color"], width=2))
    fig_loop.add_annotation(x=s["x"], y=s["y"]+0.03, text=f"<b>{s['label']}</b>", showarrow=False,
                             font=dict(color=s["color"], size=14))
    fig_loop.add_annotation(x=s["x"], y=s["y"]-0.07, text=s["sub"].replace("\n", "<br>"), showarrow=False,
                             font=dict(color=TEXT_DIM, size=10))

arrows = [(0.27, 0.8, 0.73, 0.8), (0.85, 0.65, 0.85, 0.35), (0.73, 0.2, 0.27, 0.2), (0.15, 0.35, 0.15, 0.65)]
for x0, y0, x1, y1 in arrows:
    fig_loop.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2, arrowcolor=AMBER)

fig_loop.update_xaxes(visible=False, range=[-0.05, 1.05])
fig_loop.update_yaxes(visible=False, range=[0, 1])
fig_loop.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_loop, width='stretch')

st.divider()

if not os.path.exists(LOG_PATH):
    st.info(
        "**No iterations logged yet.** This page populates automatically the first time you upload and retrain "
        "on new campaign data via the **Retrain / New Campaign** page. Go there, upload the template (or your own "
        "actuals), and click 'Retrain' — then come back here to see Plan vs Actual."
    )
    st.stop()

log = pd.read_csv(LOG_PATH)
history = json.load(open(HISTORY_PATH)) if os.path.exists(HISTORY_PATH) else []

campaigns = log["Campaign_ID"].unique().tolist()

st.markdown(f"##### 📋 {len(campaigns)} Campaign(s) Logged — {len(log)} Total Plan-vs-Actual Records")

# ---- KPI: overall accuracy across all logged iterations ----
items = []
for t in TARGETS:
    valid = log.dropna(subset=[f"Predicted_{t}", f"Actual_{t}"])
    if len(valid) == 0:
        continue
    mae = (valid[f"Actual_{t}"] - valid[f"Predicted_{t}"]).abs().mean()
    items.append({"label": f"{LABELS[t]} MAE (at time of plan)", "value": f"{mae:.2f} {UNIT[t]}"})
if items:
    kpi_strip(items)

st.write("")
st.divider()

# ---- Campaign selector ----
st.markdown("##### 🔍 Inspect a Campaign: Predicted vs Actual")
campaign_pick = st.selectbox("Campaign", campaigns)
target_pick = st.selectbox("Target", TARGETS, format_func=lambda x: LABELS[x])

sub = log[log["Campaign_ID"] == campaign_pick].dropna(subset=[f"Predicted_{target_pick}", f"Actual_{target_pick}"])

if len(sub) == 0:
    st.warning("No valid predicted/actual pairs for this campaign+target (this can happen for the very first "
               "campaign uploaded before any model existed).")
else:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("###### Predicted vs. Actual — Scatter")
        fig = go.Figure()
        max_val = max(sub[f"Predicted_{target_pick}"].max(), sub[f"Actual_{target_pick}"].max()) * 1.1
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                  line=dict(color=TEXT_DIM, dash="dot", width=1.5), name="Perfect prediction"))
        fig.add_trace(go.Scatter(
            x=sub[f"Predicted_{target_pick}"], y=sub[f"Actual_{target_pick}"], mode="markers",
            marker=dict(color=AMBER, size=9, line=dict(width=1, color=BORDER)),
            text=sub["Well_ID"] + " · " + sub["Phase"],
            hovertemplate="%{text}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
            name="Phases",
        ))
        fig.update_layout(height=380, xaxis_title=f"Predicted {LABELS[target_pick]}",
                           yaxis_title=f"Actual {LABELS[target_pick]}", showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption("Points on the dotted diagonal = perfect prediction. Points above = actual came in higher than planned. Below = actual came in lower than planned.")

    with c2:
        st.markdown("###### Error Distribution")
        errors = sub[f"Actual_{target_pick}"] - sub[f"Predicted_{target_pick}"]
        fig2 = go.Figure(go.Histogram(x=errors, nbinsx=20, marker_color=STEEL))
        fig2.add_vline(x=0, line_color=GREEN, line_dash="dash", annotation_text="No error")
        fig2.update_layout(height=380, xaxis_title=f"Actual − Predicted ({UNIT[target_pick]})", yaxis_title="Count",
                            showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    mae_c = errors.abs().mean()
    bias_c = errors.mean()
    kpi_strip([
        {"label": "Records", "value": f"{len(sub)}"},
        {"label": "MAE", "value": f"{mae_c:.2f} {UNIT[target_pick]}"},
        {"label": "Bias (mean error)", "value": f"{bias_c:+.2f} {UNIT[target_pick]}",
         "sub": "Positive = system under-predicts" if bias_c > 0 else "Negative = system over-predicts"},
    ])

st.divider()

# ---- Cross-campaign improvement: MAE trend with annotations ----
if len(campaigns) >= 1 and history:
    st.markdown("##### 📈 Accuracy Across Iterations (the proof of improvement)")
    hist_rows = []
    for h in history:
        for t in TARGETS:
            if t in h.get("metrics", {}):
                hist_rows.append({"Campaign": h["campaign_label"], "Target": LABELS[t], "MAE": h["metrics"][t]["MAE"]})
    if hist_rows:
        hdf = pd.DataFrame(hist_rows)
        fig3 = px.line(hdf, x="Campaign", y="MAE", color="Target", markers=True,
                        color_discrete_sequence=[AMBER, STEEL, GREEN])
        fig3.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=10),
                            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig3, width='stretch')
        st.caption("Each point = the model's MAE on its held-out test set AFTER incorporating that campaign's data. "
                   "A downward trend across campaigns is the literal evidence of 'improving and improving'.")

st.divider()
with st.expander("How is this different from a simulated demo?"):
    st.markdown("""
    - When you upload Campaign N's actual data, the system **first** runs the model **as it existed before
      this upload** to generate the "Predicted" values you see here — this is a true before/after comparison,
      not predictions made after the fact with hindsight.
    - **Only after** logging that comparison does the system retrain on the combined data (old + new).
    - The next campaign you upload will be compared against THIS newly retrained model — so the loop is real
      and cumulative, exactly matching the Plan → Execute → Compare → Retrain cycle requested for this project.
    """)
