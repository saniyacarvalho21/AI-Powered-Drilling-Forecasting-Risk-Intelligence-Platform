"""
dashboard/pages/1_Forecasting.py
Predict Duration, Cost, NPT for a planned phase — styled as a rig
planning console.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore", message="X has feature names")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, kpi_strip, TEXT, TEXT_DIM, AMBER, STEEL, GREEN, PANEL_ALT, BORDER

st.set_page_config(page_title="Forecasting | Drilling AI", page_icon="📈", layout="wide")
apply_theme()
page_header("PLANNING CONSOLE", "📈 Forecast Duration, Cost & NPT",
            "Enter planned conditions for an upcoming phase — predictions come from the best model per target, "
            "automatically selected from 4 trained algorithms.")

DATA_PATH = os.path.join("data", "processed", "features_data.csv")
MODELS_DIR = "models"
TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]
LABELS = {"Duration_Hours": "Duration", "Total_Cost_USD": "Cost", "NPT_Hours": "NPT"}
UNITS = {"Duration_Hours": "hrs", "Total_Cost_USD": "USD", "NPT_Hours": "hrs"}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    models = {}
    for t in TARGETS:
        meta = json.load(open(os.path.join(MODELS_DIR, f"{t}_metadata.json")))
        model = joblib.load(os.path.join(MODELS_DIR, f"{t}_best_model.pkl"))
        models[t] = {"model": model, "meta": meta}
    return models


df = load_data()
models = load_models()

with st.container(border=True):
    st.markdown("##### 🛠️ Planned Phase Conditions")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div style='color:{AMBER};font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;'>Location & Type</div>", unsafe_allow_html=True)
        basin = st.selectbox("Basin", sorted(df["Basin"].unique()))
        well_type = st.selectbox("Well Type", sorted(df["Well_Type"].unique()))
        phase = st.selectbox("Phase", sorted(df["Phase"].unique()))
        formation = st.selectbox("Formation", sorted(df["Formation"].unique()))

    with c2:
        st.markdown(f"<div style='color:{AMBER};font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;'>Risk Conditions</div>", unsafe_allow_html=True)
        formation_hardness = st.slider("Formation Hardness (1-10)", 1, 10, 5)
        equipment_failures = st.slider("Expected Equipment Failures", 0, 4, 0)
        weather_severity = st.slider("Weather Severity (1-10)", 1, 10, 5)
        mud_weight = st.slider("Mud Weight (ppg)", 8.0, 16.0, 12.0, step=0.1)

    with c3:
        st.markdown(f"<div style='color:{AMBER};font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;'>Depth & Rate</div>", unsafe_allow_html=True)
        depth_from = st.number_input("Depth From (m)", min_value=0, value=1000)
        depth_to = st.number_input("Depth To (m)", min_value=0, value=1100)
        rop = st.slider("Expected ROP (m/hr)", 0.0, 80.0, 20.0)
        meterage = max(0, depth_to - depth_from) if phase == "Drilling" else 0

    submitted = st.button("🚀 Generate Forecast", type="primary", width='stretch')

if submitted:
    depth_interval = depth_to - depth_from
    drilling_eff = (rop / formation_hardness) if formation_hardness > 0 else 0
    failure_density_guess = equipment_failures / 12.0
    weather_impact_guess = weather_severity * 12.0
    medians = df.median(numeric_only=True)

    row = {
        "Depth_From_m": depth_from, "Depth_To_m": depth_to, "Meterage_Drilled_m": meterage,
        "Formation_Hardness": formation_hardness, "Equipment_Failures": equipment_failures,
        "Weather_Severity": weather_severity, "Mud_Weight_ppg": mud_weight,
        "ROP_m_per_hr": rop if phase == "Drilling" else 0.0,
        "Daily_Rig_Rate_USD": medians["Daily_Rig_Rate_USD"],
        "Materials_Cost_USD": medians["Materials_Cost_USD"],
        "Service_Cost_USD": medians["Service_Cost_USD"],
        "Depth_Interval_m": depth_interval, "Is_Drilling_Phase": 1 if phase == "Drilling" else 0,
        "Drilling_Efficiency": drilling_eff, "Failure_Density": failure_density_guess,
        "Weather_Impact": weather_impact_guess, "Phase_Order": medians["Phase_Order"],
        "Cumulative_Duration": medians["Cumulative_Duration"], "Cumulative_NPT": medians["Cumulative_NPT"],
        "Previous_Phase_NPT": medians["Previous_Phase_NPT"], "NPT_Was_Capped": 0,
        "Sum_Cost_Components": medians["Sum_Cost_Components"],
        "NPT_Ratio": medians.get("NPT_Ratio", 0.15), "Productive_Hours": medians.get("Productive_Hours", 10.0),
        "Cost_Per_Hour": medians.get("Cost_Per_Hour", 5000.0), "Cost_Per_Meter": medians.get("Cost_Per_Meter", 0.0),
    }

    for cat_col, val in [("Basin", basin), ("Well_Type", well_type), ("Phase", phase), ("Formation", formation)]:
        means = df.groupby(cat_col)["Total_Cost_USD"].mean()
        row[f"{cat_col}_Encoded"] = means.get(val, means.mean())
    bf = f"{basin}_{formation}"
    means_bf = df.groupby("Basin_Formation")["Total_Cost_USD"].mean()
    row["Basin_Formation_Encoded"] = means_bf.get(bf, means_bf.mean())

    raw_row = {"Basin": basin, "Well_Type": well_type, "Phase": phase, "Formation": formation, "Basin_Formation": bf}

    predictions = {}
    for target in TARGETS:
        meta = models[target]["meta"]
        model = models[target]["model"]
        feature_cols = meta["feature_columns"]
        cat_features = meta.get("cat_features", [])
        model_name = meta["best_model_name"]
        X_row = {col: row.get(col, raw_row.get(col, medians.get(col, 0))) for col in feature_cols}
        X_df = pd.DataFrame([X_row])[feature_cols]
        if model_name == "CatBoost":
            for c in cat_features:
                X_df[c] = X_df[c].astype(str)
        pred = max(0, float(model.predict(X_df)[0]))
        predictions[target] = pred

    st.session_state["last_prediction"] = predictions
    st.session_state["last_inputs"] = {
        "Basin": basin, "Well_Type": well_type, "Phase": phase, "Formation": formation,
        "Formation_Hardness": formation_hardness, "Equipment_Failures": equipment_failures,
        "Weather_Severity": weather_severity,
        "NPT_Ratio_Predicted": predictions["NPT_Hours"] / max(predictions["Duration_Hours"], 0.01),
    }

    st.write("")
    st.markdown("##### 📋 Forecast Result")

    fmts = {"Duration_Hours": "{:.2f} hrs", "Total_Cost_USD": "${:,.0f}", "NPT_Hours": "{:.2f} hrs"}
    kpi_strip([
        {"label": f"Predicted {LABELS[t]}", "value": fmts[t].format(predictions[t])}
        for t in TARGETS
    ])

    st.write("")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("###### Predicted vs. Historical Distribution (this Phase type)")
        phase_hist = df[df["Phase"] == phase]
        target_pick = st.selectbox("Compare on", TARGETS, format_func=lambda x: LABELS[x], key="cmp")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=phase_hist[target_pick], nbinsx=30, marker_color=STEEL, opacity=0.7,
                                    name="Historical"))
        fig.add_vline(x=predictions[target_pick], line_color=AMBER, line_width=3,
                      annotation_text="Your forecast", annotation_font=dict(color=AMBER))
        fig.update_layout(height=320, showlegend=False, xaxis_title=LABELS[target_pick], yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("###### Where This Forecast Sits (Percentile Rank)")
        for t in TARGETS:
            phase_vals = df[df["Phase"] == phase][t]
            pct_rank = (phase_vals < predictions[t]).mean() * 100
            color = GREEN if pct_rank < 60 else AMBER if pct_rank < 85 else "#E8543D"
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:{TEXT_DIM};">
                    <span>{LABELS[t]}</span><span class="rig-mono" style="color:{color};">{pct_rank:.0f}th percentile</span>
                </div>
                <div style="background:{PANEL_ALT};border-radius:6px;height:10px;margin-top:4px;border:1px solid {BORDER};">
                    <div style="background:{color};width:{pct_rank}%;height:100%;border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Percentile rank vs. all historical phases of this same type. Higher = more extreme than typical.")

    st.info("➡️ Go to **Monte Carlo Risk** for the P10/P50/P90 uncertainty band, or **Recommendations** for risk-based actions.")

st.divider()
with st.expander("📤 Compare a completed campaign: Plan vs Actual"):
    st.markdown("Upload an actuals CSV (same columns as training data, plus `Predicted_<target>` columns) to compare plan vs reality.")
    actuals_file = st.file_uploader("Upload actuals CSV", type="csv")
    if actuals_file is not None:
        actuals_df = pd.read_csv(actuals_file)
        st.dataframe(actuals_df.head(20), width='stretch')
        for target in TARGETS:
            if target in actuals_df.columns and f"Predicted_{target}" in actuals_df.columns:
                err = (actuals_df[target] - actuals_df[f"Predicted_{target}"]).abs().mean()
                st.metric(f"{LABELS[target]} — Mean Absolute Error vs Plan", f"{err:.2f}")
