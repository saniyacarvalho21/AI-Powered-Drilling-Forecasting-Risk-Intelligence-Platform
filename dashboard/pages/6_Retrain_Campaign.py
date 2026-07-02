"""
dashboard/pages/6_Retrain_Campaign.py
Self-improving loop: upload new actuals, retrain, watch the trend.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard.theme import apply_theme, page_header, kpi_strip, TEXT, TEXT_DIM, AMBER, STEEL, GREEN, RED, PANEL_ALT, BORDER
from src.models.campaign_manager import run_campaign_update, load_history, REQUIRED_RAW_COLUMNS, RAW_PATH

st.set_page_config(page_title="Retrain | Drilling AI", page_icon="🔁", layout="wide")
apply_theme()
page_header("SELF-IMPROVING LOOP", "🔁 Retrain on New Campaign Data",
            "Feed in real results from any completed campaign. The system appends, retrains all 4 algorithms, "
            "rechecks calibration, and logs the trend — this is the measurable evidence of improvement over time.")

LABELS = {"Duration_Hours": "Duration", "Total_Cost_USD": "Cost", "NPT_Hours": "NPT"}

# ---- Step 1: template ----
with st.container(border=True):
    st.markdown("##### 1️⃣ Get the Data Template")
    st.caption("Your CSV needs these columns (one row per phase). Extra columns are ignored.")
    st.code(", ".join(REQUIRED_RAW_COLUMNS), language=None)
    template_df = pd.read_csv(RAW_PATH).head(3)
    cols_present = [c for c in REQUIRED_RAW_COLUMNS if c in template_df.columns]
    template_csv = template_df[cols_present].to_csv(index=False)
    st.download_button("⬇️ Download example template", data=template_csv,
                        file_name="new_campaign_template.csv", mime="text/csv")

st.write("")

# ---- Step 2: upload + retrain ----
with st.container(border=True):
    st.markdown("##### 2️⃣ Upload New Campaign Data (Actuals)")
    campaign_label = st.text_input("Campaign label (optional)", placeholder="e.g. Campaign_2_NorthSea_Q3")
    uploaded = st.file_uploader("Upload CSV with new actual data", type="csv")

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.dataframe(new_df.head(10), width='stretch')

        if st.button("🚀 Retrain on this data", type="primary", width='stretch'):
            with st.spinner("Appending data, rebuilding features, retraining 4 algorithms × 3 targets... ~30-90 sec"):
                try:
                    result = run_campaign_update(new_df, campaign_label or None)
                    st.success(f"Retrained on {result['n_total_rows']} total rows "
                               f"({result['n_new_rows']} new) as '{result['campaign_label']}'.")
                    st.session_state["last_campaign_result"] = result
                    st.balloons()
                except ValueError as e:
                    st.error(str(e))

st.write("")

# ---- Step 3: history ----
st.markdown("##### 3️⃣ Campaign History — Improvement Over Time")
history = load_history()

if not history:
    st.info("No campaign history yet. Run `python -m src.models.campaign_manager` once in the terminal to seed a baseline.")
else:
    rows = []
    for h in history:
        row = {"Campaign": h["campaign_label"], "Timestamp": h["timestamp"][:16].replace("T", " "),
               "Total Rows": h["n_total_rows"], "New Rows": h["n_new_rows"]}
        for target in ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]:
            if target in h["metrics"]:
                row[f"{LABELS[target]} MAE"] = h["metrics"][target]["MAE"]
                row[f"{LABELS[target]} Coverage%"] = h["coverage"].get(target)
        rows.append(row)
    hist_df = pd.DataFrame(rows)

    # KPI: latest vs first campaign comparison
    if len(hist_df) >= 2:
        st.markdown("###### 📈 Latest vs. Baseline Campaign")
        kc = st.columns(3)
        for col, target in zip(kc, ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]):
            mae_col = f"{LABELS[target]} MAE"
            if mae_col in hist_df.columns:
                first, last = hist_df[mae_col].iloc[0], hist_df[mae_col].iloc[-1]
                delta = last - first
                delta_pct = (delta / first * 100) if first != 0 else 0
                color = GREEN if delta < 0 else RED
                col.markdown(f"""
                <div class="rig-card">
                    <div style="color:{TEXT_DIM};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">{LABELS[target]} MAE Trend</div>
                    <div class="rig-mono" style="color:{AMBER};font-size:1.4rem;font-weight:700;margin-top:4px;">{last:.2f}</div>
                    <div style="color:{color};font-size:0.82rem;margin-top:2px;">{'▼' if delta<0 else '▲'} {abs(delta_pct):.1f}% vs baseline ({first:.2f})</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    st.dataframe(hist_df, width='stretch', hide_index=True)

    st.markdown("###### Trend Charts")
    target_choice = st.selectbox("Target", ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"],
                                  format_func=lambda x: LABELS[x], key="trend_target")
    mae_col = f"{LABELS[target_choice]} MAE"
    cov_col = f"{LABELS[target_choice]} Coverage%"

    c1, c2 = st.columns(2)
    with c1:
        if mae_col in hist_df.columns:
            fig = go.Figure(go.Scatter(x=hist_df["Campaign"], y=hist_df[mae_col], mode="lines+markers",
                                        line=dict(color=AMBER, width=3), marker=dict(size=9, color=AMBER)))
            fig.update_layout(height=280, title=f"{LABELS[target_choice]}: MAE per campaign (↓ = improving)",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')
    with c2:
        if cov_col in hist_df.columns:
            fig2 = go.Figure(go.Scatter(x=hist_df["Campaign"], y=hist_df[cov_col], mode="lines+markers",
                                         line=dict(color=STEEL, width=3), marker=dict(size=9, color=STEEL)))
            fig2.add_hline(y=80, line_dash="dash", line_color=GREEN, annotation_text="80% target")
            fig2.update_layout(height=280, title=f"{LABELS[target_choice]}: Coverage % per campaign",
                                margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig2, width='stretch')

    if len(hist_df) == 1:
        st.caption("Only 1 campaign logged — upload more data above to start seeing the improvement trend.")

st.divider()
with st.expander("Why this design?"):
    st.markdown("""
    - **Append, never overwrite**: the raw dataset keeps growing — mirrors how a real operator accumulates a track record.
    - **Full retrain, not incremental "online learning"**: at this data scale, full retraining takes under 2 minutes
      and is simpler, more robust, and easier to audit than incremental learning, which can drift unpredictably.
    - **Validation before training**: uploads are checked for required columns and missing values BEFORE touching
      the master dataset — a bad upload can't corrupt your data.
    - **History log**: every retrain is timestamped and recorded — this IS the audit trail proving the
      "improving and improving" requirement.
    """)
