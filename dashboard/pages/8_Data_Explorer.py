"""
dashboard/pages/8_Data_Explorer.py

Upload ANY drilling CSV → instant stats, distributions, correlation
heatmap, and AI predictions with scatter plots.
All computation happens right here in Python — no backend calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import warnings
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

st.set_page_config(page_title="Data Explorer | Drilling AI", page_icon="📂", layout="wide")

# ── inline theme ──────────────────────────────────────────────────────────────
TEXT = "#E7ECF3"; TEXT_DIM = "#93A2BD"; AMBER = "#E8A33D"; STEEL = "#5B7A9D"
GREEN = "#3DDC97"; RED = "#E8543D"; YELLOW = "#E8C53D"
PANEL = "#121B2E"; PANEL_ALT = "#16223A"; BORDER = "#243454"; BG = "#0B1220"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.stApp{{background:{BG};}}
section[data-testid="stSidebar"]{{background:{PANEL};border-right:1px solid {BORDER};}}
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}
h1,h2,h3{{font-weight:800!important;color:{TEXT}!important;}}
.rig-card{{background:linear-gradient(180deg,{PANEL_ALT} 0%,{PANEL} 100%);border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:10px;}}
div[data-testid="stMetric"]{{background:{PANEL_ALT};border:1px solid {BORDER};border-radius:10px;padding:14px;}}
div[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace!important;color:{AMBER}!important;font-weight:700!important;}}
div[data-testid="stMetricLabel"]{{color:{TEXT_DIM}!important;font-size:0.75rem!important;text-transform:uppercase;letter-spacing:0.06em;}}
div.stButton>button{{background:linear-gradient(180deg,{AMBER},{STEEL});color:#fff;border:none;font-weight:700;border-radius:8px;padding:8px 20px;}}
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
tmpl = go.layout.Template()
tmpl.layout = go.Layout(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL_ALT,
    font=dict(color=TEXT, family="Inter,sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM),
    colorway=[AMBER, STEEL, GREEN, RED, YELLOW],
    margin=dict(t=40, l=10, r=10, b=10),
)
pio.templates["drill"] = tmpl
pio.templates.default = "drill"

MODELS_DIR = "models"
TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]
LABELS  = {"Duration_Hours": "Duration (hrs)", "Total_Cost_USD": "Cost (USD)", "NPT_Hours": "NPT (hrs)"}
REQUIRED = [
    "Well_ID","Basin","Well_Type","Phase","Formation",
    "Depth_From_m","Depth_To_m","Meterage_Drilled_m","Duration_Hours",
    "NPT_Hours","ROP_m_per_hr","Mud_Weight_ppg","Formation_Hardness",
    "Equipment_Failures","Weather_Severity","Daily_Rig_Rate_USD",
    "Materials_Cost_USD","Service_Cost_USD","Total_Cost_USD",
]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f'<div style="color:{AMBER};font-size:0.72rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;">DATA EXPLORER</div>', unsafe_allow_html=True)
st.markdown("# 📂 Upload Data → Instant Analysis & Predictions")
st.markdown(f'<p style="color:{TEXT_DIM};margin-top:-8px;">Upload any drilling CSV (same column structure as training data) and get instant statistics, distributions, correlations, and AI predictions.</p>', unsafe_allow_html=True)
st.markdown(f'<hr style="border:none;height:1px;background:linear-gradient(90deg,{AMBER},{BORDER},transparent);margin:6px 0 18px 0;"/>', unsafe_allow_html=True)

# ── upload widget ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="rig-card" style="border:2px dashed {AMBER}55;">
<div style="color:{AMBER};font-weight:700;font-size:1rem;margin-bottom:6px;">📤 Upload Your Drilling CSV File</div>
<div style="color:{TEXT_DIM};font-size:0.85rem;">Accepts CSV with the same columns as the original training data. Extra columns are ignored. Maximum file size: 200 MB.</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose CSV file",
    type=["csv"],
    help="Upload a drilling dataset CSV. Must include: Well_ID, Basin, Phase, Duration_Hours, Total_Cost_USD, NPT_Hours, etc.",
    key="main_uploader"
)

if uploaded is None:
    st.markdown(f"""
    <div class="rig-card" style="text-align:center;padding:40px;">
        <div style="font-size:3rem;margin-bottom:16px;">📂</div>
        <div style="color:{TEXT};font-size:1.1rem;font-weight:700;">No file uploaded yet</div>
        <div style="color:{TEXT_DIM};font-size:0.9rem;margin-top:8px;">Click "Browse files" above or drag and drop your CSV</div>
        <div style="color:{TEXT_DIM};font-size:0.82rem;margin-top:16px;">Need a template? Go to the <b>Retrain / New Campaign</b> page and click "Download example template"</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── load and validate ─────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

missing_cols = [c for c in REQUIRED if c not in raw_df.columns]
have_targets = all(t in raw_df.columns for t in TARGETS)

if len(missing_cols) > len(REQUIRED) * 0.5:
    st.error(f"This file is missing too many required columns: {missing_cols}")
    st.stop()

st.success(f"✅ Loaded **{len(raw_df):,} rows** · **{raw_df['Well_ID'].nunique() if 'Well_ID' in raw_df.columns else '?'} wells** · **{len(raw_df.columns)} columns**")
st.write("")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "📈 Distributions",
    "🔗 Correlations",
    "🤖 AI Predictions",
    "📉 Probability Curves",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("##### Quick Stats")

    n_rows   = len(raw_df)
    n_wells  = raw_df["Well_ID"].nunique() if "Well_ID" in raw_df.columns else "N/A"
    n_miss   = int(raw_df[REQUIRED if not missing_cols else [c for c in REQUIRED if c in raw_df.columns]].isnull().sum().sum())
    n_dups   = int(raw_df.duplicated().sum())

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Rows",     f"{n_rows:,}")
    c2.metric("Unique Wells",   f"{n_wells}")
    c3.metric("Columns",        f"{len(raw_df.columns)}")
    c4.metric("Missing Values", f"{n_miss}", delta="⚠️ Fix" if n_miss > 0 else "✅ Clean", delta_color="inverse")
    c5.metric("Duplicate Rows", f"{n_dups}", delta="⚠️ Found" if n_dups > 0 else "✅ None",  delta_color="inverse")

    st.write("")

    # Data quality cards
    st.markdown("##### Data Quality")
    qc1, qc2, qc3 = st.columns(3)

    npt_bad = int((raw_df["NPT_Hours"] > raw_df["Duration_Hours"]).sum()) if all(c in raw_df.columns for c in ["NPT_Hours","Duration_Hours"]) else 0
    neg_vals = int((raw_df[["Duration_Hours","Total_Cost_USD","NPT_Hours"]].lt(0)).sum().sum()) if have_targets else 0
    zero_cost = int((raw_df["Total_Cost_USD"] == 0).sum()) if "Total_Cost_USD" in raw_df.columns else 0

    for col, label, val, good_msg, bad_msg in [
        (qc1, "NPT > Duration violations", npt_bad, "✅ No violations", f"⚠️ {npt_bad} rows — NPT exceeds phase duration"),
        (qc2, "Negative values", neg_vals, "✅ No negatives", f"⚠️ {neg_vals} negative values found"),
        (qc3, "Zero cost rows", zero_cost, "✅ No zero costs", f"⚠️ {zero_cost} rows with zero cost"),
    ]:
        color = GREEN if val == 0 else YELLOW
        col.markdown(f"""
        <div class="rig-card" style="border-left:4px solid {color};">
            <div style="color:{TEXT_DIM};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
            <div style="color:{color};font-weight:700;margin-top:4px;">{good_msg if val == 0 else bad_msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("##### Raw Data Preview (first 50 rows)")
    st.dataframe(raw_df.head(50), use_container_width=True)

    st.write("")
    st.markdown("##### Descriptive Statistics")
    st.dataframe(raw_df.describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### Distribution Explorer")

    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_options  = [c for c in ["Basin","Phase","Well_Type","Formation"] if c in raw_df.columns]

    d1, d2, d3 = st.columns(3)
    col_pick  = d1.selectbox("Column to plot", numeric_cols,
                              index=numeric_cols.index("Duration_Hours") if "Duration_Hours" in numeric_cols else 0)
    group_by  = d2.selectbox("Group / color by", ["(none)"] + cat_options)
    n_bins    = d3.slider("Number of bins", 10, 80, 40)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"###### Histogram — {col_pick}")
        if group_by == "(none)":
            fig = px.histogram(raw_df, x=col_pick, nbins=n_bins, color_discrete_sequence=[AMBER])
        else:
            fig = px.histogram(raw_df, x=col_pick, color=group_by, nbins=n_bins,
                                barmode="overlay", opacity=0.75,
                                color_discrete_sequence=[AMBER,STEEL,GREEN,RED,YELLOW,"#8C5B3D"])
        fig.update_layout(height=380, title=f"Distribution of {col_pick}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if group_by != "(none)":
            st.markdown(f"###### Box Plot — {col_pick} by {group_by}")
            fig2 = px.box(raw_df, x=group_by, y=col_pick, color=group_by,
                           color_discrete_sequence=[AMBER,STEEL,GREEN,RED,YELLOW,"#8C5B3D"])
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(f"###### Smooth Density Curve — {col_pick}")
            vals = raw_df[col_pick].dropna().values
            if len(vals) > 10:
                kde  = gaussian_kde(vals)
                xg   = np.linspace(vals.min(), vals.max(), 300)
                yg   = kde(xg)
                p10, p50, p90 = np.percentile(vals, [10, 50, 90])
                fig3 = go.Figure()
                mask_lo = xg <= p10
                mask_mi = (xg >= p10) & (xg <= p90)
                mask_hi = xg >= p90
                fig3.add_trace(go.Scatter(x=xg[mask_lo],y=yg[mask_lo],fill="tozeroy",mode="lines",line=dict(width=0),fillcolor="rgba(61,220,151,0.25)",showlegend=False))
                fig3.add_trace(go.Scatter(x=xg[mask_mi],y=yg[mask_mi],fill="tozeroy",mode="lines",line=dict(width=0),fillcolor="rgba(232,163,61,0.30)",showlegend=False))
                fig3.add_trace(go.Scatter(x=xg[mask_hi],y=yg[mask_hi],fill="tozeroy",mode="lines",line=dict(width=0),fillcolor="rgba(232,84,61,0.25)",showlegend=False))
                fig3.add_trace(go.Scatter(x=xg,y=yg,mode="lines",line=dict(color=AMBER,width=2.5),showlegend=False))
                for p,name,color in [(p10,"P10",GREEN),(p50,"P50",AMBER),(p90,"P90",RED)]:
                    fig3.add_vline(x=p,line_dash="dash",line_color=color,
                                   annotation_text=f"{name}: {p:.2f}",
                                   annotation_font=dict(color=color,size=11))
                fig3.update_layout(height=380, xaxis_title=col_pick, yaxis_title="Density")
                st.plotly_chart(fig3, use_container_width=True)

    # Phase-level summary table
    if "Phase" in raw_df.columns and col_pick in raw_df.columns:
        st.markdown(f"###### {col_pick} Summary by Phase")
        phase_stats = raw_df.groupby("Phase")[col_pick].agg(["mean","median","std","min","max"]).round(2).reset_index()
        st.dataframe(phase_stats, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("##### Full Correlation Heatmap")
    num_df = raw_df.select_dtypes(include=[np.number])
    corr   = num_df.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0,RED],[0.5,PANEL_ALT],[1,GREEN]], zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=8),
        hoverongaps=False,
    ))
    fig.update_layout(height=580, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Top Correlations with Total Cost")
    if "Total_Cost_USD" in corr.columns:
        cost_corr = corr["Total_Cost_USD"].drop("Total_Cost_USD").sort_values(key=abs, ascending=False).head(10)
        fig2 = go.Figure(go.Bar(
            x=cost_corr.values, y=cost_corr.index, orientation="h",
            marker=dict(color=[GREEN if v>0 else RED for v in cost_corr.values]),
            text=[f"{v:.3f}" for v in cost_corr.values], textposition="outside",
        ))
        fig2.update_layout(height=360, xaxis_title="Correlation with Total_Cost_USD",
                            xaxis=dict(range=[-1.1,1.1]))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Scatter: Any Two Columns")
    sc1, sc2, sc3 = st.columns(3)
    all_num = num_df.columns.tolist()
    x_col = sc1.selectbox("X axis", all_num, index=all_num.index("Duration_Hours") if "Duration_Hours" in all_num else 0)
    y_col = sc2.selectbox("Y axis", all_num, index=all_num.index("Total_Cost_USD") if "Total_Cost_USD" in all_num else 1)
    c_col = sc3.selectbox("Color by", ["(none)"]+[c for c in ["Phase","Basin","Well_Type","Formation"] if c in raw_df.columns])
    fig3  = px.scatter(raw_df, x=x_col, y=y_col,
                        color=None if c_col=="(none)" else c_col,
                        opacity=0.6, trendline="ols",
                        color_discrete_sequence=[AMBER,STEEL,GREEN,RED,YELLOW])
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("##### Run Trained AI Models on This Uploaded Data")
    st.caption("Uses the models saved in `models/` (from your training pipeline) to predict each row, then compares to actuals.")

    models_ready = all(
        os.path.exists(os.path.join(MODELS_DIR, f"{t}_best_model.pkl")) for t in TARGETS
    )

    if not models_ready:
        st.warning("No trained models found. Run `python src/models/train_models.py` first, then come back.")
        st.stop()

    if st.button("🤖 Run AI Predictions on This Data", type="primary"):
        with st.spinner("Building features and scoring all rows..."):
            try:
                sys.path.insert(0, ".")
                from src.features.feature_engineering import build_features

                feats = build_features(raw_df.copy())

                id_cols = [c for c in ["Well_ID","Phase"] if c in feats.columns]
                results_df = feats[id_cols].copy().reset_index(drop=True)

                for target in TARGETS:
                    meta  = json.load(open(os.path.join(MODELS_DIR, f"{target}_metadata.json")))
                    model = joblib.load(os.path.join(MODELS_DIR, f"{target}_best_model.pkl"))
                    fcols = meta["feature_columns"]
                    cats  = meta.get("cat_features", [])
                    mname = meta["best_model_name"]

                    miss = [c for c in fcols if c not in feats.columns]
                    if miss:
                        st.warning(f"Cannot predict {target} — missing columns: {miss}")
                        continue

                    X = feats[fcols].copy()
                    if mname == "CatBoost":
                        for c in cats:
                            X[c] = X[c].astype(str)

                    preds = model.predict(X)
                    results_df[f"Predicted_{target}"] = np.maximum(0, preds).round(2)

                    if target in feats.columns:
                        results_df[f"Actual_{target}"]    = feats[target].values
                        results_df[f"Error_{target}"]     = (feats[target].values - np.maximum(0, preds)).round(2)
                        results_df[f"AbsError_{target}"]  = np.abs(results_df[f"Error_{target}"]).round(2)

                st.session_state["pred_results"] = results_df
                st.success(f"✅ Scored {len(results_df):,} rows successfully.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    if "pred_results" in st.session_state:
        results_df = st.session_state["pred_results"]

        # MAE summary
        st.markdown("##### Accuracy Summary")
        mae_cols = st.columns(3)
        for col, target in zip(mae_cols, TARGETS):
            ae_col = f"AbsError_{target}"
            if ae_col in results_df.columns:
                mae_val = results_df[ae_col].mean()
                col.metric(f"{LABELS[target]} MAE", f"{mae_val:.2f}")

        st.write("")
        st.markdown("##### Predictions Table")
        st.dataframe(results_df, use_container_width=True)

        # Scatter: predicted vs actual for each target
        st.markdown("##### Predicted vs Actual — Scatter Plot")
        target_pick = st.selectbox("Select target", TARGETS, format_func=lambda x: LABELS[x], key="pred_scatter")
        pc, ac = f"Predicted_{target_pick}", f"Actual_{target_pick}"

        if pc in results_df.columns and ac in results_df.columns:
            valid = results_df.dropna(subset=[pc, ac])
            max_v = max(valid[pc].max(), valid[ac].max()) * 1.1

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(color=TEXT_DIM, dash="dot", width=1.5),
                name="Perfect prediction", showlegend=True,
            ))
            hover_text = (valid["Well_ID"] + " · " + valid["Phase"]).tolist() if all(c in valid.columns for c in ["Well_ID","Phase"]) else [""]*len(valid)
            fig.add_trace(go.Scatter(
                x=valid[pc], y=valid[ac], mode="markers",
                marker=dict(color=AMBER, size=7, opacity=0.7,
                            line=dict(width=0.5, color=BORDER)),
                text=hover_text,
                hovertemplate="<b>%{text}</b><br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
                name="Phases",
            ))
            fig.update_layout(
                height=440,
                title=f"{LABELS[target_pick]}: Predicted vs Actual",
                xaxis_title=f"Predicted {LABELS[target_pick]}",
                yaxis_title=f"Actual {LABELS[target_pick]}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Points on the dotted diagonal = perfect prediction. Above = under-predicted (actual came in higher). Below = over-predicted.")

            # Error distribution
            ec = f"Error_{target_pick}"
            if ec in results_df.columns:
                errors = results_df[ec].dropna()
                fig2 = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color=STEEL))
                fig2.add_vline(x=0, line_color=GREEN, line_dash="dash",
                               annotation_text="No error", annotation_font=dict(color=GREEN))
                fig2.update_layout(height=300, title="Error Distribution (Actual − Predicted)",
                                   xaxis_title="Error", yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)

        # Download
        csv_out = results_df.to_csv(index=False)
        st.download_button("⬇️ Download predictions CSV", data=csv_out,
                           file_name="drilling_predictions.csv", mime="text/csv")

        st.info("💡 To feed this data back and improve the model: go to **🔁 Retrain / New Campaign** page and upload the same file.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INTERACTIVE PROBABILITY CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("##### Interactive Probability Density Curves")
    st.caption("Hover anywhere on the curve to read the exact probability density at that value. Click the P-value lines to see where P10/P50/P90 sit. Use custom sliders for any percentile.")

    numeric_cols2 = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    group_options = ["(all data)"] + [c for c in ["Basin","Phase","Well_Type","Formation"] if c in raw_df.columns]

    pr1, pr2, pr3 = st.columns(3)
    prob_col   = pr1.selectbox("Column", numeric_cols2,
                                index=numeric_cols2.index("Duration_Hours") if "Duration_Hours" in numeric_cols2 else 0,
                                key="prob_col")
    prob_group = pr2.selectbox("Split by group", group_options, key="prob_group")
    prob_bw    = pr3.slider("Curve smoothness (bandwidth)", 0.1, 2.0, 0.5, step=0.05,
                             help="Lower = jagged (more detail), Higher = smoother")

    vals_all = raw_df[prob_col].dropna().values

    if len(vals_all) < 5:
        st.warning("Not enough data points.")
    else:
        x_min, x_max = vals_all.min(), vals_all.max()
        x_grid = np.linspace(x_min, x_max, 500)

        fig = go.Figure()

        if prob_group == "(all data)":
            groups = [("All data", raw_df)]
        else:
            groups = [(gname, raw_df[raw_df[prob_group] == gname])
                      for gname in sorted(raw_df[prob_group].dropna().unique())]

        palette = [AMBER, STEEL, GREEN, RED, YELLOW, "#8C5B3D", "#7A5B9D", "#5B9D7A"]
        for (gname, gdf), color in zip(groups, palette):
            gvals = gdf[prob_col].dropna().values
            if len(gvals) < 5:
                continue
            kde  = gaussian_kde(gvals, bw_method=prob_bw)
            ydns = kde(x_grid)
            fig.add_trace(go.Scatter(
                x=x_grid, y=ydns, mode="lines",
                line=dict(color=color, width=2.5),
                name=gname,
                hovertemplate=f"<b>{gname}</b><br>{prob_col}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
                fill="tozeroy" if len(groups) == 1 else None,
                fillcolor="rgba(232,163,61,0.15)" if len(groups) == 1 else None,
            ))

        # P-value lines for the full dataset
        p10, p50, p90 = np.percentile(vals_all, [10, 50, 90])
        kde_full = gaussian_kde(vals_all, bw_method=prob_bw)
        for p, name, color in [(p10,"P10",GREEN),(p50,"P50",AMBER),(p90,"P90",RED)]:
            yp = float(kde_full(p)[0])
            fig.add_trace(go.Scatter(
                x=[p, p], y=[0, yp], mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{name} = {p:.2f}", showlegend=True,
                hovertemplate=f"<b>{name}</b><br>Value: {p:.2f}<extra></extra>",
            ))
            fig.add_annotation(
                x=p, y=yp,
                text=f"<b>{name}</b><br>{p:.2f}",
                showarrow=True, arrowhead=2, arrowcolor=color, arrowsize=1.2,
                font=dict(color=color, size=11), bgcolor=PANEL, bordercolor=color,
                ay=-45,
            )

        fig.update_layout(
            height=500,
            title=f"Probability Density — {prob_col}",
            xaxis_title=prob_col,
            yaxis_title="Probability Density",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Custom percentile reader ────────────────────────────────────────
        st.markdown("##### 🎯 Read Any Percentile — Interactive")
        st.caption("Move the sliders to read the exact value at any percentile, or type a value to find its percentile rank.")

        rdr1, rdr2 = st.columns(2)
        with rdr1:
            st.markdown("###### Percentile → Value")
            p_input = st.slider("Select percentile", 0.0, 100.0, 50.0, step=0.1, key="pslider")
            p_value = float(np.percentile(vals_all, p_input))
            st.markdown(f"""
            <div class="rig-card" style="text-align:center;">
                <div style="color:{TEXT_DIM};font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;">P{p_input:.1f}</div>
                <div style="color:{AMBER};font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{p_value:.3f}</div>
                <div style="color:{TEXT_DIM};font-size:0.8rem;">{p_input:.1f}% of values are below this</div>
            </div>
            """, unsafe_allow_html=True)

        with rdr2:
            st.markdown("###### Value → Percentile Rank")
            col_min, col_max = float(vals_all.min()), float(vals_all.max())
            v_input = st.slider("Select value", col_min, col_max,
                                 float(np.median(vals_all)),
                                 step=(col_max-col_min)/200, key="vslider")
            pct_rank = float((vals_all < v_input).mean() * 100)
            bar_color = GREEN if pct_rank < 60 else YELLOW if pct_rank < 85 else RED
            st.markdown(f"""
            <div class="rig-card" style="text-align:center;">
                <div style="color:{TEXT_DIM};font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;">Percentile Rank of {v_input:.2f}</div>
                <div style="color:{bar_color};font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">P{pct_rank:.1f}</div>
                <div style="color:{TEXT_DIM};font-size:0.8rem;">{pct_rank:.1f}% of all values are below {v_input:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Summary stats table
        st.markdown("##### Distribution Summary")
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.percentile(vals_all, pcts)
        summary_df = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pcts],
            "Value": [f"{v:.3f}" for v in pct_vals],
            "Interpretation": [
                "Only 1% of values fall below this",
                "Very low (5th percentile)",
                "P10 — Optimistic planning estimate",
                "Lower quartile",
                "P50 — Median (most likely)",
                "Upper quartile",
                "P90 — Conservative planning estimate",
                "Very high (95th percentile)",
                "Only 1% of values exceed this",
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)