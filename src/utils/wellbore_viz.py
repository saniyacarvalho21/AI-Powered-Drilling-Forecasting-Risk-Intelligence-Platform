"""
src/utils/wellbore_viz.py

A schematic wellbore / depth-column visualization -- the kind of visual
an actual drilling engineer expects to see: a vertical shaft showing
depth intervals colored by phase, annotated with formation and key
metrics. Built with Plotly so it's interactive (hover, zoom) without
needing a heavy 3D engine.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.utils.ui import style_fig, BG, PANEL_2, BORDER, TEXT, TEXT_DIM, AMBER, TEAL, RED, GREEN

PHASE_COLORS = {
    "Setup": "#5B8DEF",
    "Rig Move": "#8A7FE8",
    "Drilling": "#E8A33D",
    "Tripping": "#34D0C6",
    "Casing": "#3DD68C",
    "Cementing": "#E5C53D",
    "Logging": "#E55DAD",
    "Completion": "#E5484D",
}

FORMATION_COLORS = {
    "Shale": "#3a2f23",
    "Sandstone": "#5a4a2f",
    "Limestone": "#4a4a55",
    "Dolomite": "#3f4a4a",
}


def wellbore_schematic(well_df: pd.DataFrame, well_id: str):
    """
    well_df: rows for ONE well, must have Phase, Depth_From_m, Depth_To_m,
             Formation, Duration_Hours, NPT_Hours columns, ordered by
             Phase_Order if available.
    """
    df = well_df.copy()
    if "Phase_Order" in df.columns:
        df = df.sort_values("Phase_Order")

    fig = go.Figure()

    # Formation background bands (behind everything)
    depth_rows = df[df["Depth_To_m"] > df["Depth_From_m"]]
    for _, row in depth_rows.iterrows():
        formation = row.get("Formation", "Unknown")
        color = FORMATION_COLORS.get(formation, "#333")
        fig.add_shape(
            type="rect", x0=-1, x1=1,
            y0=-row["Depth_To_m"], y1=-row["Depth_From_m"],
            fillcolor=color, opacity=0.5, line=dict(width=0), layer="below",
        )

    # The wellbore shaft (depth-based phases only)
    for _, row in depth_rows.iterrows():
        phase = row["Phase"]
        color = PHASE_COLORS.get(phase, "#888")
        npt_ratio = row.get("NPT_Hours", 0) / max(row.get("Duration_Hours", 1), 0.01)
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-row["Depth_From_m"], -row["Depth_To_m"]],
            mode="lines",
            line=dict(color=color, width=22),
            name=phase,
            hovertemplate=(
                f"<b>{phase}</b><br>"
                f"Depth: {row['Depth_From_m']:.0f}-{row['Depth_To_m']:.0f} m<br>"
                f"Duration: {row.get('Duration_Hours', 0):.1f} hrs<br>"
                f"NPT: {row.get('NPT_Hours', 0):.1f} hrs ({npt_ratio:.0%})<br>"
                f"Formation: {row.get('Formation', 'N/A')}"
                "<extra></extra>"
            ),
            showlegend=True,
        ))
        # NPT marker -- a red tick where non-productive time was significant
        if npt_ratio > 0.15:
            mid = -(row["Depth_From_m"] + row["Depth_To_m"]) / 2
            fig.add_trace(go.Scatter(
                x=[0.55], y=[mid], mode="markers+text",
                marker=dict(symbol="triangle-left", size=11, color=RED),
                text=[f"NPT {npt_ratio:.0%}"], textposition="middle right",
                textfont=dict(size=9, color=RED, family="JetBrains Mono, monospace"),
                showlegend=False, hoverinfo="skip",
            ))

    # Surface marker
    fig.add_annotation(x=0, y=8, text="⛽ SURFACE", showarrow=False,
                        font=dict(size=11, color=TEXT_DIM, family="JetBrains Mono, monospace"))

    max_depth = depth_rows["Depth_To_m"].max() if len(depth_rows) else 100
    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(title="Depth (m)", tickformat=",.0f", autorange=False,
                    range=[-max_depth * 1.08, 20]),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
    )
    # Flip y tick labels to positive depth values
    fig.update_yaxes(tickvals=list(range(0, -int(max_depth) - 100, -200)),
                       ticktext=[str(-v) for v in range(0, -int(max_depth) - 100, -200)])

    return style_fig(fig, height=520, title=f"Wellbore Schematic — {well_id}")


def campaign_gantt(df: pd.DataFrame, n_wells: int = 12):
    """
    A Gantt-style phase timeline across multiple wells -- gives an
    'operations command center' overview of a whole campaign at a
    glance. Uses Cumulative_Duration as a proxy timeline if no real
    dates exist.
    """
    wells = df["Well_ID"].unique()[:n_wells]
    sub = df[df["Well_ID"].isin(wells)].copy()
    if "Phase_Order" in sub.columns:
        sub = sub.sort_values(["Well_ID", "Phase_Order"])

    fig = go.Figure()
    for well in wells:
        wdf = sub[sub["Well_ID"] == well]
        start = 0
        for _, row in wdf.iterrows():
            dur = row["Duration_Hours"]
            phase = row["Phase"]
            color = PHASE_COLORS.get(phase, "#888")
            fig.add_trace(go.Bar(
                x=[dur], y=[well], base=[start], orientation="h",
                marker=dict(color=color, line=dict(width=0.5, color=BG)),
                name=phase, showlegend=(well == wells[0]),
                hovertemplate=f"<b>{phase}</b><br>{dur:.1f} hrs<extra></extra>",
            ))
            start += dur

    fig.update_layout(barmode="stack", xaxis_title="Cumulative Hours", yaxis_title=None,
                        legend=dict(orientation="h", y=1.12, font=dict(size=10)))
    return style_fig(fig, height=420, title="Campaign Timeline — Phase Sequence by Well")
