"""
src/utils/ui.py

Shared visual design system for the dashboard. Centralizing this means
every page uses the same color tokens, fonts, and component styling --
the difference between "a Streamlit demo" and "a designed product".

Design language: dark rig command-center.
  - Background: near-black blue-charcoal (#0B1220 / #121B2E)
  - Primary accent: amber / warning-lamp orange (#E8A33D) -- the color
    of rig floor warning lights and analog gauges
  - Secondary accent: teal/cyan (#34D0C6) -- "healthy system" signal,
    used for P50 / good-status / positive deltas
  - Risk red: (#E5484D) for P90 / high-severity
  - Risk green: (#3DD68C) for P10 / low-severity
  - Type: system sans for UI, tabular numerals for data (monospace
    accents on big KPI numbers, like a digital rig readout)
"""

import streamlit as st
import plotly.graph_objects as go

# ---- Color tokens ----
BG = "#0B1220"
PANEL = "#121B2E"
PANEL_2 = "#16213A"
BORDER = "#23304A"
TEXT = "#E7ECF3"
TEXT_DIM = "#8FA0BD"
AMBER = "#E8A33D"
TEAL = "#34D0C6"
RED = "#E5484D"
GREEN = "#3DD68C"
BLUE = "#5B8DEF"

PLOTLY_TEMPLATE = "plotly_dark"


def inject_global_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 0%, rgba(232,163,61,0.06), transparent 40%),
                radial-gradient(circle at 90% 10%, rgba(52,208,198,0.05), transparent 35%),
                {BG};
        }}

        section[data-testid="stSidebar"] {{
            background: {PANEL};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}

        h1, h2, h3 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            letter-spacing: -0.01em;
        }}

        .rig-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {AMBER};
            font-weight: 600;
            margin-bottom: 2px;
        }}

        .rig-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: {TEXT};
            margin: 0 0 4px 0;
            line-height: 1.15;
        }}

        .rig-subtitle {{
            color: {TEXT_DIM};
            font-size: 0.95rem;
            max-width: 760px;
            line-height: 1.5;
        }}

        .rig-divider {{
            height: 1px;
            background: linear-gradient(90deg, {AMBER}55, {BORDER} 35%, transparent);
            margin: 1.1rem 0 1.4rem 0;
            border: none;
        }}

        /* KPI card */
        .kpi-card {{
            background: linear-gradient(165deg, {PANEL_2}, {PANEL});
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px 14px 18px;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--accent, {AMBER});
            opacity: 0.85;
        }}
        .kpi-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {TEXT_DIM};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.65rem;
            font-weight: 700;
            color: {TEXT};
            line-height: 1.1;
        }}
        .kpi-delta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            margin-top: 6px;
            font-weight: 600;
        }}
        .kpi-delta.up {{ color: {RED}; }}
        .kpi-delta.down {{ color: {GREEN}; }}
        .kpi-delta.flat {{ color: {TEXT_DIM}; }}

        /* status badge */
        .badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid;
        }}
        .badge.green {{ color: {GREEN}; border-color: {GREEN}55; background: {GREEN}14; }}
        .badge.amber {{ color: {AMBER}; border-color: {AMBER}55; background: {AMBER}14; }}
        .badge.red   {{ color: {RED};   border-color: {RED}55;   background: {RED}14; }}
        .badge.teal  {{ color: {TEAL};  border-color: {TEAL}55;  background: {TEAL}14; }}

        /* panel / section container */
        .rig-panel {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 20px;
        }}

        .section-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {TEAL};
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-label::before {{
            content: "";
            width: 6px; height: 6px;
            border-radius: 50%;
            background: {TEAL};
            box-shadow: 0 0 8px {TEAL};
        }}

        /* dataframe polish */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}

        button[kind="primary"] {{
            background: linear-gradient(135deg, {AMBER}, #C97E1F) !important;
            border: none !important;
            font-weight: 600 !important;
        }}

        [data-testid="stMetricValue"] {{
            font-family: 'Space Grotesk', sans-serif;
        }}

        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {BG}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 5px; }}
    </style>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f"""
        <div class="rig-eyebrow">{eyebrow}</div>
        <div class="rig-title">{title}</div>
        <div class="rig-subtitle">{subtitle}</div>
        <hr class="rig-divider"/>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None, delta_dir: str = "flat", accent: str = AMBER):
    delta_html = ""
    if delta:
        arrow = {"up": "▲", "down": "▼", "flat": "•"}[delta_dir]
        delta_html = f'<div class="kpi-delta {delta_dir}">{arrow} {delta}</div>'
    st.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "amber"):
    return f'<span class="badge {color}">{text}</span>'


def style_fig(fig: go.Figure, height=380, title=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_2, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def risk_gauge(value: float, p10: float, p50: float, p90: float, title: str, suffix: str = ""):
    """
    An analog-style gauge (like a rig instrument dial) showing where the
    P50 point estimate sits relative to the P10-P90 risk band.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 30, "family": "Space Grotesk, sans-serif", "color": TEXT}},
        title={"text": title, "font": {"size": 13, "family": "JetBrains Mono, monospace", "color": TEXT_DIM}},
        gauge={
            "axis": {"range": [p10 * 0.85, p90 * 1.15], "tickcolor": TEXT_DIM, "tickfont": {"color": TEXT_DIM, "size": 9}},
            "bar": {"color": AMBER, "thickness": 0.32},
            "bgcolor": PANEL_2,
            "borderwidth": 0,
            "steps": [
                {"range": [p10 * 0.85, p10], "color": GREEN + "33"},
                {"range": [p10, p90], "color": TEAL + "22"},
                {"range": [p90, p90 * 1.15], "color": RED + "33"},
            ],
            "threshold": {
                "line": {"color": TEAL, "width": 3},
                "thickness": 0.85,
                "value": p50,
            },
        },
    ))
    return style_fig(fig, height=230)


def fan_chart(p10, p50, p90, label, unit_fmt="{:.1f}"):
    """
    A horizontal 'risk band' bar -- P10 to P90 range with P50 marked --
    a compact alternative/companion to the full histogram.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[p90 - p10], y=[label], base=[p10], orientation="h",
        marker=dict(color=TEAL, opacity=0.28, line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    for val, color, name in [(p10, GREEN, "P10"), (p50, AMBER, "P50"), (p90, RED, "P90")]:
        fig.add_trace(go.Scatter(
            x=[val], y=[label], mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=2, color=BG)),
            text=[f"{name}<br>{unit_fmt.format(val)}"], textposition="top center",
            textfont=dict(size=10, color=color, family="JetBrains Mono, monospace"),
            showlegend=False,
        ))
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=160)
