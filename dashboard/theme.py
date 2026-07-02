"""
dashboard/theme.py

Shared visual design system for the whole app. Every page imports
`apply_theme()` at the top and uses the component helpers below
instead of raw st.metric / st.dataframe, so the look is consistent
everywhere (one source of truth, edit once, changes everywhere).

DESIGN TOKENS
-------------
Palette  : deep rig-blue background, amber/rust accent (drill-bit
           steel + rust + warning-amber -- the actual color world of
           a drilling rig at night), green/amber/red risk semantics.
Type     : "Sans" for body/UI, a heavier weight for numbers/KPIs so
           figures read like instrumentation, not a website.
Layout   : dark control-room aesthetic -- card panels on a near-black
           background, thin amber rule lines, monospace-flavored
           numerals for data, like a real SCADA / rig dashboard.
Signature: the P10/P50/P90 "fan" gauge styling and the sidebar
           "rig status" block, reused everywhere percentile risk
           appears.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---- Color tokens ----
BG = "#0B1220"
PANEL = "#121B2E"
PANEL_ALT = "#16223A"
BORDER = "#243454"
TEXT = "#E7ECF3"
TEXT_DIM = "#93A2BD"
AMBER = "#E8A33D"
AMBER_DIM = "#C98A2C"
RUST = "#C1521F"
STEEL = "#5B7A9D"
GREEN = "#3DDC97"
RED = "#E8543D"
YELLOW = "#E8C53D"

RISK_COLOR = {"High": RED, "Medium": YELLOW, "Low": GREEN}

PLOTLY_TEMPLATE = "drilling_dark"


def _register_plotly_template():
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, -apple-system, sans-serif", size=13),
        title=dict(font=dict(color=TEXT, size=16)),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_DIM),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_DIM),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[AMBER, STEEL, GREEN, RUST, YELLOW, RED],
        margin=dict(t=50, l=10, r=10, b=10),
    )
    pio.templates[PLOTLY_TEMPLATE] = tmpl
    pio.templates.default = PLOTLY_TEMPLATE


_register_plotly_template()


def apply_theme():
    """Call once at the top of every page (after st.set_page_config)."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(1200px 600px at 10% -10%, #16223A 0%, {BG} 55%),
            {BG};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: {TEXT_DIM};
    }}

    /* Hide default Streamlit chrome we don't want */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Headings */
    h1, h2, h3 {{
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }}
    h1 {{ color: {TEXT} !important; }}
    h2, h3 {{ color: {TEXT} !important; }}

    /* Native metric widget restyle -> instrumentation look */
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, {PANEL_ALT} 0%, {PANEL} 100%);
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_DIM} !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {AMBER} !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricDelta"] svg {{ display: none; }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        font-weight: 600;
        color: {TEXT_DIM};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {AMBER} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {AMBER} !important;
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button {{
        background: linear-gradient(180deg, {AMBER} 0%, {AMBER_DIM} 100%);
        color: #1A1300;
        border: none;
        font-weight: 700;
        border-radius: 8px;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        filter: brightness(1.08);
        color: #1A1300;
    }}

    /* Expanders / containers with border */
    div[data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        background: {PANEL};
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER} !important;
        background: {PANEL};
        border-radius: 10px;
    }}

    /* Custom classes used by component helpers below */
    .rig-eyebrow {{
        color: {AMBER};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    .rig-hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, {AMBER} 0%, {BORDER} 35%, transparent 100%);
        margin: 6px 0 18px 0;
    }}
    .rig-card {{
        background: linear-gradient(180deg, {PANEL_ALT} 0%, {PANEL} 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .rig-pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .rig-pill-green {{ background: rgba(61,220,151,0.14); color: {GREEN}; border: 1px solid rgba(61,220,151,0.35); }}
    .rig-pill-amber {{ background: rgba(232,197,61,0.14); color: {YELLOW}; border: 1px solid rgba(232,197,61,0.35); }}
    .rig-pill-red   {{ background: rgba(232,84,61,0.14); color: {RED}; border: 1px solid rgba(232,84,61,0.35); }}
    .rig-mono {{
        font-family: 'JetBrains Mono', monospace;
    }}
    </style>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    """Consistent page header: small amber eyebrow + big title + rule."""
    st.markdown(f"""
    <div class="rig-eyebrow">{eyebrow}</div>
    <h1 style="margin-top:0;">{title}</h1>
    {f'<p style="color:{TEXT_DIM};margin-top:-8px;">{subtitle}</p>' if subtitle else ''}
    <hr class="rig-hr"/>
    """, unsafe_allow_html=True)


def pill(text: str, level: str = "Low") -> str:
    cls = {"High": "rig-pill-red", "Medium": "rig-pill-amber", "Low": "rig-pill-green"}.get(level, "rig-pill-amber")
    return f'<span class="rig-pill {cls}">{text}</span>'


def kpi_strip(items):
    """
    items: list of dicts {label, value, sub (optional), level (optional: High/Medium/Low for color)}
    Renders a row of instrumentation-style KPI cards (not the default st.metric look).
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color = RISK_COLOR.get(item.get("level"), AMBER) if item.get("level") else AMBER
        sub_html = f'<div style="color:{TEXT_DIM};font-size:0.78rem;margin-top:2px;">{item.get("sub","")}</div>' if item.get("sub") else ""
        col.markdown(f"""
        <div class="rig-card">
            <div style="color:{TEXT_DIM};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">{item['label']}</div>
            <div class="rig-mono" style="color:{color};font-size:1.6rem;font-weight:700;margin-top:4px;">{item['value']}</div>
            {sub_html}
        </div>
        """, unsafe_allow_html=True)


def gauge_chart(value, p10, p50, p90, title, unit="", height=240):
    """
    A rig-instrument-style gauge showing where `value` (or p50) sits
    within the P10-P90 band -- visually communicates risk position,
    not just a number.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p50,
        number={"suffix": f" {unit}", "font": {"size": 28, "color": AMBER, "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 14, "color": TEXT_DIM}},
        gauge={
            "axis": {"range": [p10 * 0.85, p90 * 1.15], "tickcolor": TEXT_DIM, "tickfont": {"color": TEXT_DIM, "size": 10}},
            "bar": {"color": AMBER, "thickness": 0.35},
            "bgcolor": PANEL_ALT,
            "borderwidth": 0,
            "steps": [
                {"range": [p10 * 0.85, p10], "color": "rgba(232,84,61,0.18)"},
                {"range": [p10, p90], "color": "rgba(61,220,151,0.12)"},
                {"range": [p90, p90 * 1.15], "color": "rgba(232,84,61,0.18)"},
            ],
            "threshold": {
                "line": {"color": RED, "width": 3},
                "thickness": 0.8,
                "value": p90,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=40, b=10, l=20, r=20),
        font=dict(color=TEXT),
    )
    return fig
