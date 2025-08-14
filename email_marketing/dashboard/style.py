"""Dashboard styling configuration.

MauBank-inspired light theme for the Streamlit dashboard.
This module centralises the look & feel via CSS variables and targeted
component rules. It is drop-in and backward compatible with the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st


@dataclass
class Theme:
    """Configuration values for the dashboard's appearance."""

    primary_font: str = "Arial, sans-serif"
    heading_font: str = "Arial, sans-serif"
    base_font_size: int = 14
    spacing_unit: int = 8
    # (bg, secondary bg, text). Kept for compatibility if used elsewhere.
    colours: Tuple[str, str, str] = ("#ffffff", "#f5f5f5", "#222222")


THEME = Theme()


def apply_theme(_: Optional[object] = None) -> None:
    """Inject MauBank-inspired light theme into Streamlit app (full coverage)."""
    st.markdown(
        """
    <style>
    /* ========= MauBank tokens ========= */
    :root {
    /* Brand */
    --brand: #002147;        /* Deep blue (aprox MauBank) */
    --brand-600: #001437;    /* Hover/active */
    --accent: #10B981;       /* Success (convención UX) */
    --accent-yellow: #FFD100;/* Acento opcional en chips/badges */
    --danger: #EF4444;

    /* Surfaces */
    --bg: #FFFFFF;
    --bg-2: #F5F5F5;
    --card: #F7F7F7;
    --border: #E5E7EB;

    /* Typography */
    --text: #222222;
    --muted: #6C757D;

    /* Focus/A11y */
    --focus-ring: rgba(0,33,71,0.35); /* brand with alpha */
    --ring-width: 2px;

    /* Shape & elevation */
    --radius: 8px;
    --shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    /* ========= Global ========= */
    html, body, .stApp, .stAppViewContainer, .main, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Arial, Helvetica, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    }
    *:focus-visible { outline: var(--ring-width) solid var(--focus-ring) !important; outline-offset: 1px; }
    @media (prefers-reduced-motion: reduce){
    *{ transition:none !important; animation:none !important; }
    }

    /* ========= Sidebar ========= */
    [data-testid="stSidebar"] { background: var(--bg-2) !important; }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] a:hover { color: var(--brand) !important; }
    [data-testid="stSidebar"] img,
    [data-testid="stSidebar"] svg {
    display: inline-block !important;
    max-width: 100% !important;
    }
    [data-testid="stSidebar"] .mo-sidebar-avatar{
    height: 72px;               /* >= image height */
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: visible !important;
    margin: 2px 0 8px 0;
    line-height: 0;
    }
    [data-testid="stSidebar"] .mo-sidebar-avatar img{
    display: block !important;
    max-height: 64px;
    width: auto;
    }

    /* ========= Layout / containers ========= */
    .block-container { padding-top: 1.25rem; }
    .stContainer, .stCard { background: transparent !important; }
    hr { border-top: 1px solid var(--border); }

    /* ========= Headings / text ========= */
    h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
    small, .stCaption, .stMarkdown em, .stMarkdown .small { color: var(--muted) !important; }

    /* ========= Buttons ========= */
    .stButton > button {
    background: var(--brand) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--brand) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
    transition: background .15s ease, border-color .15s ease, transform .05s ease-in;
    }
    .stButton > button:hover {
    background: var(--brand-600) !important;
    border-color: var(--brand-600) !important;
    }
    .stButton > button:active { transform: translateY(1px); }
    .stButton > button:disabled { opacity:.6; cursor:not-allowed; }

    /* Secondary (ghost) */
    .stButton > button[kind="secondary"]{
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    box-shadow: none !important;
    }

    /* ========= Inputs / selects ========= */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    textarea, input[type="text"], input[type="search"]{
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: var(--radius) !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #9CA3AF !important; }

    .stDateInput input, .stDatetimeInput input,
    .stSelectbox [data-baseweb="select"] > div{
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: var(--radius) !important;
    }
    [data-baseweb="select"] div:focus-within{
    box-shadow: 0 0 0 2px var(--focus-ring) !important;
    border-color: var(--brand) !important;
    }
    /* Tags en multi-select (BaseWeb) */
    [data-baseweb="tag"]{
    background: #EEF2FF !important; /* azul muy claro */
    color: #0F172A !important;
    border-radius: 999px !important;
    border: 1px solid #DBEAFE !important;
    }

    /* File uploader */
    .stFileUploader div[data-testid="stFileUploaderDropzone"]{
    background: #FFFFFF !important;
    border: 1px dashed rgba(0,0,0,0.2) !important;
    border-radius: var(--radius) !important;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover{
    border-color: var(--brand) !important;
    }

    /* Checkbox / Radio */
    .stCheckbox [data-testid="stTickbox"] > div,
    .stRadio [role="radiogroup"] > label > div:first-child{
    border: 1px solid rgba(0,0,0,0.25) !important;
    }

    /* Slider */
    [data-testid="stSlider"] [role="slider"]{ background: var(--brand) !important; }

    /* ========= Tabs ========= */
    [data-baseweb="tab-list"]{ gap: .25rem; }
    [data-baseweb="tab-list"] button{
    background: var(--bg-2) !important;
    color: var(--text) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]{
    background: var(--brand) !important;
    color: #FFFFFF !important;
    }

    /* ========= Tables / DataFrames ========= */
    thead tr th{
    background: var(--bg-2) !important;
    color: #1f2937 !important;
    }
    tbody tr td{ color: var(--text) !important; }
    tbody tr:nth-child(even) td{ background: #FAFAFA !important; }
    [data-testid="stDataFrame"] div[role="grid"]{
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: var(--radius);
    }

    /* ========= Metrics ========= */
    [data-testid="stMetricValue"]{ color: var(--text) !important; font-weight: 600; }
    [data-testid="stMetricLabel"]{ color: var(--muted) !important; }

    /* ========= Alerts / Notifications ========= */
    .stAlert > div{ border-radius: var(--radius) !important; }
    .stAlert[data-baseweb="notification"][kind="success"] > div{ border: 1px solid #b7ebc6; }
    .stAlert[data-baseweb="notification"][kind="error"] > div{ border: 1px solid #ffc2c2; }
    .stAlert[data-baseweb="notification"][kind="info"] > div{ border: 1px solid #cfe2ff; }

    /* Progress & spinner & toast */
    [data-testid="stProgress"] > div > div{ background: var(--brand) !important; }
    [data-testid="stSpinner"]{ color: var(--brand) !important; }
    [data-testid="stToast"]{ background: #111827 !important; color: #FFFFFF !important; }

    /* ========= Code blocks ========= */
    code, pre, kbd, samp{
    background: #F3F4F6 !important;
    color: #111827 !important;
    border-radius: 6px;
    }

    /* ========= Utility (opt-in) ========= */
    .mo-card{
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1rem;
    }
    .mo-muted{ color: var(--muted); }

    /* ========= Responsive tweaks ========= */
    @media (max-width: 900px){
    [data-baseweb="tab-list"]{ gap: .15rem; }
    .block-container{ padding-left: .5rem; padding-right: .5rem; }
    }
    </style>
            """,
            unsafe_allow_html=True,
        )


def get_refresh_interval() -> int:
    """Return the auto-refresh interval for the dashboard in seconds.

    Controlled via environment variable ``REFRESH_INTERVAL``.
    Defaults to 10 seconds when missing or invalid.
    """
    try:
        return int(os.environ.get("REFRESH_INTERVAL", "10"))
    except ValueError:
        return 10


# --- Brand-consistent Matplotlib theme (optional, no breaking changes) ---
def apply_matplotlib_theme() -> None:
    """Set Matplotlib rcParams to align charts with the dashboard theme."""
    try:
        import matplotlib as mpl

        mpl.rcParams.update({
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.prop_cycle": mpl.cycler(color=[
                "#002147",  # brand
                "#1D4ED8",  # secondary blue
                "#10B981",  # success
                "#F59E0B",  # amber
                "#EF4444",  # danger
            ]),
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E5E7EB",
        })
    except Exception:
        # Do not fail rendering if Matplotlib is absent or misconfigured
        pass
