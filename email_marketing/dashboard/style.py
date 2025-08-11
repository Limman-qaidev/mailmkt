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
    """Inject MauBank-inspired light theme into Streamlit app."""
    st.markdown(
        """
<style>
/* ========= CSS variables (MauBank-inspired) ========= */
:root {
  /* Brand */
  --brand: #002147;        /* deep blue (inferred) */
  --brand-600: #001437;    /* darker hover */
  --accent: #10B981;       /* success/OK; (optional: #FFD100 for yellow accents) */
  --danger: #EF4444;

  /* Surfaces */
  --bg: #FFFFFF;
  --bg-2: #F5F5F5;
  --card: #F7F7F7;

  /* Typography */
  --text: #222222;
  --muted: #6C757D;

  /* Shape & elevation */
  --radius: 8px;
  --shadow: 0 4px 12px rgba(0,0,0,0.12);
}

/* ========= Global ========= */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Arial, Helvetica, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ========= Sidebar ========= */
[data-testid="stSidebar"] {
  background: var(--bg-2) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] p {
  color: var(--text) !important;
}

/* ========= Layout / containers ========= */
.block-container { padding-top: 1.25rem; }
.stContainer, .stCard { background: transparent !important; }
hr { border-top: 1px solid rgba(0,0,0,0.08); }

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

/* Secondary */
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid rgba(0,0,0,0.2) !important;
  box-shadow: none !important;
}

/* ========= Inputs ========= */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: #FFFFFF !important;
  color: var(--text) !important;
  border: 1px solid rgba(0,0,0,0.15) !important;
  border-radius: var(--radius) !important;
}
.stDateInput input, .stDatetimeInput input,
.stSelectbox [data-baseweb="select"] > div {
  background: #FFFFFF !important;
  color: var(--text) !important;
  border: 1px solid rgba(0,0,0,0.15) !important;
  border-radius: var(--radius) !important;
}
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
  background: #FFFFFF !important;
  border: 1px dashed rgba(0,0,0,0.2) !important;
  border-radius: var(--radius) !important;
}

/* Checkbox / Radio */
.stCheckbox [data-testid="stTickbox"] > div,
.stRadio [role="radiogroup"] > label > div:first-child {
  border: 1px solid rgba(0,0,0,0.25) !important;
}
.stRadio [role="radio"][aria-checked="true"] {
  outline: 2px solid var(--brand) !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] { background: var(--brand) !important; }

/* ========= Tabs ========= */
[data-baseweb="tab-list"] { gap: .25rem; }
[data-baseweb="tab-list"] button {
  background: var(--bg-2) !important;
  color: var(--text) !important;
  border-radius: var(--radius) var(--radius) 0 0 !important;
}
[data-baseweb="tab-list"] button[aria-selected="true"] {
  background: var(--brand) !important;
  color: #FFFFFF !important;
}

/* ========= Tables / DataFrames ========= */
thead tr th {
  background: var(--bg-2) !important;
  color: #1f2937 !important;
}
tbody tr td { color: var(--text) !important; }
[data-testid="stDataFrame"] { filter: none !important; }
[data-testid="stDataFrame"] div[role="grid"] {
  border: 1px solid rgba(0,0,0,0.08) !ident;
  border-radius: var(--radius);
}

/* ========= Metrics ========= */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  color: var(--text) !important;
}

/* ========= Alerts / Notifications ========= */
.stAlert > div { border-radius: var(--radius) !important; }
.stAlert[data-baseweb="notification"][kind="success"] > div { border: 1px solid #b7ebc6; }
.stAlert[data-baseweb="notification"][kind="error"] > div { border: 1px solid #ffc2c2; }

/* ========= Code blocks ========= */
code, pre, kbd, samp {
  background: #F3F4F6 !important;
  color: #111827 !important;
  border-radius: 6px;
}

/* ========= Tooltips ========= */
[data-testid="stTooltip"] {
  background: #111827 !important;
  color: #FFFFFF !important;
}

/* ========= Cards util classes (opt-in) ========= */
.mo-card {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1rem;
}
.mo-muted { color: var(--muted); }
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
