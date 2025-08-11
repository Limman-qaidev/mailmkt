"""Dashboard styling configuration.

This module encapsulates the theme variables used throughout the Streamlit
dashboard.  Styles are defined via a dataclass and injected into the page
using ``st.markdown``.  By centralising these values you can easily
customise the look and feel of the application from a single place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import streamlit as st


@dataclass
class Theme:
    """Configuration values for the dashboard's appearance."""

    primary_font: str = "Arial, sans-serif"
    heading_font: str = "Arial, sans-serif"
    base_font_size: int = 14
    spacing_unit: int = 8
    # Define a neutral colour palette; concrete colours are intentionally
    # unspecified to enable runtime theming.  You can override these with
    # environment variables or by modifying this dataclass.
    colours: Tuple[str, str, str] = ("#ffffff", "#f5f5f5", "#333333")


THEME = Theme()


def apply_theme() -> None:
    """Inject a clean, professional look & feel using CSS variables."""
    st.markdown(
        """
        <style>
        :root{
          --brand:#2563EB;            /* azul corporativo */
          --brand-600:#1D4ED8;
          --accent:#10B981;           /* verde para estados OK */
          --danger:#EF4444;           /* rojo para avisos */
          --bg:#0B0F17;               /* fondo oscuro elegante */
          --bg-2:#0F1522;
          --card:#111827;             /* cartas */
          --text:#E5E7EB;             /* texto base */
          --muted:#94A3B8;            /* texto secundario */
          --radius:14px;
          --shadow:0 6px 20px rgba(0,0,0,0.25);
        }
        html,body,[data-testid="stAppViewContainer"]{
          background:linear-gradient(180deg,var(--bg),var(--bg-2));
          color:var(--text);
        }
        .main > div { padding-top: 8px; }
        h1,h2,h3,h4,h5 { color: var(--text)!important; letter-spacing:.2px; }
        .stMarkdown, .stText, .stCaption { color: var(--text)!important; }
        .stDataFrame, .stTable { border-radius: var(--radius); overflow:hidden; }

        /* Cards genéricas */
        .app-card{
          background:var(--card);
          border:1px solid rgba(255,255,255,.06);
          border-radius:var(--radius);
          box-shadow:var(--shadow);
          padding:18px 20px;
        }

        /* Botones */
        .stButton>button{
          background:var(--brand);
          color:white;
          border:none;
          border-radius:12px;
          padding:10px 16px;
          font-weight:600;
          transition: all .15s ease;
        }
        .stButton>button:hover{ background:var(--brand-600); transform: translateY(-1px); }
        .stButton>button[kind="secondary"]{
          background:transparent; color:var(--text);
          border:1px solid rgba(255,255,255,.12);
        }

        /* Inputs */
        .stTextInput>div>div>input, .stTextArea textarea{
          background:#0B1220; color:var(--text); border-radius:12px; border:1px solid rgba(255,255,255,.08);
        }

        /* Tabs */
        .stTabs [role="tablist"] { gap: 8px; }
        .stTabs [role="tab"] {
          background: rgba(255,255,255,.04);
          border-radius:10px;
          padding:8px 12px;
        }
        .stTabs [aria-selected="true"]{
          background: var(--brand);
          color: #fff;
        }

        /* MO specific helpers */
        .mo-hero{display:flex;gap:18px;align-items:center;margin-bottom:12px}
        .mo-kpis{display:flex;gap:16px;flex-wrap:wrap}
        .mo-kpi{flex:1;min-width:180px;background:#0B1220;padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,.06)}
        .mo-card{background:var(--card);padding:18px 20px;border-radius:var(--radius);box-shadow:var(--shadow)}
        .mo-muted{color:var(--muted);font-size:.92rem}
        </style>
        """,
        unsafe_allow_html=True,
    )



def get_refresh_interval() -> int:
    """Return the auto‑refresh interval for the dashboard in seconds.

    The interval can be configured via the ``REFRESH_INTERVAL`` environment
    variable.  If the variable is not set or invalid, a default of 60
    seconds is used.
    """
    try:
        return int(os.environ.get("REFRESH_INTERVAL", "10"))
    except ValueError:
        return 10
