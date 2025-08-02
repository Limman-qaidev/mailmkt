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
    """Inject custom CSS into the Streamlit page.

    The CSS variables declared here reference the fields of :class:`Theme`.
    They can be used in the HTML returned by Streamlit components.  This
    function should be called once per session, ideally at the beginning of
    the main app.
    """
    css = f"""
    <style>
    :root {{
        --primary-font: {THEME.primary_font};
        --heading-font: {THEME.heading_font};
        --base-font-size: {THEME.base_font_size}px;
        --spacing-unit: {THEME.spacing_unit}px;
        --colour-background: {THEME.colours[0]};
        --colour-surface: {THEME.colours[1]};
        --colour-text: {THEME.colours[2]};
    }}
    html, body, [class*="css"] {{
        font-family: var(--primary-font);
        font-size: var(--base-font-size);
        color: var(--colour-text);
        background-color: var(--colour-background);
    }}
    h1, h2, h3, h4, h5 {{
        font-family: var(--heading-font);
        margin-top: calc(var(--spacing-unit) * 2);
        margin-bottom: calc(var(--spacing-unit) * 1);
    }}
    .metric {{
        padding: calc(var(--spacing-unit) * 2);
        margin-bottom: calc(var(--spacing-unit) * 2);
        background-color: var(--colour-surface);
        border-radius: calc(var(--spacing-unit) / 2);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
