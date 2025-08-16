"""Entry point for the Streamlit dashboard and embedded servers.

When executed with ``python -m email_marketing.app`` or via ``streamlit run``,
this module starts the Streamlit interface and, optionally, a background
tracking server. The dashboard exposes views for composing campaigns and
analyzing engagement events.

Design emphasizes configurability: colours, fonts and spacing are controlled
by ``dashboard.style``. A background thread can launch the tracking API in the
same process when ``RUN_TRACKING_WITH_STREAMLIT`` is set.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from fastapi import FastAPI
import uvicorn


# Ensure project root is on PYTHONPATH so imports like
# `email_marketing.dashboard` work
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

try:
    from email_marketing.dashboard import (
        email_editor,
        stats_view,
        campaign_metrics_view,
        style,
        mo_assistant,
        recipient_insights,  # NEW: Customer 360º
    )
except ImportError as exc:
    raise ImportError("Failed to import dashboard modules") from exc

try:
    # Optional tracking server (FastAPI). If unavailable, the dashboard still
    # works.
    from email_marketing.tracking import server as tracking_server
except ImportError:
    tracking_server = None  # type: ignore[assignment]


def _start_tracking_server() -> None:
    """Start the tracking server in a background thread if enabled."""
    if tracking_server is None:
        return

    if os.getenv("RUN_TRACKING_WITH_STREAMLIT", "").lower() not in {"1", "true", "yes"}:
        return

    def _run_server() -> None:
        # Determine which attribute to use
        if hasattr(tracking_server, "app"):
            app_instance: FastAPI = tracking_server.app
        elif hasattr(tracking_server, "create_app"):
            app_instance = tracking_server.create_app()
        else:
            raise RuntimeError("Unsupported tracking_server implementation")

        if not isinstance(app_instance, FastAPI):
            raise RuntimeError(f"Expected FastAPI instance, got {type(app_instance)}")

        uvicorn.run(app_instance, host="0.0.0.0", port=8000, log_level="info")

    thread = threading.Thread(target=_run_server, name="tracking-server", daemon=True)
    thread.start()


def _render_sidebar_avatar_for_page(page: str, size_px: int = 120) -> None:
    """Render the MO avatar in the sidebar for a given page.

    This does NOT render on 'MO Assistant' (landing), to avoid duplication.
    It selects a page-specific SVG when available and falls back safely.
    """
    if page == "MO Assistant":
        return  # do not show the avatar on the landing page

    with st.sidebar:
        shown = False
        try:
            static_dir = Path(__file__).resolve().parent / "dashboard" / "static"

            # Select preferred variant per page
            preferred_map = {
                "Email Editor": "mo_bot_default.svg",
                "Statistics": "mo_bot_analyzing.svg",
                "Campaign Metrics": "mo_bot_metrics.svg",
                "Customer 360º": "mo_bot_analyzing.svg",
            }
            preferred = preferred_map.get(page, "mo_bot_default.svg")

            # Special case: if Email Editor is sending, prefer "writing" animation
            sending = bool(
                st.session_state.get("email_sending")
                or st.session_state.get("campaign_sending")
                or st.session_state.get("sending")
            )
            if page == "Email Editor" and sending:
                preferred = "mo_bot_writing.svg"

            # Try preferred SVG
            svg_path = static_dir / preferred
            if svg_path.exists() and svg_path.suffix.lower() == ".svg":
                svg = svg_path.read_text(encoding="utf-8")
                data_uri = f"data:image/svg+xml;utf8,{quote(svg)}"
                st.markdown(
                    f"""
                    <div style="height:{size_px + 16}px;display:flex;align-items:center;justify-content:center;overflow:visible;margin:2px 0 8px 0;line-height:0;">
                      <img alt="MO" width="{size_px}" height="{size_px}" src="{data_uri}" style="display:block" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                shown = True
            else:
                # Fallback to default SVG
                fallback_svg = static_dir / "mo_bot_default.svg"
                if fallback_svg.exists():
                    svg = fallback_svg.read_text(encoding="utf-8")
                    data_uri = f"data:image/svg+xml;utf8,{quote(svg)}"
                    st.markdown(
                        f"""
                        <div style="height:{size_px + 16}px;display:flex;align-items:center;justify-content:center;overflow:visible;margin:2px 0 8px 0;line-height:0;">
                          <img alt="MO" width="{size_px}" height="{size_px}" src="{data_uri}" style="display:block" />
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    shown = True
                else:
                    # Fallback to PNG if present
                    png = static_dir / "mo.png"
                    if png.exists():
                        st.image(str(png), width=size_px)
                        shown = True
        except Exception:
            shown = False

        if not shown:
            # Last-resort inline badge (always visible)
            fallback_svg_inline = """
            <svg width="128" height="128" viewBox="0 0 128 128"
              xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MO">
              <circle cx="64" cy="64" r="62" fill="#002147"
                stroke="#FFFFFF" stroke-width="4"/>
              <text x="64" y="80" font-size="56" font-family="Arial,
                Helvetica, sans-serif" font-weight="700"
                  text-anchor="middle" fill="#FFFFFF">MO</text>
            </svg>
            """.strip()
            st.markdown(
                f"""
                <div style="height:{size_px + 16}px;display:flex;align-items:center;justify-content:center;overflow:visible;margin:2px 0 8px 0;line-height:0;">
                  <img alt="MO" width="{size_px}" height="{size_px}"
                       src="data:image/svg+xml;utf8,{quote(fallback_svg_inline)}"
                       style="display:block" />
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)


def main() -> None:
    """Render the Streamlit dashboard and optionally launch the tracking API."""
    st.set_page_config(
        page_title="MauBank – Mail Watcher",
        layout="wide",
        page_icon="email_marketing/dashboard/static/maubank/logo.png",
    )
    style.apply_theme()
    if hasattr(style, "apply_matplotlib_theme"):
        style.apply_matplotlib_theme()

    if os.environ.get("RUN_TRACKING_WITH_STREAMLIT", "false").lower() in {"1", "true", "yes"}:
        _start_tracking_server()

    # -------- Deferred navigation handling (before building the selectbox) --------
    # Any sub-page can request a redirect by setting:
    #   st.session_state["nav_redirect"] = "<Target Page Name>"
    # or legacy:
    #   st.session_state["pending_nav"] = "<Target Page Name>"
    redirect_target = st.session_state.pop("nav_redirect", None) or st.session_state.pop("pending_nav", None)

    pages = (
        "MO Assistant",
        "Email Editor",
        "Statistics",
        "Campaign Metrics",
        "Customer 360º",
    )

    # If a redirect is requested, clear previous widget value so `index` applies
    default_index = 0
    if isinstance(redirect_target, str) and redirect_target in pages:
        default_index = pages.index(redirect_target)
        if "nav" in st.session_state:
            del st.session_state["nav"]  # ensure the selectbox respects `index`

    # Sidebar title and page selector
    st.sidebar.title("Mail watcher")
    page = st.sidebar.selectbox("Navigate", pages, index=default_index, key="nav")

    # Sidebar avatar (page-aware). Hidden on "MO Assistant".
    _render_sidebar_avatar_for_page(page, size_px=120)

    # -------- Routing --------
    if page == "MO Assistant":
        mo_assistant.render_mo_assistant()
    elif page == "Email Editor":
        email_editor.render_email_editor()
    elif page == "Statistics":
        stats_view.render_stats_view()
    elif page == "Campaign Metrics":
        campaign_metrics_view.render_campaign_metrics_view()
    elif page == "Customer 360º":
        recipient_insights.render_recipient_insights()
    else:
        st.write("Unsupported page selected.")


if __name__ == "__main__":
    main()
