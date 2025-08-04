"""Entry point for the Streamlit dashboard and embedded servers.

When executed with ``python -m email_marketing.app`` or via ``streamlit run``,
this module starts the Streamlit interface and, optionally, a background
tracking server.  The dashboard exposes two main views: an email editor for
crafting campaigns and a statistics view for analysing engagement events.

The design emphasises configurability: colours, fonts and spacing are
controlled by ``dashboard.style``, and refresh intervals are driven by
environment variables.  A background thread can launch the tracking API in
the same process when ``RUN_TRACKING_WITH_STREAMLIT`` is set.
"""

from __future__ import annotations
import os
import sys
import threading
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
    )
except ImportError:
    raise ImportError("Failed to import dashboard modules")

try:
    # Attempt to import the tracking server.  This import is optional;
    # if unavailable (e.g. when running in a separate container) the
    # dashboard will still function.
    from email_marketing.tracking import server as tracking_server
except ImportError:
    tracking_server = None  # type: ignore[assignment]


def _start_tracking_server() -> None:
    """Start the tracking server in a background thread.

    When the tracking module is available and the environment variable
    ``RUN_TRACKING_WITH_STREAMLIT`` evaluates to true, this helper starts
    the Flask (or FastAPI) app on port 8000.  Running a web server in a
    separate thread within Streamlit is acceptable for development but
    discouraged in production.  In production use a process manager such as
    ``gunicorn`` or run the service in its own container.
    """

    if tracking_server is None:
        return

    if (
        os.getenv("RUN_TRACKING_WITH_STREAMLIT", "").lower() not in {
            "1", "true", "yes"
        }
    ):
        return

    def _run_server() -> None:
        # Determine which attribute to use
        if hasattr(tracking_server, "app"):
            app_instance: FastAPI = tracking_server.app
        elif hasattr(tracking_server, "create_app"):
            app_instance = tracking_server.create_app()
        else:
            raise RuntimeError("Unsupported tracking_server implementation")

        # Ensure it's actually FastAPI
        if not isinstance(app_instance, FastAPI):
            raise RuntimeError(
                f"Expected FastAPI instance, got {type(app_instance)}"
                )

        # Start Uvicorn
        uvicorn.run(
            app_instance,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )

    thread = threading.Thread(
        target=_run_server,
        name="tracking-server",
        daemon=True,
    )
    thread.start()


def main() -> None:
    """
    Render the Streamlit dashboard and optionally launch the tracking API.
    """
    # Apply global Streamlit options and custom CSS defined in the style
    # module.
    st.set_page_config(page_title="Mail watcher Dashboard", layout="wide")
    style.apply_theme()

    # Optionally launch the tracking server in the same process.
    if os.environ.get("RUN_TRACKING_WITH_STREAMLIT", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        _start_tracking_server()

    # Sidebar navigation
    st.sidebar.title("Mail watcher")
    page = st.sidebar.selectbox(
        "Navigate", ("Email Editor", "Statistics", "Campaign Metrics")
    )

    if page == "Email Editor":
        email_editor.render_email_editor()
    elif page == "Statistics":
        stats_view.render_stats_view()
    elif page == "Campaign Metrics":
        campaign_metrics_view.render_campaign_metrics_view()
    else:
        st.write("Unsupported page selected.")


if __name__ == "__main__":
    main()
