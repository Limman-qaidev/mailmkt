"""Statistics view component for the Streamlit dashboard.

This module renders summary metrics and visualisations of engagement events.
Events are read from the SQLite database created by the tracking server.
Users can observe opens, clicks, unsubscribes and complaints over time.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from email_marketing.dashboard import style
from streamlit_autorefresh import st_autorefresh



def _load_events() -> pd.DataFrame:
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "email_events.db"
    )
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["msg_id", "event_type", "client_ip", "ts"])
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT msg_id, event_type, client_ip, ts FROM events", conn)
    # Convert timestamp to datetime
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def _compute_metrics(events: pd.DataFrame) -> Dict[str, int]:
    return {
        "opens": int((events["event_type"] == "open").sum()),
        "clicks": int((events["event_type"] == "click").sum()),
        "unsubscribes": int((events["event_type"] == "unsubscribe").sum()),
        "complaints": int((events["event_type"] == "complaint").sum()),
    }


def _plot_event_counts(events: pd.DataFrame) -> None:
    counts = events["event_type"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Event Counts")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def render_stats_view() -> None:
    """Render the statistics page in Streamlit."""
    st.header("Engagement Statistics")
    # Auto-refresh based on the configured interval
    #refresh_interval = style.get_refresh_interval()
    #st.experimental_rerun = st.experimental_rerun  # quiet mypy complaining
    auto_refresh = STYLE.get("auto_refresh", 60)
    if st.button("Refresh now"):
        st.experimental_rerun()


    events = _load_events()
    metrics = _compute_metrics(events)

    cols = st.columns(4)
    cols[0].metric("Opens", metrics["opens"])
    cols[1].metric("Clicks", metrics["clicks"])
    cols[2].metric("Unsubscribes", metrics["unsubscribes"])
    cols[3].metric("Complaints", metrics["complaints"])

    if not events.empty:
        _plot_event_counts(events)
        # Display a table of recent events sorted by timestamp descending
        st.subheader("Recent Events")
        st.dataframe(events.sort_values(by="ts", ascending=False))
    else:
        st.info("No events recorded yet.")
