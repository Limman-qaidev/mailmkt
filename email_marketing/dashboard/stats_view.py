"""Statistics view component for the Streamlit dashboard.

This module renders summary metrics and visualisations of engagement events.
Events are read from the SQLite database created by the tracking server.
Users can observe opens, clicks, unsubscribes and complaints over time.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from email_marketing.dashboard import style


def _load_events() -> pd.DataFrame:
    """
    Load engagement events and merge in the recipient email for each msg_id.
    """
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        "email_events.db"
    )
    # If no events DB exists yet, return an empty DataFrame with the right
    # columns
    if not os.path.exists(db_path):
        return pd.DataFrame(
            columns=["msg_id", "event_type", "client_ip", "ts", "recipient"]
        )

    # 1. Load raw events
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT msg_id, event_type, client_ip, ts FROM events", conn
        )

    # 2. Convert timestamp to datetime
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # 3. Merge in the recipient email from the email_map database
    map_db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        "email_map.db"
    )
    try:
        with sqlite3.connect(map_db_path) as conn2:
            df_map = pd.read_sql_query(
                "SELECT msg_id, recipient FROM email_map", conn2
                )
        # Perform a left join so that every event keeps its row,
        # even if no mapping is found for some msg_id
        df = df.merge(df_map, on="msg_id", how="left")
    except Exception:
        # If the mapping DB is missing or malformed, add an empty recipient
        # column
        df["recipient"] = None

    return df


def _compute_metrics(events: pd.DataFrame,
                     map_df: pd.DataFrame
                     ) -> Dict[str, int]:
    counts = {
        "opens":   int((events["event_type"] == "open").sum()),
        "clicks":  int((events["event_type"] == "click").sum()),
        "unsubscribes": int((events["event_type"] == "unsubscribe").sum()),
        "complaints":   int((events["event_type"] == "complaint").sum()),
    }

    # Merge events with send timestamps
    df = map_df[["msg_id", "send_ts"]].merge(
        events[["msg_id"]].drop_duplicates(),
        on="msg_id", how="left", indicator=True
    )
    now = datetime.utcnow()
    threshold = now - timedelta(days=7)
    # Count msg_id with no events (_merge == "left_only") AND send_ts older
    #  than threshold
    stale = df.query("_merge == 'left_only' and send_ts < @threshold")
    counts["deleted_or_spam"] = len(stale)

    return counts


def _plot_event_counts(events: pd.DataFrame) -> None:
    #counts = events["event_type"].value_counts()
    #fig, ax = plt.subplots()
    #counts.plot(kind="bar", ax=ax)
    #ax.set_title("Event Counts")
    #ax.set_xlabel("Event Type")
    #ax.set_ylabel("Count")
    #st.pyplot(fig)
    """Plot event counts as a responsive Plotly bar chart."""
    counts = events["event_type"].value_counts()
    fig = go.Figure(
        data=[go.Bar(x=counts.index, y=counts.values)],
        layout=go.Layout(
                title="Event Counts",
                xaxis=dict(title="Event Type"),
                yaxis=dict(title="Count"),
                autosize=True,
                margin=dict(l=40, r=40, t=60, b=40),
                ),
        )
    # En Streamlit, use_container_width=True hace que el gráfico ocupe
    # todo el ancho disponible de forma adaptativa.
    st.plotly_chart(fig, use_container_width=True)


def render_stats_view() -> None:
    """Render the statistics page in Streamlit.

    This function displays aggregate metrics for all recorded events and
    automatically refreshes the view at a configurable interval. The
    refresh interval is obtained via :func:`style.get_refresh_interval`.
    """
    st.header("Engagement Statistics")

    # 1) Determine refresh interval (default 60 seconds if not set).
    refresh_interval = style.get_refresh_interval()

    # 2) Inject JavaScript to reload the page every refresh_interval seconds.
    import streamlit.components.v1 as components
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        height=0,
        width=0,
    )

    # 3) Load events
    events = _load_events()

    # 4) Load mapping (msg_id → send_ts) into map_df
    map_db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "email_map.db"
    )
    with sqlite3.connect(map_db_path) as conn2:
        map_df = pd.read_sql_query(
            "SELECT msg_id, send_ts FROM email_map", conn2,
            parse_dates=["send_ts"]
        )

    # 5) Compute metrics
    metrics = _compute_metrics(events, map_df)

    # 6) Display summary metrics
    cols = st.columns(5)
    cols[0].metric("Opens", metrics["opens"])
    cols[1].metric("Clicks", metrics["clicks"])
    cols[2].metric("Unsubscribes", metrics["unsubscribes"])
    cols[3].metric("Complaints", metrics["complaints"])
    cols[4].metric("Deleted/Spam", metrics["deleted_or_spam"])

    # 5) Plot and table of recent events
    if not events.empty:
        _plot_event_counts(events)
        st.subheader("Recent Events")
        st.dataframe(
            events.sort_values(by="ts", ascending=False),
            width=3000,
            height=400
        )
    else:
        st.info("No events recorded yet.")
