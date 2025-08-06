"""Campaign analytics view for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analytics.db import load_all_data
from analytics.metrics import compute_campaign_metrics


def _default_db_paths() -> tuple[str, str, str]:
    """Return default locations for the analytics SQLite databases."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    return (
        str(data_dir / "email_events.db"),
        str(data_dir / "email_map.db"),
        str(data_dir / "campaigns.db"),
    )


def render_campaign_metrics_view() -> None:
    """Render campaign information and computed metrics."""
    st.header("Campaign Analytics")

    events_db, sends_db, campaigns_db = _default_db_paths()

    try:
        events, sends, campaigns, signups = load_all_data(
            events_db, sends_db, campaigns_db
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load data: {exc}")
        return
    events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    try:
        metrics_df = compute_campaign_metrics(sends, events, signups)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to compute metrics: {exc}")
        return

    if campaigns.empty:
        st.info("No campaign data available.")
        return
    campaign_options = campaigns["name"]
    selected = st.sidebar.selectbox("Select campaign", campaign_options)
    campaign_id = selected

    info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

    with info_tab:
        campaign_row = events[events["campaign"] == campaign_id]
        if campaign_id not in metrics_df.index:
            st.warning("Campaign not found")
        else:
            info = {
                'name': campaign_id,
                'start_date': campaign_row['event_ts'].min(),
                'end_date': campaign_row['event_ts'].max(),
            }
            st.write(info)

    # Plot using Streamlit's built-in bar chart for quick visualisation.
    with metrics_tab:
        m = metrics_df.loc[selected, :]
        if m.empty:
            st.warning("No metrics available for this campaign")
        else:
            campaign_events = events[events["campaign"] == campaign_id]

            open_events = campaign_events[
                campaign_events["event_type"] == "open"
                ]
            daily_opens = (
                open_events.assign(date=open_events["event_ts"].dt.date)
                .groupby("date")["msg_id"]
                .nunique()
                .rename("daily_opens")
            )

            click_events = campaign_events[
                campaign_events["event_type"] == "click"
                ]
            daily_clicks = (
                click_events.assign(date=click_events["event_ts"].dt.date)
                .groupby("date")["msg_id"]
                .nunique()
                .rename("daily_clicks")
            )

            signup_events = campaign_events[
                campaign_events["event_type"] == "signup"
                ]
            daily_signups = (
                signup_events.assign(date=signup_events["event_ts"].dt.date)
                .groupby("date")["msg_id"]
                .nunique()
                .rename("daily_signups")
                if not signup_events.empty
                else pd.Series(dtype=int, name="daily_signups")
            )

            daily_df = (
                pd.concat([daily_opens, daily_clicks, daily_signups], axis=1)
                .fillna(0)
                .reset_index()
            )

            funnel_values = [
                m["N_sends"],
                m["N_opens"],
                m["N_clicks"],
                m["N_signups"],
            ]
            funnel_steps = ["Sent", "Opened", "Clicked", "Signed Up"]

            k1, k2, k3, k4 = st.columns(4)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=m["open_rate"],
                    delta={"reference": 0, "relative": True},
                    gauge={"axis": {"range": [0, 1]}},
                    title={"text": "Open Rate"},
                )
            )
            k1.plotly_chart(fig, use_container_width=True)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=m["click_rate"],
                    delta={"reference": 0, "relative": True},
                    gauge={"axis": {"range": [0, 1]}},
                    title={"text": "Click Rate"},
                )
            )
            k2.plotly_chart(fig, use_container_width=True)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=m["signup_rate"],
                    delta={"reference": 0, "relative": True},
                    gauge={"axis": {"range": [0, 1]}},
                    title={"text": "Signup Rate"},
                )
            )
            k3.plotly_chart(fig, use_container_width=True)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=m["unsubscribe_rate"],
                    delta={"reference": 0, "relative": True},
                    gauge={"axis": {"range": [0, 1]}},
                    title={"text": "Unsubscribe Rate"},
                )
            )
            k4.plotly_chart(fig, use_container_width=True)

            st.subheader("Daily Engagement Over Time")

            fig_ts = px.line(
                daily_df,
                x="index",
                y=["daily_opens", "daily_clicks", "daily_signups"],
                labels={"value": "Count", "index": "Date"},
                title="Daily Opens, Clicks, and Signups",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader("Conversion Funnel")
            fig_funnel = px.funnel(
                x=funnel_values,
                y=funnel_steps,
                title="Campaign Conversion Funnel",
            )
            st.plotly_chart(fig_funnel, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
