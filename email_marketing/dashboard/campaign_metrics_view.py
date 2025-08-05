"""Campaign analytics view for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from email_marketing.analytics import db, metrics


def _default_db_paths() -> tuple[str, str, str]:
    """Return default locations for the analytics SQLite databases."""
    return (
        str(db.EVENTS_DB),
        str(db.MAP_DB),
        str(db.CAMPAIGNS_DB),
    )


def render_campaign_metrics_view() -> None:
    """Render campaign information and computed metrics."""
    st.header("Campaign Analytics")

    default_events, default_sends, default_campaigns = _default_db_paths()

    st.sidebar.subheader("Database paths")
    events_path = st.sidebar.text_input("Events DB", default_events)
    sends_path = st.sidebar.text_input("Sends DB", default_sends)
    campaigns_path = st.sidebar.text_input("Campaigns DB", default_campaigns)

    try:
        events, sends, campaigns, signups = db.load_all_data(
            events_path, sends_path, campaigns_path
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load data: {exc}")
        return

    if campaigns.empty:
        st.info("No campaign data available.")
        return

    metrics_df: pd.DataFrame = metrics.compute_campaign_metrics(
        sends=sends, events=events, signups=signups
    )

    campaign_options = {
        row["campaign_id"]: f"{row['campaign_id']} – {row['name']}"
        for _, row in campaigns.iterrows()
    }

    selection = st.selectbox(
        "Select Campaign",
        list(campaign_options.values()),
    )
    selected_id = next(
        cid for cid, label in campaign_options.items() if label == selection
    )

    info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

    campaign_row = campaigns[campaigns["campaign_id"] == selected_id].iloc[0]

    with info_tab:
        info = {
            "name": campaign_row.get("name"),
            "start_date": campaign_row.get("start_date"),
            "end_date": campaign_row.get("end_date"),
            "budget": campaign_row.get("budget"),
        }
        st.write(info)

    # Plot using Streamlit's built-in bar chart for quick visualisation.
    with metrics_tab:
        if selected_id in metrics_df.index:
            st.dataframe(metrics_df.loc[[selected_id]].reset_index())
        else:
            st.info("No metrics for selected campaign.")


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
