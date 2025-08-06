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

    """st.sidebar.subheader("Database paths")
    events_path = st.sidebar.text_input("Events DB", default_events)
    sends_path = st.sidebar.text_input("Sends DB", default_sends)
    campaigns_path = st.sidebar.text_input("Campaigns DB", default_campaigns)
    """

    try:
        events, sends, campaigns, signups = db.load_all_data(
            default_events,
            default_sends,
            default_campaigns
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load data: {exc}")
        return
    events = events.merge(
        sends, on="msg_id", how='left', suffixes=('', '_send')
        )
    events = events.merge(
        campaigns, left_on="campaign", right_on="name",
        how='left', suffixes=('', '_campaign')
    )
    events['event_ts'] = pd.to_datetime(events['event_ts'], errors='raise')

    if campaigns.empty:
        st.info("No campaign data available.")
        return

    metrics_df: pd.DataFrame = metrics.compute_campaign_metrics(
        sends=sends, events=events, signups=signups
    )

    # Create a mapping of campaign IDs to names for selection
    campaign_options = {
        row["campaign_id"]: f"{row['name']}"
        for _, row in campaigns.iterrows()
    }

    # Create a selection box for campaigns
    selection = st.selectbox(
        "Select Campaign",
        list(campaign_options.values()),
    )

    # Get the selected campaign ID
    selected_id = next(
        cid for cid, label in campaign_options.items() if label == selection
    )

    info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

    # Select campaign data
    campaign_data = events[events["campaign_id"] == selected_id]
    start_date = campaign_data["event_ts"].min()
    end_date = campaign_data["event_ts"].max()

    with info_tab:
        info = {
            "name": selection,
            "start_date": start_date,
            "end_date": end_date,
        }
        st.write(info)

    # Plot using Streamlit's built-in bar chart for quick visualisation.
    with metrics_tab:
        if selection in metrics_df.index:
            st.dataframe(metrics_df.loc[[selection]].reset_index())
        else:
            st.info("No metrics for selected campaign.")


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
