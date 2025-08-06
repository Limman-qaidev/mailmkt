"""Campaign analytics view for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

# import pandas as pd
import streamlit as st

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
    try:
        metrics_df = compute_campaign_metrics(
            sends, events, signups
            ).reset_index()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to compute metrics: {exc}")
        return

    # Create a mapping of campaign IDs to names for selection
    if (
        "campaign_id" not in metrics_df.columns and
        "campaign" in metrics_df.columns
    ):
        metrics_df = metrics_df.rename(columns={"campaign": "campaign_id"})

    if campaigns.empty:
        st.info("No campaign data available.")
        return
    campaign_options = campaigns["campaign_id"] + " – " + campaigns["name"]
    selected = st.sidebar.selectbox("Select campaign", campaign_options)
    campaign_id = selected.split(" – ")[0]

    info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

    with info_tab:
        campaign_row = campaigns[campaigns["campaign_id"] == campaign_id]
        if campaign_row.empty:
            st.warning("Campaign not found")
        else:
            info = campaign_row[
                ["name", "start_date", "end_date", "budget"]
            ].iloc[0].to_dict()
            st.write(info)

    # Plot using Streamlit's built-in bar chart for quick visualisation.
    with metrics_tab:
        m = metrics_df[metrics_df["campaign_id"] == campaign_id]
        if m.empty:
            st.warning("No metrics available for this campaign")
        else:
            st.dataframe(m)


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
