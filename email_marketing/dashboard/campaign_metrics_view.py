"""Campaign metrics view for Streamlit dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from email_marketing.analytics.metrics import compute_campaign_metrics


def render_campaign_metrics_view() -> None:
    """Render per-campaign engagement metrics.

    Uses :func:`compute_campaign_metrics` to aggregate opens, clicks and
    signups for each campaign. The result is shown both as a table and a bar
    chart so trends can be inspected visually.
    """
    st.header("Campaign Metrics")

    metrics: pd.DataFrame = compute_campaign_metrics()

    if metrics.empty:
        st.info("No campaign metrics available.")
        return

    # Display the raw numbers in a table.
    st.dataframe(metrics.reset_index())

    # Plot using Streamlit's built-in bar chart for quick visualisation.
    st.bar_chart(metrics)
