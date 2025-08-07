"""Campaign analytics view for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

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
    view_mode = st.sidebar.radio(
        "View mode", ["Single campaign", "Compare campaigns"]
        )

    info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

    if view_mode == "Single campaign":
        campaign_options = campaigns["name"]
        selected = st.sidebar.selectbox("Select campaign", campaign_options)
        campaign_id = selected

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

                start = info["start_date"]
                end = info["end_date"]
                budget = f"${info.get('budget', 0):,.0f}"

                # 2. KPI cards
                k1, k2, k3, k4 = st.columns([2, 1, 1, 1])
                k1.metric("📣 Campaign", info["name"])
                k2.metric("📅 Start Date", start.strftime("%Y-%m-%d"))
                k3.metric("📅 End Date", end.strftime("%Y-%m-%d"))
                k4.metric("💰 Budget", budget)

                # 3. Timeline
                df_tl = pd.DataFrame([{
                    "Campaign": info["name"],
                    "Start": start,
                    "End": end
                }])
                fig_tl = px.timeline(
                    df_tl, x_start="Start", x_end="End", y="Campaign"
                    )
                fig_tl.update_yaxes(visible=False)
                st.plotly_chart(fig_tl, use_container_width=True)

                # 4. Progress Bar
                total_days = (end - start).days
                elapsed = (datetime.utcnow() - start).days
                pct = max(0, min(
                    100, int(100 * elapsed / total_days)
                    )) if total_days > 0 else 0
                st.markdown("**Campaign Progress**")
                st.progress(pct)
                st.caption(f"{pct}% complete ({elapsed}/{total_days} days)")

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
                    click_events.assign(
                        date=click_events["event_ts"].dt.date
                        )
                    .groupby("date")["msg_id"]
                    .nunique()
                    .rename("daily_clicks")
                )
                signup_events = campaign_events[
                    campaign_events["event_type"] == "signup"
                    ]
                daily_signups = (
                    signup_events.assign(
                        date=signup_events["event_ts"].dt.date
                        )
                    .groupby("date")["msg_id"]
                    .nunique()
                    .rename("daily_signups")
                    if not signup_events.empty
                    else pd.Series(dtype=int, name="daily_signups")
                )
                daily_df = (
                    pd.concat(
                        [daily_opens, daily_clicks, daily_signups],
                        axis=1
                        )
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

    else:  # Compare campaigns mode
        campaign_options = campaigns["name"]
        selected_list = st.sidebar.multiselect(
            "Select campaigns to compare", campaign_options
        )
        campaign_ids = [s.split(" – ")[0] for s in selected_list]

        if not campaign_ids:
            st.warning("Select at least two campaigns to compare")
            return
        metric = st.sidebar.selectbox(
            "Metric to compare",
            ["open_rate", "click_rate", "signup_rate", "unsubscribe_rate"],
        )

        with metrics_tab:
            cmp_df = metrics_df.reset_index()
            if "campaign_id" not in cmp_df.columns:
                cmp_df = cmp_df.rename(
                    columns={cmp_df.columns[0]: "campaign_id"}
                    )
            cmp_df = cmp_df[cmp_df["campaign_id"].isin(campaign_ids)]
            fig_cmp = px.bar(
                cmp_df,
                x="campaign_id",
                y=metric,
                labels={"campaign_id": "Campaign", metric: metric.replace(
                    "_", " "
                    ).title()},
                title=f"Comparison of {metric.replace(
                    '_', ' '
                    ).title()} Across Campaigns",
                barmode="group",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            df_list = []
            df = events.merge(
                sends, on="msg_id", how="left", suffixes=("", "_send")
                )
            for cid in campaign_ids:
                send_ts = df[df["campaign"] == cid]["event_ts"].min()
                ev = df[df["campaign"] == cid]
                ev = ev.assign(days_since=((ev["event_ts"] - send_ts).dt.days))
                daily = (
                    ev.groupby(["days_since", "event_type"])
                    .size()
                    .reset_index(name="count")
                    .pivot(
                        index="days_since",
                        columns="event_type",
                        values="count"
                        )
                    .fillna(0)
                ).reset_index()
                daily["campaign_id"] = cid
                df_list.append(daily)
            ts_df = pd.concat(df_list, ignore_index=True)

            st.subheader("Normalized Daily Engagement")
            fig_ts_cmp = px.line(
                ts_df,
                x="days_since",
                y=metric.split("_")[0],
                color="campaign_id",
                labels={
                    "days_since": "Days Since Launch",
                    metric.split("_")[0]: "Count",
                },
                title=f"{metric.replace(
                    '_', ' ').title()} Over Time by Campaign",
            )
            st.plotly_chart(fig_ts_cmp, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
