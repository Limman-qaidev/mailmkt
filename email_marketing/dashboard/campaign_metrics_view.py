"""Campaign analytics view for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

from analytics.db import load_all_data
from analytics.metrics import compute_campaign_metrics


def _now_ts() -> str:
    # Espacio entre fecha y hora; optional microseconds
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def _default_db_paths() -> tuple[str, str, str]:
    """Return default locations for the analytics SQLite databases."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    return (
        str(data_dir / "email_events.db"),
        str(data_dir / "email_map.db"),
        str(data_dir / "campaigns.db"),
    )


def generate_distribution_list_by_campaign() -> None:
    """
    Create a CSV with recipients of a campaign who have not unsubscribed,
    complained, or marked the email as spam/deleted.
    """
    events_db, sends_db, campaigns_db = _default_db_paths()
    try:
        events, _, campaigns, _ = load_all_data(
            events_db, sends_db, campaigns_db
        )
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load data: {exc}")
        return

    campaign_options = campaigns["name"].values
    mail_discard = events[events["event_type"].isin([
        "unsubscribe",
        "complaint",
        "deleted_or_spam",
    ])].loc[:, ['campaign', 'email']]
    ev = events[~events["event_type"].isin(
        [
            "unsubscribe",
            "complaint",
            "deleted_or_spam",
            "send"
        ]
    )]
    patern = Path(__file__).resolve().parents[2] / "data"
    for campaign in campaign_options:
        ev_camp = ev[ev["campaign"] == campaign]
        mail_discard_camp = mail_discard[mail_discard["campaign"] == campaign]
        dist = ev_camp[~ev_camp["email"].isin(mail_discard_camp["email"])]
        dist = dist.drop_duplicates(subset=["email"])['email']

        # 4) Guardo resultado
        if not dist.empty:
            path = os.path.join(patern, f"distribution_list_{campaign}.csv")
            if os.path.exists(path):
                distro = pd.read_csv(path)
                dist = pd.concat([distro, dist]).drop_duplicates()
            dist.to_csv(path, index=False)


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

    generate_distribution_list_by_campaign()

    events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(
            signups["signup_ts"], errors="coerce"
        )
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
            selected_campaign_info = campaigns[
                campaigns["name"] == campaign_id
                ]
            if campaign_id not in metrics_df.index:
                st.warning("Campaign not found")
            else:
                info = {
                    'name': campaign_id,
                    'start_date': selected_campaign_info[
                        'start_date'
                        ].values[0],
                    'end_date': selected_campaign_info['end_date'].values[0],
                    'budget': selected_campaign_info['budget'].values[0],
                }

                start = pd.to_datetime(info["start_date"])
                end = pd.to_datetime(info["end_date"])
                budget = f"${info.get('budget', 0):,.0f}"
                # 2. KPI cards
                k1, k2, k3, k4 = st.columns([2, 1, 1, 1])
                k1.metric("📣 Campaign", info["name"])
                k2.metric("📅 Start Date", start.strftime("%Y-%m-%d"))
                k3.metric("📅 End Date", end.strftime("%Y-%m-%d"))
                k4.metric("💰 Budget", budget)

                """# 3. Timeline
                df_tl = pd.DataFrame([{
                    "Campaign": info["name"],
                    "Start": start,
                    "End": end
                }])
                fig_tl = px.timeline(
                    df_tl, x_start="Start", x_end="End", y="Campaign"
                    )
                fig_tl.update_yaxes(visible=False)
                st.plotly_chart(fig_tl, use_container_width=True)"""

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
        st.write(metrics_df)
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
                signup_events = signups[signups['campaign'] == campaign_id]
                daily_signups = (
                    signup_events.assign(
                        date=signup_events["signup_ts"].dt.date
                        )
                    .groupby("date")["signup_id"]
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
                    m["N_signups_attr"],
                ]
                funnel_steps = ["Sent", "Opened", "Clicked", "Signed Up"]

                k1, k2, k3, k4 = st.columns(4)

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=m["open_rate"],
                        delta={"reference": 0, "relative": False},
                        gauge={"axis": {"range": [0, 1]}},
                        title={"text": "Open Rate"},
                    )
                )
                k1.plotly_chart(fig, use_container_width=True)

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=m["ctr"],
                        delta={"reference": 0, "relative": False},
                        gauge={"axis": {"range": [0, 1]}},
                        # title={"text": "Click Rate"},
                        title={"text": "CTR"}
                    )
                )
                k2.plotly_chart(fig, use_container_width=True)

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=m["signup_rate"],
                        delta={"reference": 0, "relative": False},
                        gauge={"axis": {"range": [0, 1]}},
                        title={"text": "Signup Rate"},
                    )
                )
                k3.plotly_chart(fig, use_container_width=True)

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=m["unsubscribe_rate"],
                        delta={"reference": 0, "relative": False},
                        gauge={"axis": {"range": [0, 1]}},
                        title={"text": "Unsubscribe Rate"},
                    )
                )
                k4.plotly_chart(fig, use_container_width=True)
                st.subheader("Daily Engagement Over Time")
                daily_df = daily_df.sort_values("date").reindex()
                fig_ts = px.line(
                    daily_df,
                    x="date",
                    y=["daily_opens", "daily_clicks", "daily_signups"],
                    labels={"value": "Count", "date": "Date"},
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
            # ["open_rate", "click_rate", "signup_rate", "unsubscribe_rate"],
            ["open_rate", "ctr", "signup_rate", "unsubscribe_rate"],
        )

        with info_tab:
            # Filtramos todas las campañas seleccionadas
            sel = campaigns[campaigns["name"].isin(campaign_ids)]
            if sel.empty:
                st.warning("No campaign data found")
            else:
                # For each campaign, we print the KPI cards just like in
                #  single mode
                for _, row in sel.iterrows():
                    name = row["name"]
                    start = pd.to_datetime(row["start_date"])
                    end = pd.to_datetime(row["end_date"])
                    budget_str = f"${row['budget']:,.0f}"

                    st.markdown(f"### 📣 {name}")
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.metric("Campaign", name)
                    c2.metric("Start Date", start.strftime("%Y-%m-%d"))
                    c3.metric("End Date",   end.strftime("%Y-%m-%d"))
                    c4.metric("💰 Budget",  budget_str)

                    # Progress bar
                    total_days = (end - start).days
                    elapsed = (datetime.utcnow() - start).days
                    pct = (
                        max(0, min(100, int(100 * elapsed / total_days)))
                        if total_days > 0
                        else 0
                    )
                    st.progress(pct)
                    st.caption(
                        f"{pct}% complete ({elapsed}/{total_days} days)"
                        )

        with metrics_tab:
            cmp_df = metrics_df.reset_index()
            if "campaign_id" not in cmp_df.columns:
                cmp_df = cmp_df.rename(
                    columns={cmp_df.columns[0]: "campaign_id"}
                    )
            cmp_df = cmp_df[cmp_df["campaign_id"].isin(campaign_ids)]
            metric_label = metric.replace("_", " ").title()
            fig_cmp = px.bar(
                cmp_df,
                x="campaign_id",
                y=metric,
                labels={"campaign_id": "Campaign", metric: metric_label},
                title=f"Comparison of {metric_label} Across Campaigns",
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
                signup_events = signups[signups["campaign"] == cid]
                signup_ev = pd.DataFrame(
                        {
                         'campaign': cid,
                         'msg_id': signup_events['signup_id'],
                         'event_type': 'signup',
                         'event_ts': signup_events['signup_ts'],
                         'email': signup_events['email'],
                         'email_send': signup_events['email'],
                         'campaign_send': signup_events['campaign']
                        }
                )
                ev = pd.concat([ev, signup_ev], ignore_index=True)

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
            ts_df = pd.concat(df_list, ignore_index=True).fillna(0)

            st.subheader("Normalized Daily Engagement")
            event_map = {
                "open_rate": "open",
                "ctr": "click",
                "signup_rate": "signup",
                "unsubscribe_rate": "unsubscribe",
            }
            event_col = event_map.get(metric, metric.split("_")[0])
            fig_ts_cmp = px.line(
                ts_df,
                x="days_since",
                # y=metric.split("_")[0],
                y=event_col,
                color="campaign_id",
                labels={
                    "days_since": "Days Since Launch",
                    # metric.split("_")[0]: "Count",
                    event_col: "Count",
                },
                title=f"{metric_label} Over Time by Campaign",
            )
            st.plotly_chart(fig_ts_cmp, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
