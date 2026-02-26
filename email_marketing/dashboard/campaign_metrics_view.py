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


def _as_utc(x) -> pd.Timestamp:
    """Parse any datetime-like into a UTC-aware Timestamp (NaT si no válido)."""
    ts = pd.to_datetime(x, errors="coerce", utc=True)
    return ts


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
    """Render campaign analytics with single/compare and period aggregation modes (with deltas vs previous equal-length period and richer visuals)."""
    st.header("Campaign Analytics")

    events_db, sends_db, campaigns_db = _default_db_paths()

    try:
        events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load data: {exc}")
        return

    # --- Harmonize keys / types ---
    for df in (events, sends, signups):
        if "campaign" in df.columns:
            df["campaign"] = df["campaign"].astype(str)

    # msg_id -> campaign from events
    msg2camp = (
        events.loc[events["campaign"].notna(), ["msg_id", "campaign"]]
        .drop_duplicates("msg_id")
        .set_index("msg_id")["campaign"]
    )

    # Preserve variant; rewrite sends.campaign using events map when available
    if "campaign" in sends.columns:
        sends["variant"] = sends["campaign"]
    sends["campaign"] = sends["msg_id"].map(msg2camp).fillna(sends.get("campaign"))

    # Ensure timestamps
    if "send_ts" in sends.columns:
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")
    if "event_ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    elif "ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(signups["signup_ts"], errors="coerce")

    # Compatibility
    generate_distribution_list_by_campaign()

    events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    if "campaign" not in sends.columns and "campaign" in events.columns:
        msg2camp = events.dropna(subset=["msg_id", "campaign"]).drop_duplicates(
            subset=["msg_id"]
        ).set_index("msg_id")["campaign"]
        sends["campaign"] = sends["msg_id"].map(msg2camp)
    elif "campaign" in sends.columns and "campaign" in events.columns:
        msg2camp = events.dropna(subset=["msg_id", "campaign"]).drop_duplicates(
            subset=["msg_id"]
        ).set_index("msg_id")["campaign"]
        existing_campaign = sends["campaign"].copy()
        sends["campaign"] = sends["msg_id"].map(msg2camp).fillna(existing_campaign)

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
        n_opens = float(mdf["N_opens"].sum()) if "N_opens" in cols else 0.0
        n_clicks = float(mdf["N_clicks"].sum()) if "N_clicks" in cols else 0.0

        if "N_signups_attr" in cols:
            n_signups = float(mdf["N_signups_attr"].sum())
        elif "N_signups" in cols:
            n_signups = float(mdf["N_signups"].sum())
        else:
            n_signups = 0.0

        if "N_unsubscribes" in cols:
            n_unsubs = float(mdf["N_unsubscribes"].sum())
        elif "unsubscribe_rate" in cols and "N_sends" in cols:
            n_unsubs = float((mdf["unsubscribe_rate"] * mdf["N_sends"]).sum())
        else:
            if events_df is not None and "event_type" in events_df.columns:
                n_unsubs = float(events_df.query("event_type == 'unsubscribe'")["msg_id"].nunique())
            else:
                n_unsubs = 0.0

        open_rate = (n_opens / n_sends) if n_sends > 0 else 0.0
        ctr = (n_clicks / n_sends) if n_sends > 0 else 0.0
        signup_rate = (n_signups / n_sends) if n_sends > 0 else 0.0
        unsubscribe_rate = (n_unsubs / n_sends) if n_sends > 0 else 0.0

        out.update(
            N_sends=n_sends,
            N_opens=n_opens,
            N_clicks=n_clicks,
            N_signups=n_signups,
            N_unsubscribes=n_unsubs,
            open_rate=open_rate,
            ctr=ctr,
            signup_rate=signup_rate,
            unsubscribe_rate=unsubscribe_rate,
        )
        return out

    def _daily_series(e: pd.DataFrame, s: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=pd.Index([], name="date"))
        if not e.empty and "event_ts" in e.columns:
            e = e.assign(date=e["event_ts"].dt.date)
            opens = e.query("event_type == 'open'").groupby("date")["msg_id"].nunique().rename("opens")
            clicks = e.query("event_type == 'click'").groupby("date")["msg_id"].nunique().rename("clicks")
            df = pd.concat([opens, clicks], axis=1).fillna(0.0)
        if not g.empty and "signup_ts" in g.columns:
            g = g.assign(date=g["signup_ts"].dt.date)
            signups = g.groupby("date")["signup_id"].nunique().rename("signups")
            df = pd.concat([df, signups], axis=1).fillna(0.0)
        df = df.reset_index().sort_values("date")
        return df

    def _pct_change(cur: float, prev: float) -> str:
        try:
            if prev is None or prev <= 0:
                return "—"
            return f"{(cur - prev) / prev:+.1%}"
        except Exception:
            return "—"

    def _pp_change(cur_rate: float, prev_rate: float) -> str:
        """Delta en puntos porcentuales (pp)."""
        try:
            if prev_rate is None:
                return "—"
            return f"{(cur_rate - prev_rate) * 100:+.1f} pp"
        except Exception:
            return "—"

    # -------------- View selector --------------
    view_mode = st.sidebar.radio(
        "View mode",
        ["Single campaign", "Compare campaigns", "Aggregated period", "Compare periods"],
        index=0,
    )

    # ======================= MODE 1: Single =======================
    if view_mode == "Single campaign":
        try:
            metrics_df = compute_campaign_metrics(sends, events, signups)
        except Exception as exc:
            st.error(f"Failed to compute metrics: {exc}")
            return
        if campaigns.empty:
            st.info("No campaign data available.")
            return

        info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

        campaign_options = campaigns["name"]
        selected = st.sidebar.selectbox("Select campaign", campaign_options)
        campaign_id = selected

        with info_tab:
            selected_campaign_info = campaigns[campaigns["name"] == campaign_id]
            if campaign_id not in metrics_df.index or selected_campaign_info.empty:
                st.warning("Campaign not found")
            else:
                info = {
                    "name": campaign_id,
                    "start_date": selected_campaign_info["start_date"].values[0],
                    "end_date": selected_campaign_info["end_date"].values[0],
                    "budget": selected_campaign_info["budget"].values[0],
                }

                start = _as_utc(info["start_date"])
                end = _as_utc(info["end_date"])
                now_utc = pd.Timestamp.now(tz="UTC")

                budget = f"${info.get('budget', 0):,.0f}"

                k1, k2, k3, k4 = st.columns([2, 1, 1, 1])
                k1.metric("📣 Campaign", info["name"])
                k2.metric("📅 Start Date", start.tz_convert("UTC").strftime("%Y-%m-%d") if pd.notna(start) else "—")
                k3.metric("📅 End Date", end.tz_convert("UTC").strftime("%Y-%m-%d") if pd.notna(end) else "—")
                k4.metric("💰 Budget", budget)

                if pd.notna(start) and pd.notna(end) and end > start:
                    total_days = (end - start).days
                    elapsed = max(0, min(total_days, (now_utc - start).days))
                    pct = int(round(100 * elapsed / total_days)) if total_days > 0 else 0
                else:
                    total_days, elapsed, pct = 0, 0, 0

                st.markdown("**Campaign Progress**")
                st.progress(pct)
                st.caption(f"{pct}% complete ({elapsed}/{total_days} days)")

        with metrics_tab:
            m = metrics_df.loc[selected, :]
            if m.empty:
                st.warning("No metrics available for this campaign")
            else:
                campaign_events = events[events["campaign"] == campaign_id]
                open_events = campaign_events[campaign_events["event_type"] == "open"]
                daily_opens = (
                    open_events.assign(date=open_events["event_ts"].dt.date)
                    .groupby("date")["msg_id"]
                    .nunique()
                    .rename("daily_opens")
                )
                click_events = campaign_events[campaign_events["event_type"] == "click"]
                daily_clicks = (
                    click_events.assign(date=click_events["event_ts"].dt.date)
                    .groupby("date")["msg_id"]
                    .nunique()
                    .rename("daily_clicks")
                )
                signup_events = signups[signups["campaign"] == campaign_id]
                daily_signups = (
                    signup_events.assign(date=signup_events["signup_ts"].dt.date)
                    .groupby("date")["signup_id"]
                    .nunique()
                    .rename("daily_signups")
                    if not signup_events.empty
                    else pd.Series(dtype=int, name="daily_signups")
                )
                daily_df = pd.concat([daily_opens, daily_clicks, daily_signups], axis=1).fillna(0).reset_index()

                funnel_values = [m.get("N_sends", 0), m.get("N_opens", 0), m.get("N_clicks", 0), m.get("N_signups_attr", 0)]
                funnel_steps = ["Sent", "Opened", "Clicked", "Signed Up"]

                k1, k2, k3, k4 = st.columns(4)
                k1.plotly_chart(
                    go.Figure(go.Indicator(mode="gauge+number", value=m.get("open_rate", 0.0), gauge={"axis": {"range": [0, 1]}}, title={"text": "Open Rate"})),
                    use_container_width=True,
                )
                k2.plotly_chart(
                    go.Figure(go.Indicator(mode="gauge+number", value=m.get("ctr", 0.0), gauge={"axis": {"range": [0, 1]}}, title={"text": "Click Rate"})),
                    use_container_width=True,
                )
                k3.plotly_chart(
                    go.Figure(go.Indicator(mode="gauge+number", value=m.get("signup_rate", 0.0), gauge={"axis": {"range": [0, 1]}}, title={"text": "Signup Rate"})),
                    use_container_width=True,
                )
                k4.plotly_chart(
                    go.Figure(go.Indicator(mode="gauge+number", value=m.get("unsubscribe_rate", 0.0), gauge={"axis": {"range": [0, 1]}}, title={"text": "Unsubscribe Rate"})),
                    use_container_width=True,
                )

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
                fig_funnel = px.funnel(x=funnel_values, y=funnel_steps, title="Campaign Conversion Funnel")
                st.plotly_chart(fig_funnel, use_container_width=True)

        return  # end Single

    # ======================= MODE 2: Compare campaigns =======================
    if view_mode == "Compare campaigns":
        try:
            metrics_df = compute_campaign_metrics(sends, events, signups)
        except Exception as exc:
            st.error(f"Failed to compute metrics: {exc}")
            return
        if campaigns.empty:
            st.info("No campaign data available.")
            return

        info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])
        campaign_options = campaigns["name"]
        selected_list = st.sidebar.multiselect("Select campaigns to compare", campaign_options)
        campaign_ids = [s for s in selected_list]

        if not campaign_ids:
            st.warning("Select at least two campaigns to compare")
            return
        metric = st.sidebar.selectbox(
            "Metric to compare",
            ["open_rate", "ctr", "signup_rate", "unsubscribe_rate"],
        )

        with info_tab:
            sel = campaigns[campaigns["name"].isin(campaign_ids)]
            if sel.empty:
                st.warning("No campaign data found")
            else:
                for _, row in sel.iterrows():
                    name = row["name"]
                    start = _as_utc(row.get("start_date"))
                    end = _as_utc(row.get("end_date"))
                    budget_str = f"${(row.get('budget') or 0):,.0f}"

                    st.markdown(f"### 📣 {name}")
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.metric("Campaign", name)
                    c2.metric("Start Date", start.tz_convert("UTC").strftime("%Y-%m-%d") if pd.notna(start) else "—")
                    c3.metric("End Date",   end.tz_convert("UTC").strftime("%Y-%m-%d") if pd.notna(end) else "—")
                    c4.metric("💰 Budget",  budget_str)

                    now_utc = pd.Timestamp.now(tz="UTC")
                    if pd.notna(start) and pd.notna(end) and end > start:
                        total_days = (end - start).days
                        elapsed = max(0, min(total_days, (now_utc - start).days))
                        pct = int(round(100 * elapsed / total_days)) if total_days > 0 else 0
                    else:
                        total_days, elapsed, pct = 0, 0, 0

                    st.progress(pct)
                    st.caption(f"{pct}% complete ({elapsed}/{total_days} days)")

        with metrics_tab:
            cmp_df = metrics_df.reset_index()
            if "campaign_id" not in cmp_df.columns:
                cmp_df = cmp_df.rename(columns={cmp_df.columns[0]: "campaign_id"})
            cmp_df = cmp_df[cmp_df["campaign_id"].isin(campaign_ids)]
            fig_cmp = px.bar(
                cmp_df,
                x="campaign_id",
                y=metric,
                labels={"campaign_id": "Campaign", metric: metric.replace("_", " ").title()},
                title=f"Comparison of {metric.replace('_', ' ').title()} Across Campaigns",
                barmode="group",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Normalized daily engagement since launch (include signups)
            df_events = events.copy()
            if "event_ts" not in df_events.columns and "ts" in df_events.columns:
                df_events["event_ts"] = pd.to_datetime(df_events["ts"], errors="coerce")
            else:
                df_events["event_ts"] = pd.to_datetime(df_events["event_ts"], errors="coerce")

            df_sends = sends.copy()
            if "send_ts" in df_sends.columns:
                df_sends["send_ts"] = pd.to_datetime(df_sends["send_ts"], errors="coerce")

            df = df_events.merge(df_sends, on="msg_id", how="left", suffixes=("", "_send"))

            df_list = []
            for cid in campaign_ids:
                ev = df[df["campaign"] == cid].copy()

                # Append signups as pseudo-events for this campaign
                su = pd.DataFrame()
                if not signups.empty and "campaign" in signups.columns:
                    su = signups[signups["campaign"] == cid].copy()
                    if not su.empty:
                        su["event_ts"] = pd.to_datetime(su.get("signup_ts"), errors="coerce")
                        su = su.loc[:, ["event_ts"]]
                        su["event_type"] = "signup"
                        ev = pd.concat([ev, su], ignore_index=True, sort=False)

                # Baseline: first send_ts for the campaign; fallback to earliest event_ts
                base_ts = pd.NaT
                if "campaign" in df_sends.columns and "send_ts" in df_sends.columns:
                    s_all = df_sends.loc[df_sends["campaign"] == cid, "send_ts"]
                    if not s_all.empty:
                        base_ts = s_all.min()
                if pd.isna(base_ts) and "event_ts" in ev.columns:
                    base_ts = ev["event_ts"].min()

                if pd.isna(base_ts):
                    continue  # nothing to plot for this campaign

                ev = ev[pd.notna(ev["event_ts"])].copy()
                ev["days_since"] = (ev["event_ts"] - base_ts).dt.days

                daily = (
                    ev.groupby(["days_since", "event_type"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="days_since", columns="event_type", values="count")
                    .fillna(0)
                    .reset_index()
                )
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
            if event_col not in ts_df.columns:
                st.info(
                    f"No time-series data available for '{event_col}' yet."
                )
            else:
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
                    title=f"{metric.replace(
                        '_', ' ').title()} Over Time by Campaign",
                )
                st.plotly_chart(fig_ts_cmp, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
