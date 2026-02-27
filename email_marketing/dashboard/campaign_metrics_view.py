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


@st.cache_data(show_spinner=False)
def _cached_load_all_data(events_db: str, sends_db: str, campaigns_db: str):
    return load_all_data(events_db, sends_db, campaigns_db)


@st.cache_data(show_spinner=False)
def _cached_compute_metrics(sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame):
    return compute_campaign_metrics(sends, events, signups)

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
        #events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)
        events, sends, campaigns, signups = _cached_load_all_data(
                events_db, sends_db, campaigns_db
            )
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

    # Aplica el mapeo de msg_id→campaign desde los eventos
    mapped = sends["msg_id"].map(msg2camp)

    # Si existe la columna campaign, combina; en caso contrario, usa el mapeo tal cual
    if "campaign" in sends.columns:
        sends["campaign"] = mapped.combine_first(sends["campaign"])
    else:
        sends["campaign"] = mapped

    #sends["campaign"] = sends["msg_id"].map(msg2camp).fillna(sends.get("campaign"))

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

    # -------------- Helpers --------------
    def _date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
        candidates = []
        if "send_ts" in sends.columns:
            candidates += [sends["send_ts"].min(), sends["send_ts"].max()]
        if "event_ts" in events.columns:
            candidates += [events["event_ts"].min(), events["event_ts"].max()]
        if "signup_ts" in signups.columns:
            candidates += [signups["signup_ts"].min(), signups["signup_ts"].max()]
        candidates = [c for c in candidates if pd.notna(c)]
        if not candidates:
            now = pd.Timestamp.utcnow().normalize()
            return now - pd.Timedelta(days=30), now
        return min(candidates), max(candidates)

    def _filter_period(
        sends_df: pd.DataFrame,
        events_df: pd.DataFrame,
        signups_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        campaign_filter: set[str] | None,
        filter_mode: str = "exclude",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        s = sends_df.copy()
        e = events_df.copy()
        g = signups_df.copy()
        if "send_ts" in s.columns:
            s = s[(s["send_ts"] >= start) & (s["send_ts"] <= end)]
        if "event_ts" in e.columns:
            e = e[(e["event_ts"] >= start) & (e["event_ts"] <= end)]
        if "signup_ts" in g.columns:
            g = g[(g["signup_ts"] >= start) & (g["signup_ts"] <= end)]

        if campaign_filter:
            if filter_mode == "include":
                s = s[s["campaign"].isin(campaign_filter)]
                e = e[e["campaign"].isin(campaign_filter)]
                g = g[g["campaign"].isin(campaign_filter)]
            else:
                s = s[~s["campaign"].isin(campaign_filter)]
                e = e[~e["campaign"].isin(campaign_filter)]
                g = g[~g["campaign"].isin(campaign_filter)]
        return s, e, g

    def _aggregate_metrics(
        mdf: pd.DataFrame,
        events_df: pd.DataFrame | None = None,
        sends_df: pd.DataFrame | None = None,
    ) -> dict:
        out: dict[str, float] = {}
        cols = set(mdf.columns)

        n_sends = float(mdf["N_sends"].sum()) if "N_sends" in cols else (
            float(len(sends_df["msg_id"].unique())) if (sends_df is not None and "msg_id" in sends_df.columns) else 0.0
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
        # -----------------------------
        # Campaign selection (safe)
        # -----------------------------

        # Determine the campaign label column in campaigns.db
        if "campaign" in campaigns.columns:
            campaign_col = "campaign"
        elif "name" in campaigns.columns:
            campaign_col = "name"
        elif "campaign_id" in campaigns.columns:
            # fallback: still allow selection by ID if names are missing
            campaign_col = "campaign_id"
        else:
            st.error("campaigns.db has no recognizable campaign identifier column.")
            return

        campaign_names_all = campaigns[campaign_col].astype(str).tolist()

        # Only allow campaigns that actually have metrics (i.e., exist in metrics_df.index)
        available_campaigns = [c for c in campaign_names_all if c in metrics_df.index]

        if not available_campaigns:
            st.info(
                "No campaigns with metrics are available. "
                "This usually means there are no matching events/sends in the analytics database."
            )
            return

        # If Streamlit kept an old selection that no longer exists, reset it
        state_key = "campaign_metrics_selected"
        if st.session_state.get(state_key) not in available_campaigns:
            st.session_state[state_key] = available_campaigns[0]

        selected = st.selectbox(
            "Select campaign",
            options=available_campaigns,
            key=state_key,
        )
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
            if selected not in metrics_df.index:
                st.warning(
                    f"No metrics available for '{selected}'. "
                    "It exists in campaigns.db but has no matching events in the analytics DB (possibly pruned)."
                )
                return

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
            #metrics_df = compute_campaign_metrics(sends, events, signups)
            metrics_df = _cached_compute_metrics(sends, events, signups)
        except Exception as exc:
            st.error(f"Failed to compute metrics: {exc}")
            return
        if campaigns.empty:
            st.info("No campaign data available.")
            return

        info_tab, metrics_tab = st.tabs(["Campaign Info", "Metrics"])

        # -----------------------------
        # Campaign selection (safe)
        # Only campaigns that have metrics are selectable
        # -----------------------------
        if "name" not in campaigns.columns:
            st.error("campaigns.db must contain a 'name' column.")
            return

        campaign_names_all = campaigns["name"].astype(str).tolist()
        available_campaigns = [c for c in campaign_names_all if c in metrics_df.index]

        if not available_campaigns:
            st.info(
                "No campaigns with metrics are available. "
                "This usually means there are no matching events/sends in the analytics database."
            )
            return

        # Keep only valid selections if Streamlit cached old values
        state_key = "cmp_campaigns_selected"
        prev = st.session_state.get(state_key, [])
        if isinstance(prev, list):
            st.session_state[state_key] = [c for c in prev if c in available_campaigns]

        selected_list = st.sidebar.multiselect(
            "Select campaigns to compare",
            options=available_campaigns,
            default=st.session_state.get(state_key, []) or available_campaigns[: min(2, len(available_campaigns))],
            key=state_key,
        )

        campaign_ids = [str(s) for s in selected_list]

        if len(campaign_ids) < 2:
            st.warning("Select at least two campaigns to compare.")
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
            cmp_df = metrics_df.reset_index(names="campaign_id")
            cmp_df = cmp_df[cmp_df["campaign_id"].astype(str).isin(campaign_ids)]
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

            if df_list:
                ts_df = pd.concat(df_list, ignore_index=True).fillna(0)

                event_map = {
                    "open_rate": "open",
                    "ctr": "click",
                    "signup_rate": "signup",
                    "unsubscribe_rate": "unsubscribe",
                }
                event_col = event_map.get(metric, "open")

                # If the chosen metric column is missing (e.g., no signups), add zeros to avoid crash
                if event_col not in ts_df.columns:
                    ts_df[event_col] = 0

                st.subheader("Normalized Daily Engagement")
                fig_ts_cmp = px.line(
                    ts_df,
                    x="days_since",
                    y=event_col,
                    color="campaign_id",
                    labels={"days_since": "Days Since Launch", event_col: "Count"},
                    title=f"{metric.replace('_', ' ').title()} Over Time by Campaign",
                )
                st.plotly_chart(fig_ts_cmp, use_container_width=True)

        return  # end Compare campaigns

    # ======================= MODE 3/4: Period-based =======================
    # Shared sidebar controls for periods
    global_start, global_end = _date_bounds()

    def _sidebar_period_controls(prefix: str) -> tuple[pd.Timestamp, pd.Timestamp, set[str], str]:
        st.sidebar.markdown(f"**{prefix} period**")
        start = st.sidebar.date_input(
            f"{prefix} start",
            value=global_start.date(),
            min_value=global_start.date(),
            max_value=global_end.date(),
            key=f"{prefix}_start",
        )
        end = st.sidebar.date_input(
            f"{prefix} end",
            value=global_end.date(),
            min_value=global_start.date(),
            max_value=global_end.date(),
            key=f"{prefix}_end",
        )
        avail = sorted(events["campaign"].dropna().unique().tolist())
        mode = st.sidebar.radio(
            f"{prefix} filter mode",
            ["exclude", "include"],
            help="Exclude removes selected campaigns; Include keeps only selected.",
            horizontal=True,
            key=f"{prefix}_filter_mode",
        )
        picked = set(st.sidebar.multiselect(f"{prefix} campaigns", avail, key=f"{prefix}_camps"))
        return pd.to_datetime(start), pd.to_datetime(end), picked, mode

    # ---------- Aggregated period ----------
    if view_mode == "Aggregated period":
        a_start, a_end, a_set, a_mode = _sidebar_period_controls("A")

        sA, eA, gA = _filter_period(sends, events, signups, a_start, a_end, a_set, a_mode)
        try:
            mA = compute_campaign_metrics(sA, eA, gA)
        except Exception as exc:
            st.error(f"Failed to compute metrics in period A: {exc}")
            return
        aggA = _aggregate_metrics(mA, eA, sA)

        # Previous equal-length period for A
        period_days = max(1, (a_end.normalize() - a_start.normalize()).days + 1)
        prev_end = a_start - pd.Timedelta(seconds=1)
        prev_start = prev_end - pd.Timedelta(days=period_days - 1)

        sP, eP, gP = _filter_period(sends, events, signups, prev_start, prev_end, a_set, a_mode)
        try:
            mP = compute_campaign_metrics(sP, eP, gP)
        except Exception:
            mP = pd.DataFrame()
        aggP = _aggregate_metrics(mP, eP, sP) if not mP.empty else {
            "N_sends": 0.0, "N_opens": 0.0, "N_clicks": 0.0, "N_signups": 0.0, "N_unsubscribes": 0.0,
            "open_rate": 0.0, "ctr": 0.0, "signup_rate": 0.0, "unsubscribe_rate": 0.0
        }

        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.subheader(f"Aggregate KPIs ({a_start.date()} → {a_end.date()})")
            st.caption(f"Previous period: {prev_start.date()} → {prev_end.date()}")
        with top_right:
            if hasattr(st, "popover"):
                with st.popover("ℹ️ KPI deltas"):
                    st.write(
                        "- The percentage next to each KPI compares **this period** vs the **previous equal-length period**.\n"
                        "- Counts show relative change: (A − Prev) / Prev.\n"
                        "- Green ↑ means improvement; for **Unsubs** lower is better (inverse coloring)."
                    )
            else:
                st.caption("ℹ️ Deltas compare to previous equal-length period. Counts show relative change; Unsubs use inverse coloring.")

        # Row 1: Counts with delta %
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Sent", f"{int(aggA['N_sends']):,}", delta=_pct_change(aggA["N_sends"], aggP.get("N_sends", 0.0)))
        k2.metric("Opened", f"{int(aggA['N_opens']):,}", delta=_pct_change(aggA["N_opens"], aggP.get("N_opens", 0.0)))
        k3.metric("Clicked", f"{int(aggA['N_clicks']):,}", delta=_pct_change(aggA["N_clicks"], aggP.get("N_clicks", 0.0)))
        k4.metric("Signed up", f"{int(aggA['N_signups']):,}", delta=_pct_change(aggA["N_signups"], aggP.get("N_signups", 0.0)))
        k5.metric("Unsubs", f"{int(aggA['N_unsubscribes']):,}", delta=_pct_change(aggA["N_unsubscribes"], aggP.get("N_unsubscribes", 0.0)), delta_color="inverse")

        # Row 2: Rates with delta in pp
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Open rate", f"{aggA['open_rate']:.1%}", delta=_pp_change(aggA["open_rate"], aggP["open_rate"]))
        r2.metric("Click rate (CTR)", f"{aggA['ctr']:.1%}", delta=_pp_change(aggA["ctr"], aggP["ctr"]))
        r3.metric("Signup rate", f"{aggA['signup_rate']:.1%}", delta=_pp_change(aggA["signup_rate"], aggP["signup_rate"]))
        r4.metric("Unsubscribe rate", f"{aggA['unsubscribe_rate']:.1%}", delta=_pp_change(aggA["unsubscribe_rate"], aggP["unsubscribe_rate"]), delta_color="inverse")

        # Daily series (period A)
        dailyA = _daily_series(eA, sA, gA)
        st.subheader("Daily engagement (period A)")
        if dailyA.empty:
            st.info("No events in this period.")
        else:
            figA = px.line(
                dailyA,
                x="date",
                y=[c for c in ["opens", "clicks", "signups"] if c in dailyA.columns],
                labels={"value": "Count", "date": "Date"},
                title=f"Daily Opens/Clicks/Signups ({a_start.date()} → {a_end.date()})",
            )
            st.plotly_chart(figA, use_container_width=True)

        st.subheader("Top campaigns in period A")
        if not mA.empty:
            cols_show = [c for c in ["N_sends", "N_opens", "N_clicks", "N_signups_attr", "open_rate", "ctr", "signup_rate", "unsubscribe_rate"] if c in mA.columns]
            tableA = mA.sort_values(by=[c for c in ["N_sends", "N_clicks"] if c in mA.columns], ascending=False)[cols_show]
            st.dataframe(tableA, use_container_width=True)

        with st.expander("Export"):
            if not dailyA.empty:
                csv_ts = dailyA.to_csv(index=False).encode("utf-8")
                st.download_button("Download daily time series (CSV)", csv_ts, file_name="periodA_daily.csv", mime="text/csv")
            if not mA.empty:
                csv_m = mA.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button("Download per-campaign metrics (CSV)", csv_m, file_name="periodA_campaign_metrics.csv", mime="text/csv")

        return

    # ---------- Compare periods ----------
    if view_mode == "Compare periods":
        a_start, a_end, a_set, a_mode = _sidebar_period_controls("A")
        b_start, b_end, b_set, b_mode = _sidebar_period_controls("B")

        # Filter
        sA, eA, gA = _filter_period(sends, events, signups, a_start, a_end, a_set, a_mode)
        sB, eB, gB = _filter_period(sends, events, signups, b_start, b_end, b_set, b_mode)

        # Metrics per period
        try:
            mA = compute_campaign_metrics(sA, eA, gA)
            mB = compute_campaign_metrics(sB, eB, gB)
        except Exception as exc:
            st.error(f"Failed to compute metrics for periods: {exc}")
            return

        aggA = _aggregate_metrics(mA, eA, sA)
        aggB = _aggregate_metrics(mB, eB, sB)

        # Previous equal-length periods for A and B
        daysA = max(1, (a_end.normalize() - a_start.normalize()).days + 1)
        prevA_end = a_start - pd.Timedelta(seconds=1)
        prevA_start = prevA_end - pd.Timedelta(days=daysA - 1)

        daysB = max(1, (b_end.normalize() - b_start.normalize()).days + 1)
        prevB_end = b_start - pd.Timedelta(seconds=1)
        prevB_start = prevB_end - pd.Timedelta(days=daysB - 1)

        sAp, eAp, gAp = _filter_period(sends, events, signups, prevA_start, prevA_end, a_set, a_mode)
        sBp, eBp, gBp = _filter_period(sends, events, signups, prevB_start, prevB_end, b_set, b_mode)

        try:
            mAp = compute_campaign_metrics(sAp, eAp, gAp)
        except Exception:
            mAp = pd.DataFrame()
        try:
            mBp = compute_campaign_metrics(sBp, eBp, gBp)
        except Exception:
            mBp = pd.DataFrame()

        aggAp = _aggregate_metrics(mAp, eAp, sAp) if not mAp.empty else {
            "N_sends": 0.0, "N_opens": 0.0, "N_clicks": 0.0, "N_signups": 0.0, "N_unsubscribes": 0.0,
            "open_rate": 0.0, "ctr": 0.0, "signup_rate": 0.0, "unsubscribe_rate": 0.0
        }
        aggBp = _aggregate_metrics(mBp, eBp, sBp) if not mBp.empty else {
            "N_sends": 0.0, "N_opens": 0.0, "N_clicks": 0.0, "N_signups": 0.0, "N_unsubscribes": 0.0,
            "open_rate": 0.0, "ctr": 0.0, "signup_rate": 0.0, "unsubscribe_rate": 0.0
        }

        # Help
        if hasattr(st, "popover"):
            with st.popover("ℹ️ KPI deltas"):
                st.write(
                    "- Each period's KPI delta compares against its **own previous equal-length period**.\n"
                    "- Counts show relative change: (Period − Previous) / Previous.\n"
                    "- For **Unsubs**, lower is better (inverse coloring)."
                )
        else:
            st.caption("ℹ️ Deltas compare each period with its own previous equal-length period; Unsubs use inverse coloring.")

        # KPI comparison: counts (row 1) + rates (row 2)
        st.subheader("Aggregate KPIs")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Period A**  \n{a_start.date()} → {a_end.date()}")
            st.caption(f"Prev: {prevA_start.date()} → {prevA_end.date()}")
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Sent", f"{int(aggA['N_sends']):,}", delta=_pct_change(aggA["N_sends"], aggAp["N_sends"]))
            a2.metric("Opened", f"{int(aggA['N_opens']):,}", delta=_pct_change(aggA["N_opens"], aggAp["N_opens"]))
            a3.metric("Clicked", f"{int(aggA['N_clicks']):,}", delta=_pct_change(aggA["N_clicks"], aggAp["N_clicks"]))
            a4.metric("Signed up", f"{int(aggA['N_signups']):,}", delta=_pct_change(aggA["N_signups"], aggAp["N_signups"]))
            a5.metric("Unsubs", f"{int(aggA['N_unsubscribes']):,}", delta=_pct_change(aggA["N_unsubscribes"], aggAp["N_unsubscribes"]), delta_color="inverse")
            ar1, ar2, ar3, ar4 = st.columns(4)
            ar1.metric("Open rate", f"{aggA['open_rate']:.1%}", delta=_pp_change(aggA["open_rate"], aggAp["open_rate"]))
            ar2.metric("CTR", f"{aggA['ctr']:.1%}", delta=_pp_change(aggA["ctr"], aggAp["ctr"]))
            ar3.metric("Signup rate", f"{aggA['signup_rate']:.1%}", delta=_pp_change(aggA["signup_rate"], aggAp["signup_rate"]))
            ar4.metric("Unsubs rate", f"{aggA['unsubscribe_rate']:.1%}", delta=_pp_change(aggA["unsubscribe_rate"], aggAp["unsubscribe_rate"]), delta_color="inverse")

        with c2:
            st.markdown(f"**Period B**  \n{b_start.date()} → {b_end.date()}")
            st.caption(f"Prev: {prevB_start.date()} → {prevB_end.date()}")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Sent", f"{int(aggB['N_sends']):,}", delta=_pct_change(aggB["N_sends"], aggBp["N_sends"]))
            b2.metric("Opened", f"{int(aggB['N_opens']):,}", delta=_pct_change(aggB["N_opens"], aggBp["N_opens"]))
            b3.metric("Clicked", f"{int(aggB['N_clicks']):,}", delta=_pct_change(aggB["N_clicks"], aggBp["N_clicks"]))
            b4.metric("Signed up", f"{int(aggB['N_signups']):,}", delta=_pct_change(aggB["N_signups"], aggBp["N_signups"]))
            b5.metric("Unsubs", f"{int(aggB['N_unsubscribes']):,}", delta=_pct_change(aggB["N_unsubscribes"], aggBp["N_unsubscribes"]), delta_color="inverse")
            br1, br2, br3, br4 = st.columns(4)
            br1.metric("Open rate", f"{aggB['open_rate']:.1%}", delta=_pp_change(aggB["open_rate"], aggBp["open_rate"]))
            br2.metric("CTR", f"{aggB['ctr']:.1%}", delta=_pp_change(aggB["ctr"], aggBp["ctr"]))
            br3.metric("Signup rate", f"{aggB['signup_rate']:.1%}", delta=_pp_change(aggB["signup_rate"], aggBp["signup_rate"]))
            br4.metric("Unsubs rate", f"{aggB['unsubscribe_rate']:.1%}", delta=_pp_change(aggB["unsubscribe_rate"], aggBp["unsubscribe_rate"]), delta_color="inverse")

        # ---------- Comparative visuals (more useful than raw time overlay) ----------
        st.subheader("Comparative visuals")

        # 1) Butterfly chart (counts) A vs B
        metrics_counts = ["Sent", "Opened", "Clicked", "Signed up", "Unsubs"]
        valsA = [aggA["N_sends"], aggA["N_opens"], aggA["N_clicks"], aggA["N_signups"], aggA["N_unsubscribes"]]
        valsB = [aggB["N_sends"], aggB["N_opens"], aggB["N_clicks"], aggB["N_signups"], aggB["N_unsubscribes"]]

        fig_bfly = go.Figure()
        fig_bfly.add_trace(go.Bar(
            y=metrics_counts, x=valsA, name="Period A", orientation="h", hovertemplate="A %{y}: %{x:,.0f}<extra></extra>"
        ))
        fig_bfly.add_trace(go.Bar(
            y=metrics_counts, x=[-v for v in valsB], name="Period B", orientation="h", hovertemplate="B %{y}: %{customdata:,.0f}<extra></extra>",
            customdata=valsB
        ))
        fig_bfly.update_layout(
            barmode="relative",
            title="Counts: Period A (right) vs Period B (left)",
            xaxis_title="Count (A positive, B negative)",
            yaxis_title="",
        )
        st.plotly_chart(fig_bfly, use_container_width=True)

        # 2) Rates comparison bars (A vs B)
        df_rates = pd.DataFrame({
            "metric": ["Open rate", "CTR", "Signup rate", "Unsubs rate"],
            "A": [aggA["open_rate"], aggA["ctr"], aggA["signup_rate"], aggA["unsubscribe_rate"]],
            "B": [aggB["open_rate"], aggB["ctr"], aggB["signup_rate"], aggB["unsubscribe_rate"]],
        })
        dfm = df_rates.melt(id_vars="metric", var_name="period", value_name="rate")
        fig_rates = px.bar(
            dfm, x="metric", y="rate", color="period", barmode="group",
            text=dfm["rate"].map(lambda r: f"{r:.1%}"),
            title="Rates comparison",
            labels={"rate": "Rate", "metric": "Metric"},
        )
        fig_rates.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_rates, use_container_width=True)

        # Export
        with st.expander("Export"):
            comp_counts = pd.DataFrame({"metric": metrics_counts, "A": valsA, "B": valsB})
            st.download_button("Download counts (CSV)", comp_counts.to_csv(index=False).encode("utf-8"), file_name="periods_counts.csv", mime="text/csv")
            st.download_button("Download rates (CSV)", df_rates.to_csv(index=False).encode("utf-8"), file_name="periods_rates.csv", mime="text/csv")

        return


if __name__ == "__main__":  # pragma: no cover - manual execution
    render_campaign_metrics_view()
