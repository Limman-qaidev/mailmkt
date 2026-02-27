# email_marketing/dashboard/recipient_insights.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import re
import os

from email_marketing.analytics.db import load_all_data
from email_marketing.analytics.user_metrics import (
    build_user_aggregates,
    compute_eb_rates,
)

import sqlite3


def _extract_topic_from_text(text: str) -> str:
    """Extract topic label from a subject or campaign name."""
    if not isinstance(text, str):
        return ""
    s = text.strip().lower()
    # Keep it simple and deterministic (adjust your rules if needed)
    if "loan" in s:
        return "Loans"
    if "mortgage" in s:
        return "Mortgage"
    if "savings" in s:
        return "Savings Account"
    return ""


def _build_campaign_topic_map(campaigns_df: pd.DataFrame) -> dict[str, str]:
    """Map campaign_id -> topic."""
    if campaigns_df is None or campaigns_df.empty:
        return {}
    if "campaign_id" not in campaigns_df.columns:
        # Try best-effort fallback
        campaigns_df = campaigns_df.rename(columns={campaigns_df.columns[0]: "campaign_id"})
    name_col = "campaign_name" if "campaign_name" in campaigns_df.columns else "campaign_id"
    out: dict[str, str] = {}
    for _, row in campaigns_df.iterrows():
        cid = str(row["campaign_id"])
        out[cid] = _extract_topic_from_text(str(row.get(name_col, cid)))
    return out


def _prepare_ev_sd_with_topics(
    events_df: pd.DataFrame,
    sends_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (events_df_enriched, sends_df_enriched) adding 'topic' column if possible.
    Never fails: if it can't enrich, it returns the original dfs with topic="".
    """
    if events_df is None or events_df.empty:
        return events_df, sends_df

    topic_map = _build_campaign_topic_map(campaigns_df)

    # Ensure campaign_id exists or can be derived
    if "campaign_id" not in events_df.columns and "campaign" in events_df.columns:
        events_df = events_df.rename(columns={"campaign": "campaign_id"})

    if "campaign_id" in events_df.columns:
        events_df = events_df.copy()
        events_df["topic"] = events_df["campaign_id"].astype(str).map(topic_map).fillna("")
    else:
        events_df = events_df.copy()
        events_df["topic"] = ""

    if sends_df is not None and not sends_df.empty:
        if "campaign_id" not in sends_df.columns and "campaign" in sends_df.columns:
            sends_df = sends_df.rename(columns={"campaign": "campaign_id"})
        sends_df = sends_df.copy()
        if "campaign_id" in sends_df.columns:
            sends_df["topic"] = sends_df["campaign_id"].astype(str).map(topic_map).fillna("")
        else:
            sends_df["topic"] = ""

    return events_df, sends_df


def _topic_corpus(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build a topic-level corpus from events (for Customer 360)."""
    if events_df is None or events_df.empty or "topic" not in events_df.columns:
        return pd.DataFrame()
    # Example: counts by topic + event_type
    cols = [c for c in ["topic", "event_type"] if c in events_df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return (
        events_df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "count"], ascending=[True, False])
    )


def _owners_series(campaigns_df: pd.DataFrame) -> pd.Series:
    """Return Series indexed by campaign_id with owner (if available)."""
    if campaigns_df is None or campaigns_df.empty:
        return pd.Series(dtype="object")
    if "campaign_id" not in campaigns_df.columns:
        campaigns_df = campaigns_df.rename(columns={campaigns_df.columns[0]: "campaign_id"})
    owner_col = "owner" if "owner" in campaigns_df.columns else None
    if owner_col is None:
        return pd.Series(dtype="object")
    s = campaigns_df.set_index("campaign_id")[owner_col]
    return s


@st.cache_data(show_spinner=False)
def load_table(db_path: str, query: str) -> pd.DataFrame:
    """Load a table/query from SQLite and return a DataFrame (cached)."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(query, con)

@st.cache_data(show_spinner=False)
def compute_metrics_cached(events: pd.DataFrame, email_map: pd.DataFrame) -> pd.DataFrame:
    """Compute campaign metrics (cached). Keep this pure/deterministic."""
    # ... your existing computations ...
    return metrics_df

def _db_fingerprint(paths: Tuple[str, str, str]) -> str:
    """
    Build a stable fingerprint for cache invalidation based on file mtime + size.
    This avoids hashing huge DataFrames on every rerun.
    """
    parts: list[str] = []
    for p in paths:
        try:
            st_ = os.stat(p)
            parts.append(f"{p}:{st_.st_mtime_ns}:{st_.st_size}")
        except FileNotFoundError:
            parts.append(f"{p}:missing")
    return "|".join(parts)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load_all_data_by_fp(fp: str, events_db: str, sends_db: str, campaigns_db: str):
    """
    Cached DB load. 'fp' is used only as a cache key.
    """
    return load_all_data(events_db, sends_db, campaigns_db)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_customer360_bundle_by_fp(
    fp: str, events_db: str, sends_db: str, campaigns_db: str
):
    """
    Cached bundle for Customer 360 page.
    Heavy transforms live here so reruns don't recompute them.
    NOTE: fp is only used as a cache key.
    """
    events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)

    # Normalize dtypes once
    for df_ in (events, sends, campaigns, signups):
        if "campaign" in df_.columns:
            df_["campaign"] = df_["campaign"].astype(str)

    # Ensure email in events (join from sends if needed)
    events_prepared = _prepare_events_with_email(events, sends)

    # Build topic-aware versions + topic corpus + ownership map
    ev_t, sd_t, _ = _prepare_ev_sd_with_topics(events_prepared, sends, campaigns)
    corpus = _topic_corpus(ev_t, sd_t, signups, campaigns)
    owners_mi = _owners_series(ev_t, signups, campaigns)

    # Global user aggregates (Recipient Detail tab)
    users_df = build_user_aggregates(events_prepared, sends, signups)
    if not users_df.empty:
        users_df = compute_eb_rates(users_df)

    return events_prepared, sends, campaigns, signups, ev_t, sd_t, corpus, owners_mi, users_df


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_customer360_artifacts(fp: str, events_db: str, sends_db: str, campaigns_db: str):
    events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)

    for df_ in (events, sends, signups):
        if "campaign" in df_.columns:
            df_["campaign"] = df_["campaign"].astype(str)

    events = _prepare_events_with_email(events, sends)
    ev_t, sd_t, _ = _prepare_ev_sd_with_topics(events, sends, campaigns)
    corpus = _topic_corpus(ev_t, sd_t, signups)
    owners_mi = _owners_series(ev_t, signups)

    users_df = build_user_aggregates(events, sends, signups)
    users_df = compute_eb_rates(users_df) if not users_df.empty else users_df

    return corpus, owners_mi, users_df


    # ---------------------- Small helpers ----------------------
def _ensure_num_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure columns exist and are numeric (filled with 0.0)."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _recipient_topic_metrics(
    picked_email: str,
    events: pd.DataFrame,
    sends: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-recipient metrics by topic: counts and rates."""
    if not picked_email:
        return pd.DataFrame()

    # Normaliza email y event_ts
    ev = events.copy()
    ev["email"] = ev["email"].astype(str).str.strip().str.lower()
    ts_col = "event_ts" if "event_ts" in ev.columns else ("ts" if "ts" in ev.columns else None)
    if ts_col:
        ev["event_ts"] = pd.to_datetime(ev[ts_col], errors="coerce", utc=True)

    sd = sends.copy()
    # Hallar columna de recipient en sends
    rc = "recipient" if "recipient" in sd.columns else ("email" if "email" in sd.columns else None)
    if rc:
        sd["email"] = sd[rc].astype(str).str.strip().str.lower()
    else:
        sd["email"] = pd.NA
    # msg_id -> campaign desde events
    if "campaign" in ev.columns and "msg_id" in ev.columns:
        msg2camp = (
            ev.loc[ev["campaign"].notna(), ["msg_id", "campaign"]]
            .drop_duplicates("msg_id")
            .set_index("msg_id")["campaign"]
        )
    else:
        msg2camp = pd.Series(dtype=str)

    # Mapa campaign -> topic
    camp2topic = _build_campaign_topic_map(campaigns)

    # ---- Sends por topic para este email ----
    sd_e = sd[(sd["email"] == picked_email)].copy()
    if "msg_id" in sd_e.columns and not msg2camp.empty:
        sd_e["campaign"] = sd_e["msg_id"].map(msg2camp)
    else:
        sd_e["campaign"] = sd_e.get("campaign", None)
    sd_e["topic"] = sd_e["campaign"].map(camp2topic).fillna(
        sd_e["campaign"].map(_extract_topic_from_text).fillna("Other")
    )
    sends_topic = (
        sd_e.dropna(subset=["topic"])
            .groupby("topic")["msg_id"].nunique()
            .rename("N_sends_topic")
    )

    # ---- Eventos opens/clicks/unsubs por topic para este email ----
    ev_e = ev[(ev["email"] == picked_email)].copy()
    ev_e["topic"] = ev_e["campaign"].map(camp2topic).fillna(
        ev_e["campaign"].map(_extract_topic_from_text).fillna("Other")
    )
    # únicos msg por topic y tipo de evento
    def _count(evtype: str) -> pd.Series:
        m = ev_e[ev_e["event_type"].astype(str).str.lower().eq(evtype)]
        return m.dropna(subset=["topic"]).groupby("topic")["msg_id"].nunique()

    opens_t = _count("open").rename("N_opens_topic")
    clicks_t = _count("click").rename("N_clicks_topic")
    unsubs_t = _count("unsubscribe").rename("N_unsubs_topic")

    df = (
        pd.DataFrame(index=pd.Index([], name="topic"))
        .join(sends_topic, how="outer")
        .join(opens_t, how="outer")
        .join(clicks_t, how="outer")
        .join(unsubs_t, how="outer")
        .fillna(0)
        .reset_index()
    )
    if df.empty:
        return df

    # tasas por topic (con protección por 0)
    df["open_rate"] = (df["N_opens_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["ctr"] = (df["N_clicks_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["unsub_rate"] = (df["N_unsubs_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)

    # ordenar por envíos y apertura
    df = df.sort_values(["N_sends_topic", "N_opens_topic"], ascending=[False, False]).reset_index(drop=True)
    return df


def _default_db_paths() -> tuple[str, str, str]:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    return (
        str(data_dir / "email_events.db"),
        str(data_dir / "email_map.db"),
        str(data_dir / "campaigns.db"),
    )


def _as_utc(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)


def _prepare_events_with_email(events: pd.DataFrame, sends: pd.DataFrame) -> pd.DataFrame:
    """Ensure events have an 'email' column (fallback joining sends if needed)."""
    ev = events.copy()
    if "email" not in ev.columns or ev["email"].isna().all():
        if "msg_id" in ev.columns and "msg_id" in sends.columns:
            # Choose recipient-like column in sends
            rc = "recipient" if "recipient" in sends.columns else ("email" if "email" in sends.columns else None)
            if rc:
                ev = ev.merge(sends[["msg_id", rc]].rename(columns={rc: "email"}), on="msg_id", how="left")
    ev["email"] = ev["email"].astype(str).str.strip().str.lower()
    # Timestamp
    ts_col = "event_ts" if "event_ts" in ev.columns else ("ts" if "ts" in ev.columns else None)
    if ts_col:
        ev["event_ts"] = pd.to_datetime(ev[ts_col], errors="coerce", utc=True)
    return ev


def _heatmap_df(ev: pd.DataFrame, email: str, event_type: str) -> pd.DataFrame:
    """Build hour×weekday matrix for a given email and event type."""
    if ev.empty:
        return pd.DataFrame()
    df = ev[
        (ev["email"] == email)
        & (ev["event_type"].astype(str).str.lower() == event_type)
        & (ev["event_ts"].notna())
    ][["event_ts"]].copy()
    if df.empty:
        return pd.DataFrame()
    df["hour"] = df["event_ts"].dt.hour
    df["dow"] = df["event_ts"].dt.dayofweek  # 0=Mon
    mat = df.groupby(["dow", "hour"]).size().reset_index(name="count")
    # Friendly labels
    mat["weekday"] = mat["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    return mat[["weekday", "hour", "count"]]


def _request_nav_to_editor(recipients: list[str], subject: str) -> None:
    st.session_state["mo_recipients"] = recipients
    st.session_state["mo_subject_live"] = subject
    st.session_state.pop("nav", None)
    st.session_state["pending_nav"] = "Email Editor"
    st.rerun()


def render_recipient_insights() -> None:
    """Customer 360º (Recipient Insights) with tabs:
       - Campaign Planning (topic audience + best next campaigns)
       - Recipient Detail (per email)

    Probability: p_signup (Empirical-Bayes) per topic.
    Registered: detected via 'signup' events and/or 'signups' table.
    """
    import re
    import plotly.express as px

    # ---------------------- Internal constants (hidden) ----------------------
    ALPHA = 5.0          # EB prior strength
    TOPN_DEFAULT = 500   # default top N in topic audience

    # ---------------------- Load & harmonize data ----------------------
    st.title("Customer 360º")

    events_db, sends_db, campaigns_db = _default_db_paths()
    try:
        fp = _db_fingerprint((events_db, sends_db, campaigns_db))
        (
            events,
            sends,
            campaigns,
            signups,
            ev_t,
            sd_t,
            corpus,
            owners_mi,
            users_df,
        ) = _cached_customer360_bundle_by_fp(fp, events_db, sends_db, campaigns_db)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    if users_df.empty:
        st.info("No recipient data available.")
        return
    users_df = compute_eb_rates(users_df)

    # --- persistent audience state (for Campaign Planning) ---
    st.session_state.setdefault("aud_built", False)
    st.session_state.setdefault("aud_base", None)
    st.session_state.setdefault("aud_topic", "")

    # selection controls default values
    st.session_state.setdefault("aud_sel_mode", "Top N")
    st.session_state.setdefault("aud_sel_n", 500)
    st.session_state.setdefault("aud_sel_pct", 20)
    st.session_state.setdefault("aud_sel_target", 50.0)
    st.session_state.setdefault("aud_bands_pick", ["Very High","High"])

    # ========================== Tabs ==========================
    tab_campaign, tab_recipient = st.tabs(["Campaign Planning", "Recipient Detail"])

    # ======================= TAB: Recipient Detail =======================
    with tab_recipient:
        st.markdown("#### Recipient detail")
        emails_sorted = users_df["email"].dropna().astype(str).sort_values().unique().tolist()
        use_select = len(emails_sorted) <= 10000
        if use_select:
            picked = st.selectbox("Find recipient", options=[""] + emails_sorted, index=0)
        else:
            picked = st.text_input("Find recipient (type exact email; case-insensitive)").strip().lower()
            if picked and picked not in set(emails_sorted):
                st.warning("Email not found in current data.")

        if not picked:
            st.info("Pick a recipient to see their 360º view.")
        else:
            row = users_df.loc[users_df["email"] == picked]
            if row.empty:
                st.warning("Recipient not found after filters.")
            else:
                r = row.iloc[0]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Sends", f"{int(r['N_sends']):,}")
                k2.metric("Opens", f"{int(r['N_opens']):,}", delta=f"{r['open_rate_raw']:.1%}")
                k3.metric("Clicks", f"{int(r['N_clicks']):,}", delta=f"{r['click_rate_raw']:.1%}")
                k4.metric("Unsubs", f"{int(r['N_unsubscribes']):,}", delta=f"{r['unsubscribe_rate_raw']:.1%}", delta_color="inverse")
                k5.metric("EB open rate", f"{r['open_rate_eb']:.1%}")

                # ---- Per-topic detail with Registered + p_signup (EB) ----
                st.subheader("Topic interest (per recipient)")
                sd_e = sd_t[sd_t["email"] == picked].copy()
                ev_e = ev_t[ev_t["email"] == picked].copy()

                sends_topic = sd_e.dropna(subset=["topic"]).groupby("topic")["msg_id"].nunique().rename("N_sends_topic")
                def _cnt(evtype: str) -> pd.Series:
                    m = ev_e[ev_e["event_type"].str.lower().eq(evtype)]
                    return m.dropna(subset=["topic"]).groupby("topic")["msg_id"].nunique()

                opens_t = _cnt("open").rename("N_opens_topic")
                clicks_t = _cnt("click").rename("N_clicks_topic")
                unsubs_t = _cnt("unsubscribe").rename("N_unsubs_topic")
                signups_t = _cnt("signup").rename("N_signups_topic")

                tdf = (
                    pd.DataFrame(index=pd.Index([], name="topic"))
                    .join(sends_topic, how="outer")
                    .join(opens_t, how="outer")
                    .join(clicks_t, how="outer")
                    .join(unsubs_t, how="outer")
                    .join(signups_t, how="outer")
                    .fillna(0.0)
                    .reset_index()
                )

                if tdf.empty:
                    st.info("No topic-level information is available for this recipient.")
                else:
                    # raw rates
                    tdf["open_rate"] = (tdf["N_opens_topic"] / tdf["N_sends_topic"]).where(tdf["N_sends_topic"] > 0, other=0.0)
                    tdf["ctr"] = (tdf["N_clicks_topic"] / tdf["N_sends_topic"]).where(tdf["N_sends_topic"] > 0, other=0.0)
                    tdf["unsub_rate"]= (tdf["N_unsubs_topic"] / tdf["N_sends_topic"]).where(tdf["N_sends_topic"] > 0, other=0.0)

                    # merge p0_signup from corpus, compute p_signup (EB)
                    tdf = tdf.merge(corpus[["topic", "p0_signup"]], on="topic", how="left").fillna({"p0_signup": 0.01})
                    S_loc = tdf["N_sends_topic"].astype(float)
                    Y_loc = tdf["N_signups_topic"].astype(float)
                    tdf["p_signup"] = (Y_loc + ALPHA * tdf["p0_signup"]) / (S_loc + ALPHA)

                    # Registered from owners_mi
                    def _reg(t: str) -> bool:
                        try:
                            return bool(owners_mi.loc[(picked, t)])
                        except Exception:
                            return False
                    tdf["Registered"] = tdf["topic"].map(_reg)

                    # View controls
                    col_tt1, col_tt2 = st.columns([1, 1])
                    top_only = col_tt1.checkbox("Top topics only", value=True, help="Show only the K most relevant topics for this recipient.")
                    top_k = int(col_tt2.number_input("K", min_value=3, max_value=30, value=8, step=1))

                    tdf = tdf.sort_values(["p_signup", "N_sends_topic"], ascending=[False, False])
                    topic_view = tdf.head(top_k) if top_only else tdf

                    # Heatmap of rates (visual summary)
                    heat = topic_view.melt(
                        id_vars=["topic"],
                        value_vars=["open_rate", "ctr", "unsub_rate"],
                        var_name="metric",
                        value_name="value",
                    )
                    metric_labels = {"open_rate": "Open rate", "ctr": "CTR", "unsub_rate": "Unsub rate"}
                    heat["metric"] = heat["metric"].map(metric_labels)
                    counts = topic_view[["topic", "N_sends_topic", "N_opens_topic", "N_clicks_topic", "N_unsubs_topic"]]
                    heat = heat.merge(counts, on="topic", how="left")

                    c1_v, c2_v = st.columns([2, 3], gap="large")
                    with c1_v:
                        st.caption("Rates heatmap (per topic)")
                        fig_hm = px.density_heatmap(
                            heat,
                            x="metric", y="topic", z="value",
                            histfunc="avg", text_auto=True, color_continuous_scale="Blues", title=None,
                        )
                        fig_hm.update_layout(coloraxis_colorbar=dict(title="Rate"))
                        fig_hm.update_traces(
                            hovertemplate=(
                                "Topic: %{y}<br>Metric: %{x}<br>Value: %{z:.1%}"
                                "<br>Sends: %{customdata[0]:,}"
                                "<br>Opens: %{customdata[1]:,}"
                                "<br>Clicks: %{customdata[2]:,}"
                                "<br>Unsubs: %{customdata[3]:,}"
                            ),
                            customdata=heat[["N_sends_topic", "N_opens_topic", "N_clicks_topic", "N_unsubs_topic"]].values,
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)

                    with c2_v:
                        st.caption("Per-topic detail")
                        pretty = topic_view.rename(columns={
                            "N_sends_topic": "Sends",
                            "N_opens_topic": "Opens",
                            "N_clicks_topic": "Clicks",
                            "N_unsubs_topic": "Unsubs",
                            "open_rate": "Open rate",
                            "ctr": "CTR",
                            "unsub_rate": "Unsub rate",
                            "p_signup": "p_signup",
                        })
                        st.dataframe(
                            pretty[["topic", "Registered", "Sends", "Opens", "Clicks", "Unsubs", "Open rate", "CTR", "Unsub rate", "p_signup"]],
                            use_container_width=True, height=360,
                        )

    # ======================= TAB: Campaign Planning =======================
    with tab_campaign:

        st.markdown("#### Campaign Planning")

        # Goal: acquisition excludes existing owners for the chosen topic
        goal = st.radio(
            "Goal",
            ["New acquisition (exclude existing customers)", "All recipients"],
            index=0,
            help=("‘New acquisition’: excludes emails already registered in that "
                "product/topic. ‘All recipients’: includes all eligible recipients."),
            horizontal=False,
        )
        exclude_owners = goal.startswith("New acquisition")

        # --- Topic audience ---
        st.subheader("Topic audience")
        topic_options = corpus["topic"].dropna().astype(str).sort_values().unique().tolist()
        if not topic_options:
            st.info("No topics available to build an audience.")
        else:
            chosen_topic = st.selectbox(
                "Topic",
                options=topic_options,
                index=0,
                help="Select the topic/campaign for which you want to build an audience.",
            )

            # How many we ultimately want to keep (used later in selection phase)
            topN_default = 500
            st.session_state.setdefault("aud_sel_n", topN_default)

            # ---- Refinement controls (persisted) ----
            with st.expander("Refine audience (optional)"):
                col_f1, col_f2, col_f3 = st.columns([1.1, 1.1, 1.2])

                col_f1.number_input(
                    "Min. sends in topic",
                    min_value=0,
                    value=st.session_state.get("refine_min_s_topic", 1),  # recommended default 1
                    step=1,
                    key="refine_min_s_topic",
                    help="Minimum number of historical sends in this topic to consider the recipient.",
                )

                col_f2.number_input(
                    "Max. days since last open in topic (0 = disabled)",
                    min_value=0,
                    value=st.session_state.get("refine_max_recency_days", 0),
                    step=10,
                    key="refine_max_recency_days",
                    help="0 = no recency filter. If >0, requires an open within that number of days (in this topic).",
                )

                domain_options = sorted([d for d in users_df["domain"].dropna().unique().tolist() if isinstance(d, str)])
                col_f3.multiselect(
                    "Domains (include; empty=all)",
                    options=domain_options,
                    default=st.session_state.get("refine_domains_pick", []),
                    key="refine_domains_pick",
                    help="Restrict the audience to specific email domains.",
                )

            # ---------- PHASE A: Build base audience (runs once when you click the button) ----------
            if st.button("Build audience", key="build_aud"):
                # Read refinements from session (robust on reruns)
                min_s_topic       = int(st.session_state.get("refine_min_s_topic", 1))
                max_recency_days  = int(st.session_state.get("refine_max_recency_days", 0))
                domains_pick      = st.session_state.get("refine_domains_pick", []) or []

                # Slice frames for the chosen topic
                sd_topic = sd_t[sd_t["topic"] == chosen_topic].copy()
                ev_topic = ev_t[ev_t["topic"] == chosen_topic].copy()

                # p0 for this topic (fallbacks)
                def _clamp(x, lo, hi):
                    try:
                        return float(max(lo, min(hi, x)))
                    except Exception:
                        return lo

                S_tot = float(sd_topic["msg_id"].nunique())
                O_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("open")]["msg_id"].nunique())
                C_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("click")]["msg_id"].nunique())
                Y_ev_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("signup")]["msg_id"].nunique())

                # Optional: merge signups table, mapped to topic, without double counting (use the max as a prudent bound)
                if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                    sgn = signups.copy()
                    sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                    camp2topic = _build_campaign_topic_map(campaigns)
                    sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
                    Y_tab_tot = float(sgn[sgn["topic"] == chosen_topic]["signup_id"].nunique())
                else:
                    Y_tab_tot = 0.0

                Y_tot = max(Y_ev_tot, Y_tab_tot)

                p0_open  = _clamp((O_tot / S_tot) if S_tot > 0 else 0.05, 1e-4, 0.9)
                p0_click = _clamp((C_tot / S_tot) if S_tot > 0 else 0.02, 1e-4, 0.9)
                p0_sc    = _clamp((Y_tot / C_tot) if C_tot > 0 else 0.10, 1e-4, 0.9)  # P(signup|click)

                # Owners in this topic: events (signup) + signups table mapped to topic
                own_topic = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("signup")]
                    .groupby("email")["msg_id"].nunique().gt(0)
                )
                if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                    sgn = signups.copy()
                    sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                    camp2topic = _build_campaign_topic_map(campaigns)
                    sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
                    own_tab = (
                        sgn[sgn["topic"] == chosen_topic]
                        .groupby("email")["signup_id"].nunique().gt(0)
                    )
                    own_topic = own_topic.combine(own_tab, lambda a, b: bool(a) or bool(b)).fillna(False)

                # Unique counts per email in this topic
                sends_u   = sd_topic.groupby("email")["msg_id"].nunique().rename("S")
                opens_u   = ev_topic[ev_topic["event_type"].str.lower().eq("open")]       .groupby("email")["msg_id"].nunique().rename("O")
                clicks_u  = ev_topic[ev_topic["event_type"].str.lower().eq("click")]      .groupby("email")["msg_id"].nunique().rename("C")
                unsubs_u  = ev_topic[ev_topic["event_type"].str.lower().eq("unsubscribe")].groupby("email")["msg_id"].nunique().rename("U")
                complaints_u = ev_topic[ev_topic["event_type"].str.lower().eq("complaint")].groupby("email")["msg_id"].nunique().rename("Q")

                # Y: 0/1 "has signup" from events + table (combined without double counting)
                signups_ev = ev_topic[ev_topic["event_type"].str.lower().eq("signup")].groupby("email")["msg_id"].nunique()
                if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                    sgn = signups.copy()
                    sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                    camp2topic = _build_campaign_topic_map(campaigns)
                    sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
                    signups_tab = sgn[sgn["topic"] == chosen_topic].groupby("email")["signup_id"].nunique()
                else:
                    signups_tab = pd.Series(dtype=float)

                Y_ind = (signups_ev > 0).astype(int).combine((signups_tab > 0).astype(int), lambda a, b: int(bool(a) or bool(b))).rename("Y")

                # Assemble audience base (outer joins)
                df = (
                    pd.DataFrame(index=pd.Index([], name="email"))
                    .join(sends_u,      how="outer")
                    .join(Y_ind,        how="outer")
                    .join(opens_u,      how="outer")
                    .join(clicks_u,     how="outer")
                    .join(unsubs_u,     how="outer")
                    .join(complaints_u, how="outer")
                    .reset_index()
                )
                # Ensure numeric columns exist
                for col in ("S","Y","O","C","U","Q"):
                    if col not in df.columns:
                        df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

                if df.empty or df["S"].sum() == 0:
                    st.warning("No eligible recipients for this topic.")
                else:
                    # EB-smoothed probability model
                    ALPHA_O, ALPHA_C, ALPHA_SC = 2.0, 2.0, 2.0
                    EPS = 1e-9

                    S_ser = df["S"]
                    O_ser = df["O"]
                    C_ser = df["C"]
                    Y_ser = df["Y"]

                    p_open_hat  = (O_ser + ALPHA_O * p0_open)  / (S_ser + ALPHA_O)
                    p_click_hat = (C_ser + ALPHA_C * p0_click) / (S_ser + ALPHA_C)
                    p_c_given_o = (p_click_hat / p_open_hat.clip(lower=EPS)).clip(upper=1.0)
                    p_sc_hat    = (Y_ser + ALPHA_SC * p0_sc) / (C_ser + ALPHA_SC)

                    df["p_signup"] = (p_open_hat * p_c_given_o * p_sc_hat).fillna(0.0)

                    # Eligibility filters (except Top N/%/Target)
                    # 1) min exposure
                    df = df[df["S"] >= float(min_s_topic)]
                    # 2) exclude owners if acquisition
                    if exclude_owners:
                        mask_owner = df["email"].map(own_topic).fillna(False)
                        df = df[~mask_owner]
                    # 3) exclude unsub/complaint
                    if "U" not in df.columns: df["U"] = 0.0
                    if "Q" not in df.columns: df["Q"] = 0.0
                    df = df[(df["U"] == 0.0) & (df["Q"] == 0.0)]
                    # 4) recency within topic
                    if max_recency_days > 0 and not ev_topic.empty and "event_ts" in ev_topic.columns:
                        last_open = ev_topic[ev_topic["event_type"].str.lower().eq("open")].groupby("email")["event_ts"].max()
                        now_utc = pd.Timestamp.now(tz="UTC")
                        recency_ok = last_open.apply(lambda ts: (now_utc - ts).days <= max_recency_days if pd.notna(ts) else False)
                        df = df[df["email"].isin(recency_ok[recency_ok].index)]
                    # 5) domain filter
                    if domains_pick:
                        df = df[df["email"].str.split("@").str[-1].isin(domains_pick)]

                    if df.empty:
                        st.warning("No eligible recipients after applying filters.")
                    else:
                        # Persist sorted base audience once (deterministic ordering)
                        df_base = df.sort_values(by=["p_signup", "S"], ascending=[False, False]).reset_index(drop=True)
                        st.session_state["aud_base"]  = df_base
                        st.session_state["aud_topic"] = chosen_topic
                        st.session_state["aud_built"] = True
                        st.rerun()

            # ---------- PHASE B: Always-on selection UI (won't reset on radio/slider changes) ----------
            base = st.session_state.get("aud_base")
            if st.session_state.get("aud_built") and isinstance(base, pd.DataFrame) and not base.empty:
                st.markdown(f"**Topic:** {st.session_state.get('aud_topic','')}")

                # Probability bands based on current base distribution
                q = base["p_signup"].quantile([0.50, 0.75, 0.90]).to_dict()
                q50, q75, q90 = float(q.get(0.50, 0.0)), float(q.get(0.75, 0.0)), float(q.get(0.90, 0.0))

                def _band(p: float) -> str:
                    if p >= q90: return "Very High"
                    if p >= q75: return "High"
                    if p >= q50: return "Medium"
                    return "Low"

                dfb = base.copy()
                dfb["prob_band"] = dfb["p_signup"].astype(float).map(_band)

                band_sizes = (
                    dfb.groupby("prob_band")["email"].nunique()
                    .reindex(["Very High","High","Medium","Low"])
                    .fillna(0).astype(int)
                )
                st.caption("Band sizes → " + " | ".join([f"{b}: {band_sizes.get(b,0):,}" for b in ["Very High","High","Medium","Low"]]))

                bands_pick = st.multiselect(
                    "Include probability bands",
                    options=["Very High","High","Medium","Low"],
                    default=st.session_state.get("aud_bands_pick", ["Very High","High"]),
                    key="aud_bands_pick",
                    help="Choose which probability bands to include in the audience."
                )
                if bands_pick:
                    dfb = dfb[dfb["prob_band"].isin(bands_pick)]

                # ======= Interactive selection over the built audience =======
                base = st.session_state.get("aud_base")
                if st.session_state.get("aud_built") and isinstance(base, pd.DataFrame) and not base.empty:
                    st.markdown(f"**Topic:** {st.session_state.get('aud_topic','')}")

                    # Probability bands (ya calculadas arriba en dfb)
                    try:
                        aud_work = dfb.copy()
                    except NameError:
                        aud_work = base.copy()

                    if aud_work is None or aud_work.empty:
                        st.info("No recipients match the current filters. Adjust filters or bands.")
                    else:
                        # Asegurar dtype numérico para ordenar/quantiles (no altera tu lógica)
                        aud_work["p_signup"] = pd.to_numeric(aud_work["p_signup"], errors="coerce").fillna(0.0)

                        st.markdown("**How many to target?**")
                        c1, c2, _sp = st.columns([1.6, 1.2, 0.2])

                        selection_mode = c1.radio(
                            "Selection mode",
                            ["Top N", "Top %", "Expected signups target"],
                            horizontal=True,
                            key="aud_sel_mode",
                            help=("Top N: keep the N highest probabilities. "
                                "Top %: keep the top X percentile by probability. "
                                "Expected signups target: keep the smallest set whose summed p_signup reaches your target.")
                        )

                        # Orden determinista por probabilidad; empates por S
                        df_sorted = aud_work.sort_values(by=["p_signup", "S"], ascending=[False, False]).reset_index(drop=True)

                        # === Widgets con keys separadas y sincronización explícita ===
                        if selection_mode == "Top N":
                            n_value = c2.number_input(
                                "N",
                                min_value=1,
                                value=int(st.session_state.get("aud_sel_n", 500)),
                                step=10,
                                key="aud_sel_n_w",        # <- key de WIDGET distinta
                                format="%d",
                            )
                            # sincroniza estado de negocio
                            st.session_state["aud_sel_n"] = int(n_value)
                            n_keep = max(1, min(st.session_state["aud_sel_n"], len(df_sorted)))
                            df_sel = df_sorted.iloc[:n_keep].reset_index(drop=True)

                        elif selection_mode == "Top %":
                            pct_value = c2.slider(
                                "Percent",
                                min_value=1,
                                max_value=100,
                                value=int(st.session_state.get("aud_sel_pct", 20)),
                                step=1,
                                key="aud_sel_pct_w",      # <- key de WIDGET distinta
                            )
                            st.session_state["aud_sel_pct"] = int(pct_value)
                            pct_keep = st.session_state["aud_sel_pct"]
                            cutoff = float(df_sorted["p_signup"].quantile(1 - pct_keep / 100.0)) if len(df_sorted) else 1.0
                            df_sel = df_sorted[df_sorted["p_signup"] >= cutoff].reset_index(drop=True)

                        else:  # Expected signups target
                            target_value = c2.number_input(
                                "Target signups",
                                min_value=0.5,
                                value=float(st.session_state.get("aud_sel_target", 50.0)),
                                step=1.0,
                                key="aud_sel_target_w",   # <- key de WIDGET distinta
                            )
                            st.session_state["aud_sel_target"] = float(target_value)
                            target = st.session_state["aud_sel_target"]
                            cum = df_sorted["p_signup"].cumsum().values
                            pos = int(np.searchsorted(cum, target, side="left"))
                            pos = max(0, min(pos, len(df_sorted) - 1))
                            df_sel = df_sorted.iloc[:pos + 1].reset_index(drop=True)

                        # KPIs + tabla de la selección actual
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Eligible (base)", f"{len(df_sorted):,}")
                        k2.metric("Selected", f"{len(df_sel):,}")
                        k3.metric("Expected signups", f"{df_sel['p_signup'].sum():.1f}")

                        st.dataframe(
                            df_sel[["email", "S", "Y", "p_signup", "O", "C"]],
                            use_container_width=True,
                            height=360,
                        )

                        # Export + Send to Email Editor for the selection
                        exp_col, mo_col = st.columns([1, 2])
                        topic_slug = re.sub(r"[^a-z0-9]+", "_", str(st.session_state.get("aud_topic", "")).lower()).strip("_") or "topic"
                        csv_name = f"distribution_list_topic_{topic_slug}_selected.csv"

                        exp_col.download_button(
                            "Download selected (CSV)",
                            data=df_sel[["email"]].to_csv(index=False).encode("utf-8"),
                            file_name=csv_name,
                            mime="text/csv",
                        )

                        ed_subject = mo_col.text_input(
                            "Subject to use in Email Editor",
                            value=f"[{(st.session_state.get('aud_topic') or 'Campaign').title()}] New Campaign",
                            key="subject_topic_selected",
                            help="Will be prefilled in the Email Editor.",
                        )

                        if st.button("Send to Email Editor", type="primary", key="send_selected_to_email_editor"):
                            # Email Editor prefill: these are exactly the keys it pop()s on load
                            st.session_state["mo_recipients"] = df_sel["email"].astype(str).tolist()
                            # Email Editor mira primero 'mo_subject_live' y luego 'mo_subject'
                            st.session_state["mo_subject_live"] = ed_subject
                            st.session_state["mo_topic"] = st.session_state.get("aud_topic") or ""
                            st.session_state["nav_redirect"] = "Email Editor"
                            st.rerun()

                        # (Opcional) clear
                        if st.button("Clear audience", key="clear_aud"):
                            st.session_state["aud_built"] = False
                            st.session_state["aud_base"] = None
                            st.session_state["aud_topic"] = ""
                            st.rerun()

            else:
                st.info("Build an audience to see selection controls.")
