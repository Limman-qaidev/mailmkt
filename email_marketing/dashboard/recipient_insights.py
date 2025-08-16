# email_marketing/dashboard/recipient_insights.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re

from email_marketing.analytics.db import load_all_data
from email_marketing.analytics.user_metrics import (
    build_user_aggregates,
    compute_eb_rates,
)


def _extract_topic_from_text(text: str) -> str:
    """Heuristic: [Topic] prefix, else left segment before — / - / : ."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    m = re.match(r"^\s*\[([^\]]+)\]\s*", s)
    if m:
        return m.group(1).strip()
    for sep in ("—", "-", ":"):
        if sep in s:
            return s.split(sep, 1)[0].strip()
    return " ".join(s.split()[:3]).strip()  # último recurso

def _build_campaign_topic_map(campaigns: pd.DataFrame) -> dict[str, str]:
    """Map campaign name -> topic (prefer 'topic' column; else parse 'subject'/'name')."""
    if campaigns is None or campaigns.empty:
        return {}
    cols = {c.lower(): c for c in campaigns.columns}
    name_col = cols.get("name") or next((c for c in campaigns.columns if c.lower().startswith("name")), None)
    topic_col = cols.get("topic")
    subject_col = cols.get("subject")

    cmap: dict[str, str] = {}
    for _, row in campaigns.iterrows():
        name = str(row.get(name_col, "")).strip() if name_col else ""
        topic = str(row.get(topic_col, "")).strip() if topic_col else ""
        if not topic:
            subj = str(row.get(subject_col, "")).strip() if subject_col else ""
            topic = _extract_topic_from_text(subj or name)
        if name:
            cmap[name] = topic or "Other"
    return cmap

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


def _request_nav_to_editor(recipients: List[str], subject: str) -> None:
    """Store recipients/subject and request navigation to Email Editor."""
    st.session_state["mo_recipients"] = recipients
    st.session_state["mo_subject_live"] = subject
    # Use a deferred routing flag; app.py should consume this **before** building the sidebar.
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

    # ---------------------- Small helpers ----------------------
    def _ensure_num_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Ensure columns exist and are numeric (filled with 0.0)."""
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    def _extract_topic_from_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        m = re.match(r"^\s*\[([^\]]+)\]\s*", s)
        if m:
            return m.group(1).strip()
        for sep in ("—", "-", ":"):
            if sep in s:
                return s.split(sep, 1)[0].strip()
        return " ".join(s.split()[:3]).strip()

    def _build_campaign_topic_map(campaigns_df: pd.DataFrame) -> dict[str, str]:
        if campaigns_df is None or campaigns_df.empty:
            return {}
        cols = {c.lower(): c for c in campaigns_df.columns}
        name_col = cols.get("name") or next((c for c in campaigns_df.columns if c.lower().startswith("name")), None)
        topic_col = cols.get("topic")
        subject_col = cols.get("subject")
        cmap: dict[str, str] = {}
        for _, rowc in campaigns_df.iterrows():
            name = str(rowc.get(name_col, "")).strip() if name_col else ""
            topic = str(rowc.get(topic_col, "")).strip() if topic_col else ""
            if not topic:
                subj = str(rowc.get(subject_col, "")).strip() if subject_col else ""
                topic = _extract_topic_from_text(subj or name)
            if name:
                cmap[name] = topic or "Other"
        return cmap

    def _prepare_ev_sd_with_topics(
        events_df: pd.DataFrame,
        sends_df: pd.DataFrame,
        campaigns_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
        ev = events_df.copy()
        sd = sends_df.copy()

        ev["email"] = ev["email"].astype(str).str.strip().str.lower()
        if "event_ts" not in ev.columns and "ts" in ev.columns:
            ev["event_ts"] = pd.to_datetime(ev["ts"], errors="coerce", utc=True)

        rc = "recipient" if "recipient" in sd.columns else ("email" if "email" in sd.columns else None)
        if rc:
            sd["email"] = sd[rc].astype(str).str.strip().str.lower()
        else:
            sd["email"] = pd.NA

        # msg_id -> campaign via events
        if "campaign" in ev.columns and "msg_id" in ev.columns:
            msg2camp = (
                ev.loc[ev["campaign"].notna(), ["msg_id", "campaign"]]
                .drop_duplicates("msg_id")
                .set_index("msg_id")["campaign"]
            )
        else:
            msg2camp = pd.Series(dtype=str)

        if "msg_id" in sd.columns and not msg2camp.empty:
            sd["campaign"] = sd["msg_id"].map(msg2camp)

        camp2topic = _build_campaign_topic_map(campaigns_df)
        ev["topic"] = ev["campaign"].map(camp2topic).fillna(ev["campaign"].map(_extract_topic_from_text))
        sd["topic"] = sd["campaign"].map(camp2topic).fillna(sd["campaign"].map(_extract_topic_from_text))
        return ev, sd, camp2topic

    def _topic_corpus(ev: pd.DataFrame, sd: pd.DataFrame, signups_df: pd.DataFrame) -> pd.DataFrame:
        """Corpus per topic: S_total, Y_total (signups) and p0_signup."""
        sends_t = sd.dropna(subset=["topic"]).groupby("topic")["msg_id"].nunique().rename("S_total")

        # signups from events
        y_ev = (
            ev[ev["event_type"].str.lower().eq("signup")]
            .dropna(subset=["topic"])
            .groupby("topic")["msg_id"].nunique()
            .rename("Y_ev")
        )
        # signups from signups table (map campaign->topic first)
        if not signups_df.empty and "campaign" in signups_df.columns:
            sgn = signups_df.copy()
            sgn["campaign"] = sgn["campaign"].astype(str)
            camp2topic = _build_campaign_topic_map(campaigns)
            sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
            y_tab = sgn.dropna(subset=["topic"]).groupby("topic")["signup_id"].nunique().rename("Y_tab")
        else:
            y_tab = pd.Series(dtype=float, name="Y_tab")

        corp = (
            pd.DataFrame(index=pd.Index([], name="topic"))
            .join(sends_t, how="outer")
            .join(y_ev, how="outer")
            .join(y_tab, how="outer")
            .fillna(0.0)
            .reset_index()
        )
        corp["Y_total"] = corp["Y_ev"] + corp["Y_tab"]
        corp["p0_signup"] = (corp["Y_total"] / corp["S_total"]).where(corp["S_total"] > 0, other=0.01)
        return corp[["topic", "S_total", "Y_total", "p0_signup"]]

    def _owners_series(ev: pd.DataFrame, signups_df: pd.DataFrame) -> pd.Series:
        """(email, topic) -> bool if the user has registered (events or table)."""
        ev_su = ev[ev["event_type"].str.lower().eq("signup")]
        own_ev = (ev_su.groupby(["email", "topic"])["msg_id"].nunique() > 0)

        if not signups_df.empty and {"campaign", "email"}.issubset(signups_df.columns):
            sgn = signups_df.copy()
            sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
            camp2topic = _build_campaign_topic_map(campaigns)
            sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
            own_tab = (sgn.dropna(subset=["topic"]).groupby(["email", "topic"])["signup_id"].nunique() > 0)
        else:
            own_tab = pd.Series(dtype=bool)

        owners = own_ev.combine(own_tab, func=lambda a, b: bool(a) or bool(b)).fillna(False)
        owners.name = "owner"
        return owners

    # ---------------------- Load & harmonize data ----------------------
    st.title("Customer 360º")

    events_db, sends_db, campaigns_db = _default_db_paths()
    try:
        events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    for df_ in (events, sends, signups):
        if "campaign" in df_.columns:
            df_["campaign"] = df_["campaign"].astype(str)

    # Guarantee email in events
    events = _prepare_events_with_email(events, sends)

    # Topic-prepared versions
    ev_t, sd_t, _ = _prepare_ev_sd_with_topics(events, sends, campaigns)
    corpus = _topic_corpus(ev_t, sd_t, signups)
    owners_mi = _owners_series(ev_t, signups)  # MultiIndex (email, topic) -> bool

    # Global aggregates per email (for the Recipient Detail tab)
    users_df = build_user_aggregates(events, sends, signups)
    if users_df.empty:
        st.info("No recipient data available.")
        return
    users_df = compute_eb_rates(users_df)

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

            topN = int(
                st.number_input(
                    "Top N recipients to keep",
                    min_value=10,
                    value=TOPN_DEFAULT,
                    step=10,
                    help="Cut the final audience to the top N by registration probability (p_signup).",
                )
            )

            # ---- Refinement controls (persisted in session_state) ----
            with st.expander("Refine audience (optional)"):
                col_f1, col_f2, col_f3, col_f4 = st.columns([1.1, 1.1, 1.2, 1.2])

                col_f1.number_input(
                    "Min. sends in topic",
                    min_value=0,
                    value=st.session_state.get("refine_min_s_topic", 1),  # default 1 recomendado
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

                domain_options = sorted(
                    [d for d in users_df["domain"].dropna().unique().tolist() if isinstance(d, str)]
                )
                col_f3.multiselect(
                    "Domains (include; empty=all)",
                    options=domain_options,
                    default=st.session_state.get("refine_domains_pick", []),
                    key="refine_domains_pick",
                    help="Restrict the audience to specific email domains.",
                )

                # NEW: minimum probability threshold (0 disables)
                col_f4.slider(
                    "Min. signup probability (EB)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("refine_min_prob", 0.0),
                    step=0.01,
                    key="refine_min_prob",
                    help="Exclude recipients with p_signup below this threshold. 0 disables this filter.",
                    )

            if st.button("Build audience"):
                # --- Read refinements from session (robust on reruns)
                min_s_topic = int(st.session_state.get("refine_min_s_topic", 1))
                max_recency_days = int(st.session_state.get("refine_max_recency_days", 0))
                domains_pick = st.session_state.get("refine_domains_pick", []) or []
                min_p_signup = float(st.session_state.get("refine_min_p_signup", 0.0))

                # Slice frames for the chosen topic
                sd_topic = sd_t[sd_t["topic"] == chosen_topic].copy()
                ev_topic = ev_t[ev_t["topic"] == chosen_topic].copy()

                # Owners in this topic: events (signup) + signups table mapped to topic
                own_topic = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("signup")]
                    .groupby("email")["msg_id"].nunique().gt(0)
                )
                if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                    sgn = signups.copy()
                    sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                    camp2topic = _build_campaign_topic_map(campaigns)
                    sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(
                        sgn["campaign"].map(_extract_topic_from_text)
                    )
                    own_tab = (
                        sgn[sgn["topic"] == chosen_topic]
                        .groupby("email")["signup_id"].nunique().gt(0)
                    )
                    own_topic = own_topic.combine(own_tab, lambda a, b: bool(a) or bool(b)).fillna(False)

                # Unique counts per email (topic)
                sends_u = sd_topic.groupby("email")["msg_id"].nunique().rename("S")
                opens_u = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("open")]
                    .groupby("email")["msg_id"].nunique().rename("O")
                )
                clicks_u = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("click")]
                    .groupby("email")["msg_id"].nunique().rename("C")
                )
                unsubs_u = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("unsubscribe")]
                    .groupby("email")["msg_id"].nunique().rename("U")
                )
                complaints_u = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("complaint")]
                    .groupby("email")["msg_id"].nunique().rename("Q")
                )

                # Signups per email: combina eventos y tabla (indicador 0/1 sin doble conteo)
                signups_ev = (
                    ev_topic[ev_topic["event_type"].str.lower().eq("signup")]
                    .groupby("email")["msg_id"].nunique().rename("Y_ev")
                )
                if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                    sgn = signups.copy()
                    sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                    camp2topic = _build_campaign_topic_map(campaigns)
                    sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(
                        sgn["campaign"].map(_extract_topic_from_text)
                    )
                    signups_tab = (
                        sgn[sgn["topic"] == chosen_topic]
                        .groupby("email")["signup_id"].nunique().rename("Y_tab")
                    )
                else:
                    signups_tab = pd.Series(dtype=float, name="Y_tab")

                Y = (
                    (signups_ev > 0).astype(int)
                    .combine((signups_tab > 0).astype(int), lambda a, b: int(bool(a) or bool(b)))
                    .rename("Y")
                )

                # Assemble audience table (OUTER merges to avoid missing columns)
                df = (
                    pd.DataFrame(index=pd.Index([], name="email"))
                    .join(sends_u,      how="outer")
                    .join(Y,            how="outer")
                    .join(opens_u,      how="outer")
                    .join(clicks_u,     how="outer")
                    .join(unsubs_u,     how="outer")
                    .join(complaints_u, how="outer")
                    .reset_index()
                )

                # Ensure all numeric cols exist + are numeric (prevents KeyError 'U')
                for col in ("S", "Y", "O", "C", "U", "Q"):
                    if col not in df.columns:
                        df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

                if df.empty or df["S"].sum() == 0:
                    st.warning("No eligible recipients for this topic.")
                else:
                    # ---------- Topic-level priors (open/click/click→signup) ----------
                    def _clamp(x, lo, hi):
                        try:
                            return float(max(lo, min(hi, x)))
                        except Exception:
                            return lo

                    S_tot = float(sd_topic["msg_id"].nunique())
                    O_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("open")]["msg_id"].nunique())
                    C_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("click")]["msg_id"].nunique())
                    Y_ev_tot = float(ev_topic[ev_topic["event_type"].str.lower().eq("signup")]["msg_id"].nunique())

                    if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                        sgn = signups.copy()
                        camp2topic = _build_campaign_topic_map(campaigns)
                        sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(sgn["campaign"].map(_extract_topic_from_text))
                        Y_tab_tot = float(sgn[sgn["topic"] == chosen_topic]["signup_id"].nunique())
                    else:
                        Y_tab_tot = 0.0

                    Y_tot = max(Y_ev_tot, Y_tab_tot)
                    p0_open = _clamp(O_tot / S_tot if S_tot > 0 else 0.05, 1e-4, 0.9)
                    p0_click = _clamp(C_tot / S_tot if S_tot > 0 else 0.02, 1e-4, 0.9)
                    p0_sc = _clamp(Y_tot / C_tot if C_tot > 0 else 0.10, 1e-4, 0.9)  # P(signup|click)

                    # ---------- EB-smoothed per user ----------
                    ALPHA_O, ALPHA_C, ALPHA_SC = 2.0, 2.0, 2.0
                    EPS = 1e-9

                    S_ser = df["S"].astype(float)
                    O_ser = df["O"].astype(float)
                    C_ser = df["C"].astype(float)
                    Y_ser = df["Y"].astype(float)

                    p_open_hat = (O_ser + ALPHA_O * p0_open)  / (S_ser + ALPHA_O)
                    p_click_hat = (C_ser + ALPHA_C * p0_click) / (S_ser + ALPHA_C)
                    p_c_given_o = (p_click_hat / p_open_hat.clip(lower=EPS)).clip(upper=1.0)
                    p_sc_hat = (Y_ser + ALPHA_SC * p0_sc) / (C_ser + ALPHA_SC)

                    df["p_signup"] = (p_open_hat * p_c_given_o * p_sc_hat).fillna(0.0)

                    # ---------- Filters ----------
                    # 1) minimum exposure in topic
                    df = df[df["S"] >= float(min_s_topic)]

                    # 2) exclude owners (new acquisition)
                    if exclude_owners:
                        df = df[~df["email"].map(own_topic).fillna(False)]

                    # 3) exclude unsub/complaint in topic
                    df = df[(df["U"] == 0.0) & (df["Q"] == 0.0)]

                    # 4) recency (in-topic, optional)
                    if max_recency_days > 0 and not ev_topic.empty and "event_ts" in ev_topic.columns:
                        last_open = (
                            ev_topic[ev_topic["event_type"].str.lower().eq("open")]
                            .groupby("email")["event_ts"].max()
                        )
                        now_utc = pd.Timestamp.now(tz="UTC")
                        recency_ok = last_open.apply(lambda ts: (now_utc - ts).days <= max_recency_days if pd.notna(ts) else False)
                        df = df[df["email"].isin(recency_ok[recency_ok].index)]

                    # 5) domain filter
                    if domains_pick:
                        df = df[df["email"].str.split("@").str[-1].isin(domains_pick)]

                    # 6) probability threshold
                    if min_p_signup > 0:
                        df = df[df["p_signup"] >= min_p_signup]

                    if df.empty:
                        st.warning("No eligible recipients after applying filters.")
                    else:
                        # Final ordering + top N
                        df["p_signup"] = pd.to_numeric(df["p_signup"], errors="coerce").fillna(0.0)
                        df = df.sort_values(by=["p_signup", "S"], ascending=[False, False]).head(topN).reset_index(drop=True)

                        b1, b2 = st.columns([1, 1])
                        with b1:
                            st.subheader("Final recipients")
                            st.dataframe(
                                df[["email", "S", "Y", "p_signup", "O", "C"]],
                                use_container_width=True,
                                height=360,
                            )
                        with b2:
                            st.subheader("Summary")
                            st.markdown(
                                f"- **Topic:** {chosen_topic}\n"
                                f"- **Goal:** {'New acquisition' if exclude_owners else 'All recipients'}\n"
                                f"- **Top N:** {topN}\n"
                                f"- **Min S per user-topic:** {min_s_topic}\n"
                                f"- **Min p_signup:** {min_p_signup:.2%}"
                            )

                        # Export + Send to MO
                        topic_slug = re.sub(r"[^a-z0-9]+", "_", chosen_topic.lower()).strip("_") or "topic"
                        csv_name = f"distribution_list_topic_{topic_slug}.csv"
                        csv_bytes = df[["email"]].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download topic audience (CSV)",
                            data=csv_bytes,
                            file_name=csv_name,
                            mime="text/csv",
                        )

                        colA, colB = st.columns([1, 2])
                        mo_subject = colB.text_input(
                            "Subject to use in MO/Editor",
                            value=f"[{chosen_topic.title()}] New Campaign",
                            key="subject_topic_audience",
                            help="Will be prefilled when navigating to MO Assistant / Email Editor.",
                        )
                        if colA.button("Send to MO Assistant", type="primary", key="send_topic_aud_to_mo"):
                            st.session_state["mo_recipients"] = df["email"].astype(str).tolist()
                            st.session_state["mo_subject_live"] = mo_subject
                            st.session_state["mo_topic"] = chosen_topic
                            st.session_state["nav_redirect"] = "MO Assistant"
                            st.rerun()

        # --- Best next campaigns ---
        st.subheader("Best next campaigns")
        st.caption("Ranking of topics by expected signups (sum of p_signup across eligible recipients).")

        # Build per-(email,topic) S,Y and EB p_signup for ALL topics
        grp_s = (
            sd_t.dropna(subset=["topic"])
            .groupby(["email", "topic"])["msg_id"]
            .nunique()
            .rename("S")
        )
        grp_y = (
            ev_t[ev_t["event_type"].str.lower().eq("signup")]
            .dropna(subset=["topic"])
            .groupby(["email", "topic"])["msg_id"]
            .nunique()
            .rename("Y")
        )
        m = (
            pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["email", "topic"]))
            .join(grp_s, how="outer")
            .join(grp_y, how="outer")
            .fillna(0.0)
            .reset_index()
        )
        if m.empty:
            st.info("Not enough data to rank topics.")
        else:
            # p0 per topic (fallback 1%)
            m = m.merge(corpus[["topic", "p0_signup"]], on="topic", how="left").fillna({"p0_signup": 0.01})
            S_all, Y_all = m["S"].astype(float), m["Y"].astype(float)
            m["p_signup"] = (Y_all + ALPHA * m["p0_signup"]) / (S_all + ALPHA)

            # Owners matrix by (email, topic) from events + signups table
            owners_ev = (
                ev_t[ev_t["event_type"].str.lower().eq("signup")]
                .groupby(["email", "topic"])["msg_id"].nunique().gt(0)
            )
            owners_tab = pd.Series(dtype=bool)
            if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                sgn = signups.copy()
                sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                camp2topic = _build_campaign_topic_map(campaigns)
                sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(
                    sgn["campaign"].map(_extract_topic_from_text)
                )
                owners_tab = sgn.groupby(["email", "topic"])["signup_id"].nunique().gt(0)

            owners_all = owners_ev.combine(owners_tab, lambda a, b: bool(a) or bool(b)).fillna(False)

            # Exclude owners if acquisition goal selected
            if exclude_owners:
                def _is_owner_row(row) -> bool:
                    try:
                        return bool(owners_all.loc[(row["email"], row["topic"])])
                    except Exception:
                        return False
                m = m[~m.apply(_is_owner_row, axis=1)]

            port = (
                m.groupby("topic")
                .agg(Eligible=("email", "nunique"),
                     Expected_signups=("p_signup", "sum"),
                     Mean_p_signup=("p_signup", "mean"))
                .reset_index()
                .sort_values(["Expected_signups", "Mean_p_signup"], ascending=[False, False])
            )
            st.dataframe(port.head(20), use_container_width=True, height=360)

            # Quick CTA to build & send the top-ranked topic
            if not port.empty:
                c_sel, c_top, c_subj = st.columns([1.4, 1.0, 2.0])
                topic_choices = port["topic"].tolist()
                chosen_from_rank = c_sel.selectbox("Choose topic to send", options=topic_choices, index=0)
                topN_rank = int(c_top.number_input("Top N", min_value=10, value=TOPN_DEFAULT, step=10))
                subject_rank = c_subj.text_input(
                    "Subject for MO/Editor",
                    value=f"[{chosen_from_rank.title()}] New Campaign",
                    key="subject_best_next",
                )

                if st.button("Build & Send to MO Assistant", type="primary", key="build_send_best_next"):
                    # Reuse per-topic build quickly
                    sd_topic = sd_t[sd_t["topic"] == chosen_from_rank].copy()
                    ev_topic = ev_t[ev_t["topic"] == chosen_from_rank].copy()
                    rowc = corpus.loc[corpus["topic"] == chosen_from_rank]
                    p0_signup = float(rowc["p0_signup"].iloc[0]) if not rowc.empty else 0.01

                    # owners for that topic
                    own_topic = (
                        ev_topic[ev_topic["event_type"].str.lower().eq("signup")]
                        .groupby("email")["msg_id"].nunique().gt(0)
                    )
                    if not signups.empty and {"campaign", "email"}.issubset(signups.columns):
                        sgn = signups.copy()
                        sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
                        camp2topic = _build_campaign_topic_map(campaigns)
                        sgn["topic"] = sgn["campaign"].map(camp2topic).fillna(
                            sgn["campaign"].map(_extract_topic_from_text)
                        )
                        own_tab = (
                            sgn[sgn["topic"] == chosen_from_rank]
                            .groupby("email")["signup_id"].nunique().gt(0)
                        )
                        own_topic = own_topic.combine(own_tab, lambda a, b: bool(a) or bool(b)).fillna(False)

                    sends_u = sd_topic.groupby("email")["msg_id"].nunique().rename("S")
                    signups_u = (
                        ev_topic[ev_topic["event_type"].str.lower().eq("signup")]
                        .groupby("email")["msg_id"].nunique().rename("Y")
                    )
                    unsubs_u = (
                        ev_topic[ev_topic["event_type"].str.lower().eq("unsubscribe")]
                        .groupby("email")["msg_id"].nunique().rename("U")
                    )
                    complaints_u = (
                        ev_topic[ev_topic["event_type"].str.lower().eq("complaint")]
                        .groupby("email")["msg_id"].nunique().rename("Q")
                    )

                    df_rank = (
                        pd.DataFrame(index=pd.Index([], name="email"))
                        .join(sends_u, how="outer")
                        .join(signups_u, how="outer")
                        .join(unsubs_u, how="outer")
                        .join(complaints_u, how="outer")
                        .reset_index()
                    )
                    # >>> FIX here as well
                    df_rank = _ensure_num_cols(df_rank, ["S", "Y", "U", "Q"])

                    if df_rank.empty:
                        st.warning("No eligible recipients for the selected topic.")
                    else:
                        S_r = df_rank["S"]
                        Y_r = df_rank["Y"]
                        df_rank["p_signup"] = (Y_r + ALPHA * p0_signup) / (S_r + ALPHA)

                        # basic eligibility (same as above)
                        df_rank = df_rank[df_rank["S"] >= 1.0]
                        df_rank = df_rank[(df_rank["U"] == 0.0) & (df_rank["Q"] == 0.0)]
                        if exclude_owners:
                            df_rank = df_rank[~df_rank["email"].map(own_topic).fillna(False)]

                        df_rank["p_signup"] = pd.to_numeric(df_rank["p_signup"], errors="coerce").fillna(0.0)
                        df_rank = df_rank.sort_values(by=["p_signup", "S"], ascending=[False, False]).head(topN_rank)

                        st.session_state["mo_recipients"] = df_rank["email"].astype(str).tolist()
                        st.session_state["mo_subject_live"] = subject_rank
                        st.session_state["mo_topic"] = chosen_from_rank
                        st.session_state["nav_redirect"] = "MO Assistant"
                        st.rerun()
