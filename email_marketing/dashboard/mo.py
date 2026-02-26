# email_marketing/dashboard/mo.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import streamlit as st

from email_marketing.analytics.db import load_all_data


def _default_db_paths() -> Tuple[str, str, str]:
    """Return default locations for the SQLite databases."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    return (
        str(data_dir / "email_events.db"),
        str(data_dir / "email_map.db"),
        str(data_dir / "campaigns.db"),
    )


def _slugify(text: str) -> str:
    """File-system friendly slug for campaign/segment names."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "campaign"


def _build_distribution_list(
    campaign_name: str, *, include_all_if_new: bool = True
) -> pd.DataFrame:
    """
    Build a per-campaign distribution list.

    Logic:
    - Base universe for the campaign: emails that were **sent** for that
      campaign (events[event_type == 'send' & campaign == name]).
    - If there are no sends yet and `include_all_if_new` is True, fall back to
      **all known emails** from the sends table.
    - Exclude any email with (unsubscribe|complaint|deleted_or_spam)
      **for that same campaign**.
    """
    events_db, sends_db, campaigns_db = _default_db_paths()
    events, sends, _, _ = load_all_data(events_db, sends_db, campaigns_db)

    # Ensure expected columns
    for df, cols in [(events, {"campaign", "event_type", "email"}),
                     (sends, {"email"})]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in dataframe")

    # Base: emails that were **sent** for this campaign
    base = events[
        (events["campaign"] == campaign_name) & (events["event_type"] == "send")
    ]["email"].dropna().drop_duplicates()

    if base.empty and include_all_if_new:
        # Fallback: all known emails if campaign is new
        base = sends["email"].dropna().drop_duplicates()

    # Emails to exclude (by campaign)
    bad = events[
        (events["campaign"] == campaign_name)
        & (events["event_type"].isin(["unsubscribe", "complaint", "deleted_or_spam"]))
    ]["email"].dropna().drop_duplicates()

    keep = base[~base.isin(bad)].sort_values().reset_index(drop=True)
    return pd.DataFrame({"email": keep})


def render_mo_chat() -> None:
    """MO front screen – a simple guided 'bot-like' starter."""
    st.title("🤖 Meet MO")
    st.caption("Your assistant to launch targeted email campaigns.")

    # Quick chips
    cols = st.columns(4)
    quick = None
    for label, col in zip(["Loans", "Insurance", "Cards", "General"], cols):
        if col.button(label):
            quick = label

    seg = st.text_input(
        "Tell MO what campaign you want to create (e.g., 'Loans')",
        value=quick or "",
        placeholder="Loans / Insurance / Cards / …",
    ).strip()

    include_all = st.checkbox(
        "If the campaign is new, start from ALL users",
        value=True,
        help="If there are no previous 'send' events for this campaign, "
             "use every known email as the starting list.",
    )

    if st.button("Generate distribution and go to Email Editor", disabled=(not seg)):
        campaign_name = seg
        try:
            dist_df = _build_distribution_list(campaign_name, include_all_if_new=include_all)
        except Exception as exc:
            st.error(f"Could not build the distribution list: {exc}")
            return

        # Save CSV (optional but handy)
        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(campaign_name)
        csv_path = data_dir / f"distribution_list_{slug}.csv"
        dist_df.to_csv(csv_path, index=False)

        st.success(f"Distribution created for '{campaign_name}': {len(dist_df)} recipients.")
        st.caption(f"Saved to: {csv_path}")

        # Prefill for the Email Editor and 'navigate'
        st.session_state["prefill_campaign"] = campaign_name
        st.session_state["prefill_recipients"] = dist_df["email"].tolist()
        st.session_state["prefill_subject"] = f"{campaign_name.title()} — Newsletter"
        st.session_state["prefill_html"] = (
            "<p>Hello {{ name }},</p><p>We have news for you about "
            f"{campaign_name.title()}.</p>"
        )
        st.session_state["nav"] = "Email Editor"
        st.experimental_rerun()

    with st.expander("What MO will do"):
        st.markdown(
            "- Builds a distribution list for the chosen campaign.\n"
            "- Excludes users who unsubscribed, complained or marked as "
            "spam for that campaign.\n"
            "- If the campaign is new (no sends yet), it can start from "
            "all known emails.\n"
            "- Prefills the Email Editor (recipients, subject, HTML, "
            "campaign)."
        )
