"""Email editor component for the Streamlit dashboard.

This module defines a Streamlit view that lets users upload a recipient
list, compose an HTML message, and trigger the sending of a campaign via
SMTP (or Mailgun if later enabled). Recipient lists can be provided in CSV
or Excel format and are displayed back to the user for verification.

MO integration:
- If MO Assistant preloaded a list/subject, a third source mode
  "From MO Assistant (preloaded)" becomes available automatically.
- During sending, session flags are set so the sidebar avatar switches to
  the "writing" animation (mo_bot_writing.svg).
"""

from __future__ import annotations

import os
import time
import urllib.parse
import uuid
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from email_marketing.ab_testing import assign_variant
from email_marketing.analytics.recommend import get_distribution_list
# from email_marketing.mailer.mailgun_sender import MailgunSender
from email_marketing.mailer.smtp_sender import SMTPSender


# ============================ Utilities ============================

def _now_ts() -> str:
    """UTC timestamp as 'YYYY-mm-dd HH:MM:SS.ffffff' (no 'T')."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def _load_recipients(upload: Optional[Any]) -> List[str]:
    """Load recipient addresses from an uploaded CSV/Excel (first column).

    Non-email values are ignored. Returns a list of strings.
    """
    if upload is None:
        return []
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        return []

    emails: List[str] = []
    for val in df.iloc[:, 0].astype(str):
        if "@" in val:
            emails.append(val.strip())
    return emails


def _db_paths_for_send() -> tuple[str, str]:
    """Return paths to email_events.db and email_map.db under /data."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    events_db = str(data_dir / "email_events.db")
    email_map_db = str(data_dir / "email_map.db")
    return events_db, email_map_db


def _upsert_email_map(
    email_map_db: str,
    msg_id: str,
    recipient: str,
    variant: str | None,
    ts_str: str,
) -> None:
    """Insert/replace row in email_map with backward-compatible schema."""
    with sqlite3.connect(email_map_db) as conn:
        cols = [r[1].lower() for r in conn.execute("PRAGMA table_info(email_map)").fetchall()]
        has_send_ts = "send_ts" in cols

        if has_send_ts:
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant, send_ts) "
                "VALUES (?, ?, ?, ?)",
                (msg_id, recipient, variant, ts_str),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant) "
                "VALUES (?, ?, ?)",
                (msg_id, recipient, variant),
            )
        conn.commit()


def _log_send_event(
    events_db: str,
    msg_id: str,
    campaign: str,
    client_ip: str = "0.0.0.0",
    ts_str: str | None = None,
) -> None:
    """Insert a 'send' event into events table (schema-compatible)."""
    if ts_str is None:
        ts_str = _now_ts()
    with sqlite3.connect(events_db) as conn:
        conn.execute(
            """
            INSERT INTO events (msg_id, event_type, client_ip, ts, campaign)
            VALUES (?, 'send', ?, ?, ?)
            """,
            (msg_id, client_ip, ts_str, campaign),
        )
        conn.commit()


# ================== MO: sending flags for sidebar avatar ==================

def _mo_set_sending_flags(value: bool) -> None:
    """Set/clear sending flags so the sidebar avatar switches to 'writing'."""
    for key in ("email_sending", "campaign_sending", "sending"):
        st.session_state[key] = bool(value)


@contextmanager
def mo_sending_state() -> None:
    """Context manager to toggle sending flags during the send window."""
    _mo_set_sending_flags(True)
    try:
        yield
    finally:
        _mo_set_sending_flags(False)


# ============================ Main view ============================

def render_email_editor() -> None:
    """Render the email editor page in Streamlit (backward compatible)."""
    st.header("Email Campaign Editor")

    # Initialise preview cache holder
    if "preview_list" not in st.session_state:
        st.session_state["preview_list"] = []

    # Prefill from MO (consume once; safe if absent)
    mo_recipients = st.session_state.pop("mo_recipients", None)
    # Prefer the live-edited subject if present; fallback to mo_subject
    mo_subject = st.session_state.pop("mo_subject_live",
                                      st.session_state.pop("mo_subject", ""))
    mo_topic = st.session_state.pop("mo_topic", "")


    # Recipient source modes (add MO preloaded mode if available)
    has_mo = bool(mo_recipients)
    modes = ["Upload list", "By campaign type"]
    if has_mo:
        modes.insert(0, "From MO Assistant (preloaded)")

    mode = st.radio("Recipient source", modes, index=0 if has_mo else 0)

    recipients: List[str] = []

    # ---------- Mode: From MO Assistant (preloaded) ----------
    if has_mo and mode == "From MO Assistant (preloaded)":
        recipients = list(mo_recipients or [])
        st.success(
            f"MO preloaded {len(recipients)} recipients"
            f"{(' for topic: ' + mo_topic) if mo_topic else ''}."
        )
        # Domain filter (optional, consistent with other mode)
        if recipients:
            domains = sorted({e.split("@")[-1] for e in recipients if "@" in e})
            selected = st.multiselect("Filter by domain", domains, key="mo_domain_filter")
            view = [e for e in recipients if not selected or e.split("@")[-1] in selected]
            st.dataframe(pd.DataFrame({"email": view}))

    # ---------- Mode: Upload list ----------
    elif mode == "Upload list":
        upload = st.file_uploader(
            "Upload recipient list (CSV or Excel)",
            type=["csv", "xls", "xlsx"],
        )
        recipients = _load_recipients(upload)
        if recipients:
            st.success(f"Loaded {len(recipients)} recipients.")
            st.dataframe(pd.DataFrame({"email": recipients}))

    # ---------- Mode: By campaign type (recommendation) ----------
    else:
        campaign_id = st.text_input(
            "Campaign ID",
            help="Identifier of the campaign used to build the recommended list.",
        )
        if st.button("Preview"):
            if not campaign_id:
                st.warning("Please provide a Campaign ID.")
            else:
                try:
                    recs = list(get_distribution_list(campaign_id, 1.0))
                    st.session_state["preview_list"] = recs
                    st.success(f"Loaded {len(recs)} recommended recipients.")
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")

        # Optional domain filter over preview_list
        if st.session_state["preview_list"]:
            domains = sorted({e.split("@")[-1] for e in st.session_state["preview_list"] if "@" in e})
            selected = st.multiselect("Filter by domain", domains, key="rec_domain_filter")
            recipients = [
                e for e in st.session_state["preview_list"]
                if not selected or e.split("@")[-1] in selected
            ]
            st.success(f"Using {len(recipients)} recipients after filtering.")
            if recipients:
                st.dataframe(pd.DataFrame({"email": recipients}))

    # ---------- Compose ----------
    subject = st.text_input("Subject", max_chars=200, value=mo_subject or "")
    html_body = st.text_area(
        "HTML Body",
        height=300,
        placeholder="<p>Hello {{ name }}, welcome to our newsletter.</p>",
    )

    st.markdown("---")

    # ---------- Send campaign ----------
    can_send = bool(recipients) and bool(subject) and bool(html_body)
    send_button = st.button("Send Email", type="primary", disabled=not can_send, key="mo_send_button")
    if not send_button:
        return

    # Instantiate sender (keep SMTP by default)
    sender = SMTPSender()
    # If you later enable Mailgun:
    # sender = MailgunSender() if sender_choice == "Mailgun" else SMTPSender()

    # Tracking URL (can be made configurable)
    tracking_url = os.environ.get("TRACKING_URL", "https://track.jonathansalgadonieto.com").strip()

    total = len(recipients)
    progress = st.progress(0.0)
    events_db_path, email_map_db_path = _db_paths_for_send()

    # Toggle MO "writing" state during the whole send loop
    with mo_sending_state():
        with st.spinner("Sending emails..."):
            for i, email in enumerate(recipients, start=1):
                # Assign variant and generate msg_id
                variant = assign_variant(email)
                msg_id = uuid.uuid4().hex

                # a) Build open-pixel/logo tag
                timestamp = int(time.time())
                logo_qs = urllib.parse.urlencode({"msg_id": msg_id, "ts": timestamp, "campaign": subject})
                logo_tag = f'<p><img src="{tracking_url}/logo?{logo_qs}" alt="Company Logo" width="200"/></p>'

                # b) Build click link
                click_qs = urllib.parse.urlencode({"msg_id": msg_id, "url": "https://example.com", "campaign": subject})
                click_tag = f'<p><a href="{tracking_url}/click?{click_qs}">Click here</a></p>'

                # c) Build unsubscribe link
                unsub_qs = urllib.parse.urlencode({"msg_id": msg_id, "campaign": subject})
                unsub_tag = f'<p><a href="{tracking_url}/unsubscribe?{unsub_qs}">Unsubscribe</a></p>'

                # d) Build complaint link
                comp_qs = urllib.parse.urlencode({"msg_id": msg_id, "campaign": subject})
                complaint_tag = f'<p><a href="{tracking_url}/complaint?{comp_qs}">Report spam</a></p>'

                # e) Assemble full HTML
                full_html = f"""<!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body>
                    {logo_tag}
                    {html_body}
                    {click_tag}
                    {unsub_tag}
                    {complaint_tag}
                </body>
                </html>"""

                # f) Send the email
                try:
                    sender.send_email(
                        recipient=email,
                        msg_id=msg_id,
                        html=full_html,
                        subject=subject,
                        variant=variant,
                    )
                    ts_str = _now_ts()
                    _log_send_event(events_db_path, msg_id, subject, "0.0.0.0", ts_str)
                    _upsert_email_map(email_map_db_path, msg_id, email, variant, ts_str)
                except Exception as exc:
                    st.error(f"Failed to send to {email}: {exc}")

                # g) Update progress
                progress.progress(i / total)

    # Post-send feedback
    if hasattr(st, "toast"):
        st.toast(f"Campaign sent to {total} recipients.")
    st.success("Campaign sent.")
