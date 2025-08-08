"""Email editor component for the Streamlit dashboard.

This module defines a Streamlit view that lets users upload a recipient
list, compose an HTML message and trigger the sending of a campaign via
either SMTP or Mailgun.  Recipient lists can be provided in CSV or Excel
format and are displayed back to the user for verification.
"""

from __future__ import annotations

import os
import urllib.parse
import time
from pathlib import Path
from datetime import datetime
import sqlite3

import uuid
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from email_marketing.ab_testing import assign_variant
# from email_marketing.analytics import calibration
# from email_marketing.analytics import model as analytics_model
from email_marketing.analytics.recommend import get_distribution_list

# Uncomment if needed add MailgunSender
# from email_marketing.mailer.mailgun_sender import MailgunSender
from email_marketing.mailer.smtp_sender import SMTPSender


def _now_ts() -> str:
    # Espacio entre fecha y hora; optional microseconds
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def _load_recipients(upload: Optional[Any]) -> List[str]:
    """
    Load recipient addresses from an uploaded file.

    Supports CSV and Excel formats.  Assumes the first column contains email
    addresses.  Non‑email values are ignored.  Returns a list of strings.
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
    """Devuelve rutas de email_events.db y email_map.db (carpeta /data)."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    events_db = str(data_dir / "email_events.db")
    email_map_db = str(data_dir / "email_map.db")
    return events_db, email_map_db


def _upsert_email_map(email_map_db: str, msg_id: str, recipient: str,
                      variant: str | None, ts_str: str) -> None:
    """
    Inserta/actualiza fila en email_map.
    Detecta columnas existentes para ser compatible con tu esquema
    (con o sin send_ts, con o sin campaign_id).
    """
    with sqlite3.connect(email_map_db) as conn:
        cols = [r[1].lower() for r in conn.execute(
            "PRAGMA table_info(email_map)"
            ).fetchall()]
        has_send_ts = "send_ts" in cols

        if has_send_ts:
            conn.execute(
                "INSERT OR REPLACE INTO email_map "
                "(msg_id, recipient, variant, send_ts) VALUES (?, ?, ?, ?)",
                (msg_id, recipient, variant, ts_str),
            )
        else:
            # Esquema antiguo sin send_ts
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant)"
                " VALUES (?, ?, ?)",
                (msg_id, recipient, variant),
            )
        conn.commit()


def _log_send_event(events_db: str, msg_id: str, campaign: str,
                    client_ip: str = "0.0.0.0", ts_str: str | None = None
                    ) -> None:
    """
    Inserta el evento 'send' en la tabla events.
    Usa formato de timestamp 'YYYY-mm-dd HH:MM:SS.SSSSSS' (sin 'T') para
      evitar el error de pandas/sqlite.
    """
    if ts_str is None:
        ts_str = _now_ts()

    with sqlite3.connect(events_db) as conn:
        # id es autoincrement, por eso nominamos columnas
        conn.execute(
            """
            INSERT INTO events (msg_id, event_type, client_ip, ts, campaign)
            VALUES (?, 'send', ?, ?, ?)
            """,
            (msg_id, client_ip, ts_str, campaign),
        )
        conn.commit()


def render_email_editor() -> None:
    """Render the email editor page in Streamlit."""
    st.header("Email Campaign Editor")

    if "preview_list" not in st.session_state:
        st.session_state["preview_list"] = []

    # 1) Upload recipient list
    mode = st.radio("Recipient source", ["Upload list", "By campaign type"])
    recipients: List[str] = []
    if mode == "Upload list":
        upload = st.file_uploader(
            "Upload recipient list (CSV or Excel)",
            type=["csv", "xls", "xlsx"],
        )
        recipients = _load_recipients(upload)
        if recipients:
            st.success(f"Loaded {len(recipients)} recipients.")
            st.dataframe(pd.DataFrame({"email": recipients}))
    else:
        campaign_id = st.text_input(
            "Campaign ID",
            help="Identifier of the campaign used to "
            "build the campaign list.",
        )
        """threshold = st.slider(
            "Recommendation threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=(
                "Minimum probability required to include a recipient in the "
                "recommended list."
                ),
        )"""
        if st.button("Preview") and campaign_id:
            try:
                recipients = get_distribution_list(campaign_id, 1.0)
                st.success(
                    f"Loaded {len(recipients)} recommended recipients."
                )
                if recipients:
                    st.dataframe(pd.DataFrame({"email": recipients}))
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")

        if st.session_state["preview_list"]:
            domains = sorted({
                email.split("@")[-1] for email in st.session_state[
                    "preview_list"
                    ]
                })
            selected = st.multiselect("Filter by domain", domains)
            recipients = [
                e
                for e in st.session_state["preview_list"]
                if not selected or e.split("@")[-1] in selected
            ]
            st.success(f"Loaded {len(recipients)} recommended recipients.")
            if recipients:
                st.dataframe(pd.DataFrame({"email": recipients}))

    # 2) Compose subject and HTML body
    subject = st.text_input("Subject", max_chars=200)
    html_body = st.text_area(
        "HTML Body",
        height=300,
        placeholder="<p>Hello {{ name }}, welcome to our newsletter.</p>",
    )

    # 3) Choose sender and analytics actions
    # Uncomment if needed adding MailgunSender
    # sender_choice = st.selectbox("Sender", ["SMTP", "Mailgun"])
    """st.sidebar.subheader("Analytics")
    if st.sidebar.button("Recalculate weights"):
        calibration.recalculate_weights()
        st.sidebar.success("Weights recalibrated")
    if st.sidebar.button("Retrain model"):
        analytics_model.train_model()
        st.sidebar.success("Model trained")"""
    send_button = st.button(
        "Send Email", disabled=not recipients or not html_body
        )
    if not send_button:
        return

    # 4) Instantiate the chosen sender
    # Uncomment if needed add MailgunSender
    sender = SMTPSender()
    """if sender_choice == "SMTP":
        sender = SMTPSender()
    else:
        try:
            sender = MailgunSender()
        except Exception as exc:
            st.error(f"Error initializing Mailgun sender: {exc}")
            return"""

    # 5) Determine tracking URL
    # Override tracking URL manually in the UI if needed
    default_tracking_url = os.environ.get(
        "TRACKING_URL",
        "https://track.jonathansalgadonieto.com"
        )
    # tracking_url = st.text_input(
    #     "Tracking URL",
    #     value=default_tracking_url,
    #     help="URL pública de tu servidor de tracking"
    # ).strip()
    tracking_url = default_tracking_url
    # 6) Debug: show the exact URLs that will be embedded
    # sample_id = uuid.uuid4().hex
    # pixel_debug = f"{tracking_url}/pixel?msg_id={sample_id}"
    # click_debug = f"{tracking_url}/click?{urllib.parse.urlencode(
    # {'msg_id': sample_id, 'url': 'https://example.com'})}"
    # unsub_debug = f"{tracking_url}/unsubscribe?msg_id={sample_id}"
    # complaint_debug = f"{tracking_url}/complaint?msg_id={sample_id}"

    """st.markdown("### 🔍 Debug: Embedded Tracking URLs")
    st.write("**Pixel URL:**", pixel_debug)
    st.write("**Click URL:**", click_debug)
    st.write("**Unsubscribe URL:**", unsub_debug)
    st.write("**Complaint URL:**", complaint_debug)"""
    st.markdown("---")

    # 7) Send emails with progress bar
    total = len(recipients)
    progress = st.progress(0.0)

    events_db_path, email_map_db_path = _db_paths_for_send()

    for i, email in enumerate(recipients, start=1):
        # Assign variant and generate msg_id
        variant = assign_variant(email)
        msg_id = uuid.uuid4().hex

        # a) Build open-pixel tag
        timestamp = int(time.time())
        # pixel_tag = (
        #    f'<img src="{tracking_url}/pixel?msg_id={msg_id}&ts={timestamp}" '
        #    'width="1" height="1" alt="" border="0" '
        #    'style="display:block; visibility:hidden;"/>'
        # )
        logo_qs = urllib.parse.urlencode(
            {"msg_id": msg_id,
             "ts": timestamp,
             "campaign": subject}
             )
        logo_tag = (
            f'<p><img src="{tracking_url}/logo?{logo_qs}" '
            'alt="Company Logo" width="200"/></p>'
        )

        # b) Build click link
        click_qs = urllib.parse.urlencode(
            {"msg_id": msg_id, "url": "https://example.com",
             "campaign": subject}
            )
        click_tag = (
            f'<p><a href="{tracking_url}/click?{click_qs}">Click here</a></p>'
        )

        # c) Build unsubscribe link
        unsub_qs = urllib.parse.urlencode(
            {"msg_id": msg_id,
             "campaign": subject}
             )
        unsub_tag = (
            f'<p><a href="{tracking_url}/unsubscribe?{unsub_qs}">Unsubscribe'
            '</a></p>'
        )

        # d) Build complaint link
        comp_qs = urllib.parse.urlencode(
            {"msg_id": msg_id,
             "campaign": subject}
             )
        complaint_tag = (
            f'<p><a href="{tracking_url}/complaint?{comp_qs}">Report spam'
            '</a></p>'
        )
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
                    </html>
                    """
        # >>> PREVIEW: solo para i==1, muestro el HTML que voy a enviar <<<
        # if i == 1:
        #     st.subheader("📧 HTML Preview (first recipient)")
        #     st.code(full_html, language="html")
        #     st.markdown("---")
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
            _upsert_email_map(
                email_map_db_path, msg_id, email, variant, ts_str
                )

        except Exception as exc:
            st.error(f"Failed to send to {email}: {exc}")

        # g) Update progress
        progress.progress(i / total)

    st.success("Campaign sent.")
