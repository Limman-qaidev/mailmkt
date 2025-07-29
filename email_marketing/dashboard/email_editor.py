"""Email editor component for the Streamlit dashboard.

This module defines a Streamlit view that lets users upload a recipient
list, compose an HTML message and trigger the sending of a campaign via
either SMTP or Mailgun.  Recipient lists can be provided in CSV or Excel
format and are displayed back to the user for verification.
"""

from __future__ import annotations

import uuid
from typing import List

import pandas as pd
import streamlit as st

from email_marketing.mailer.mailgun_sender import MailgunSender  # type: ignore
from email_marketing.mailer.smtp_sender import SMTPSender
from email_marketing.ab_testing import assign_variant


def _load_recipients(upload) -> List[str]:
    """Load recipient addresses from an uploaded file.

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


def render_email_editor() -> None:
    """Render the email editor page in Streamlit."""
    st.header("Email Campaign Editor")
    upload = st.file_uploader(
        "Upload recipient list (CSV or Excel)",
        type=["csv", "xls", "xlsx"],
    )
    recipients = _load_recipients(upload)
    if recipients:
        st.success(f"Loaded {len(recipients)} recipients.")
        st.dataframe(pd.DataFrame({"email": recipients}))

    subject = st.text_input("Subject", max_chars=200)
    html = st.text_area(
        "HTML Body",
        height=300,
        placeholder="<p>Hello {{ name }}, welcome to our newsletter.</p>",
    )
    sender_choice = st.selectbox("Sender", ["SMTP", "Mailgun"])
    send_button = st.button("Send Email", disabled=not recipients or not html)
    if send_button:
        if sender_choice == "SMTP":
            sender = SMTPSender()
        else:
            try:
                sender = MailgunSender()
            except Exception as exc:
                st.error(f"Error initialising Mailgun sender: {exc}")
                return

        progress = st.progress(0.0)
        for i, email in enumerate(recipients):
            # Assign A/B variant and generate a unique message ID.
            variant = assign_variant(email)
            msg_id = uuid.uuid4().hex
            try:
                sender.send_email(
                    recipient=email,
                    msg_id=msg_id,
                    html=html,
                    subject=subject,
                    variant=variant,
                )
            except Exception as exc:
                st.error(f"Failed to send to {email}: {exc}")
            progress.progress((i + 1) / len(recipients))
        st.success("Campaign sent.")
