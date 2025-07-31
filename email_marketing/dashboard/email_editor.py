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
import os
import urllib.parse

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

    # 1) Upload recipient list
    upload = st.file_uploader(
        "Upload recipient list (CSV or Excel)",
        type=["csv", "xls", "xlsx"],
    )
    recipients = _load_recipients(upload)
    if recipients:
        st.success(f"Loaded {len(recipients)} recipients.")
        st.dataframe(pd.DataFrame({"email": recipients}))

    # 2) Compose subject and HTML body
    subject = st.text_input("Subject", max_chars=200)
    html_body = st.text_area(
        "HTML Body",
        height=300,
        placeholder="<p>Hello {{ name }}, welcome to our newsletter.</p>",
    )

    # 3) Choose sender
    sender_choice = st.selectbox("Sender", ["SMTP", "Mailgun"])
    send_button = st.button("Send Email", disabled=not recipients or not html_body)
    if not send_button:
        return

    # 4) Instantiate the chosen sender
    if sender_choice == "SMTP":
        sender = SMTPSender()
    else:
        try:
            sender = MailgunSender()
        except Exception as exc:
            st.error(f"Error initializing Mailgun sender: {exc}")
            return

    # 5) Determine tracking URL
    # Override tracking URL manually in the UI if needed
    default_tracking_url = os.environ.get("TRACKING_URL", "http://localhost:8000")
    tracking_url = st.text_input(
        "Tracking URL",
        value=default_tracking_url,
        help="Enter the public URL where your tracking server is reachable (e.g. ngrok or LAN IP)"
    ).strip()

    # 6) Debug: show the exact URLs that will be embedded
    sample_id = uuid.uuid4().hex
    pixel_debug = f"{tracking_url}/pixel?msg_id={sample_id}"
    click_debug = f"{tracking_url}/click?{urllib.parse.urlencode({'msg_id': sample_id, 'url': 'https://example.com'})}"
    unsub_debug = f"{tracking_url}/unsubscribe?msg_id={sample_id}"
    complaint_debug = f"{tracking_url}/complaint?msg_id={sample_id}"

    st.markdown("### 🔍 Debug: Embedded Tracking URLs")
    st.write("**Pixel URL:**", pixel_debug)
    st.write("**Click URL:**", click_debug)
    st.write("**Unsubscribe URL:**", unsub_debug)
    st.write("**Complaint URL:**", complaint_debug)
    st.markdown("---")

    # 7) Send emails with progress bar
    total = len(recipients)
    progress = st.progress(0.0)

    # DEBUG: confirm tracking_url in UI
    st.write("**Using Tracking URL:**", tracking_url)
    for i, email in enumerate(recipients, start=1):
        # Assign variant and generate msg_id
        variant = assign_variant(email)
        msg_id = uuid.uuid4().hex

        # a) Build open-pixel tag
        pixel_tag = (
            f'<img src="{tracking_url}/pixel?msg_id={msg_id}" '
            'width="1" height="1" alt="" style="display:none;"/>'
        )

        # b) Build click link
        click_qs = urllib.parse.urlencode({"msg_id": msg_id, "url": "https://example.com"})
        click_tag = f'<p><a href="{tracking_url}/click?{click_qs}">Click here</a></p>'

        # c) Build unsubscribe link
        unsub_qs = urllib.parse.urlencode({"msg_id": msg_id})
        unsub_tag = f'<p><a href="{tracking_url}/unsubscribe?{unsub_qs}">Unsubscribe</a></p>'

        # d) Build complaint link
        comp_qs = urllib.parse.urlencode({"msg_id": msg_id})
        complaint_tag = f'<p><a href="{tracking_url}/complaint?{comp_qs}">Report spam</a></p>'

        # e) Assemble full HTML
        full_html = pixel_tag + html_body + click_tag + unsub_tag + complaint_tag

        # f) Send the email
        try:
            sender.send_email(
                recipient=email,
                msg_id=msg_id,
                html=full_html,
                subject=subject,
                variant=variant,
            )
        except Exception as exc:
            st.error(f"Failed to send to {email}: {exc}")

        # g) Update progress
        progress.progress(i / total)

    st.success("Campaign sent.")
