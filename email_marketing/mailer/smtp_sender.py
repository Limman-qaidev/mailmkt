"""SMTP-based email sender implementation.

This module provides ``SMTPSender``, a concrete implementation of
``EmailSender`` that uses Python's ``smtplib`` to deliver messages via an
SMTP server.  It stores a mapping between the generated ``msg_id`` and the
recipient in a local SQLite database for subsequent tracking.

Environment variables used:

* ``SMTP_HOST`` – hostname of the SMTP server.
* ``SMTP_PORT`` – port number; defaults to 587.
* ``SMTP_USERNAME`` – username for authentication.
* ``SMTP_PASSWORD`` – password for authentication.
"""

from __future__ import annotations

import os
import smtplib
import sqlite3
from email.message import EmailMessage
from typing import Optional

from email_marketing.mailer import EmailSender


class SMTPSender(EmailSender):
    """SMTP implementation of the ``EmailSender`` interface."""

    def __init__(self) -> None:
        self._host = os.environ.get("SMTP_HOST", "localhost")
        self._port = int(os.environ.get("SMTP_PORT", "587"))
        self._username = os.environ.get("SMTP_USERNAME")
        self._password = os.environ.get("SMTP_PASSWORD")
        self._db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "email_map.db"
        )

        # Ensure the mapping table exists.
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_map (
                    msg_id TEXT PRIMARY KEY,
                    recipient TEXT NOT NULL,
                    variant TEXT
                )
                """
            )

    def _store_mapping(self, msg_id: str, recipient: str, variant: Optional[str]) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant) VALUES (?, ?, ?)",
                (msg_id, recipient, variant),
            )
            conn.commit()

    def send_email(
        self,
        recipient: str,
        msg_id: str,
        html: str,
        subject: str = "",
        text: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> None:
        """Send an email via SMTP and record the message mapping.

        Args:
            recipient: Target email address.
            msg_id: Unique message identifier.
            html: HTML body of the email.
            subject: Subject line of the email.
            text: Optional plain‑text alternative.
            variant: Optional A/B variant label for experimentation.

        Raises:
            smtplib.SMTPException: If delivery fails.
        """
        # Persist mapping for tracking before sending; if sending fails this entry
        # may remain orphaned but facilitates debugging.
        self._store_mapping(msg_id, recipient, variant)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._username or ""
        msg["To"] = recipient
        msg["Message-ID"] = msg_id
        if text:
            msg.set_content(text)
            msg.add_alternative(html, subtype="html")
        else:
            msg.set_content(html, subtype="html")

        with smtplib.SMTP(self._host, self._port) as smtp:
            smtp.starttls()
            if self._username and self._password:
                smtp.login(self._username, self._password)
            smtp.send_message(msg)


__all__ = ["SMTPSender"]
