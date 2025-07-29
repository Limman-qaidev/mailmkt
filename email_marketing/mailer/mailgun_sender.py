"""Mailgun-based email sender implementation.

This module defines ``MailgunSender``, which sends email via the Mailgun
HTTP API.  It persists ``msg_id`` \u2194 recipient mappings in the same SQLite
database used by the SMTP sender, allowing the tracking subsystem to
correlate engagement events with users.  See the Mailgun API documentation
for details on the parameters accepted.

Environment variables used:

* ``MAILGUN_API_KEY`` – API key for Mailgun
* ``MAILGUN_DOMAIN`` – Domain configured in Mailgun
* ``MAILGUN_BASE_URL`` – Optional base URL; defaults to the official API
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

import requests

from email_marketing.mailer import EmailSender


class MailgunSender(EmailSender):
    """Mailgun implementation of the ``EmailSender`` interface."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("MAILGUN_API_KEY")
        self._domain = os.environ.get("MAILGUN_DOMAIN")
        self._base_url = os.environ.get(
            "MAILGUN_BASE_URL", f"https://api.mailgun.net/v3/{self._domain}"
        )
        if not self._api_key or not self._domain:
            raise ValueError("MAILGUN_API_KEY and MAILGUN_DOMAIN must be set")

        self._db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "email_map.db"
        )
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
        """Send an email via the Mailgun API and record the mapping.

        Args:
            recipient: Target email address.
            msg_id: Unique message identifier.
            html: HTML body of the email.
            subject: Subject line.
            text: Optional plain‑text alternative.
            variant: Optional A/B variant label.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        self._store_mapping(msg_id, recipient, variant)

        url = f"{self._base_url}/messages"
        auth = ("api", self._api_key)
        data = {
            "from": f"Mailmkt <mail@{self._domain}>",
            "to": [recipient],
            "subject": subject,
            "html": html,
            "o:tracking-id": msg_id,
        }
        if text:
            data["text"] = text

        response = requests.post(url, auth=auth, data=data, timeout=10)
        try:
            response.raise_for_status()
        except Exception:
            # Attach response body to the raised exception for easier debugging.
            raise


__all__ = ["MailgunSender"]
