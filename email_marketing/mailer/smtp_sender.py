"""Updated SMTP-based email sender implementation.

This module provides `SMTPSender`, a concrete implementation of
`EmailSender` that uses Python's `smtplib` to deliver messages via an
SMTP server.  It stores a mapping between the generated `msg_id` and the
recipient in a local SQLite database for subsequent tracking.

Environment variables used:

* `SMTP_HOST`/`SMTP_SERVER` – hostname of the SMTP server.
* `SMTP_PORT`/`SMTP_SERVER_PORT` – port number; defaults to 587.
* `SMTP_USERNAME`/`SMTP_USER` – username for authentication.
* `SMTP_PASSWORD`/`SMTP_APP_PWD` – password for authentication.
* `SMTP_USE_SSL` – when set to "true"/"1"/"yes", connects via SMTP over SSL
  (port 465).  When false or unset, STARTTLS is used on the configured port.
* `SMTP_DEBUG` – when set to a truthy value ("1", "true", "yes") the
  underlying `smtplib` client will output detailed debug information to
  stderr.  This can help diagnose connection and authentication problems.

Multiple aliases are supported for convenience.  The implementation reads
the first defined variable in each pair and falls back to a sensible
default if none are set.  If no host is provided but a username is
configured, the domain part of the username is used to infer the SMTP
host (e.g. `user@gmail.com` → `smtp.gmail.com`).  The default port
is 587 (STARTTLS) unless `SMTP_USE_SSL` is `true`, in which case you
should specify port 465.
"""

from __future__ import annotations

import os
import smtplib
import sqlite3
from email.message import EmailMessage
from typing import Optional

from email_marketing.mailer import EmailSender


class SMTPSender(EmailSender):
    """SMTP implementation of the `EmailSender` interface."""

    def __init__(self) -> None:
        # Read SMTP configuration from environment variables.  Support
        # multiple aliases for backwards compatibility and different naming
        # conventions.  If no variables are defined, fall back to sensible
        # defaults.  The priority is:
        # 1. SMTP_HOST / SMTP_PORT
        # 2. SMTP_SERVER (for host) /
        #    SMTP_USER / SMTP_APP_PWD (for credentials)
        self._host = (
            os.environ.get("SMTP_HOST")
            or os.environ.get("SMTP_SERVER")
            or ""
        )
        self._port = int(
            os.environ.get("SMTP_PORT")
            or os.environ.get("SMTP_SERVER_PORT")
            or "0"
        )
        # Username can be provided via SMTP_USERNAME or SMTP_USER
        self._username = (
            os.environ.get("SMTP_USERNAME") or os.environ.get("SMTP_USER")
            )
        # Password can be provided via SMTP_PASSWORD or SMTP_APP_PWD
        self._password = (
            os.environ.get("SMTP_PASSWORD") or os.environ.get("SMTP_APP_PWD")
        )
        # Use SSL instead of STARTTLS if explicitly requested.
        self._use_ssl = (
            os.environ.get("SMTP_USE_SSL", "false")
            .lower() in {"1", "true", "yes"}
        )
        # Enable SMTP client debug output if SMTP_DEBUG is truthy.
        self._debug = (
            os.environ.get("SMTP_DEBUG", "true").lower() in {
                "1", "true", "yes"
                }
        )
        # Path to SQLite database for mapping msg_id to recipient
        self._db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "email_map.db",
        )

        self._init_db()

        # Infer host and port from the email domain if not explicitly set.
        if not self._host:
            domain = None
            if self._username and "@" in self._username:
                domain = self._username.split("@", 1)[1].lower()
            if domain:
                provider_hosts = {
                    "gmail.com": "smtp.gmail.com",
                    "outlook.com": "smtp-mail.outlook.com",
                    "hotmail.com": "smtp-mail.outlook.com",
                    "live.com": "smtp-mail.outlook.com",
                    "yahoo.com": "smtp.mail.yahoo.com",
                }
                self._host = provider_hosts.get(domain, f"smtp.{domain}")
            else:
                self._host = "localhost"
        if self._port == 0:
            self._port = 587

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            # 1) Ensure table exists with send_ts column
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_map (
                    msg_id TEXT PRIMARY KEY,
                    recipient TEXT NOT NULL,
                    variant TEXT,
                    send_ts DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 2) Detect if send_ts column is missing and add it
            cur = conn.execute("PRAGMA table_info(email_map)")
            cols = [row[1] for row in cur.fetchall()]
            if "send_ts" not in cols:
                conn.execute(
                    "ALTER TABLE email_map ADD COLUMN send_ts DATETIME"
                )
            conn.commit()

    def _store_mapping(self, msg_id: str,
                       recipient: str,
                       variant: Optional[str]) -> None:
        """Insert or update the mapping and record the send timestamp."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO email_map "
                "(msg_id, recipient, variant, send_ts) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
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
        """Send an email via SMTP and record the message mapping."""
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

        try:
            # Use SSL or plain SMTP with STARTTLS depending on
            # configuration.
            if self._use_ssl:
                smtp_conn: smtplib.SMTP = smtplib.SMTP_SSL(
                    self._host, self._port
                )
            else:
                smtp_conn = smtplib.SMTP(self._host, self._port)
            with smtp_conn as smtp:
                if self._debug:
                    smtp.set_debuglevel(1)
                smtp.ehlo()
                if not self._use_ssl:
                    smtp.starttls()
                    smtp.ehlo()
                if self._username and self._password:
                    smtp.login(self._username, self._password)

                # <<< DEBUG: muestra el HTML completo antes de enviarlo >>>
                if os.environ.get("SMTP_DEBUG", "").lower() in ("1", "true", "yes"):
                    print(f"\n[SMTP DEBUG] OUTGOING HTML for msg_id={msg_id}:\n{html}\n")
                smtp.send_message(msg)
        except Exception as exc:
            raise RuntimeError(
                (
                    f"Failed to connect or send email via SMTP server at "
                    f"{self._host}:{self._port}: {exc}"
                )
            ) from exc


__all__ = ["SMTPSender"]
