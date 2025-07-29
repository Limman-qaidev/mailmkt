"""Example task for polling a spam folder via IMAP.

This script demonstrates how one might monitor a mailbox for messages that
land in the spam folder.  Integrating this polling mechanism into RQ or
Celery allows automated responses when a spam complaint occurs.  The
implementation here only logs the presence of new messages and does not
perform any actions on them.
"""

from __future__ import annotations

import imaplib
import os
from typing import List, Tuple


def poll_spam_folder() -> List[Tuple[str, str]]:
    """Connect to an IMAP server and return message IDs from the spam folder.

    The IMAP server and credentials are read from environment variables:
    ``IMAP_HOST``, ``IMAP_PORT``, ``IMAP_USERNAME`` and ``IMAP_PASSWORD``.

    Returns:
        A list of tuples of the form ``(uid, subject)`` for each new
        message found in the spam folder.  The function does not delete or
        mark messages as read.
    """
    host = os.environ.get("IMAP_HOST")
    port = int(os.environ.get("IMAP_PORT", "993"))
    user = os.environ.get("IMAP_USERNAME")
    password = os.environ.get("IMAP_PASSWORD")
    if not all([host, user, password]):
        return []

    uids: List[Tuple[str, str]] = []
    with imaplib.IMAP4_SSL(host, port) as imap:
        imap.login(user, password)
        imap.select("Spam")  # Some providers use "Junk" or "Spam"
        typ, data = imap.search(None, "ALL")
        if typ != "OK" or not data or not data[0]:
            return []
        for uid in data[0].split():
            uid_str = uid.decode("utf-8")
            typ, msg_data = imap.fetch(uid, "(BODY[HEADER.FIELDS (SUBJECT)])")
            subject = ""
            if typ == "OK" and msg_data:
                # Extract subject line
                for part in msg_data:
                    if isinstance(part, tuple):
                        header = part[1].decode("utf-8", errors="ignore")
                        if header.lower().startswith("subject:"):
                            subject = header.split(":", 1)[1].strip()
                            break
            uids.append((uid_str, subject))
    return uids


if __name__ == "__main__":
    messages = poll_spam_folder()
    for uid, subject in messages:
        print(f"Spam message {uid}: {subject}")
