"""Abstract interface and implementations for sending email messages.

This subpackage defines a common ``send_email`` interface along with two
concrete implementations: one using the standard SMTP protocol and another
targeting the Mailgun HTTP API.  Client code can select an implementation
based on configuration without changing the calling semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class EmailSender(ABC):
    """Abstract base class for email senders.

    Implementations must provide a ``send_email`` method.  The arguments
    mirror those commonly used in email marketing: a recipient address, a
    message identifier, the HTML body, an optional subject and an optional
    plain‑text alternative.
    """

    @abstractmethod
    def send_email(
        self,
        recipient: str,
        msg_id: str,
        html: str,
        subject: str = "",
        text: Optional[str] = None,
    ) -> None:
        """Send a single email message.

        Args:
            recipient: The target email address.
            msg_id: A unique identifier for this message; used to track events.
            html: The HTML content of the message.
            subject: The email subject line.
            text: Optional plain‑text version.

        Raises:
            Any implementation specific exceptions on failure.
        """
        raise NotImplementedError


__all__ = ["EmailSender"]
