"""Background workflows and scheduled tasks.

This package collects long‑running operations that are triggered by
engagement events.  For example, follow‑up emails can be sent when a
recipient clicks a link, or cleaning tasks can be run periodically.
Workers are expected to be executed via an RQ or Celery worker.  See
``seed_polling.py`` for an example.
"""

from __future__ import annotations

__all__ = ["seed_polling"]
