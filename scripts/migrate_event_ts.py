"""One-off migration script to normalise event timestamps and campaigns.

This utility updates the ``events`` table replacing ISO ``T`` separators with
spaces so that SQLite's default timestamp parser and pandas can read the
values.  It also backfills missing ``campaign`` values from ``email_map`` when
possible.  Run it once after deploying the new analytics pipeline::

    python scripts/migrate_event_ts.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1] / "email_marketing" / "data"
EVENTS_DB = BASE_DIR / "email_events.db"
MAP_DB = BASE_DIR / "email_map.db"


def migrate() -> None:
    if not EVENTS_DB.exists():
        return
    with sqlite3.connect(EVENTS_DB) as conn:
        # Normalise timestamp format
        conn.execute(
            "UPDATE events SET ts = REPLACE(ts, 'T', ' ') WHERE ts LIKE '%T%'"
        )

        # Backfill campaign from email_map when missing
        if MAP_DB.exists():
            conn.execute(f"ATTACH DATABASE '{MAP_DB}' AS mapdb")
            conn.execute(
                """
                UPDATE events
                   SET campaign = (
                       SELECT m.campaign FROM mapdb.email_map AS m
                        WHERE m.msg_id = events.msg_id
                   )
                 WHERE (campaign IS NULL OR campaign = '')
                   AND EXISTS (
                       SELECT 1 FROM mapdb.email_map AS m
                        WHERE m.msg_id = events.msg_id AND m.campaign IS NOT NULL
                   )
                """
            )
            conn.execute("DETACH DATABASE mapdb")

        conn.commit()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    migrate()

