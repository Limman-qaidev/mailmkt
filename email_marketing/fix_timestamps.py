"""Normalize timestamp formats in legacy databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1] / "data"
EVENTS_DB = BASE_DIR / "email_events.db"
MAP_DB = BASE_DIR / "email_map.db"


def _ensure_email_map_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(email_map)")}
    if "send_ts" not in cols:
        conn.execute("ALTER TABLE email_map ADD COLUMN send_ts TEXT")
    if "campaign" not in cols:
        conn.execute("ALTER TABLE email_map ADD COLUMN campaign TEXT")


def fix_events() -> None:
    if EVENTS_DB.exists():
        with sqlite3.connect(EVENTS_DB) as conn:
            conn.execute(
                "UPDATE events SET ts = REPLACE(ts, 'T', ' ') "
                "WHERE ts LIKE '%T%'"
            )
            conn.commit()


def fix_email_map() -> None:
    if MAP_DB.exists():
        with sqlite3.connect(MAP_DB) as conn:
            _ensure_email_map_columns(conn)
            conn.execute(
                "UPDATE email_map SET send_ts = REPLACE(send_ts, 'T', ' ') "
                "WHERE send_ts LIKE '%T%'"
            )
            conn.commit()


def main() -> None:
    fix_events()
    fix_email_map()


if __name__ == "__main__":
    main()
