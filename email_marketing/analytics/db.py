"""Database helpers for the analytics module.

These functions reuse the SQLite files already employed by the main
application.  Each helper returns a :class:`pandas.DataFrame` ready for
further processing by the analytics pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Paths to the existing SQLite databases
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EVENTS_DB = DATA_DIR / "email_events.db"
MAP_DB = DATA_DIR / "email_map.db"


def get_connection(path: str) -> sqlite3.Connection:
    """Return a connection to the SQLite database at ``path``.

    The connection uses ``sqlite3.Row`` as the row factory so columns can
    be accessed by name.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_read_query(path: Path, query: str) -> pd.DataFrame:
    """Execute ``query`` against ``path`` and return a DataFrame.

    If the database or table does not exist, an empty DataFrame with the
    appropriate columns is returned instead of raising an exception.
    """
    if not path.exists():
        return pd.DataFrame()
    with get_connection(str(path)) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame()


def load_event_log(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the email event log.

    Parameters
    ----------
    path:
        Optional override for the database file.  Defaults to
        :data:`EVENTS_DB`.
    """
    db_path = path or EVENTS_DB
    return _safe_read_query(db_path, "SELECT * FROM events")


def load_send_log(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the send log mapping ``msg_id`` to recipients and campaigns."""
    db_path = path or MAP_DB
    return _safe_read_query(db_path, "SELECT * FROM email_map")


def load_campaigns(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the list of campaigns.

    Expects a table named ``campaigns`` with at least the columns
    ``campaign_id`` and ``name``.  If unavailable, an empty DataFrame is
    returned.
    """
    db_path = path or MAP_DB
    return _safe_read_query(db_path, "SELECT * FROM campaigns")


def load_user_signups(path: Optional[Path] = None) -> pd.DataFrame:
    """Load user signup information.

    Expects a table ``user_signups`` with columns ``email`` and
    ``campaign_id``.  Missing tables yield an empty DataFrame.
    """
    db_path = path or MAP_DB
    return _safe_read_query(db_path, "SELECT * FROM user_signups")
