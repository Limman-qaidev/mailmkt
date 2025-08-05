"""Database helpers for the analytics module.

These functions reuse the SQLite files already employed by the main
application.  Each helper returns a :class:`pandas.DataFrame` ready for
further processing by the analytics pipeline.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Paths to the existing SQLite databases
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EVENTS_DB = DATA_DIR / "email_events.db"
MAP_DB = DATA_DIR / "email_map_old.db"
CAMPAIGNS_DB = DATA_DIR / "campaigns.db"

LOGGER = logging.getLogger(__name__)


def get_connection(path: str) -> sqlite3.Connection:
    """Return a connection to the SQLite database at ``path``.

    The connection uses ``sqlite3.Row`` as the row factory and enables
    ``PARSE_DECLTYPES`` so SQLite types are converted into appropriate
    Python objects (e.g. ``datetime``).
    """
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_read_query(path: Path, query: str) -> pd.DataFrame:
    """Execute ``query`` against ``path`` and return a DataFrame.

    Parameters
    ----------
    path:
        Location of the SQLite database.
    query:
        SQL statement to execute.

    Returns
    -------
    pd.DataFrame
        Result of the query.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    sqlite3.DatabaseError
        If executing the query fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    with get_connection(str(path)) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except sqlite3.DatabaseError as exc:  # pragma: no cover - defensive
            LOGGER.error("Query failed for %s: %s", path, exc)
            raise


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
    """Load the send log mapping ``msg_id`` to recipients and campaigns.

    The table is expected to expose ``campaign_id``, ``msg_id`` and
    ``email`` columns.  Missing tables yield an empty DataFrame.
    """
    db_path = path or MAP_DB
    return _safe_read_query(
        db_path, "SELECT campaign_id, msg_id, email, send_ts FROM send_log"
    )


def load_campaigns(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the list of campaigns.

    Expects a table named ``campaigns`` with at least the columns
    ``campaign_id`` and ``name``.  If unavailable, an empty DataFrame is
    returned.
    """
    db_path = path or CAMPAIGNS_DB
    return _safe_read_query(db_path, "SELECT * FROM campaigns")


def load_user_signups(path: Optional[Path] = None) -> pd.DataFrame:
    """Load user signup information.

    Expects a table ``user_signup`` with columns ``email`` and
    ``campaign_id``.  Missing tables yield an empty DataFrame.
    """
    db_path = path or CAMPAIGNS_DB
    return _safe_read_query(db_path, "SELECT * FROM user_signup")


def load_all_data(
    events_db: str, sends_db: str, campaigns_db: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load core tables from the three analytics databases.

    Parameters
    ----------
    events_db:
        Path to the database containing ``event_log``.
    sends_db:
        Path to the database containing ``send_log``.
    campaigns_db:
        Path to the database containing ``campaigns`` and ``user_signup``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for events, sends, campaigns and signups respectively,
        with column structures::

            events   (campaign_id, msg_id, event_type, event_ts)
            sends    (campaign_id, msg_id, email, send_ts)
            campaigns(campaign_id, name, start_date, end_date, budget)
            signups  (signup_id, campaign_id, client_name, email)
    """
    for path_str in (events_db, sends_db, campaigns_db):
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Database not found: {path_str}")

    events = _safe_read_query(
        Path(events_db),
        "SELECT campaign_id, msg_id, event_type, event_ts FROM events",
    )
    sends = _safe_read_query(
        Path(sends_db),
        "SELECT campaign_id, msg_id, email, send_ts FROM send_log",
    )
    campaigns = _safe_read_query(
        Path(campaigns_db),
        (
            "SELECT campaign_id, name, start_date, end_date, budget "
            "FROM campaigns"
        ),
    )
    signups = _safe_read_query(
        Path(campaigns_db),
        "SELECT signup_id, campaign_id, client_name, email FROM user_signup",
    )
    return events, sends, campaigns, signups
