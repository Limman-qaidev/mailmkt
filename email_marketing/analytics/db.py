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
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EVENTS_DB = DATA_DIR / "email_events.db"
MAP_DB = DATA_DIR / "email_map.db"
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

    The schema of the underlying table can vary between deployments.  We
    therefore attempt a couple of known options and normalise the result to
    expose at least ``msg_id`` and ``email`` columns.  If no recognised table
    is found, an empty :class:`~pandas.DataFrame` is returned.
    """

    db_path = path or MAP_DB

    queries = [
        "SELECT campaign_id, msg_id, email, send_ts FROM email_map",
        "SELECT msg_id, recipient AS email FROM email_map",
    ]

    for query in queries:
        try:
            df = _safe_read_query(db_path, query)
            break
        except sqlite3.DatabaseError:
            df = pd.DataFrame()
    else:  # pragma: no cover - defensive, should not happen
        df = pd.DataFrame()

    if df.empty:
        return df

    # Standardise column names if needed
    if "recipient" in df.columns and "email" not in df.columns:
        df = df.rename(columns={"recipient": "email"})
    if "campaign" in df.columns and "campaign_id" not in df.columns:
        df = df.rename(columns={"campaign": "campaign_id"})

    return df


def load_campaigns(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the list of campaigns.

    Expects a table named ``campaigns`` with at least the columns
    ``campaign_id`` and ``name``.  If unavailable, an empty DataFrame is
    returned.
    """
    db_path = path or CAMPAIGNS_DB
    try:
        return _safe_read_query(
            db_path,
            "SELECT campaign_id, name, start_date, end_date, budget "
            "FROM campaigns",
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        LOGGER.warning("campaigns table missing in %s", db_path)
        return pd.DataFrame(
            columns=["campaign_id", "name", "start_date", "end_date", "budget"]
        )


def load_user_signups(path: Optional[Path] = None) -> pd.DataFrame:
    """Load user signup information.

    Expects a table ``user_signup`` with columns ``email`` and
    ``campaign_id``.  Missing tables yield an empty DataFrame.
    """
    db_path = path or CAMPAIGNS_DB
    try:
        return _safe_read_query(
            db_path,
            "SELECT signup_id, campaign_id, client_name, email "
            "FROM user_signup",
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        LOGGER.warning("user_signup table missing in %s", db_path)
        return pd.DataFrame(
            columns=["signup_id", "campaign_id", "client_name", "email"]
        )


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

    # Load raw tables
    events = _safe_read_query(Path(events_db), "SELECT * FROM events")
    sends = load_send_log(Path(sends_db))

    # Normalise column names
    if "event_ts" not in events.columns and "ts" in events.columns:
        events = events.rename(columns={"ts": "event_ts"})

    if "campaign_id" not in events.columns and "campaign" in events.columns:
        events = events.rename(columns={"campaign": "campaign_id"})

    if not sends.empty:
        if "email" not in sends.columns and "recipient" in sends.columns:
            sends = sends.rename(columns={"recipient": "email"})
        if "campaign_id" not in sends.columns:
            sends = sends.merge(
                events[["msg_id", "campaign_id"]].drop_duplicates(),
                on="msg_id",
                how="left",
            )
        if "email" not in events.columns:
            events = events.merge(
                sends[["msg_id", "email"]], on="msg_id", how="left"
            )

    cols = ["campaign_id", "msg_id", "event_type", "event_ts"]
    if "email" in events.columns:
        cols.append("email")
    campaigns = load_campaigns(Path(campaigns_db))
    signups = load_user_signups(Path(campaigns_db))
    return events, sends, campaigns, signups
