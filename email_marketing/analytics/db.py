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

    The connection uses ``sqlite3.Row`` for convenient column access but does
    not enable any automatic type conversion.  All parsing is deferred to
    pandas.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _read_events(events_db: str) -> pd.DataFrame:
    """
    Read the events table, tolerating timestamps with 'T' (ISO) and without
    'T'.
    Don't use PARSE_DECLTYPES so SQLite doesn't try to convert before we do.
    """
    with sqlite3.connect(events_db) as conn:
        df = pd.read_sql_query(
            "SELECT msg_id, event_type, client_ip, ts, campaign FROM events",
            conn,
        )
    # Parse with pandas, which understands both formats
    df["event_ts"] = pd.to_datetime(
        df["ts"].str.replace("T", " ", regex=False), errors="coerce"
        )
    return df


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
    """Load the mapping from ``msg_id`` to recipient addresses.

    Two table layouts are supported:

    ``send_log`` with columns ``campaign_id``, ``msg_id``, ``email`` and
    ``send_ts``;
    and ``email_map`` with just ``msg_id`` and ``recipient``.  Column names are
    normalised so that at least ``msg_id`` and ``email`` are present in the
    returned DataFrame.  If neither table exists an empty DataFrame is
    returned.
    """

    db_path = path or MAP_DB

    with get_connection(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "send_log" in tables:
            df = pd.read_sql_query(
                "SELECT campaign_id, msg_id, email, send_ts FROM send_log",
                conn,
            )
        elif "email_map" in tables:
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(email_map)"
                    ).fetchall()
            }
            select_cols = ["msg_id", "recipient AS email"]
            if "variant" in cols:
                select_cols.append("variant")
            if "send_ts" in cols:
                select_cols.append("send_ts")
            if "campaign" in cols:
                select_cols.append("campaign")
            query = "SELECT " + ", ".join(select_cols) + " FROM email_map"
            df = pd.read_sql_query(query, conn)
        else:
            return pd.DataFrame()

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
    """Load and harmonise data from the three analytics databases.

    The function expects the following schemas:

    ``events``
        Columns ``msg_id``, ``event_type``, ``client_ip``, ``ts`` and
        ``campaign``.
    ``email_map``
        Columns ``msg_id`` and ``recipient``.
    ``campaigns`` and ``user_signup``
        Linked via ``campaign_id``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for events, sends, campaigns and signups respectively.
    """

    for path_str in (events_db, sends_db, campaigns_db):
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Database not found: {path_str}")

    # Load raw tables
    events = _read_events(events_db)
    sends = load_send_log(Path(sends_db))

    # Normalise column names
    if "event_ts" not in events.columns and "ts" in events.columns:
        events = events.rename(columns={"ts": "event_ts"})

    # Attach campaign information to sends and email addresses to events
    if not sends.empty:
        if "email" not in sends.columns and "recipient" in sends.columns:
            sends = sends.rename(columns={"recipient": "email"})
        if (
            "campaign_id" in events.columns and
            "campaign_id" not in sends.columns
        ):
            sends = sends.merge(
                events[["msg_id", "campaign_id"]].drop_duplicates(),
                on="msg_id",
                how="left",
            )
        if "campaign" in events.columns and "campaign" not in sends.columns:
            sends = sends.merge(
                events[["msg_id", "campaign"]].drop_duplicates(),
                on="msg_id",
                how="left",
            )
        if "email" not in events.columns:
            events = events.merge(
                sends[["msg_id", "email"]], on="msg_id", how="left"
            )

    # Reduce event columns to essentials
    keep_cols = [
        c
        for c in [
            "campaign_id",
            "campaign",
            "msg_id",
            "event_type",
            "event_ts",
            "email",
        ]
        if c in events.columns
    ]
    events = events[keep_cols]

    campaigns = _safe_read_query(
        Path(campaigns_db), "SELECT * FROM campaigns"
    )

    signups = _safe_read_query(
        Path(campaigns_db),
        "SELECT * FROM user_signup",
    ).merge(campaigns, on="campaign_id", how="left")
    signups = signups.rename(columns={"name": "campaign"})

    # Después de leer signups
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(signups["signup_ts"],
                                              errors="coerce")

    # Si envías el send_ts desde email_map
    if "send_ts" in sends.columns:
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")

    # Si events trae 'ts' (texto), ya sea aquí o en la vista:
    if "ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")

    return events, sends, campaigns, signups
