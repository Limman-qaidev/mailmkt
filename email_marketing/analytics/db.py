# db.py
"""Database helpers for the analytics module.

If the environment variable NEON_URL is set, data is read from Postgres (Neon)
via SQLAlchemy. Otherwise we fall back to the existing local SQLite files.

Tables expected (same schema as your local DBs):
  - events(msg_id, event_type, client_ip, ts, campaign)
  - email_map(msg_id, recipient, variant?, send_ts?, campaign?)
    or send_log(campaign_id?, msg_id, email, send_ts)
  - campaigns(campaign_id, name, start_date, end_date, budget)
  - user_signup(signup_id, campaign_id, client_name, email, signup_ts?)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Paths to the existing SQLite databases
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EVENTS_DB = DATA_DIR / "email_events.db"
MAP_DB = DATA_DIR / "email_map.db"
CAMPAIGNS_DB = DATA_DIR / "campaigns.db"

LOGGER = logging.getLogger(__name__)

# ---------------------- Engine / routing ----------------------
_ENGINE: Engine | None = None


def _get_engine() -> Engine | None:
    """Create (once) and return a SQLAlchemy Engine if NEON_URL is set."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    url = os.getenv("NEON_URL", "").strip()
    if url:
        # Example: postgresql+psycopg://USER:PWD@HOST/DB?sslmode=require
        _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def _use_neon() -> bool:
    return bool(os.getenv("NEON_URL"))


# ---------------------- SQLite helpers ----------------------
def get_connection(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_read_query(path: Path, query: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    with get_connection(str(path)) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except sqlite3.DatabaseError as exc:  # pragma: no cover
            LOGGER.error("Query failed for %s: %s", path, exc)
            raise


def _read_events_sqlite(events_db: str) -> pd.DataFrame:
    with sqlite3.connect(events_db) as conn:
        df = pd.read_sql_query(
            "SELECT msg_id, event_type, client_ip, ts, campaign FROM events",
            conn,
        )
    # Normalize event_ts (works for 'YYYY-mm-dd HH:MM:SS' and with 'T')
    if "ts" in df.columns:
        df["event_ts"] = pd.to_datetime(
            df["ts"].astype(str).str.replace("T", " ", regex=False),
            errors="coerce",
        )
    return df


def load_event_log(path: Optional[Path] = None) -> pd.DataFrame:
    """SQLite fallback: load the email event log."""
    db_path = path or EVENTS_DB
    return _safe_read_query(db_path, "SELECT * FROM events")


def load_send_log(path: Optional[Path] = None) -> pd.DataFrame:
    """SQLite fallback: load send_log/email_map with normalized columns."""
    db_path = path or MAP_DB
    with get_connection(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "send_log" in tables:
            df = pd.read_sql_query(
                "SELECT campaign_id, msg_id, email, send_ts FROM send_log", conn
            )
        elif "email_map" in tables:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(email_map)").fetchall()}
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
    db_path = path or CAMPAIGNS_DB
    try:
        return _safe_read_query(
            db_path, "SELECT campaign_id, name, start_date, end_date, budget FROM campaigns"
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        LOGGER.warning("campaigns table missing in %s", db_path)
        return pd.DataFrame(columns=["campaign_id", "name", "start_date", "end_date", "budget"])


def load_user_signups(path: Optional[Path] = None) -> pd.DataFrame:
    db_path = path or CAMPAIGNS_DB
    try:
        return _safe_read_query(
            db_path,
            "SELECT signup_id, campaign_id, client_name, email, signup_ts FROM user_signup",
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        LOGGER.warning("user_signup table missing in %s", db_path)
        return pd.DataFrame(columns=["signup_id", "campaign_id", "client_name", "email", "signup_ts"])


# ---------------------- Postgres (Neon) helpers ----------------------
def _pg_table_exists(engine: Engine, table: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = :t
        LIMIT 1
    """)
    with engine.connect() as con:
        res = con.execute(q, {"t": table}).first()
        return res is not None


def _read_events_pg(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT msg_id, event_type, client_ip, ts, campaign FROM events",
        con=engine,
    )
    # Normalize event_ts
    if "ts" in df.columns:
        # ts might already be timestamp; coerce anyway
        df["event_ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    return df


def _read_sends_pg(engine: Engine) -> pd.DataFrame:
    if _pg_table_exists(engine, "send_log"):
        q = "SELECT campaign_id, msg_id, email, send_ts FROM send_log"
        df = pd.read_sql_query(q, con=engine)
    elif _pg_table_exists(engine, "email_map"):
        # Discover columns to keep parity with SQLite branch
        cols = pd.read_sql_query(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='email_map'
            """,
            con=engine,
        )["column_name"].tolist()
        select_cols = ["msg_id", "recipient AS email"]
        if "variant" in cols:
            select_cols.append("variant")
        if "send_ts" in cols:
            select_cols.append("send_ts")
        if "campaign" in cols:
            select_cols.append("campaign")
        q = "SELECT " + ", ".join(select_cols) + " FROM email_map"
        df = pd.read_sql_query(q, con=engine)
    else:
        df = pd.DataFrame()
    return df


def _read_campaigns_pg(engine: Engine) -> pd.DataFrame:
    if not _pg_table_exists(engine, "campaigns"):
        return pd.DataFrame(columns=["campaign_id", "name", "start_date", "end_date", "budget"])
    return pd.read_sql_query(
        "SELECT campaign_id, name, start_date, end_date, budget FROM campaigns",
        con=engine,
    )


def _read_signups_pg(engine: Engine) -> pd.DataFrame:
    if not _pg_table_exists(engine, "user_signup"):
        return pd.DataFrame(columns=["signup_id", "campaign_id", "client_name", "email", "signup_ts"])
    return pd.read_sql_query(
        "SELECT signup_id, campaign_id, client_name, email, signup_ts FROM user_signup",
        con=engine,
    )


# ---------------------- Orchestrator ----------------------
def load_all_data(
    events_db: str, sends_db: str, campaigns_db: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and harmonize data from either Neon (if NEON_URL) or local SQLite."""

    if _use_neon():
        engine = _get_engine()
        if engine is None:
            raise RuntimeError("NEON_URL is set but SQLAlchemy engine could not be created.")

        # --- Read raw tables from Postgres
        events = _read_events_pg(engine)
        sends = _read_sends_pg(engine)
        campaigns = _read_campaigns_pg(engine)
        signups = _read_signups_pg(engine)

    else:
        # --- SQLite (legacy/local)
        for path_str in (events_db, sends_db, campaigns_db):
            if not Path(path_str).exists():
                raise FileNotFoundError(f"Database not found: {path_str}")

        events = _read_events_sqlite(events_db)
        sends = load_send_log(Path(sends_db))
        campaigns = _safe_read_query(Path(campaigns_db), "SELECT * FROM campaigns")
        signups = _safe_read_query(Path(campaigns_db), "SELECT * FROM user_signup")

    # ---------------- Harmonize columns ----------------
    # events: ensure event_ts present
    if "event_ts" not in events.columns and "ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")

    # sends: ensure email col
    if not sends.empty and "email" not in sends.columns and "recipient" in sends.columns:
        sends = sends.rename(columns={"recipient": "email"})

    # attach email to events (via msg_id) if missing
    if "email" not in events.columns and not sends.empty:
        events = events.merge(sends[["msg_id", "email"]].drop_duplicates(), on="msg_id", how="left")

    # reduce event columns to essentials
    keep_cols = [c for c in ["campaign_id", "campaign", "msg_id", "event_type", "event_ts", "email"] if c in events.columns]
    events = events[keep_cols]

    # signups join campaigns for human-readable campaign name
    if not campaigns.empty and "campaign_id" in campaigns.columns:
        signups = signups.merge(campaigns, on="campaign_id", how="left")
        signups = signups.rename(columns={"name": "campaign"})

    # parse timestamps
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(signups["signup_ts"], errors="coerce")
    if "send_ts" in sends.columns:
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")

    return events, sends, campaigns, signups
