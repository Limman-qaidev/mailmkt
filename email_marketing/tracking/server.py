from __future__ import annotations

"""Web server for tracking email engagement events.

This module exposes a typed API using FastAPI.  It records events such as
opens (via a pixel), clicks, unsubscribes and complaints into a local
SQLite database.  The server can be run standalone::

    uvicorn email_marketing.tracking.server:app --reload

or embedded inside the Streamlit dashboard through :func:`create_app`.
"""
# email_marketing/tracking/server.py
"""Web server for tracking email engagement events (SQLite local / Postgres Neon)."""

import base64
import datetime as dt
from datetime import datetime
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, EmailStr

# Optional Postgres support (Neon)
PG_URL = os.getenv("DATABASE_URL", "").strip()
try:
    import psycopg  # psycopg 3
except Exception:
    psycopg = None  # will stay None if not installed


def _now_ts_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


# ---------- Storage backend (SQLite or Postgres) ----------
class _Store:
    def __init__(self) -> None:
        self.kind = "pg" if (PG_URL and psycopg) else "sqlite"
        if self.kind == "sqlite":
            base_dir: Path = Path(__file__).resolve().parent.parent
            self.events_db = str(base_dir / "data" / "email_events.db")
            self.map_db = str(base_dir / "data" / "email_map.db")
            os.makedirs(os.path.dirname(self.events_db), exist_ok=True)
            self._init_sqlite()
        else:
            self._init_pg()

    # ---- SQLite ----
    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.events_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    client_ip TEXT,
                    ts TEXT NOT NULL,
                    campaign TEXT
                )
                """
            )
            # ensure 'campaign' exists (old DBs)
            cols = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
            if "campaign" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN campaign TEXT")
            conn.commit()

    # ---- Postgres (Neon) ----
    def _pg_conn(self):
        if not psycopg:
            raise RuntimeError("psycopg is not installed but DATABASE_URL is set.")
        # Force SSL on Neon if not present
        dsn = PG_URL if "sslmode=" in PG_URL else (PG_URL + ("&" if "?" in PG_URL else "?") + "sslmode=require")
        return psycopg.connect(dsn)

    def _init_pg(self) -> None:
        with self._pg_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id BIGSERIAL PRIMARY KEY,
                    msg_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    client_ip TEXT,
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    campaign TEXT
                )
                """
            )
            # email_map debe existir porque lo migraste; no la creamos aquí para no pisarla.
            conn.commit()

    # ---- Shared ops ----
    def fetch_send_ts(self, msg_id: str) -> Optional[dt.datetime]:
        """Return send_ts from email_map if available, else None."""
        try:
            if self.kind == "sqlite":
                if not os.path.exists(self.map_db):
                    return None
                with sqlite3.connect(self.map_db) as conn:
                    row = conn.execute(
                        "SELECT send_ts FROM email_map WHERE msg_id=? LIMIT 1", (msg_id,)
                    ).fetchone()
                if not row or not row[0]:
                    return None
                try:
                    return dt.datetime.fromisoformat(row[0])  # ISO text -> datetime
                except Exception:
                    return None
            else:
                with self._pg_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT send_ts FROM email_map WHERE msg_id=%s LIMIT 1", (msg_id,))
                    row = cur.fetchone()
                if not row or row[0] is None:
                    return None
                val = row[0]
                # Neon might return timestamp or string; normalize
                if isinstance(val, dt.datetime):
                    return val.replace(tzinfo=None)
                try:
                    return dt.datetime.fromisoformat(str(val))
                except Exception:
                    return None
        except Exception:
            return None

    def insert_event(self, msg_id: str, event_type: str, client_ip: Optional[str], campaign: Optional[str]) -> None:
        if self.kind == "sqlite":
            with sqlite3.connect(self.events_db) as conn:
                conn.execute(
                    "INSERT INTO events (msg_id, event_type, client_ip, ts, campaign) VALUES (?, ?, ?, ?, ?)",
                    (msg_id, event_type, client_ip, _now_ts_str(), campaign),
                )
                conn.commit()
        else:
            with self._pg_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO events (msg_id, event_type, client_ip, campaign) VALUES (%s, %s, %s, %s)",
                    (msg_id, event_type, client_ip, campaign),
                )
                conn.commit()


store = _Store()


# ---------- FastAPI app ----------
app = FastAPI(title="Mailmkt Tracking API")


class ClickEvent(BaseModel):
    msg_id: str
    url: str


class MsgEvent(BaseModel):
    msg_id: str


class SubscribeRequest(BaseModel):
    email: EmailStr


def _should_count_open(request: Request) -> bool:
    """Avoid counting Gmail proxy prefetch as opens."""
    ua = request.headers.get("User-Agent", "")
    if "GoogleImageProxy" in ua and not request.headers.get("X-Forwarded-For"):
        return False
    return True


def _record_event(msg_id: str, event_type: str, client_ip: Optional[str], campaign: Optional[str]) -> None:
    # Optional grace period for 'open' after send
    send_ts = store.fetch_send_ts(msg_id)
    if event_type == "open" and send_ts is not None:
        grace_seconds = int(os.environ.get("OPEN_EVENT_GRACE_PERIOD_SECONDS", "30"))
        if (dt.datetime.utcnow() - send_ts) < dt.timedelta(seconds=grace_seconds):
            return
    store.insert_event(msg_id, event_type, client_ip, campaign)


# --- Pixel (1x1 gif) ---
@app.get("/pixel", response_class=Response, summary="Tracking pixel")
async def pixel(request: Request, msg_id: str, campaign: Optional[str] = None, ts: Optional[str] = None) -> Response:
    client_ip = request.client.host if request.client else None
    if _should_count_open(request):
        _record_event(msg_id, "open", client_ip, campaign)

    gif_b64 = "R0lGODlhAQABAPAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return Response(content=base64.b64decode(gif_b64), media_type="image/gif", headers=headers)


@app.head("/pixel", include_in_schema=False)
async def pixel_head(request: Request, msg_id: str) -> Response:
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "image/gif",
    }
    return Response(status_code=200, headers=headers)


# --- Click ---
@app.get("/click", response_class=RedirectResponse, summary="Record click via GET and redirect")
async def click_get(request: Request, msg_id: str, url: str, campaign: Optional[str] = None) -> RedirectResponse:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "click", client_ip, campaign)
    return RedirectResponse(url)


@app.head("/click", include_in_schema=False)
async def click_head(request: Request, msg_id: str, url: str, campaign: Optional[str] = None) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "click", client_ip, campaign)
    return Response(status_code=307, headers={"Location": url})


# --- Unsubscribe ---
@app.get("/unsubscribe", response_class=PlainTextResponse, summary="Handle unsubscribe via GET")
async def unsubscribe_get(request: Request, msg_id: str, campaign: Optional[str] = None) -> PlainTextResponse:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return PlainTextResponse("You have been unsubscribed")


@app.head("/unsubscribe", include_in_schema=False)
async def unsubscribe_head(request: Request, msg_id: str, campaign: Optional[str] = None) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return Response(status_code=200)


# --- Complaint ---
@app.get("/complaint", response_class=PlainTextResponse, summary="Handle complaint via GET")
async def complaint_get(request: Request, msg_id: str, campaign: Optional[str] = None) -> PlainTextResponse:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "complaint", client_ip, campaign)
    return PlainTextResponse("Thank you, your complaint has been recorded")


@app.head("/complaint", include_in_schema=False)
async def complaint_head(request: Request, msg_id: str, campaign: Optional[str] = None) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "complaint", client_ip, campaign)
    return Response(status_code=200)


# --- Logo (counts open + serves PNG) ---
@app.get("/logo", summary="Serve corporate logo and record open", response_class=FileResponse)
async def logo(request: Request, msg_id: str, ts: Optional[str] = None, campaign: Optional[str] = None) -> FileResponse:
    client_ip = request.client.host if request.client else None
    if _should_count_open(request):
        _record_event(msg_id, "open", client_ip, campaign)
    logo_path = Path(__file__).resolve().parent / "static" / "corporate_logo.png"
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return FileResponse(path=logo_path, media_type="image/png", headers=headers)


@app.head("/logo", include_in_schema=False)
async def logo_head(request: Request, msg_id: str, ts: Optional[str] = None) -> Response:
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "image/png",
    }
    return Response(status_code=200, headers=headers)


# --- Dummy subscribe/confirm (unchanged) ---
class SubscribeRequest(BaseModel):
    email: EmailStr


@app.post("/subscribe", response_class=PlainTextResponse, summary="Request double opt-in")
async def subscribe(req: SubscribeRequest) -> str:
    token = uuid.uuid4().hex
    return token


@app.get("/confirm/{token}", response_class=PlainTextResponse, summary="Confirm subscription")
async def confirm(token: str) -> str:
    return f"subscription confirmed: {token}"


def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app
