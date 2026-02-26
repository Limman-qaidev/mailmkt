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

def _get_db_path() -> str:
    # Base dir: .../email_marketing
    base_dir: Path = Path(__file__).resolve().parent.parent
    # Events DB in .../email_marketing/data/email_events.db
    return str(base_dir / "data" / "email_events.db")


def _ensure_campaign_column() -> None:
    """
    Make sure the 'events' table has a 'campaign' column.
    If it doesn't exist, add it.
    """
    db = _get_db_path()
    with sqlite3.connect(db) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
        if "campaign" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN campaign TEXT")
            conn.commit()


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
            """
        )


def _record_event(
    msg_id: str, event_type: str, client_ip: Optional[str], campaign: Optional[str]
) -> None:
    """Insert a new event into the SQLite database."""
    with sqlite3.connect(_get_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO events
            (msg_id, event_type, client_ip, ts, campaign)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                event_type,
                client_ip,
                dt.datetime.utcnow().isoformat(),
                campaign,
            ),
        )
        conn.commit()


store = _Store()

_init_db()
_ensure_campaign_column()


class ClickEvent(BaseModel):
    msg_id: str
    url: str


class MsgEvent(BaseModel):
    msg_id: str


class SubscribeRequest(BaseModel):
    email: EmailStr


@app.get("/pixel", response_class=Response, summary="Tracking pixel")
async def pixel(
    request: Request,
    msg_id: str,
    campaign: Optional[str] = None,
    ts: Optional[str] = None,  # parámetro anti-cache
) -> Response:
    """Return a 1×1 GIF, record an 'open', y evitar caching."""
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "open", client_ip, campaign)

    # Decode the 1×1 transparent GIF
    gif_b64 = "R0lGODlhAQABAPAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    pixel_bytes = base64.b64decode(gif_b64)


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

    return Response(content=pixel_bytes, media_type="image/gif", headers=headers)


@app.head("/pixel", include_in_schema=False)
async def pixel_head(request: Request, msg_id: str) -> Response:
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "image/gif",
    }
    return Response(status_code=200, headers=headers)


@app.get("/click", response_class=RedirectResponse, summary="Record click via GET and redirect")
async def click_get(
    request: Request,
    msg_id: str,
    url: str,
    campaign: Optional[str] = None,
) -> RedirectResponse:
    """Record a click event via GET and redirect to the target URL."""
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "click", client_ip, campaign)
    return RedirectResponse(url)


@app.get("/unsubscribe", response_class=PlainTextResponse, summary="Handle unsubscribe via GET")
async def unsubscribe_get(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> PlainTextResponse:
    """Record an unsubscribe event via GET and confirm."""
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return PlainTextResponse("You have been unsubscribed")


@app.get("/complaint", response_class=PlainTextResponse, summary="Handle complaint via GET")
async def complaint_get(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> PlainTextResponse:
    """Record a spam complaint event via GET and confirm."""
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "complaint", client_ip, campaign)
    return PlainTextResponse("Thank you, your complaint has been recorded")


@app.get(
    "/logo",
    summary="Serve corporate logo and record open",
    response_class=FileResponse,
)
async def logo(
    request: Request, msg_id: str, ts: Optional[str] = None, campaign: Optional[str] = None
) -> FileResponse:
    """
    Serve the corporate logo PNG and record an 'open' event.
    Query params:
    - msg_id: identifier of the message
    - ts: timestamp to bust proxy cache
    """
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "open", client_ip, campaign)

    logo_path = Path(__file__).resolve().parent / "static" / "logo.png"
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return FileResponse(path=logo_path, media_type="image/png", headers=headers)


@app.head("/logo", include_in_schema=False)
async def logo_head(request: Request, msg_id: str, ts: Optional[str] = None) -> Response:
    """
    Respond to HEAD so proxies (Gmail, Outlook) puedan validar la imagen.
    También grabamos el 'open' aquí.
    """
    # client_ip = request.client.host if request.client else None
    print(f"[DEBUG] LOGO HEAD HIT msg_id={msg_id}")
    # _record_event(msg_id, "open", client_ip)

    # Solo devolvemos cabeceras, sin cuerpo, con Content-Type + anti-cache
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "image/png",
    }
    return Response(status_code=200, headers=headers)


@app.post("/subscribe", response_class=PlainTextResponse, summary="Request double opt-in")
async def subscribe(req: SubscribeRequest) -> str:
    """Initiate a double opt‑in process.


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


# --- Click HEAD handler ---
@app.head("/click", include_in_schema=False)
async def click_head(
    request: Request, msg_id: str, url: str, campaign: Optional[str] = None
) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "click", client_ip, campaign)
    return Response(status_code=307, headers={"Location": url})


# --- Unsubscribe HEAD handler ---
@app.head("/unsubscribe", include_in_schema=False)
async def unsubscribe_head(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return Response(status_code=200)


# --- Complaint HEAD handler ---
@app.head("/complaint", include_in_schema=False)
async def complaint_head(request: Request, msg_id: str, campaign: Optional[str] = None) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "complaint", client_ip, campaign)
    return Response(status_code=200)
