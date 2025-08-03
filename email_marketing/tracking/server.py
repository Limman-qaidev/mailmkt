
"""Web server for tracking email engagement events.

This module exposes a typed API using FastAPI.  It records events such as
opens (via a pixel), clicks, unsubscribes and complaints into a local
SQLite database.  The server can be run standalone::

    uvicorn email_marketing.tracking.server:app --reload

or embedded inside the Streamlit dashboard through :func:`create_app`.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, EmailStr


def _get_db_path() -> str:
    # Base dir: .../email_marketing
    base_dir: Path = Path(__file__).resolve().parent.parent
    # Events DB in .../email_marketing/data/email_events.db
    return str(base_dir / "data" / "email_events.db")


def _get_map_db_path() -> str:
    """Return path to the msg_id → recipient mapping database."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    return str(base_dir / "data" / "email_map.db")


def _ensure_campaign_column() -> None:
    """
    Make sure the 'events' table has a 'campaign' column.
    If it doesn't exist, add it.
    """
    db = _get_db_path()
    with sqlite3.connect(db) as conn:
        cols = [row[1] for row in conn.execute(
            "PRAGMA table_info(events)").fetchall()
            ]
        if "campaign" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN campaign TEXT")
            conn.commit()


def _init_db() -> None:
    os.makedirs(os.path.dirname(_get_db_path()), exist_ok=True)
    with sqlite3.connect(_get_db_path()) as conn:
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


def _record_event(msg_id: str,
                  event_type: str,
                  client_ip: Optional[str],
                  campaign: Optional[str]
                  ) -> None:
    """Insert a new event into the SQLite database."""
    send_ts: Optional[dt.datetime] = None
    map_db = _get_map_db_path()
    if os.path.exists(map_db):
        try:
            with sqlite3.connect(map_db) as conn_map:
                row = conn_map.execute(
                    "SELECT send_ts FROM email_map WHERE msg_id=?", (msg_id,)
                ).fetchone()
            if row and row[0]:
                send_ts = dt.datetime.fromisoformat(row[0])
        except sqlite3.Error:
            send_ts = None

    if event_type == "open" and send_ts is not None:
        grace_seconds = int(
            os.environ.get("OPEN_EVENT_GRACE_PERIOD_SECONDS", "60")
            )
        if (
            dt.datetime.utcnow() - send_ts < dt.timedelta(
                seconds=grace_seconds
            )
        ):
            return
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


app = FastAPI(title="Mailmkt Tracking API")

_init_db()


class ClickEvent(BaseModel):
    msg_id: str
    url: str


class MsgEvent(BaseModel):
    msg_id: str


class SubscribeRequest(BaseModel):
    email: EmailStr


def _should_count_open(request: Request) -> bool:
    """Return True if the request looks like a real user opening the email.

    Some mail providers (notably Gmail) prefetch remote images using a
    special user agent (``GoogleImageProxy``) as soon as the message is
    received.  This causes opens to be recorded before the recipient actually
    views the email.  To mitigate this, ignore requests from known proxy
    agents so only real client loads are counted as opens.
    """

    ua = request.headers.get("User-Agent", "")
    blocked_agents = ["GoogleImageProxy"]
    return not any(agent in ua for agent in blocked_agents)


@app.get("/pixel", response_class=Response,
         summary="Tracking pixel")
async def pixel(
        request: Request,
        msg_id: str,
        campaign: Optional[str] = None,
        ts: Optional[str] = None  # parámetro anti-cache
        ) -> Response:
    """Return a 1×1 GIF, record an 'open', y evitar caching."""
    client_ip = request.client.host if request.client else None
    if _should_count_open(request):
        _record_event(msg_id, "open", client_ip, campaign)

    # Decode the 1×1 transparent GIF
    gif_b64 = "R0lGODlhAQABAPAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    pixel_bytes = base64.b64decode(gif_b64)

    # Headers to force no-cache
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    return Response(content=pixel_bytes, media_type="image/gif",
                    headers=headers)


@app.head("/pixel", include_in_schema=False)
async def pixel_head(request: Request, msg_id: str) -> Response:
    headers = {
         "Cache-Control": "no-cache, no-store, must-revalidate",
         "Pragma": "no-cache",
         "Expires": "0",
         "Content-Type": "image/gif",
    }
    return Response(status_code=200, headers=headers)


@app.get("/click", response_class=RedirectResponse,
         summary="Record click via GET and redirect")
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


@app.get("/unsubscribe", response_class=PlainTextResponse,
         summary="Handle unsubscribe via GET")
async def unsubscribe_get(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> PlainTextResponse:
    """Record an unsubscribe event via GET and confirm."""
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return PlainTextResponse("You have been unsubscribed")


@app.get("/complaint", response_class=PlainTextResponse,
         summary="Handle complaint via GET")
async def complaint_get(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> PlainTextResponse:
    """Record a spam complaint event via GET and confirm."""
    client_ip = request.client.host if request.client else None
    # print(f"[DEBUG] COMPLAINT HIT msg_id={msg_id}")
    _record_event(msg_id, "complaint", client_ip, campaign)
    return PlainTextResponse("Thank you, your complaint has been recorded")


@app.get(
    "/logo",
    summary="Serve corporate logo and record open",
    response_class=FileResponse,
)
async def logo(
    request: Request,
    msg_id: str,
    ts: Optional[str] = None,
    campaign: Optional[str] = None
) -> FileResponse:
    """
    Serve the corporate logo PNG and record an 'open' event.
    Query params:
    - msg_id: identifier of the message
    - ts: timestamp to bust proxy cache
    """
    client_ip = request.client.host if request.client else None
    if _should_count_open(request):
        _record_event(msg_id, "open", client_ip, campaign)
    # Location of the static logo file
    logo_path = (
        Path(__file__).resolve().parent / "static" / "corporate_logo.png"
    )
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return FileResponse(path=logo_path, media_type="image/png",
                        headers=headers)


# HEAD handler para el logo, para que Gmail proxy valide correctamente
@app.head("/logo", include_in_schema=False)
async def logo_head(request: Request,
                    msg_id: str,
                    ts: Optional[str] = None) -> Response:
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


@app.post("/subscribe", response_class=PlainTextResponse,
          summary="Request double opt-in")
async def subscribe(req: SubscribeRequest) -> str:
    """Initiate a double opt‑in process.

    Generates a confirmation token and would normally send a confirmation
    email.  Here we simply return the token for demonstration purposes.
    """
    token = uuid.uuid4().hex
    return token


@app.get("/confirm/{token}", response_class=PlainTextResponse,
         summary="Confirm subscription")
async def confirm(token: str) -> str:
    """Confirm a subscription given a token generated by :func:`subscribe`."""
    return f"subscription confirmed: {token}"


def create_app() -> FastAPI:
    """Return the FastAPI application instance.

    Exposing this function allows the tracking server to be embedded in
    arbitrary host environments (e.g. Streamlit).  It simply returns the
    module level ``app``.
    """
    return app


# --- Click HEAD handler ---
@app.head("/click", include_in_schema=False)
async def click_head(
    request: Request, msg_id: str, url: str, campaign: Optional[str] = None
) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "click", client_ip, campaign)
    # Devolvemos sólo la cabecera de redirección (301/307) sin cuerpo
    return Response(status_code=307, headers={"Location": url})


# --- Unsubscribe HEAD handler ---
async def unsubscribe_head(
    request: Request, msg_id: str, campaign: Optional[str] = None
) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "unsubscribe", client_ip, campaign)
    return Response(status_code=200)


# --- Complaint HEAD handler ---
@app.head("/complaint", include_in_schema=False)
async def complaint_head(
    request: Request,
    msg_id: str,
    campaign: Optional[str] = None

) -> Response:
    client_ip = request.client.host if request.client else None
    _record_event(msg_id, "complaint", client_ip, campaign)
    return Response(status_code=200)
