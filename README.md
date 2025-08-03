# Mailmkt – Modular Email Marketing Platform

Mailmkt is a self‑hosted, fully modular email marketing platform inspired by commercial tools such as HubSpot.  It provides an extensible framework for composing and sending campaigns, tracking engagement, running A/B tests, and orchestrating follow‑up workflows.  The project is written in **Python 3.13** with strict type annotations and is structured around clearly separated sub‑packages.

## Architecture Overview

The repository follows a layered architecture, with each major concern isolated into its own package:

```text
email_marketing/
├─ app.py                # Streamlit dashboard and embedded server launcher
├─ mailer/               # Sending abstractions for SMTP or Mailgun
│   ├─ __init__.py
│   ├─ smtp_sender.py    # Implements send_email() via smtplib
│   └─ mailgun_sender.py # Implements send_email() via Mailgun API
├─ tracking/             # Engagement tracking server (Flask/FastAPI)
│   ├─ __init__.py
│   └─ server.py
├─ dashboard/            # Streamlit components for the UI
│   ├─ __init__.py
│   ├─ email_editor.py   # Rich HTML editor and list uploader
│   ├─ stats_view.py     # Metric tiles and plots
│   └─ style.py          # Theme configuration and CSS injection
├─ workflows/            # Background job orchestration
│   ├─ __init__.py
│   └─ seed_polling.py   # Example IMAP polling task
├─ ab_testing/           # A/B assignment helpers
│   └─ __init__.py
├─ data/                 # SQLite databases and migrations
│   └─ __init__.py
├─ tests/                # Pytest based unit tests
└─ ci/                   # GitHub Actions workflows

root/
├─ requirements.txt      # Runtime and development dependencies
├─ pyproject.toml        # Configuration for black, isort and mypy
├─ Dockerfile            # Build the application container
├─ docker-compose.yml    # Orchestration for local development
└─ README.md             # Project documentation
```

### Data Stores

Two SQLite databases are used by default:

* `email_map.db` maps each `msg_id` to its recipient and A/B variant.
* `email_events.db` records events (`msg_id`, `event_type`, `client_ip`, `ts`).

Alembic can manage schema migrations; see `data/migrations/` for scripts.

### Tracking Service

The tracking API is served by Flask (or FastAPI) and exposes typed endpoints:

* **GET `/pixel`** – returns a 1×1 transparent GIF and logs a _display_ event.
* **POST `/click`** – records a click for a given `msg_id` and target URL.
* **POST `/unsubscribe`** – handles unsubscription requests.
* **POST `/complaint`** – records spam complaints.

These events feed the dashboard and can trigger workflows.

### Dashboard

The dashboard is implemented with Streamlit.  It includes:

* **Email editor** – upload recipient lists (CSV/XLSX) and craft HTML bodies.  The editor uses the theme defined in `style.py`.
* **Stats view** – summary metrics (opens, clicks, unsubscribe rate) with Matplotlib charts and a table of recent events.
* **Auto‑refresh** – the refresh interval is configurable via environment variables or the style module.

### Advanced Features

* **Double opt‑in** – `POST /subscribe` issues a confirmation token, and `GET /confirm/<token>` activates the subscriber.
* **A/B testing** – the `ab_testing` package assigns recipients to variants and extends `email_map.db` with a `variant` column.
* **Workflows** – background workers (via RQ or Celery) consume the event stream to schedule follow‑ups.  An example IMAP polling task (`workflows/seed_polling.py`) demonstrates how to check a spam folder.

## Getting Started

### Local Development

1. **Clone** this repository and create a virtual environment (Python 3.13).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run database migrations (optional).  For a fresh start the SQLite files are created automatically.
4. Launch the app:
   ```bash
   streamlit run email_marketing/app.py
   ```

The tracking server can be run separately with:
```bash
python -m email_marketing.tracking.server
```

### Docker Compose

For a full stack (dashboard, tracking service and Redis worker), use:

```bash
docker-compose up --build
```

Navigate to <http://localhost:8501> for the dashboard and <http://localhost:8000/docs> for the tracking API when using FastAPI.

### Environment Variables

The following environment variables configure runtime behaviour:

| Variable          | Description                                |
|-------------------|--------------------------------------------|
| `SMTP_HOST` / `SMTP_SERVER`           | Hostname of the SMTP server.  Either variable may be used.             |
| `SMTP_PORT` / `SMTP_SERVER_PORT`      | Port for the SMTP server.  Defaults to 587 if unset.                  |
| `SMTP_USERNAME` / `SMTP_USER`         | Username for SMTP authentication.  The application checks both names. |
| `MAILGUN_API_KEY` | API key for Mailgun                        |
| `MAILGUN_DOMAIN`  | Domain configured in Mailgun               |
| `TRACKING_URL`    | Base URL of the tracking server            |
| `REDIS_URL`       | Redis connection URI for workflows         |
| `REFRESH_INTERVAL`| Seconds between dashboard refreshes        |
| `OPEN_EVENT_GRACE_PERIOD_SECONDS` | Ignore opens occurring within this many seconds of sending |

When unspecified, sensible defaults are used or the feature is disabled.

## Contributing

This project uses **black**, **isort**, **flake8** and **mypy** for code quality.  A GitHub Actions workflow (in the `ci/` directory) verifies formatting, type checking and unit tests on every push.  Unit tests live in `tests/` and are run with `pytest`.  Test coverage should remain above 90 % to ensure reliability.

## License

This repository is provided for educational purposes and does not carry a warranty of fitness for production use.
