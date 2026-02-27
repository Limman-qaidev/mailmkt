import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from email_marketing.analytics.db import load_all_data, load_period_data


def _seed_sqlite(events_db: Path, sends_db: Path, campaigns_db: Path) -> None:
    with sqlite3.connect(events_db) as conn:
        conn.execute("CREATE TABLE events (msg_id TEXT, event_type TEXT, client_ip TEXT, ts TEXT, campaign TEXT)")
        conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?)",
            [
                ("m1", "open", "1.1.1.1", "2025-01-01 10:00:00", "A"),
                ("m2", "click", "1.1.1.2", "2025-01-02 11:00:00", "B"),
            ],
        )
    with sqlite3.connect(sends_db) as conn:
        conn.execute("CREATE TABLE email_map (msg_id TEXT, recipient TEXT, campaign TEXT, send_ts TEXT)")
        conn.executemany(
            "INSERT INTO email_map VALUES (?,?,?,?)",
            [
                ("m1", "a@x.com", "A", "2025-01-01 09:00:00"),
                ("m2", "b@x.com", "B", "2025-01-02 09:00:00"),
            ],
        )
    with sqlite3.connect(campaigns_db) as conn:
        conn.execute("CREATE TABLE campaigns (campaign_id TEXT, name TEXT, start_date TEXT, end_date TEXT, budget REAL)")
        conn.execute("CREATE TABLE user_signup (signup_id TEXT, campaign_id TEXT, client_name TEXT, email TEXT, signup_ts TEXT)")
        conn.executemany(
            "INSERT INTO campaigns VALUES (?,?,?,?,?)",
            [
                ("cA", "A", "2025-01-01", "2025-01-10", 100.0),
                ("cB", "B", "2025-01-01", "2025-01-10", 100.0),
            ],
        )
        conn.execute("INSERT INTO user_signup VALUES (?,?,?,?,?)", ("s1", "cA", "Client", "a@x.com", "2025-01-03 00:00:00"))


def test_load_all_data_and_period_data_parity(tmp_path: Path) -> None:
    events_db = tmp_path / "events.db"
    sends_db = tmp_path / "sends.db"
    campaigns_db = tmp_path / "campaigns.db"
    _seed_sqlite(events_db, sends_db, campaigns_db)

    events, sends, campaigns, signups = load_all_data(str(events_db), str(sends_db), str(campaigns_db))
    assert not events.empty and not sends.empty and not campaigns.empty
    assert "event_ts" in events.columns
    assert "email" in sends.columns

    p_events, p_sends, _, _ = load_period_data(
        start=pd.Timestamp("2025-01-01 00:00:00"),
        end=pd.Timestamp("2025-01-01 23:59:59"),
        campaign_filter=["A"],
        filter_mode="include",
        events_db=str(events_db),
        sends_db=str(sends_db),
        campaigns_db=str(campaigns_db),
    )
    assert set(p_events["campaign"].astype(str).unique()) == {"A"}
    assert set(p_sends["campaign"].astype(str).unique()) == {"A"}
