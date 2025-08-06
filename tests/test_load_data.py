import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from email_marketing.analytics import db, metrics


def _create_events_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE events (campaign_id TEXT, msg_id TEXT, "
        "event_type TEXT, event_ts TIMESTAMP)"
    )
    cur.executemany(
        "INSERT INTO events VALUES (?,?,?,?)",
        [
            ("c1", "m1", "open", "2020-01-01 00:00:00"),
            ("c1", "m1", "click", "2020-01-01 00:01:00"),
            ("c1", "m1", "unsubscribe", "2020-01-01 00:02:00"),
            ("c1", "m1", "complaint", "2020-01-01 00:03:00"),
        ],
    )
    conn.commit()
    conn.close()


def _create_sends_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE send_log (campaign_id TEXT, msg_id TEXT, "
        "email TEXT, send_ts TIMESTAMP)"
    )
    cur.execute(
        "INSERT INTO send_log VALUES (?,?,?,?)",
        ("c1", "m1", "a@example.com", "2020-01-01 00:00:00"),
    )
    conn.commit()
    conn.close()


def _create_campaigns_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE campaigns (campaign_id TEXT, name TEXT, "
        "start_date TEXT, end_date TEXT, budget REAL)"
    )
    cur.execute(
        "CREATE TABLE user_signup (signup_id TEXT, campaign_id TEXT, "
        "client_name TEXT, email TEXT)"
    )
    cur.execute(
        "INSERT INTO campaigns VALUES (?,?,?,?,?)",
        ("c1", "Test", "2020-01-01", "2020-01-02", 100.0),
    )
    cur.execute(
        "INSERT INTO user_signup VALUES (?,?,?)",
        ("s1", "c1", "Alice", "a@example.com"),
    )
    conn.commit()
    conn.close()


def test_load_all_data(tmp_path: Path) -> None:
    events_db = tmp_path / "events.db"
    sends_db = tmp_path / "sends.db"
    campaigns_db = tmp_path / "campaigns.db"

    _create_events_db(events_db)
    _create_sends_db(sends_db)
    _create_campaigns_db(campaigns_db)

    events, sends, campaigns, signups = db.load_all_data(
        str(events_db), str(sends_db), str(campaigns_db)
    )

    assert not events.empty
    assert "campaign_id" in events.columns
    assert not sends.empty
    assert not campaigns.empty

    metrics_df = metrics.compute_campaign_metrics(sends, events, signups)
    assert "open_rate" in metrics_df.columns
    assert metrics_df.loc["c1", "N_open_signups"] == 1
    assert metrics_df.loc["c1", "open_signup_rate"] == 1.0
