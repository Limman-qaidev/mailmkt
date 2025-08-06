import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from email_marketing.analytics import db, metrics


def _create_events_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE events (msg_id TEXT, event_type TEXT, "
        "client_ip TEXT, ts TIMESTAMP, campaign TEXT)"
    )
    cur.executemany(
        "INSERT INTO events VALUES (?,?,?,?)",
        [
            ("m1", "open", "1.1.1.1", "2020-01-01 00:00:00", "Test"),
            ("m1", "click", "1.1.1.1", "2020-01-01 00:01:00", "Test"),
            ("m1", "unsubscribe", "1.1.1.1", "2020-01-01 00:02:00", "Test"),
            ("m1", "complaint", "1.1.1.1", "2020-01-01 00:03:00", "Test"),
        ],
    )
    conn.commit()
    conn.close()


def _create_sends_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE email_map ( msg_id TEXT PRIMARY KEY, "
        "recipient TEXT NOT NULL)"
    )
    cur.execute(
        "INSERT INTO email_map VALUES (?,?)",
        ("m1", "a@example.com"),
    )
    conn.commit()
    conn.close()


def _create_campaigns_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE campaigns (campaign_id TEXT, name TEXT)"
    )
    cur.execute(
        "CREATE TABLE user_signup (signup_id TEXT, campaign_id TEXT, "
        "client_name TEXT, email TEXT)"
    )
    cur.execute(
        "INSERT INTO campaigns VALUES (?,?)",
        ("c1", "Test"),
    )
    cur.execute(
        "INSERT INTO user_signup VALUES (?,?,?, ?)",
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
    assert "campaign" in events.columns
    assert "email" in events.columns
    assert not sends.empty
    assert "campaign" in sends.columns
    assert not campaigns.empty

    metrics_df = metrics.compute_campaign_metrics(sends, events, signups)
    assert "open_rate" in metrics_df.columns
    assert metrics_df.loc["Test", "N_open_signups"] == 1
    assert metrics_df.loc["Test", "open_signup_rate"] == 1.0


def test_default_data_dir() -> None:
    """The analytics module should look for databases under the package."""
    expected = Path(db.__file__).resolve().parent.parent / "data"
    assert db.DATA_DIR == expected
