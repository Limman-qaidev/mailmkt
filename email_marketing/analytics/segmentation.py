import sqlite3
import pandas as pd


def generate_segments(
        events_db: str,
        email_map_db: str,
        campaigns_db: str
        ) -> dict[str, list[str]]:
    """
    Connects to the three SQLite databases and returns a dict with three keys:
    - "unsubscribed": list of recipient emails who ever unsubscribed
    (event_type="unsubscribe")
    - "interested": list of recipient emails who did not unsubscribe but have
    at least one "open" or "click" event OR appear in user_signup (only those
      signups whose email exists in email_map.recipient)
    - "not_interested": everyone else
    """
    # 1. Load event_log and email_map, merge on msg_id
    conn_ev = sqlite3.connect(events_db, detect_types=sqlite3.PARSE_DECLTYPES)
    events = pd.read_sql("SELECT msg_id, event_type FROM event_log", conn_ev)
    conn_ev.close()

    conn_map = sqlite3.connect(
        email_map_db, detect_types=sqlite3.PARSE_DECLTYPES
        )
    email_map = pd.read_sql(
        "SELECT msg_id, recipient FROM email_map",
        conn_map
        )
    conn_map.close()

    df = events.merge(email_map, on="msg_id", how="left")

    # 2. Load user_signup, filter to those in email_map.recipient
    conn_c = sqlite3.connect(
        campaigns_db,
        detect_types=sqlite3.PARSE_DECLTYPES
        )
    signups = pd.read_sql("SELECT email FROM user_signup", conn_c)
    conn_c.close()
    signups = signups[signups["email"].isin(email_map["recipient"])]

    # 3. Build segments
    unsub = set(df[df["event_type"] == "unsubscribe"]["recipient"].dropna())
    engaged = set(
        df[df["event_type"].isin(["open", "click"])]["recipient"].dropna()
        )
    signed = set(signups["email"])
    interested = (engaged | signed) - unsub
    all_recipients = set(email_map["recipient"].dropna())
    not_interested = all_recipients - unsub - interested

    return {
        "unsubscribed": sorted(unsub),
        "interested": sorted(interested),
        "not_interested": sorted(not_interested),
    }


if __name__ == "__main__":
    segments = generate_segments(
        "data/email_events.db",
        "data/email_map.db",
        "data/campaigns.db"
        )
    for segment, emails in segments.items():
        print(f"{segment}: {len(emails)}")
