import sqlite3

import polars as pl


def generate_segments(
    events_db: str,
    email_map_db: str,
    campaigns_db: str,
) -> dict[str, list[str]]:
    """Return unsubscribed/interested/not_interested recipient segments."""
    conn_ev = sqlite3.connect(events_db, detect_types=sqlite3.PARSE_DECLTYPES)
    conn_map = sqlite3.connect(email_map_db, detect_types=sqlite3.PARSE_DECLTYPES)
    conn_c = sqlite3.connect(campaigns_db, detect_types=sqlite3.PARSE_DECLTYPES)

    try:
        events = pl.read_database("SELECT msg_id, event_type FROM event_log", conn_ev)
        email_map = pl.read_database("SELECT msg_id, recipient FROM email_map", conn_map)
        signups = pl.read_database("SELECT email FROM user_signup", conn_c)
    finally:
        conn_ev.close()
        conn_map.close()
        conn_c.close()

    if events.is_empty() or email_map.is_empty():
        return {"unsubscribed": [], "interested": [], "not_interested": []}

    df = events.join(email_map, on="msg_id", how="left")
    valid_recipients = email_map.select("recipient").drop_nulls().unique()
    signups = signups.filter(pl.col("email").is_in(valid_recipients.get_column("recipient")))

    unsub = set(
        df.filter(pl.col("event_type") == "unsubscribe")
        .get_column("recipient")
        .drop_nulls()
        .to_list()
    )
    engaged = set(
        df.filter(pl.col("event_type").is_in(["open", "click"]))
        .get_column("recipient")
        .drop_nulls()
        .to_list()
    )
    signed = set(signups.get_column("email").drop_nulls().to_list()) if not signups.is_empty() else set()
    interested = (engaged | signed) - unsub
    all_recipients = set(valid_recipients.get_column("recipient").to_list())
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
        "data/campaigns.db",
    )
    for segment, emails in segments.items():
        print(f"{segment}: {len(emails)}")
