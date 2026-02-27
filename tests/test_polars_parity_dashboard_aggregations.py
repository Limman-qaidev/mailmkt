import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "email_marketing") not in sys.path:
    sys.path.insert(0, str(ROOT / "email_marketing"))

from email_marketing.dashboard.campaign_metrics_view import _daily_series_from_frames
from email_marketing.dashboard.recipient_insights import (
    _map_topics_from_lookup,
    _recipient_topic_detail_from_rollup,
    _topic_email_rollup,
    _topic_level_totals,
    _topic_lookup_from_unique_values,
)


def test_daily_series_from_frames_parity() -> None:
    events = pd.DataFrame(
        {
            "msg_id": ["m1", "m2", "m2"],
            "event_type": ["open", "open", "click"],
            "event_ts": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-02 12:00:00"]),
        }
    )
    signups = pd.DataFrame(
        {
            "signup_id": ["s1"],
            "signup_ts": pd.to_datetime(["2025-01-02 15:00:00"]),
        }
    )
    out = _daily_series_from_frames(events, signups)
    assert {"date", "opens", "clicks", "signups"}.issubset(out.columns)
    assert float(out["opens"].sum()) == 2.0
    assert float(out["clicks"].sum()) == 1.0
    assert float(out["signups"].sum()) == 1.0


def test_topic_rollups_parity_columns() -> None:
    ev_t = pd.DataFrame(
        {
            "topic": ["Loans", "Loans", "Cards"],
            "email": ["a@x.com", "a@x.com", "b@x.com"],
            "event_type": ["open", "click", "unsubscribe"],
            "msg_id": ["m1", "m1", "m2"],
            "event_ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
        }
    )
    sd_t = pd.DataFrame(
        {
            "topic": ["Loans", "Cards"],
            "email": ["a@x.com", "b@x.com"],
            "msg_id": ["m1", "m2"],
        }
    )
    signups_t = pd.DataFrame(
        {
            "topic": ["Loans"],
            "topic_norm": ["loans"],
            "email": ["a@x.com"],
            "signup_id": ["s1"],
        }
    )

    totals = _topic_level_totals(ev_t, sd_t, signups_t)
    rollup = _topic_email_rollup(ev_t, sd_t, signups_t)

    assert {"topic", "topic_norm", "S_tot", "O_tot", "C_tot", "Y_ev_tot", "Y_tab_tot"}.issubset(totals.columns)
    assert {"topic", "topic_norm", "email", "S", "O", "C", "U", "Q", "Y_ev", "Y", "owner", "last_open_ts"}.issubset(rollup.columns)


def test_topic_lookup_unique_and_vectorized_mapping() -> None:
    topic_map = {"CMP_LOANS_001": "Loans"}
    values = [
        pd.Series(["CMP_LOANS_001", "Cards - Welcome", "Cards - Welcome", None]),
        pd.Series(["cmp_loans_001", "Mortgages: Intro"]),
    ]
    lookup = _topic_lookup_from_unique_values(values, topic_map)

    assert lookup["CMP_LOANS_001"] == "Loans"
    assert lookup["cmp_loans_001"] == "Loans"
    assert lookup["Cards - Welcome"] == "Cards"
    assert lookup["Mortgages: Intro"] == "Mortgages"

    df = pd.DataFrame({"campaign": ["CMP_LOANS_001", "Cards - Welcome", None]})
    mapped = _map_topics_from_lookup(df, lookup)
    assert mapped.tolist() == ["Loans", "Cards", ""]


def test_recipient_topic_detail_from_rollup_parity() -> None:
    rollup = pd.DataFrame(
        {
            "topic": ["Loans", "Cards", "Loans"],
            "topic_norm": ["loans", "cards", "loans"],
            "email": ["a@x.com", "a@x.com", "b@x.com"],
            "S": [10, 4, 2],
            "O": [7, 2, 1],
            "C": [3, 1, 0],
            "U": [0, 1, 0],
            "Q": [0, 0, 0],
            "Y_ev": [2, 0, 0],
            "Y": [1, 0, 0],
            "owner": [True, False, False],
        }
    )
    corpus = pd.DataFrame({"topic": ["Loans", "Cards"], "p0_signup": [0.2, 0.1]})

    out = _recipient_topic_detail_from_rollup("a@x.com", rollup, corpus, alpha=5.0)
    assert not out.empty
    assert {
        "topic",
        "N_sends_topic",
        "N_opens_topic",
        "N_clicks_topic",
        "N_unsubs_topic",
        "N_signups_topic",
        "open_rate",
        "ctr",
        "unsub_rate",
        "p_signup",
        "Registered",
    }.issubset(out.columns)

    loans = out.loc[out["topic"] == "Loans"].iloc[0]
    assert float(loans["N_signups_topic"]) == 2.0
    assert bool(loans["Registered"]) is True
    expected = (2.0 + 5.0 * 0.2) / (10.0 + 5.0)
    assert abs(float(loans["p_signup"]) - expected) < 1e-12
