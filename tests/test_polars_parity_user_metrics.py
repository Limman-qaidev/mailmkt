import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from email_marketing.analytics.user_metrics import build_user_aggregates, compute_eb_rates


def test_build_user_aggregates_and_eb_rates_parity_columns() -> None:
    events = pd.DataFrame(
        {
            "msg_id": ["m1", "m1", "m2"],
            "event_type": ["open", "click", "unsubscribe"],
            "event_ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
            "email": ["a@x.com", "a@x.com", "b@x.com"],
        }
    )
    sends = pd.DataFrame(
        {
            "msg_id": ["m1", "m2"],
            "email": ["a@x.com", "b@x.com"],
            "send_ts": pd.to_datetime(["2024-12-31", "2025-01-01"], utc=True),
        }
    )
    signups = pd.DataFrame({"signup_id": ["s1"], "email": ["a@x.com"]})

    users = build_user_aggregates(events, sends, signups)
    assert not users.empty
    assert {"email", "N_sends", "N_opens", "N_clicks", "N_unsubscribes", "N_signups"}.issubset(users.columns)

    row_a = users.loc[users["email"] == "a@x.com"].iloc[0]
    assert int(row_a["N_sends"]) == 1
    assert int(row_a["N_opens"]) == 1
    assert int(row_a["N_clicks"]) == 1
    assert int(row_a["N_signups"]) == 1

    eb = compute_eb_rates(users)
    assert {"open_rate_eb", "click_rate_eb", "unsub_rate_eb"}.issubset(eb.columns)
    assert (eb["open_rate_eb"] >= 0).all() and (eb["open_rate_eb"] <= 1).all()
