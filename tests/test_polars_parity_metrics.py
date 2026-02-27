import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from email_marketing.analytics.metrics import compute_campaign_metrics


def test_compute_campaign_metrics_parity_shape_and_values() -> None:
    sends = pd.DataFrame(
        {
            "campaign": ["A", "A", "B"],
            "msg_id": ["m1", "m2", "m3"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "send_ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        }
    )
    events = pd.DataFrame(
        {
            "campaign": ["A", "A", "A", "A", "B"],
            "msg_id": ["m1", "m1", "m2", "m2", "m3"],
            "event_type": ["open", "open", "open", "click", "unsubscribe"],
            "event_ts": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"]),
            "email": ["a@x.com", "a@x.com", "b@x.com", "b@x.com", "c@x.com"],
        }
    )
    signups = pd.DataFrame(
        {
            "campaign": ["A"],
            "email": ["b@x.com"],
            "signup_id": ["s1"],
            "signup_ts": pd.to_datetime(["2025-01-04"]),
        }
    )

    out = compute_campaign_metrics(sends, events, signups)

    assert set(["A", "B"]).issubset(set(out.index.astype(str)))
    assert out.loc["A", "N_sends"] == 2
    assert out.loc["A", "N_opens"] == 2
    assert out.loc["A", "N_clicks"] == 1
    assert out.loc["A", "N_signups_attr"] == 1
    assert out.loc["B", "N_sends"] == 1
    assert out.loc["B", "N_unsubscribes"] == 1
    assert (out["open_rate"] >= 0).all() and (out["open_rate"] <= 1).all()
    assert (out["signup_rate_per_send"] >= 0).all() and (out["signup_rate_per_send"] <= 1).all()
