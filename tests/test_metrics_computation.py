import pandas as pd

from email_marketing.analytics.metrics import compute_campaign_metrics


def test_compute_campaign_metrics_handles_universe_and_rates() -> None:
    sends = pd.DataFrame(
        {
            "campaign": ["C"],
            "msg_id": ["m1"],
            "email": ["a@example.com"],
            "send_ts": [pd.Timestamp("2020-01-01 00:00:00")],
        }
    )

    events = pd.DataFrame(
        {
            "campaign": ["C", "C", "C"],
            "msg_id": ["m1", "m1", "m2"],
            "event_type": ["open", "click", "open"],
            "event_ts": pd.to_datetime(
                ["2020-01-01 00:01:00", "2020-01-01 00:02:00", "2020-01-01 00:03:00"]
            ),
        }
    )

    signups = pd.DataFrame(
        {
            "campaign": ["C", "C"],
            "email": ["a@example.com", "a@example.com"],
            "signup_ts": pd.to_datetime(
                ["2020-01-01 00:04:00", "2019-12-31 23:50:00"]
            ),
        }
    )

    metrics = compute_campaign_metrics(sends, events, signups)
    m = metrics.loc["C"]

    # Only events matching sends should count (m2 open ignored)
    assert m["N_opens"] == 1
    assert m["N_clicks"] == 1

    # Only signup after send_ts should be attributed
    assert m["N_signups_attr"] == 1

    # Rates are bounded
    for col in [
        "open_rate",
        "ctr",
        "ctor",
        "unsubscribe_rate",
        "signup_rate",
        "signup_rate_per_send",
    ]:
        assert 0 <= m[col] <= 1

