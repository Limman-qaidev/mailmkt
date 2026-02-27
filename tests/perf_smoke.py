"""Quick performance smoke checks for local profiling (non-blocking).

Run:
    python tests/perf_smoke.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "email_marketing")

from email_marketing.analytics.metrics import compute_campaign_metrics
from email_marketing.dashboard.campaign_metrics_view import (
    _cached_campaign_bundle_by_fp,
    _cached_period_daily_by_fp,
    _cached_period_metrics_by_fp,
    _db_fingerprint,
    _default_db_paths,
)
from email_marketing.dashboard.recipient_insights import (
    _prepare_ev_sd_with_topics,
    _prepare_events_with_email,
    _prepare_signups_with_topics,
    _build_campaign_topic_map,
    _topic_level_totals,
    _topic_email_rollup,
)


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"{label}: {dt:.4f}s")
    return out, dt


def main() -> None:
    events_db, sends_db, campaigns_db = _default_db_paths()
    fp = str(_db_fingerprint((events_db, sends_db, campaigns_db)))
    print("DB:", events_db, sends_db, campaigns_db)

    (bundle, _) = _timed(
        "bundle cold",
        lambda: _cached_campaign_bundle_by_fp(fp, events_db, sends_db, campaigns_db),
    )
    (_bundle2, _) = _timed(
        "bundle warm",
        lambda: _cached_campaign_bundle_by_fp(fp, events_db, sends_db, campaigns_db),
    )
    events, sends, campaigns, signups, _metrics = bundle

    (_m, _) = _timed("compute_campaign_metrics", lambda: compute_campaign_metrics(sends, events, signups))

    start = str(min(sends["send_ts"].min(), events["event_ts"].min()).date()) if not sends.empty and not events.empty else "2024-01-01"
    end = str(max(sends["send_ts"].max(), events["event_ts"].max()).date()) if not sends.empty and not events.empty else "2024-12-31"

    (_pm, _) = _timed(
        "period metrics cold",
        lambda: _cached_period_metrics_by_fp(fp, events_db, sends_db, campaigns_db, start, end, tuple(), "exclude"),
    )
    (_pd, _) = _timed(
        "period daily cold",
        lambda: _cached_period_daily_by_fp(fp, events_db, sends_db, campaigns_db, start, end, tuple(), "exclude"),
    )

    evp, _ = _timed("prepare_events_with_email", lambda: _prepare_events_with_email(events, sends))
    (ev_t_sd_t, t_prepare) = _timed("prepare_ev_sd_with_topics", lambda: _prepare_ev_sd_with_topics(evp, sends, campaigns))
    ev_t, sd_t = ev_t_sd_t
    camp2topic = _build_campaign_topic_map(campaigns)
    signups_t, _ = _timed("prepare_signups_with_topics", lambda: _prepare_signups_with_topics(signups, camp2topic))
    (_tt, _) = _timed("topic_level_totals", lambda: _topic_level_totals(ev_t, sd_t, signups_t))
    (_tr, _) = _timed("topic_email_rollup", lambda: _topic_email_rollup(ev_t, sd_t, signups_t))

    print("\nRows:")
    print(f"events={len(events)} sends={len(sends)} campaigns={len(campaigns)} signups={len(signups)}")
    print(f"ev_t={len(ev_t)} sd_t={len(sd_t)} signups_t={len(signups_t)}")
    print(f"\nKey hotspot prepare_ev_sd_with_topics={t_prepare:.4f}s")


if __name__ == "__main__":
    main()
