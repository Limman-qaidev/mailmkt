"""Computation of raw campaign metrics."""

from __future__ import annotations

import pandas as pd

from . import db


def compute_campaign_metrics() -> pd.DataFrame:
    """Aggregate engagement metrics for each campaign.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``campaign`` with columns ``opens``, ``clicks``
        and ``signups``.
    """
    events = db.load_event_log()
    sends = db.load_send_log()
    signups = db.load_user_signups()

    if events.empty or sends.empty:
        return pd.DataFrame(
            columns=["campaign", "opens", "clicks", "signups"])\
            .set_index("campaign")

    merged = events.merge(sends, on="msg_id", how="left")
    metrics = (
        merged.pivot_table(
            index="campaign",
            columns="event_type",
            values="msg_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .rename(columns={"open": "opens", "click": "clicks"})
        .reset_index()
    )

    if not signups.empty:
        signup_counts = signups.groupby("campaign_id")["email"].nunique()
        metrics = metrics.merge(
            signup_counts.rename("signups"),
            left_on="campaign",
            right_index=True,
            how="left",
        )
    metrics["signups"] = metrics["signups"].fillna(0).astype(int)
    metrics = metrics.set_index("campaign")
    return metrics[["opens", "clicks", "signups"]]
