"""Feature engineering for campaign recommendation models."""

from __future__ import annotations

import pandas as pd

from . import db


FEATURE_COLUMNS = ["opens", "clicks", "time_to_first_open"]


def _compute_time_to_first_open(
        events: pd.DataFrame,
        sends: pd.DataFrame
        ) -> pd.Series:
    """Return time from send to first open for each recipient in seconds."""
    open_events = events[events["event_type"] == "open"].copy()
    if open_events.empty:
        return pd.Series(dtype=float)
    first_open = open_events.groupby("msg_id")["ts"].min()
    merged = first_open.to_frame().merge(
        sends[["msg_id", "send_ts"]], left_index=True, right_on="msg_id"
    )
    delta = (pd.to_datetime(merged["ts"]) - pd.to_datetime(merged["send_ts"]))
    return delta.dt.total_seconds().rename("time_to_first_open")


def build_features_for_campaign(campaign_id: str) -> pd.DataFrame:
    """Build per-recipient features for ``campaign_id``.

    Parameters
    ----------
    campaign_id:
        Identifier of the campaign to analyse.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``email`` plus those defined in
        :data:`FEATURE_COLUMNS` and ``signed_up``.
    """
    events = db.load_event_log()
    sends = db.load_send_log()
    signups = db.load_user_signups()

    if sends.empty:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    sends = sends[sends["campaign"] == campaign_id]
    if sends.empty:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    relevant_events = events[events["campaign"] == campaign_id]
    opens = (
        relevant_events[relevant_events["event_type"] == "open"]
        .groupby("msg_id")
        .size()
        .rename("opens")
    )
    clicks = (
        relevant_events[relevant_events["event_type"] == "click"]
        .groupby("msg_id")
        .size()
        .rename("clicks")
    )
    time_to_open = _compute_time_to_first_open(relevant_events, sends)

    feats = sends.set_index("msg_id").join([opens, clicks, time_to_open])
    feats[FEATURE_COLUMNS] = feats[FEATURE_COLUMNS].fillna(0)

    if not signups.empty:
        signup_flag = (
            signups[signups["campaign_id"] == campaign_id]
            .set_index("email")
            .assign(signed_up=1)
        )
        feats = feats.join(signup_flag, on="recipient")
    feats["signed_up"] = feats["signed_up"].fillna(0).astype(int)

    return feats.reset_index().rename(columns={"recipient": "email"})[
        ["email", *FEATURE_COLUMNS, "signed_up"]
    ]
