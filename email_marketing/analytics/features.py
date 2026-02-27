"""Feature engineering for campaign recommendation models."""

from __future__ import annotations

import pandas as pd
import polars as pl

from . import db


FEATURE_COLUMNS = ["opens", "clicks", "time_to_first_open"]


def _compute_time_to_first_open(events: pd.DataFrame, sends: pd.DataFrame) -> pd.Series:
    """Return time from send to first open for each message in seconds."""
    if events.empty or sends.empty or "msg_id" not in events.columns or "msg_id" not in sends.columns:
        return pd.Series(dtype=float, name="time_to_first_open")

    ev_pl = pl.from_pandas(events, include_index=False)
    sd_pl = pl.from_pandas(sends, include_index=False)

    ts_col = "ts" if "ts" in ev_pl.columns else ("event_ts" if "event_ts" in ev_pl.columns else None)
    if ts_col is None or "send_ts" not in sd_pl.columns:
        return pd.Series(dtype=float, name="time_to_first_open")

    first_open = (
        ev_pl.filter(pl.col("event_type") == "open")
        .group_by("msg_id")
        .agg(pl.col(ts_col).min().alias("first_open_ts"))
    )
    if first_open.is_empty():
        return pd.Series(dtype=float, name="time_to_first_open")

    merged = first_open.join(sd_pl.select(["msg_id", "send_ts"]).unique(), on="msg_id", how="inner")
    if merged.is_empty():
        return pd.Series(dtype=float, name="time_to_first_open")

    merged = merged.with_columns(
        (
            (
                pl.col("first_open_ts").cast(pl.Datetime, strict=False)
                - pl.col("send_ts").cast(pl.Datetime, strict=False)
            ).dt.total_milliseconds()
            / 1000.0
        ).alias("time_to_first_open")
    )
    out = merged.select(["msg_id", "time_to_first_open"]).to_pandas().set_index("msg_id")["time_to_first_open"]
    out.name = "time_to_first_open"
    return out


def build_features_for_campaign(campaign_id: str) -> pd.DataFrame:
    """Build per-recipient features for ``campaign_id``."""
    events = db.load_event_log()
    sends = db.load_send_log()
    signups = db.load_user_signups()

    if sends.empty:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    if "campaign" not in sends.columns:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    sends = sends[sends["campaign"] == campaign_id]
    if sends.empty:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    relevant_events = events[events["campaign"] == campaign_id] if "campaign" in events.columns else events.iloc[0:0]
    ev_pl = pl.from_pandas(relevant_events, include_index=False) if not relevant_events.empty else pl.DataFrame()
    sd_pl = pl.from_pandas(sends, include_index=False)

    if "msg_id" not in sd_pl.columns:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    opens = (
        ev_pl.filter(pl.col("event_type") == "open").group_by("msg_id").len().rename({"len": "opens"})
        if not ev_pl.is_empty() and {"event_type", "msg_id"}.issubset(ev_pl.columns)
        else pl.DataFrame(schema={"msg_id": pl.Utf8, "opens": pl.Int64})
    )
    clicks = (
        ev_pl.filter(pl.col("event_type") == "click").group_by("msg_id").len().rename({"len": "clicks"})
        if not ev_pl.is_empty() and {"event_type", "msg_id"}.issubset(ev_pl.columns)
        else pl.DataFrame(schema={"msg_id": pl.Utf8, "clicks": pl.Int64})
    )
    ttf = _compute_time_to_first_open(relevant_events, sends)
    ttf_pl = (
        pl.from_pandas(ttf.reset_index().rename(columns={"index": "msg_id"}), include_index=False)
        if not ttf.empty
        else pl.DataFrame(schema={"msg_id": pl.Utf8, "time_to_first_open": pl.Float64})
    )

    feats = (
        sd_pl.join(opens, on="msg_id", how="left")
        .join(clicks, on="msg_id", how="left")
        .join(ttf_pl, on="msg_id", how="left")
    )
    for c in FEATURE_COLUMNS:
        if c not in feats.columns:
            feats = feats.with_columns(pl.lit(0.0).alias(c))
    feats = feats.with_columns([pl.col(c).fill_null(0.0) for c in FEATURE_COLUMNS])

    recipient_col = "recipient" if "recipient" in sends.columns else ("email" if "email" in sends.columns else None)
    if recipient_col is None:
        return pd.DataFrame(columns=["email", *FEATURE_COLUMNS, "signed_up"])

    if not signups.empty and {"campaign_id", "email"}.issubset(signups.columns):
        signup_flag = (
            pl.from_pandas(signups[signups["campaign_id"] == campaign_id][["email"]], include_index=False)
            .with_columns(pl.lit(1).alias("signed_up"))
            .unique(subset=["email"])
        )
        feats = feats.join(signup_flag, left_on=recipient_col, right_on="email", how="left")
        if recipient_col != "email" and "email" in feats.columns:
            feats = feats.drop("email")
        feats = feats.with_columns(pl.col("signed_up").fill_null(0).cast(pl.Int64))
    else:
        feats = feats.with_columns(pl.lit(0).cast(pl.Int64).alias("signed_up"))

    out = feats.to_pandas()
    out = out.rename(columns={recipient_col: "email"})
    if "email_right" in out.columns and "email" in out.columns:
        out = out.drop(columns=["email_right"])
    return out[["email", *FEATURE_COLUMNS, "signed_up"]]
