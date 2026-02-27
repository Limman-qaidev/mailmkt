"""Computation of campaign engagement metrics.

Polars is used internally for faster multi-threaded, columnar aggregations.
Public outputs remain pandas for compatibility with the dashboard/UI.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import polars as pl

__all__ = ["compute_campaign_metrics"]

_OUTPUT_NUMERIC_COLS: Final = [
    "N_sends",
    "N_opens",
    "N_clicks",
    "N_unsubscribes",
    "N_complaints",
    "N_signups_attr",
]
_RATE_COLS: Final = [
    "open_rate",
    "ctr",
    "ctor",
    "unsubscribe_rate",
    "signup_rate",
    "signup_rate_per_send",
]


def _empty_output() -> pd.DataFrame:
    cols = _OUTPUT_NUMERIC_COLS + _RATE_COLS
    return pd.DataFrame(columns=cols).astype(
        {
            "N_sends": int,
            "N_opens": int,
            "N_clicks": int,
            "N_unsubscribes": int,
            "N_complaints": int,
            "N_signups_attr": int,
            "open_rate": float,
            "ctr": float,
            "ctor": float,
            "unsubscribe_rate": float,
            "signup_rate": float,
            "signup_rate_per_send": float,
        }
    )


def _normalise_campaign_column(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    alt = "campaign" if group_col == "campaign_id" else "campaign_id"
    if group_col not in df.columns and alt in df.columns:
        df = df.rename(columns={alt: group_col})
    return df


def _prepare_inputs(
    sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if sends.empty:
        return sends, events, signups, "campaign"

    group_col = "campaign_id" if "campaign_id" in sends.columns else "campaign"
    sends = _normalise_campaign_column(sends.copy(), group_col)
    events = _normalise_campaign_column(events.copy(), group_col)
    signups = _normalise_campaign_column(signups.copy(), group_col)

    for df in (sends, events, signups):
        if group_col in df.columns:
            df[group_col] = df[group_col].astype(str)

    if "send_ts" in sends.columns:
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")
    if "event_ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    elif "ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(signups["signup_ts"], errors="coerce")

    return sends, events, signups, group_col


def _event_counts_polars(events_pl: pl.DataFrame, sends_pl: pl.DataFrame, group_col: str) -> pl.DataFrame:
    if events_pl.is_empty() or not {group_col, "msg_id", "event_type"}.issubset(set(events_pl.columns)):
        return pl.DataFrame(schema={group_col: pl.Utf8})

    sends_keys = sends_pl.select([group_col, "msg_id"]).unique()
    aligned = (
        events_pl.join(sends_keys, on=[group_col, "msg_id"], how="inner")
        .with_columns(pl.col("event_type").cast(pl.Utf8))
        .unique(subset=[group_col, "event_type", "msg_id"])
    )
    if aligned.is_empty():
        return pl.DataFrame(schema={group_col: pl.Utf8})

    counts = (
        aligned.group_by([group_col, "event_type"])
        .agg(pl.col("msg_id").n_unique().alias("n"))
        .pivot(index=group_col, on="event_type", values="n", aggregate_function="sum")
        .fill_null(0)
    )
    rename_map = {
        "open": "N_opens",
        "click": "N_clicks",
        "unsubscribe": "N_unsubscribes",
        "complaint": "N_complaints",
    }
    valid_renames = {k: v for k, v in rename_map.items() if k in counts.columns}
    if valid_renames:
        counts = counts.rename(valid_renames)
    return counts


def _signup_counts_polars(signups_pl: pl.DataFrame, sends_pl: pl.DataFrame, group_col: str) -> pl.DataFrame:
    if signups_pl.is_empty():
        return pl.DataFrame(schema={group_col: pl.Utf8, "N_signups_attr": pl.Int64})
    if "email" not in signups_pl.columns or "email" not in sends_pl.columns:
        return pl.DataFrame(schema={group_col: pl.Utf8, "N_signups_attr": pl.Int64})

    merge_cols = [c for c in (group_col, "email", "send_ts") if c in sends_pl.columns]
    merge_keys = [c for c in (group_col, "email") if c in merge_cols]
    if not merge_keys:
        return pl.DataFrame(schema={group_col: pl.Utf8, "N_signups_attr": pl.Int64})

    signups_attr = signups_pl.join(
        sends_pl.select(merge_cols).unique(),
        on=merge_keys,
        how="inner",
    )

    if {"signup_ts", "send_ts"}.issubset(set(signups_attr.columns)):
        signups_attr = signups_attr.filter(pl.col("signup_ts") >= pl.col("send_ts"))

    signups_attr = signups_attr.unique(subset=merge_keys)
    if signups_attr.is_empty():
        return pl.DataFrame(schema={group_col: pl.Utf8, "N_signups_attr": pl.Int64})

    return signups_attr.group_by(group_col).agg(
        pl.col("email").n_unique().cast(pl.Int64).alias("N_signups_attr")
    )


def compute_campaign_metrics(
    sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-campaign counts and bounded rates."""
    if sends.empty:
        return _empty_output()

    sends, events, signups, group_col = _prepare_inputs(sends, events, signups)
    if "msg_id" not in sends.columns:
        return _empty_output()

    sends_pl = pl.from_pandas(sends, include_index=False)
    events_pl = pl.from_pandas(events, include_index=False) if not events.empty else pl.DataFrame()
    signups_pl = pl.from_pandas(signups, include_index=False) if not signups.empty else pl.DataFrame()

    send_counts = sends_pl.group_by(group_col).agg(
        pl.col("msg_id").n_unique().cast(pl.Int64).alias("N_sends")
    )
    event_counts = _event_counts_polars(events_pl, sends_pl, group_col)
    signup_counts = _signup_counts_polars(signups_pl, sends_pl, group_col)

    idx_frames: list[pl.DataFrame] = [send_counts.select(group_col)]
    if not event_counts.is_empty() and group_col in event_counts.columns:
        idx_frames.append(event_counts.select(group_col))
    if not signup_counts.is_empty() and group_col in signup_counts.columns:
        idx_frames.append(signup_counts.select(group_col))
    idx = pl.concat(idx_frames).unique()

    metrics = (
        idx.join(send_counts, on=group_col, how="left")
        .join(event_counts, on=group_col, how="left")
        .join(signup_counts, on=group_col, how="left")
    )

    for col in _OUTPUT_NUMERIC_COLS:
        if col not in metrics.columns:
            metrics = metrics.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
    metrics = metrics.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in _OUTPUT_NUMERIC_COLS])

    denom_sends = pl.when(pl.col("N_sends") > 0).then(pl.col("N_sends")).otherwise(1).cast(pl.Float64)
    denom_opens = pl.when(pl.col("N_opens") > 0).then(pl.col("N_opens")).otherwise(1).cast(pl.Float64)
    denom_clicks = pl.when(pl.col("N_clicks") > 0).then(pl.col("N_clicks")).otherwise(1).cast(pl.Float64)

    metrics = metrics.with_columns(
        [
            (pl.col("N_opens") / denom_sends).clip(0.0, 1.0).alias("open_rate"),
            (pl.col("N_clicks") / denom_sends).clip(0.0, 1.0).alias("ctr"),
            (pl.col("N_clicks") / denom_opens).clip(0.0, 1.0).alias("ctor"),
            (pl.col("N_unsubscribes") / denom_sends).clip(0.0, 1.0).alias("unsubscribe_rate"),
            (
                pl.when(pl.col("N_clicks") > 0)
                .then(pl.col("N_signups_attr"))
                .otherwise(0)
                .cast(pl.Float64)
                / denom_clicks
            )
            .clip(0.0, 1.0)
            .alias("signup_rate"),
            (pl.col("N_signups_attr") / denom_sends).clip(0.0, 1.0).alias("signup_rate_per_send"),
        ]
    )

    select_cols = [group_col, *_OUTPUT_NUMERIC_COLS, *_RATE_COLS]
    out = metrics.select([c for c in select_cols if c in metrics.columns]).to_pandas()
    if group_col in out.columns:
        out = out.set_index(group_col)

    for c in _OUTPUT_NUMERIC_COLS:
        if c in out.columns:
            out[c] = out[c].astype(int)
    for c in _RATE_COLS:
        if c in out.columns:
            out[c] = out[c].astype(float)

    return out
