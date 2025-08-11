"""Computation of raw campaign engagement metrics.

This module exposes :func:`compute_campaign_metrics`, a robust implementation
that normalises keys and timestamps, attributes signups only to matching sends
and clips all resulting rates to the ``[0, 1]`` interval.  Previous versions
could produce metrics greater than 1 due to missing joins or inconsistent
inputs.
"""

# email_marketing/analytics/metrics.py
from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = ["compute_campaign_metrics"]

# Columns we’ll guarantee in the output
_OUTPUT_NUMERIC_COLS: Final = [
    "N_sends",
    "N_opens",
    "N_clicks",
    "N_unsubscribes",
    "N_complaints",
    "N_signups_attr",
]


def _normalise_campaign_column(
    df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Ensure the dataframe has the desired campaign column name.

    If the opposite naming exists (``campaign`` vs ``campaign_id``), it is
    renamed to ``group_col``. Otherwise the dataframe is returned unchanged.

    Args:
        df: Input dataframe.
        group_col: Desired campaign column name: ``"campaign_id"`` or
            ``"campaign"``.

    Returns:
        A dataframe with a consistent campaign column name.
    """
    if df.empty:
        return df
    alt = "campaign" if group_col == "campaign_id" else "campaign_id"
    if group_col not in df.columns and alt in df.columns:
        df = df.rename(columns={alt: group_col})
    return df


def compute_campaign_metrics(
    sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-campaign engagement counts and rates.

    The function aligns inputs on a shared campaign identifier (either
    ``campaign_id`` or ``campaign``), restricts events/signups to the universe
    of sent emails, and returns robust counts plus bounded rates.

    Args:
        sends: DataFrame with at least ``msg_id`` and a campaign column
            (``campaign_id`` or ``campaign``). Optionally ``send_ts`` and
            ``email`` for signup attribution.
        events: DataFrame with ``msg_id``, ``event_type`` and a campaign
            column. Optionally ``event_ts`` (or ``ts``) and ``email``.
        signups: DataFrame with a campaign column and ``email``. Optionally
            ``signup_ts``. Signups are attributed only when a matching send
            exists (same campaign+email) and, if timestamps are present,
            when ``signup_ts >= send_ts``.

    Returns:
        DataFrame indexed by campaign with counts and rates:
        counts: N_sends, N_opens, N_clicks, N_unsubscribes, N_complaints,
                N_signups_attr
        rates:  open_rate, ctr, ctor, unsubscribe_rate, signup_rate,
                signup_rate_per_send
    """
    # Empty guard: return a correctly typed empty frame
    if sends.empty:
        cols = _OUTPUT_NUMERIC_COLS + [
            "open_rate",
            "ctr",
            "ctor",
            "unsubscribe_rate",
            "signup_rate",
            "signup_rate_per_send",
        ]
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

    # Choose campaign key and normalise inputs
    group_col = "campaign_id" if "campaign_id" in sends.columns else "campaign"
    sends = _normalise_campaign_column(sends, group_col)
    events = _normalise_campaign_column(events, group_col)
    signups = _normalise_campaign_column(signups, group_col)

    # Parse timestamps leniently (strings -> datetime where present)
    if "send_ts" in sends.columns:
        sends = sends.copy()
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")
    if "event_ts" in events.columns:
        events = events.copy()
        events["event_ts"] = pd.to_datetime(
            events["event_ts"], errors="coerce"
            )
    elif "ts" in events.columns:
        events = events.copy()
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")
    if "signup_ts" in signups.columns:
        signups = signups.copy()
        signups["signup_ts"] = pd.to_datetime(
            signups["signup_ts"], errors="coerce"
        )

    # -------------------------
    # 1) Base universe: sends
    # -------------------------
    send_counts = (
        sends.groupby(group_col)["msg_id"].nunique().rename("N_sends")
    )

    # --------------------------------------------
    # 2) Events aligned to universe (msg_id+camp)
    # --------------------------------------------
    if not events.empty and {"msg_id", group_col}.issubset(events.columns):
        events_aligned = events.merge(
            sends[[group_col, "msg_id"]],
            on=[group_col, "msg_id"],
            how="inner",
            validate="many_to_many",
        )
        # Deduplicate per (campaign, event_type, msg_id)
        events_aligned = events_aligned.drop_duplicates(
            subset=[group_col, "event_type", "msg_id"]
        )
    else:
        events_aligned = pd.DataFrame(
            columns=[group_col, "event_type", "msg_id"]
            )

    if events_aligned.empty:
        event_counts = pd.DataFrame(index=send_counts.index)
    else:
        event_counts = (
            events_aligned.groupby([group_col, "event_type"])["msg_id"]
            .nunique()
            .unstack(fill_value=0)
            .rename(
                columns={
                    "open": "N_opens",
                    "click": "N_clicks",
                    "unsubscribe": "N_unsubscribes",
                    "complaint": "N_complaints",
                }
            )
        )

    # --------------------------------------------------
    # 3) Attribute signups to sends (campaign+email+ts)
    # --------------------------------------------------
    # Start with an empty frame to keep the variable defined.
    signups_attr = pd.DataFrame(columns=[group_col, "email"])

    if (
        not signups.empty
        and "email" in signups.columns
        and "email" in sends.columns
    ):
        # Minimal merge keys always include campaign+email
        merge_cols = [
            c for c in [group_col, "email", "send_ts"] if c in sends.columns
            ]
        merge_keys = [c for c in [group_col, "email"] if c in merge_cols]

        if merge_keys:
            signups_attr = signups.merge(
                sends[merge_cols].drop_duplicates(),
                on=merge_keys,
                how="inner",
                validate="many_to_many",
            )

            # If both timestamps exist, require signup_ts >= send_ts
            if {"signup_ts", "send_ts"}.issubset(signups_attr.columns):
                signups_attr = signups_attr[
                    signups_attr["signup_ts"] >= signups_attr["send_ts"]
                ]

            signups_attr = signups_attr.drop_duplicates(subset=merge_keys)

    signup_counts = (
        signups_attr.groupby(group_col)["email"]
        .nunique()
        .rename("N_signups_attr")
        if not signups_attr.empty
        else pd.Series(dtype=int, name="N_signups_attr")
    )

    # ---------------------------------------
    # 4) Align indices and assemble metrics
    # ---------------------------------------
    idx = send_counts.index
    if not events_aligned.empty:
        idx = idx.union(event_counts.index)
    if not signup_counts.empty:
        idx = idx.union(signup_counts.index)

    send_df = send_counts.reindex(idx, fill_value=0).to_frame()
    events_df = (
        event_counts.reindex(idx, fill_value=0)
        if not events_aligned.empty
        else pd.DataFrame(index=idx)
    )
    signups_df = signup_counts.reindex(idx, fill_value=0).to_frame()

    metrics = pd.concat([send_df, events_df, signups_df], axis=1)

    # Ensure all expected numeric columns exist
    for col in _OUTPUT_NUMERIC_COLS:
        if col not in metrics.columns:
            metrics[col] = 0

    # Types
    metrics = metrics.fillna(0).astype(
        {
            "N_sends": int,
            "N_opens": int,
            "N_clicks": int,
            "N_unsubscribes": int,
            "N_complaints": int,
            "N_signups_attr": int,
        }
    )

    # -----------------------
    # 5) Rate calculations
    # -----------------------
    # All rates clipped to [0, 1]
    denom_sends = metrics["N_sends"].where(metrics["N_sends"] > 0, 1)
    denom_opens = metrics["N_opens"].where(metrics["N_opens"] > 0, 1)
    denom_clicks = metrics["N_clicks"].where(metrics["N_clicks"] > 0, 1)

    metrics["open_rate"] = (metrics["N_opens"] / denom_sends).clip(0, 1)
    metrics["ctr"] = (metrics["N_clicks"] / denom_sends).clip(0, 1)
    metrics["ctor"] = (metrics["N_clicks"] / denom_opens).clip(0, 1)
    metrics["unsubscribe_rate"] = (
        metrics["N_unsubscribes"] / denom_sends
    ).clip(0, 1)

    # Signups:
    # - signup_rate: signups per clicker (0 if no clicks)
    # - signup_rate_per_send: signups per send
    metrics["signup_rate"] = (
        metrics["N_signups_attr"].where(
            metrics["N_clicks"] > 0, 0) / denom_clicks
    ).clip(0, 1)
    metrics["signup_rate_per_send"] = (
        metrics["N_signups_attr"] / denom_sends
    ).clip(0, 1)

    return metrics
