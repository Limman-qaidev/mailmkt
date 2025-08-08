"""Computation of raw campaign engagement metrics.

This module exposes :func:`compute_campaign_metrics`, a robust implementation
that normalises keys and timestamps, attributes signups only to matching sends
and clips all resulting rates to the ``[0, 1]`` interval.  Previous versions
could produce metrics greater than 1 due to missing joins or inconsistent
inputs.
"""

from __future__ import annotations

import pandas as pd


def _normalise_campaign_column(
        df: pd.DataFrame, group_col: str
        ) -> pd.DataFrame:
    """Ensure ``df`` contains a column named ``group_col``.

    If the alternative campaign column is present (``campaign`` vs
    ``campaign_id``) it is renamed.  This keeps different data sources
    compatible with the computation function.
    """

    alt = "campaign" if group_col == "campaign_id" else "campaign_id"
    if group_col not in df.columns and alt in df.columns:
        df = df.rename(columns={alt: group_col})
    return df


def compute_campaign_metrics(
    sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-campaign engagement counts and rates.

    Parameters
    ----------
    sends, events, signups:
        DataFrames describing sent emails, subsequent engagement events and
        user signups respectively.  ``sends`` and ``events`` must share
        ``msg_id`` values.
    """

    if sends.empty:
        cols = [
            "N_sends",
            "N_opens",
            "N_clicks",
            "N_unsubscribes",
            "N_complaints",
            "N_signups_attr",
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

    group_col = "campaign_id" if "campaign_id" in sends.columns else "campaign"

    # Normalise campaign column names across inputs
    sends = _normalise_campaign_column(sends, group_col)
    events = _normalise_campaign_column(events, group_col)
    signups = _normalise_campaign_column(signups, group_col)

    # Parse timestamps leniently
    if "send_ts" in sends.columns:
        sends["send_ts"] = pd.to_datetime(sends["send_ts"], errors="coerce")
    if "event_ts" in events.columns:
        events["event_ts"] = pd.to_datetime(
            events["event_ts"], errors="coerce"
            )
    elif "ts" in events.columns:
        events["event_ts"] = pd.to_datetime(events["ts"], errors="coerce")
    if "signup_ts" in signups.columns:
        signups["signup_ts"] = pd.to_datetime(
            signups["signup_ts"], errors="coerce"
            )

    # Universe: sends
    send_counts = sends.groupby(group_col)["msg_id"].nunique(
    ).rename("N_sends")

    # Only count events that correspond to a send
    events = events.merge(
        sends[[group_col, "msg_id"]], on=[group_col, "msg_id"], how="inner"
    )
    events = events.drop_duplicates(subset=[group_col, "event_type", "msg_id"])

    if events.empty:
        event_counts = pd.DataFrame(index=send_counts.index)
    else:
        event_counts = (
            events.groupby([group_col, "event_type"])["msg_id"]
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
        # Attribute signups only when there is a corresponding send
        if signups.empty or "email" not in signups.columns:
            signups_attr = pd.DataFrame(columns=[group_col, "email"])
        else:
            merge_cols = [
                c for c in [
                    group_col,
                    "email", "send_ts"
                    ] if c in sends.columns
                ]
            signups_attr = signups.merge(
                sends[merge_cols], on=[group_col, "email"], how="inner"
            )
        if (
            "signup_ts" in signups_attr.columns and
            "send_ts" in signups_attr.columns
        ):
            signups_attr = signups_attr[
                signups_attr["signup_ts"] >= signups_attr["send_ts"]
            ]
        signups_attr = signups_attr.drop_duplicates(
            subset=[group_col, "email"]
            )
    signup_counts = (
        signups_attr.groupby(group_col)["email"].nunique(
        ).rename("N_signups_attr")
        if not signups_attr.empty
        else pd.Series(dtype=int, name="N_signups_attr")
    )

    metrics = pd.concat([send_counts, event_counts, signup_counts], axis=1)
    for col in [
        "N_opens",
        "N_clicks",
        "N_unsubscribes",
        "N_complaints",
        "N_signups_attr"
    ]:
        if col not in metrics:
            metrics[col] = 0
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

    # Rates – clip to [0, 1] to avoid pathological values
    metrics["open_rate"] = (
        metrics["N_opens"] / metrics["N_sends"].where(
            metrics["N_sends"] > 0, 1
            )
    ).clip(0, 1)
    metrics["ctr"] = (
        metrics["N_clicks"] / metrics["N_sends"].where(
            metrics["N_sends"] > 0, 1
            )
    ).clip(0, 1)
    metrics["ctor"] = (
        metrics["N_clicks"].where(metrics["N_opens"] > 0, 0)
        / metrics["N_opens"].where(metrics["N_opens"] > 0, 1)
    ).clip(0, 1)
    metrics["unsubscribe_rate"] = (
        metrics["N_unsubscribes"]
        / metrics["N_sends"].where(metrics["N_sends"] > 0, 1)
    ).clip(0, 1)
    metrics["signup_rate"] = (
        metrics["N_signups_attr"].where(metrics["N_clicks"] > 0, 0)
        / metrics["N_clicks"].where(metrics["N_clicks"] > 0, 1)
    ).clip(0, 1)
    metrics["signup_rate_per_send"] = (
        metrics["N_signups_attr"]
        / metrics["N_sends"].where(metrics["N_sends"] > 0, 1)
    ).clip(0, 1)

    return metrics
