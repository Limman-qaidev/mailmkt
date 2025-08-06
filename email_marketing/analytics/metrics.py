"""Computation of raw campaign engagement metrics."""

from __future__ import annotations

import pandas as pd


def compute_campaign_metrics(
    sends: pd.DataFrame, events: pd.DataFrame, signups: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-campaign counts and rates of engagement events.

    The three input DataFrames must share a common campaign identifier.  This
    can either be a ``campaign_id`` column or a human readable ``campaign``
    name.  The function automatically picks whichever is available.

    Parameters
    ----------
    sends:
        DataFrame containing at least ``campaign_id`` and ``msg_id`` columns.
    events:
        DataFrame with ``campaign_id``, ``msg_id`` and ``event_type`` columns.
    signups:
        DataFrame with ``campaign_id`` identifying user signups.

    Returns
    -------
    pd.DataFrame
        One row per ``campaign_id`` with counts of sends, opens, clicks,
        unsubscribes, complaints and signups, along with corresponding rates.
    """

    group_col = "campaign_id" if "campaign_id" in sends.columns else "campaign"

    send_counts = sends.groupby(
        group_col
        )["msg_id"].nunique().rename("N_sends")

    if events.empty:
        event_counts = pd.DataFrame(index=send_counts.index)
    else:
        event_counts = (
            events.pivot_table(
                index=group_col,
                columns="event_type",
                values="msg_id",
                aggfunc="nunique",
                fill_value=0,
            ).rename(
                columns={
                    "open": "N_opens",
                    "click": "N_clicks",
                    "unsubscribe": "N_unsubscribes",
                    "complaint": "N_complaints",
                }
            )
        )

    signup_counts = (
        signups.groupby(group_col)[
            "signup_id"
            ].nunique().rename("N_signups")
        if not signups.empty
        else pd.Series(dtype=int, name="N_signups")
    )

    open_signups = (
        events.loc[events["event_type"] == "open", [group_col, "email"]]
        .dropna(subset=["email"])
        .merge(
            signups[[group_col, "email"]],
            on=[group_col, "email"],
            how="inner",
        )
        .drop_duplicates()
        .groupby(group_col)
        .size()
        .rename("N_open_signups")
        if "email" in events.columns and not events.empty and not signups.empty
        else pd.Series(dtype=int, name="N_open_signups")
    )

    metrics = pd.concat(
        [send_counts, event_counts, signup_counts, open_signups], axis=1
    )
    for col in [
        "N_opens",
        "N_clicks",
        "N_unsubscribes",
        "N_complaints",
        "N_signups",
        "N_open_signups",
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
            "N_signups": int,
            "N_open_signups": int,
        }
    )

    metrics["open_rate"] = metrics["N_opens"] / metrics["N_sends"].where(
        metrics["N_sends"] > 0, 1
    )
    metrics["click_rate"] = metrics.apply(
        lambda r: r["N_clicks"] / r["N_opens"] if r["N_opens"] > 0 else 0,
        axis=1,
    )
    metrics["unsubscribe_rate"] = metrics["N_unsubscribes"] / metrics[
        "N_sends"
    ].where(metrics["N_sends"] > 0, 1)
    metrics["signup_rate"] = metrics["N_signups"] / metrics["N_sends"].where(
        metrics["N_sends"] > 0, 1
    )
    metrics["open_signup_rate"] = metrics["N_open_signups"] / metrics[
        "N_opens"
    ].where(metrics["N_opens"] > 0, 1)

    return metrics
