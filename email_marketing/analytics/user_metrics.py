# email_marketing/analytics/user_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import pandas as pd


@dataclass(frozen=True)
class EBPrior:
    """Beta prior hyperparameters for empirical Bayes."""
    alpha: float = 0.5
    beta: float = 0.5


def _ensure_ts(series: pd.Series, *, utc: bool = True) -> pd.Series:
    """Parse to pandas datetime with optional UTC awareness."""
    out = pd.to_datetime(series, errors="coerce", utc=utc)
    return out


def _normalize_email(s: pd.Series) -> pd.Series:
    """Lowercase/strip emails; leave NaNs untouched."""
    s = s.astype(str).str.strip().str.lower()
    # Basic sanity: keep only entries that look like emails; others -> NaN
    mask = s.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True)
    s = s.where(mask, other=pd.NA)
    return s


def _recipient_col(df: pd.DataFrame) -> Optional[str]:
    """Heuristically find the recipient/email column name in a DF."""
    for c in ("recipient", "email", "to"):
        if c in df.columns:
            return c
    # Try fuzzy
    cand = [c for c in df.columns if "mail" in c.lower()]
    return cand[0] if cand else None


def _event_ts_col(events: pd.DataFrame) -> str:
    """Choose best timestamp column in events."""
    for c in ("event_ts", "ts", "timestamp"):
        if c in events.columns:
            return c
    return "event_ts"  # will be created later if absent


def _event_type_col(events: pd.DataFrame) -> str:
    for c in ("event_type", "type"):
        if c in events.columns:
            return c
    return "event_type"


def _subject_or_topic_cols(campaigns: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    subj = None
    topic = None
    for c in campaigns.columns:
        cl = c.lower()
        if subj is None and "subject" in cl:
            subj = c
        if topic is None and "topic" in cl:
            topic = c
    return subj, topic


def build_user_aggregates(
    events: pd.DataFrame,
    sends: pd.DataFrame,
    signups: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate lifetime metrics per email (robust to schema variants).

    Returns
    -------
    pd.DataFrame
        One row per email with counts, raw rates, recencies and last timestamps.
        Columns:
        ['email','domain','N_sends','N_opens','N_clicks','N_unsubscribes','N_signups',
         'open_rate_raw','click_rate_raw','unsubscribe_rate_raw',
         'last_send_ts','last_open_ts','last_click_ts','last_unsub_ts',
         'recency_open_days','recency_click_days','recency_unsub_days']
    """
    ev = events.copy()
    sd = sends.copy()
    sg = signups.copy()

    # --- Normalize timestamps ---
    ev_ts_col = _event_ts_col(ev)
    if ev_ts_col not in ev.columns and "ts" in ev.columns:
        ev_ts_col = "ts"
    ev[ev_ts_col] = _ensure_ts(ev.get(ev_ts_col, pd.Series([])))
    if "send_ts" in sd.columns:
        sd["send_ts"] = _ensure_ts(sd["send_ts"])
    if "signup_ts" in sg.columns:
        sg["signup_ts"] = _ensure_ts(sg["signup_ts"])

    # --- Normalize email columns ---
    ev_type_col = _event_type_col(ev)
    ev_email_col = _recipient_col(ev)
    if ev_email_col is None:
        # Try to inherit from sends via msg_id mapping if present
        if "msg_id" in ev.columns and "msg_id" in sd.columns:
            rc = _recipient_col(sd)
            if rc:
                ev = ev.merge(sd[["msg_id", rc]].rename(columns={rc: "email_from_sends"}),
                              on="msg_id", how="left")
                ev_email_col = "email_from_sends"
    if ev_email_col is None:
        # give up -> create empty so groupbys don't explode
        ev["email_tmp"] = pd.NA
        ev_email_col = "email_tmp"

    ev["email"] = _normalize_email(ev[ev_email_col])
    sd_email_col = _recipient_col(sd)
    if sd_email_col is None:
        # some older datasets store the recipient also under "email"
        if "msg_id" in sd.columns and "msg_id" in ev.columns:
            # if needed we could backfill from events send rows
            pass
    else:
        sd["email"] = _normalize_email(sd[sd_email_col])

    if "email" not in sd.columns:
        # fallback from events: take event_type == 'send'
        mask_send = ev[ev_type_col].astype(str).str.lower().eq("send")
        sends_from_events = (
            ev.loc[mask_send, ["msg_id", "email", ev_ts_col]]
            .rename(columns={ev_ts_col: "send_ts"})
            .dropna(subset=["email"])
        )
        sd = sd.copy()
        sd = pd.concat([sd, sends_from_events], ignore_index=True)
        sd = sd.drop_duplicates(subset=["msg_id", "email"], keep="last")

    # --- Counts per email ---
    # Sends: by unique msg_id/email
    if "email" in sd.columns:
        sends_g = sd.dropna(subset=["email"]).groupby("email")["msg_id"].nunique().rename("N_sends")
        last_send = sd.dropna(subset=["email"]).groupby("email")["send_ts"].max().rename("last_send_ts")
    else:
        sends_g = pd.Series(dtype=int, name="N_sends")
        last_send = pd.Series(dtype="datetime64[ns, UTC]", name="last_send_ts")

    # Open/click/unsub counts (unique msg per email)
    et = ev_type_col
    ev_base = ev.dropna(subset=["email"])
    opens_g = (
        ev_base[ev_base[et].astype(str).str.lower().eq("open")]
        .groupby("email")["msg_id"].nunique().rename("N_opens")
    )
    clicks_g = (
        ev_base[ev_base[et].astype(str).str.lower().eq("click")]
        .groupby("email")["msg_id"].nunique().rename("N_clicks")
    )
    unsubs_g = (
        ev_base[ev_base[et].astype(str).str.lower().eq("unsubscribe")]
        .groupby("email")["msg_id"].nunique().rename("N_unsubscribes")
    )
    last_open = (
        ev_base[ev_base[et].astype(str).str.lower().eq("open")]
        .groupby("email")[ev_ts_col].max().rename("last_open_ts")
    )
    last_click = (
        ev_base[ev_base[et].astype(str).str.lower().eq("click")]
        .groupby("email")[ev_ts_col].max().rename("last_click_ts")
    )
    last_unsub = (
        ev_base[ev_base[et].astype(str).str.lower().eq("unsubscribe")]
        .groupby("email")[ev_ts_col].max().rename("last_unsub_ts")
    )

    # Signups per email if available
    if "email" in sg.columns:
        sg["email"] = _normalize_email(sg["email"])
        signups_g = sg.dropna(subset=["email"]).groupby("email")["signup_id" if "signup_id" in sg.columns else "email"].nunique().rename("N_signups")
    else:
        signups_g = pd.Series(dtype=int, name="N_signups")

    # Combine
    df = (
        pd.DataFrame(index=pd.Index([], name="email"))
        .join(sends_g, how="outer")
        .join(opens_g, how="outer")
        .join(clicks_g, how="outer")
        .join(unsubs_g, how="outer")
        .join(signups_g, how="outer")
        .join(last_send, how="outer")
        .join(last_open, how="outer")
        .join(last_click, how="outer")
        .join(last_unsub, how="outer")
        .reset_index()
    ).fillna(0)

    # Types
    for c in ["N_sends", "N_opens", "N_clicks", "N_unsubscribes", "N_signups"]:
        if c in df.columns:
            df[c] = df[c].astype(int)
    # Rates (raw)
    df["open_rate_raw"] = (df["N_opens"] / df["N_sends"]).where(df["N_sends"] > 0, other=0.0)
    df["click_rate_raw"] = (df["N_clicks"] / df["N_sends"]).where(df["N_sends"] > 0, other=0.0)
    df["unsubscribe_rate_raw"] = (df["N_unsubscribes"] / df["N_sends"]).where(df["N_sends"] > 0, other=0.0)

    # Recency in days from now (UTC)
    now = pd.Timestamp.now(tz="UTC")
    def _recency(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([None] * len(df), name=f"recency_{col}_days")
        return ((now - pd.to_datetime(df[col], errors="coerce", utc=True)).dt.days).astype("Int64")

    df["recency_open_days"] = _recency("last_open_ts")
    df["recency_click_days"] = _recency("last_click_ts")
    df["recency_unsub_days"] = _recency("last_unsub_ts")

    # Domain
    df["domain"] = df["email"].astype(str).str.split("@").str[-1]

    # Order columns
    cols_order = [
        "email", "domain",
        "N_sends", "N_opens", "N_clicks", "N_unsubscribes", "N_signups",
        "open_rate_raw", "click_rate_raw", "unsubscribe_rate_raw",
        "last_send_ts", "last_open_ts", "last_click_ts", "last_unsub_ts",
        "recency_open_days", "recency_click_days", "recency_unsub_days",
    ]
    df = df[[c for c in cols_order if c in df.columns]].copy()
    return df.sort_values(["N_sends", "N_opens"], ascending=[False, False]).reset_index(drop=True)


def _fit_beta_moments(successes: pd.Series, trials: pd.Series) -> EBPrior:
    """Estimate (alpha,beta) by method of moments from global rates."""
    # Avoid zero/one extremes
    p = (successes.sum() + 0.5) / (trials.sum() + 1.0)
    # Variance heuristic (tunable)
    v = max(1e-5, p * (1 - p) / 50.0)
    # Solve moments for Beta
    # p = a/(a+b), v = ab/[(a+b)^2 (a+b+1)]
    # Let k = a+b => a = pk, b = (1-p)k
    # v = p(1-p)/(k+1)  => k = p(1-p)/v - 1
    k = max(1.0, (p * (1 - p)) / v - 1.0)
    alpha = max(0.5, p * k)
    beta = max(0.5, (1 - p) * k)
    return EBPrior(alpha=alpha, beta=beta)


def compute_eb_rates(
    df_users: pd.DataFrame,
    *,
    prior_open: Optional[EBPrior] = None,
    prior_click: Optional[EBPrior] = None,
    prior_unsub: Optional[EBPrior] = None,
) -> pd.DataFrame:
    """Add empirical-Bayes posterior means for open/click/unsub rates.

    Posterior mean for Beta-Binomial:
        E[theta | k, n] = (k + alpha) / (n + alpha + beta)

    If priors are not supplied, they are estimated globally via moments.
    """
    df = df_users.copy()
    # Estimate priors if missing
    if prior_open is None:
        prior_open = _fit_beta_moments(df["N_opens"], df["N_sends"].clip(lower=0))
    if prior_click is None:
        prior_click = _fit_beta_moments(df["N_clicks"], df["N_sends"].clip(lower=0))
    if prior_unsub is None:
        prior_unsub = _fit_beta_moments(df["N_unsubscribes"], df["N_sends"].clip(lower=0))

    # EB posterior means
    a, b = prior_open.alpha, prior_open.beta
    df["open_rate_eb"] = (df["N_opens"] + a) / (df["N_sends"] + a + b)
    a, b = prior_click.alpha, prior_click.beta
    df["click_rate_eb"] = (df["N_clicks"] + a) / (df["N_sends"] + a + b)
    a, b = prior_unsub.alpha, prior_unsub.beta
    df["unsub_rate_eb"] = (df["N_unsubscribes"] + a) / (df["N_sends"] + a + b)

    return df
