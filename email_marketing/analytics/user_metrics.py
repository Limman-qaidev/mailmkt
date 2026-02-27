from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class EBPrior:
    """Beta prior hyperparameters for empirical Bayes."""

    alpha: float = 0.5
    beta: float = 0.5


def _ensure_ts(series: pd.Series, *, utc: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=utc)


def _normalize_email(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    mask = s.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True)
    return s.where(mask, other=pd.NA)


def _recipient_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("recipient", "email", "to"):
        if c in df.columns:
            return c
    cand = [c for c in df.columns if "mail" in c.lower()]
    return cand[0] if cand else None


def _event_ts_col(events: pd.DataFrame) -> str:
    for c in ("event_ts", "ts", "timestamp"):
        if c in events.columns:
            return c
    return "event_ts"


def _event_type_col(events: pd.DataFrame) -> str:
    for c in ("event_type", "type"):
        if c in events.columns:
            return c
    return "event_type"


def _join_all_by_email(frames: list[pl.DataFrame]) -> pl.DataFrame:
    valid = [f for f in frames if not f.is_empty() and "email" in f.columns]
    if not valid:
        return pl.DataFrame(schema={"email": pl.Utf8})

    emails = pl.concat([f.select("email") for f in valid]).unique()
    out = emails
    for f in valid:
        cols = [c for c in f.columns if c != "email"]
        out = out.join(f.select(["email", *cols]), on="email", how="left")
    return out


def _count_unique(pl_df: pl.DataFrame, group: str, target: str, name: str) -> pl.DataFrame:
    if pl_df.is_empty() or target not in pl_df.columns:
        return pl.DataFrame(schema={group: pl.Utf8, name: pl.Int64})
    return pl_df.group_by(group).agg(pl.col(target).n_unique().cast(pl.Int64).alias(name))


def build_user_aggregates(
    events: pd.DataFrame,
    sends: pd.DataFrame,
    signups: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate lifetime metrics per email (schema-compatible output)."""
    ev = events.copy()
    sd = sends.copy()
    sg = signups.copy()

    ev_ts_col = _event_ts_col(ev)
    if ev_ts_col not in ev.columns and "ts" in ev.columns:
        ev_ts_col = "ts"
    ev[ev_ts_col] = _ensure_ts(ev.get(ev_ts_col, pd.Series([], dtype="object")))
    if "send_ts" in sd.columns:
        sd["send_ts"] = _ensure_ts(sd["send_ts"])
    if "signup_ts" in sg.columns:
        sg["signup_ts"] = _ensure_ts(sg["signup_ts"])

    ev_type_col = _event_type_col(ev)
    ev_email_col = _recipient_col(ev)
    if ev_email_col is None and {"msg_id"}.issubset(ev.columns) and "msg_id" in sd.columns:
        rc = _recipient_col(sd)
        if rc:
            ev = ev.merge(
                sd[["msg_id", rc]].rename(columns={rc: "email_from_sends"}),
                on="msg_id",
                how="left",
            )
            ev_email_col = "email_from_sends"
    if ev_email_col is None:
        ev["email_tmp"] = pd.NA
        ev_email_col = "email_tmp"
    ev["email"] = _normalize_email(ev[ev_email_col])

    sd_email_col = _recipient_col(sd)
    if sd_email_col is not None:
        sd["email"] = _normalize_email(sd[sd_email_col])

    if "email" not in sd.columns:
        mask_send = ev[ev_type_col].astype(str).str.lower().eq("send")
        sends_from_events = (
            ev.loc[mask_send, ["msg_id", "email", ev_ts_col]]
            .rename(columns={ev_ts_col: "send_ts"})
            .dropna(subset=["email"])
        )
        sd = pd.concat([sd, sends_from_events], ignore_index=True)
        keep_cols = [c for c in ("msg_id", "email") if c in sd.columns]
        if keep_cols:
            sd = sd.drop_duplicates(subset=keep_cols, keep="last")

    if "email" in sg.columns:
        sg["email"] = _normalize_email(sg["email"])

    ev_pl = pl.from_pandas(ev, include_index=False) if not ev.empty else pl.DataFrame()
    sd_pl = pl.from_pandas(sd, include_index=False) if not sd.empty else pl.DataFrame()
    sg_pl = pl.from_pandas(sg, include_index=False) if not sg.empty else pl.DataFrame()

    sends_base = sd_pl.filter(pl.col("email").is_not_null()) if "email" in sd_pl.columns else pl.DataFrame()
    if not sends_base.is_empty():
        msg_col = "msg_id" if "msg_id" in sends_base.columns else "email"
        sends_g = sends_base.group_by("email").agg(
            pl.col(msg_col).n_unique().cast(pl.Int64).alias("N_sends")
        )
        last_send = (
            sends_base.group_by("email")
            .agg(pl.col("send_ts").max().alias("last_send_ts"))
            if "send_ts" in sends_base.columns
            else pl.DataFrame(schema={"email": pl.Utf8, "last_send_ts": pl.Datetime})
        )
    else:
        sends_g = pl.DataFrame(schema={"email": pl.Utf8, "N_sends": pl.Int64})
        last_send = pl.DataFrame(schema={"email": pl.Utf8, "last_send_ts": pl.Datetime})

    ev_base = ev_pl.filter(pl.col("email").is_not_null()) if "email" in ev_pl.columns else pl.DataFrame()
    if not ev_base.is_empty() and ev_type_col in ev_base.columns:
        et_col = pl.col(ev_type_col).cast(pl.Utf8).str.to_lowercase()
        msg_col = "msg_id" if "msg_id" in ev_base.columns else "email"
        opens = _count_unique(ev_base.filter(et_col == "open"), "email", msg_col, "N_opens")
        clicks = _count_unique(ev_base.filter(et_col == "click"), "email", msg_col, "N_clicks")
        unsubs = _count_unique(ev_base.filter(et_col == "unsubscribe"), "email", msg_col, "N_unsubscribes")

        if ev_ts_col in ev_base.columns:
            last_open = ev_base.filter(et_col == "open").group_by("email").agg(pl.col(ev_ts_col).max().alias("last_open_ts"))
            last_click = ev_base.filter(et_col == "click").group_by("email").agg(pl.col(ev_ts_col).max().alias("last_click_ts"))
            last_unsub = ev_base.filter(et_col == "unsubscribe").group_by("email").agg(pl.col(ev_ts_col).max().alias("last_unsub_ts"))
        else:
            last_open = pl.DataFrame(schema={"email": pl.Utf8, "last_open_ts": pl.Datetime})
            last_click = pl.DataFrame(schema={"email": pl.Utf8, "last_click_ts": pl.Datetime})
            last_unsub = pl.DataFrame(schema={"email": pl.Utf8, "last_unsub_ts": pl.Datetime})
    else:
        opens = pl.DataFrame(schema={"email": pl.Utf8, "N_opens": pl.Int64})
        clicks = pl.DataFrame(schema={"email": pl.Utf8, "N_clicks": pl.Int64})
        unsubs = pl.DataFrame(schema={"email": pl.Utf8, "N_unsubscribes": pl.Int64})
        last_open = pl.DataFrame(schema={"email": pl.Utf8, "last_open_ts": pl.Datetime})
        last_click = pl.DataFrame(schema={"email": pl.Utf8, "last_click_ts": pl.Datetime})
        last_unsub = pl.DataFrame(schema={"email": pl.Utf8, "last_unsub_ts": pl.Datetime})

    if not sg_pl.is_empty() and "email" in sg_pl.columns:
        sg_base = sg_pl.filter(pl.col("email").is_not_null())
        if "signup_id" in sg_base.columns:
            signups_g = sg_base.group_by("email").agg(pl.col("signup_id").n_unique().cast(pl.Int64).alias("N_signups"))
        else:
            signups_g = sg_base.group_by("email").agg(pl.col("email").n_unique().cast(pl.Int64).alias("N_signups"))
    else:
        signups_g = pl.DataFrame(schema={"email": pl.Utf8, "N_signups": pl.Int64})

    out_pl = _join_all_by_email([sends_g, opens, clicks, unsubs, signups_g, last_send, last_open, last_click, last_unsub])
    if out_pl.is_empty():
        cols = [
            "email",
            "domain",
            "N_sends",
            "N_opens",
            "N_clicks",
            "N_unsubscribes",
            "N_signups",
            "open_rate_raw",
            "click_rate_raw",
            "unsubscribe_rate_raw",
            "last_send_ts",
            "last_open_ts",
            "last_click_ts",
            "last_unsub_ts",
            "recency_open_days",
            "recency_click_days",
            "recency_unsub_days",
        ]
        return pd.DataFrame(columns=cols)

    for c in ("N_sends", "N_opens", "N_clicks", "N_unsubscribes", "N_signups"):
        if c not in out_pl.columns:
            out_pl = out_pl.with_columns(pl.lit(0).cast(pl.Int64).alias(c))
    out_pl = out_pl.with_columns(
        [
            pl.col("N_sends").fill_null(0).cast(pl.Int64),
            pl.col("N_opens").fill_null(0).cast(pl.Int64),
            pl.col("N_clicks").fill_null(0).cast(pl.Int64),
            pl.col("N_unsubscribes").fill_null(0).cast(pl.Int64),
            pl.col("N_signups").fill_null(0).cast(pl.Int64),
        ]
    )

    denom = pl.when(pl.col("N_sends") > 0).then(pl.col("N_sends")).otherwise(1).cast(pl.Float64)
    out_pl = out_pl.with_columns(
        [
            (pl.col("N_opens") / denom).alias("open_rate_raw"),
            (pl.col("N_clicks") / denom).alias("click_rate_raw"),
            (pl.col("N_unsubscribes") / denom).alias("unsubscribe_rate_raw"),
        ]
    )

    out = out_pl.to_pandas()
    last_open_ser = out["last_open_ts"] if "last_open_ts" in out.columns else pd.Series(pd.NaT, index=out.index)
    last_click_ser = out["last_click_ts"] if "last_click_ts" in out.columns else pd.Series(pd.NaT, index=out.index)
    last_unsub_ser = out["last_unsub_ts"] if "last_unsub_ts" in out.columns else pd.Series(pd.NaT, index=out.index)
    now = pd.Timestamp.now(tz="UTC")
    out["recency_open_days"] = ((now - pd.to_datetime(last_open_ser, errors="coerce", utc=True)).dt.days).astype("Int64")
    out["recency_click_days"] = ((now - pd.to_datetime(last_click_ser, errors="coerce", utc=True)).dt.days).astype("Int64")
    out["recency_unsub_days"] = ((now - pd.to_datetime(last_unsub_ser, errors="coerce", utc=True)).dt.days).astype("Int64")
    out["domain"] = out["email"].astype(str).str.split("@").str[-1]

    cols_order = [
        "email",
        "domain",
        "N_sends",
        "N_opens",
        "N_clicks",
        "N_unsubscribes",
        "N_signups",
        "open_rate_raw",
        "click_rate_raw",
        "unsubscribe_rate_raw",
        "last_send_ts",
        "last_open_ts",
        "last_click_ts",
        "last_unsub_ts",
        "recency_open_days",
        "recency_click_days",
        "recency_unsub_days",
    ]
    for col in cols_order:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols_order]
    out = out.sort_values(["N_sends", "N_opens"], ascending=[False, False]).reset_index(drop=True)
    return out


def _fit_beta_moments(successes: pd.Series, trials: pd.Series) -> EBPrior:
    p = (successes.sum() + 0.5) / (trials.sum() + 1.0)
    v = max(1e-5, p * (1 - p) / 50.0)
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
    """Add empirical-Bayes posterior means for open/click/unsub rates."""
    if df_users.empty:
        return df_users.copy()

    df = df_users.copy()
    if prior_open is None:
        prior_open = _fit_beta_moments(df["N_opens"], df["N_sends"].clip(lower=0))
    if prior_click is None:
        prior_click = _fit_beta_moments(df["N_clicks"], df["N_sends"].clip(lower=0))
    if prior_unsub is None:
        prior_unsub = _fit_beta_moments(df["N_unsubscribes"], df["N_sends"].clip(lower=0))

    pl_df = pl.from_pandas(df, include_index=False).with_columns(
        [
            ((pl.col("N_opens") + prior_open.alpha) / (pl.col("N_sends") + prior_open.alpha + prior_open.beta)).alias("open_rate_eb"),
            ((pl.col("N_clicks") + prior_click.alpha) / (pl.col("N_sends") + prior_click.alpha + prior_click.beta)).alias("click_rate_eb"),
            ((pl.col("N_unsubscribes") + prior_unsub.alpha) / (pl.col("N_sends") + prior_unsub.alpha + prior_unsub.beta)).alias("unsub_rate_eb"),
        ]
    )
    return pl_df.to_pandas()
