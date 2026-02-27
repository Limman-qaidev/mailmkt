# email_marketing/dashboard/recipient_insights.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import polars as pl
import streamlit as st
import re
import os

from email_marketing.analytics.db import load_all_data
from email_marketing.analytics.user_metrics import (
    build_user_aggregates,
    compute_eb_rates,
)

def _extract_topic_from_text(text: str) -> str:
    """Heuristic topic parser from campaign/subject text."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""

    # [Topic] Subject...
    m = re.match(r"^\s*\[([^\]]+)\]\s*", s)
    if m:
        return m.group(1).strip()

    # Topic — Subject / Topic - Subject / Topic: Subject
    for sep in ("—", "-", ":"):
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            if left:
                return left

    # Last resort: take the first words as a readable topic label.
    return " ".join(s.split()[:3]).strip()


def _build_campaign_topic_map(campaigns_df: pd.DataFrame) -> dict[str, str]:
    """Map campaign identifiers/names to topic."""
    if campaigns_df is None or campaigns_df.empty:
        return {}
    if "campaign_id" not in campaigns_df.columns:
        # Try best-effort fallback
        campaigns_df = campaigns_df.rename(columns={campaigns_df.columns[0]: "campaign_id"})

    explicit_topic_col = "topic" if "topic" in campaigns_df.columns else None
    text_cols = [c for c in ("subject", "campaign_name", "name", "campaign") if c in campaigns_df.columns]
    key_cols = [c for c in ("campaign_id", "campaign", "campaign_name", "name") if c in campaigns_df.columns]

    out: dict[str, str] = {}

    def _add_key(key: str, topic: str) -> None:
        k = str(key).strip()
        if not k:
            return
        out[k] = topic
        out[k.lower()] = topic

    for _, row in campaigns_df.iterrows():
        cid = str(row.get("campaign_id", "")).strip()

        topic = ""
        if explicit_topic_col:
            raw_topic = row.get(explicit_topic_col, "")
            topic = "" if pd.isna(raw_topic) else str(raw_topic).strip()

        if not topic:
            for c in text_cols:
                topic = _extract_topic_from_text(str(row.get(c, "")))
                if topic:
                    break
        if not topic and cid:
            topic = _extract_topic_from_text(cid)
        if not topic:
            continue

        for c in key_cols:
            _add_key(str(row.get(c, "")), topic)
        _add_key(cid, topic)

    return out


def _topic_lookup_from_unique_values(
    series_list: list[pd.Series],
    topic_map: dict[str, str],
    invalid_tokens: set[str] | None = None,
) -> dict[str, str]:
    """Build topic lookup from unique raw values, inferring only unresolved keys once."""
    invalid = invalid_tokens or {"", "nan", "none", "null"}
    lookup: dict[str, str] = {}

    # Seed lookup from explicit campaign->topic map (exact + lowercase).
    for k, v in (topic_map or {}).items():
        key = str(k).strip()
        if not key or key.lower() in invalid:
            continue
        topic = "" if pd.isna(v) else str(v).strip()
        if not topic or topic.lower() in invalid:
            continue
        lookup[key] = topic
        lookup[key.lower()] = topic

    # Infer only for unique unresolved values.
    unique_values: set[str] = set()
    for s in series_list:
        if s is None:
            continue
        vals = (
            pd.Series(s)
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda x: (~x.str.lower().isin(invalid)) & (x != "")]
            .unique()
            .tolist()
        )
        unique_values.update(vals)

    for raw in unique_values:
        if raw in lookup or raw.lower() in lookup:
            continue
        inferred = _extract_topic_from_text(raw)
        topic = str(inferred).strip() if inferred is not None else ""
        if topic and topic.lower() not in invalid:
            lookup[raw] = topic
            lookup[raw.lower()] = topic

    return lookup


def _map_topics_from_lookup(
    df: pd.DataFrame,
    topic_lookup: dict[str, str],
    *,
    source_cols: tuple[str, ...] = ("campaign", "campaign_id"),
    invalid_tokens: set[str] | None = None,
) -> pd.Series:
    """Vectorized topic mapping using prebuilt lookup."""
    invalid = invalid_tokens or {"", "nan", "none", "null"}
    topic = pd.Series("", index=df.index, dtype="object")
    for source_col in source_cols:
        if source_col not in df.columns:
            continue
        raw_src = df[source_col]
        raw = raw_src.astype("string").fillna("").str.strip()
        raw_lower = raw.str.lower()
        valid_raw = (raw != "") & ~raw_lower.isin(invalid)
        mapped = raw_lower.map(topic_lookup).fillna("")
        mapped = mapped.where(valid_raw, "")
        topic = topic.where(topic.astype(str).str.strip() != "", mapped)
    return topic.fillna("").astype(str).str.strip()


def _prepare_ev_sd_with_topics(
    events_df: pd.DataFrame,
    sends_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (events_df_enriched, sends_df_enriched) adding 'topic' column if possible.
    Never fails: if it can't enrich, it returns the original dfs with topic="".
    """
    topic_map = _build_campaign_topic_map(campaigns_df)

    ev = events_df.copy() if events_df is not None else pd.DataFrame()
    sd = sends_df.copy() if sends_df is not None else pd.DataFrame()

    if not ev.empty:
        if "email" in ev.columns:
            ev["email"] = ev["email"].astype(str).str.strip().str.lower()
        ts_col = "event_ts" if "event_ts" in ev.columns else ("ts" if "ts" in ev.columns else None)
        if ts_col and "event_ts" not in ev.columns:
            ev["event_ts"] = pd.to_datetime(ev[ts_col], errors="coerce", utc=True)

    invalid_tokens = {"", "nan", "none", "null"}

    if not sd.empty:
        rc = "recipient" if "recipient" in sd.columns else ("email" if "email" in sd.columns else None)
        if rc:
            sd["email"] = sd[rc].astype(str).str.strip().str.lower()
        else:
            sd["email"] = pd.NA

        if "msg_id" in sd.columns and not ev.empty and {"msg_id", "campaign"}.issubset(ev.columns):
            msg2camp = (
                ev.loc[ev["campaign"].notna(), ["msg_id", "campaign"]]
                .drop_duplicates("msg_id")
                .set_index("msg_id")["campaign"]
            )
            if not msg2camp.empty:
                mapped_campaign = sd["msg_id"].map(msg2camp)
                if "campaign" in sd.columns:
                    existing = sd["campaign"]
                    existing_txt = existing.astype(str).str.strip()
                    existing_valid = existing.notna() & ~existing_txt.str.lower().isin(invalid_tokens)
                    sd["campaign"] = existing.where(existing_valid, mapped_campaign)
                else:
                    sd["campaign"] = mapped_campaign

    lookup_series: list[pd.Series] = []
    for df_ in (ev, sd):
        for c in ("campaign", "campaign_id"):
            if c in df_.columns:
                lookup_series.append(df_[c])
    topic_lookup = _topic_lookup_from_unique_values(lookup_series, topic_map, invalid_tokens)

    if not ev.empty:
        ev["topic"] = _map_topics_from_lookup(ev, topic_lookup, invalid_tokens=invalid_tokens)
    else:
        ev["topic"] = pd.Series(dtype="object")

    if not sd.empty:
        sd["topic"] = _map_topics_from_lookup(sd, topic_lookup, invalid_tokens=invalid_tokens)
    else:
        sd["topic"] = pd.Series(dtype="object")

    return ev, sd


def _topic_corpus(ev: pd.DataFrame, sd: pd.DataFrame, signups_df: pd.DataFrame, campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """Corpus per topic: S_total, Y_total (signups) and p0_signup."""
    sends_t = (
        sd.dropna(subset=["topic"])
        .assign(topic=lambda d: d["topic"].astype(str).str.strip())
        .loc[lambda d: d["topic"] != ""]
        .groupby("topic")["msg_id"]
        .nunique()
        .rename("S_total")
    ) if not sd.empty and {"topic", "msg_id"}.issubset(sd.columns) else pd.Series(dtype=float, name="S_total")

    y_ev = (
        ev[ev["event_type"].astype(str).str.lower().eq("signup")]
        .dropna(subset=["topic"])
        .assign(topic=lambda d: d["topic"].astype(str).str.strip())
        .loc[lambda d: d["topic"] != ""]
        .groupby("topic")["msg_id"]
        .nunique()
        .rename("Y_ev")
    ) if not ev.empty and {"event_type", "topic", "msg_id"}.issubset(ev.columns) else pd.Series(dtype=float, name="Y_ev")

    if not signups_df.empty and ("campaign" in signups_df.columns or "campaign_id" in signups_df.columns):
        camp2topic = _build_campaign_topic_map(campaigns_df)
        sgn = _prepare_signups_with_topics(signups_df, camp2topic)
        id_col = "signup_id" if "signup_id" in sgn.columns else None
        if id_col:
            y_tab = (
                sgn.dropna(subset=["topic"])
                .assign(topic=lambda d: d["topic"].astype(str).str.strip())
                .loc[lambda d: d["topic"] != ""]
                .groupby("topic")[id_col]
                .nunique()
                .rename("Y_tab")
            )
        else:
            y_tab = (
                sgn.dropna(subset=["topic"])
                .assign(topic=lambda d: d["topic"].astype(str).str.strip())
                .loc[lambda d: d["topic"] != ""]
                .groupby("topic")
                .size()
                .rename("Y_tab")
            )
    else:
        y_tab = pd.Series(dtype=float, name="Y_tab")

    corp = (
        pd.DataFrame(index=pd.Index([], name="topic"))
        .join(sends_t, how="outer")
        .join(y_ev, how="outer")
        .join(y_tab, how="outer")
        .fillna(0.0)
        .reset_index()
    )

    if corp.empty:
        return pd.DataFrame(columns=["topic", "S_total", "Y_total", "p0_signup"])

    corp["Y_total"] = corp["Y_ev"] + corp["Y_tab"]
    corp["p0_signup"] = (corp["Y_total"] / corp["S_total"]).where(corp["S_total"] > 0, other=0.01)
    return corp[["topic", "S_total", "Y_total", "p0_signup"]]


def _owners_series(ev: pd.DataFrame, signups_df: pd.DataFrame, campaigns_df: pd.DataFrame) -> pd.Series:
    """(email, topic) -> bool if the user has registered (events or table)."""
    if ev is None:
        ev = pd.DataFrame()
    if signups_df is None:
        signups_df = pd.DataFrame()

    if not ev.empty and {"event_type", "email", "topic", "msg_id"}.issubset(ev.columns):
        ev_su = ev[ev["event_type"].astype(str).str.lower().eq("signup")]
        own_ev = (
            ev_su.dropna(subset=["email", "topic"])
            .assign(
                email=lambda d: d["email"].astype(str).str.strip().str.lower(),
                topic=lambda d: d["topic"].astype(str).str.strip(),
            )
            .loc[lambda d: d["topic"] != ""]
            .groupby(["email", "topic"])["msg_id"]
            .nunique()
            .gt(0)
        )
    else:
        own_ev = pd.Series(dtype=bool)

    if not signups_df.empty and {"email"}.issubset(signups_df.columns) and (
        "campaign" in signups_df.columns or "campaign_id" in signups_df.columns
    ):
        camp2topic = _build_campaign_topic_map(campaigns_df)
        sgn = _prepare_signups_with_topics(signups_df, camp2topic)
        id_col = "signup_id" if "signup_id" in sgn.columns else None
        if id_col:
            own_tab = sgn.groupby(["email", "topic"])[id_col].nunique().gt(0)
        else:
            own_tab = sgn.groupby(["email", "topic"]).size().gt(0)
    else:
        own_tab = pd.Series(dtype=bool)

    if own_ev.empty and own_tab.empty:
        owners = pd.Series(dtype=bool)
    elif own_ev.empty:
        owners = own_tab.fillna(False)
    elif own_tab.empty:
        owners = own_ev.fillna(False)
    else:
        owners = own_ev.combine(own_tab, lambda a, b: bool(a) or bool(b)).fillna(False)
    owners.name = "owner"
    return owners


def _prepare_signups_with_topics(
    signups_df: pd.DataFrame,
    camp2topic: dict[str, str],
    topic_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize signups and enrich with topic/topic_norm."""
    if signups_df is None or signups_df.empty:
        return pd.DataFrame(columns=["email", "topic", "topic_norm", "signup_id"])

    sgn = signups_df.copy()
    if "email" in sgn.columns:
        sgn["email"] = sgn["email"].astype(str).str.strip().str.lower()
    else:
        sgn["email"] = pd.NA

    invalid_tokens = {"", "nan", "none", "null"}
    if topic_lookup is None:
        lookup_series = [sgn[c] for c in ("campaign", "campaign_id") if c in sgn.columns]
        topic_lookup = _topic_lookup_from_unique_values(lookup_series, camp2topic, invalid_tokens)

    sgn["topic"] = _map_topics_from_lookup(
        sgn,
        topic_lookup,
        source_cols=("campaign", "campaign_id"),
        invalid_tokens=invalid_tokens,
    )
    sgn = sgn[sgn["topic"] != ""].copy()
    sgn["topic_norm"] = sgn["topic"].str.lower()
    return sgn


def _topic_level_totals(ev_t: pd.DataFrame, sd_t: pd.DataFrame, signups_t: pd.DataFrame) -> pd.DataFrame:
    """Build per-topic totals used by Campaign Planning priors."""
    topic_labels: list[pl.DataFrame] = []
    keys: list[pl.DataFrame] = []
    s_tot = pl.DataFrame(schema={"topic_norm": pl.Utf8, "S_tot": pl.Float64})
    o_tot = pl.DataFrame(schema={"topic_norm": pl.Utf8, "O_tot": pl.Float64})
    c_tot = pl.DataFrame(schema={"topic_norm": pl.Utf8, "C_tot": pl.Float64})
    y_ev_tot = pl.DataFrame(schema={"topic_norm": pl.Utf8, "Y_ev_tot": pl.Float64})
    y_tab_tot = pl.DataFrame(schema={"topic_norm": pl.Utf8, "Y_tab_tot": pl.Float64})

    if sd_t is not None and not sd_t.empty and {"topic", "msg_id"}.issubset(sd_t.columns):
        sd = pl.from_pandas(sd_t, include_index=False).with_columns(
            [
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
            ]
        ).filter(pl.col("topic") != "")
        if not sd.is_empty():
            keys.append(sd.select("topic_norm"))
            topic_labels.append(sd.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            s_tot = sd.group_by("topic_norm").agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("S_tot"))

    if ev_t is not None and not ev_t.empty and {"topic", "event_type", "msg_id"}.issubset(ev_t.columns):
        ev = pl.from_pandas(ev_t, include_index=False).with_columns(
            [
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
                pl.col("event_type").cast(pl.Utf8).str.to_lowercase().alias("event_type"),
            ]
        ).filter(pl.col("topic") != "")
        if not ev.is_empty():
            keys.append(ev.select("topic_norm"))
            topic_labels.append(ev.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            o_tot = ev.filter(pl.col("event_type") == "open").group_by("topic_norm").agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("O_tot"))
            c_tot = ev.filter(pl.col("event_type") == "click").group_by("topic_norm").agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("C_tot"))
            y_ev_tot = ev.filter(pl.col("event_type") == "signup").group_by("topic_norm").agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("Y_ev_tot"))

    if signups_t is not None and not signups_t.empty and "topic_norm" in signups_t.columns:
        sg = pl.from_pandas(signups_t, include_index=False).with_columns(
            [
                pl.col("topic_norm").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic") if "topic" in signups_t.columns else pl.lit(None).cast(pl.Utf8).alias("topic"),
            ]
        ).filter(pl.col("topic_norm") != "")
        if not sg.is_empty():
            keys.append(sg.select("topic_norm"))
            if "topic" in sg.columns:
                topic_labels.append(sg.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            if "signup_id" in sg.columns:
                y_tab_tot = sg.group_by("topic_norm").agg(pl.col("signup_id").n_unique().cast(pl.Float64).alias("Y_tab_tot"))
            else:
                y_tab_tot = sg.group_by("topic_norm").len().rename({"len": "Y_tab_tot"}).with_columns(pl.col("Y_tab_tot").cast(pl.Float64))

    if not keys:
        return pd.DataFrame(columns=["topic", "topic_norm", "S_tot", "O_tot", "C_tot", "Y_ev_tot", "Y_tab_tot"])

    base = pl.concat(keys, how="vertical_relaxed").unique()
    labels = (
        pl.concat(topic_labels, how="vertical_relaxed")
        .group_by("topic_norm")
        .agg(pl.col("topic").drop_nulls().first().alias("topic"))
        if topic_labels
        else pl.DataFrame(schema={"topic_norm": pl.Utf8, "topic": pl.Utf8})
    )

    out = (
        base.join(s_tot, on="topic_norm", how="left")
        .join(o_tot, on="topic_norm", how="left")
        .join(c_tot, on="topic_norm", how="left")
        .join(y_ev_tot, on="topic_norm", how="left")
        .join(y_tab_tot, on="topic_norm", how="left")
        .join(labels, on="topic_norm", how="left")
        .with_columns(
            [
                pl.col("topic").fill_null(pl.col("topic_norm")),
                pl.col("S_tot").fill_null(0.0),
                pl.col("O_tot").fill_null(0.0),
                pl.col("C_tot").fill_null(0.0),
                pl.col("Y_ev_tot").fill_null(0.0),
                pl.col("Y_tab_tot").fill_null(0.0),
            ]
        )
    )
    return out.select(["topic", "topic_norm", "S_tot", "O_tot", "C_tot", "Y_ev_tot", "Y_tab_tot"]).to_pandas()


def _topic_email_rollup(ev_t: pd.DataFrame, sd_t: pd.DataFrame, signups_t: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregate per (topic,email): S,O,C,U,Q,Y,owner,last_open_ts."""
    keys: list[pl.DataFrame] = []
    topic_labels: list[pl.DataFrame] = []
    s_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "S": pl.Float64})
    o_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "O": pl.Float64})
    c_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "C": pl.Float64})
    u_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "U": pl.Float64})
    q_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "Q": pl.Float64})
    y_ev_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "Y_ev": pl.Float64})
    y_tab_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "Y_tab": pl.Float64})
    last_open_df = pl.DataFrame(schema={"topic_norm": pl.Utf8, "email": pl.Utf8, "last_open_ts": pl.Datetime})

    if sd_t is not None and not sd_t.empty and {"topic", "email", "msg_id"}.issubset(sd_t.columns):
        sd = pl.from_pandas(sd_t, include_index=False).with_columns(
            [
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic"),
                pl.col("email").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("email"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
            ]
        ).filter((pl.col("topic") != "") & (pl.col("email") != ""))
        if not sd.is_empty():
            keys.append(sd.select(["topic_norm", "email"]))
            topic_labels.append(sd.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            s_df = sd.group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("S"))

    if ev_t is not None and not ev_t.empty and {"topic", "email", "event_type", "msg_id"}.issubset(ev_t.columns):
        ev = pl.from_pandas(ev_t, include_index=False).with_columns(
            [
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic"),
                pl.col("email").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("email"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
                pl.col("event_type").cast(pl.Utf8).str.to_lowercase().alias("event_type"),
            ]
        ).filter((pl.col("topic") != "") & (pl.col("email") != ""))
        if not ev.is_empty():
            keys.append(ev.select(["topic_norm", "email"]))
            topic_labels.append(ev.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            o_df = ev.filter(pl.col("event_type") == "open").group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("O"))
            c_df = ev.filter(pl.col("event_type") == "click").group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("C"))
            u_df = ev.filter(pl.col("event_type") == "unsubscribe").group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("U"))
            q_df = ev.filter(pl.col("event_type") == "complaint").group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("Q"))
            y_ev_df = ev.filter(pl.col("event_type") == "signup").group_by(["topic_norm", "email"]).agg(pl.col("msg_id").n_unique().cast(pl.Float64).alias("Y_ev"))
            if "event_ts" in ev.columns:
                last_open_df = ev.filter(pl.col("event_type") == "open").group_by(["topic_norm", "email"]).agg(pl.col("event_ts").max().alias("last_open_ts"))

    if signups_t is not None and not signups_t.empty and {"topic_norm", "email"}.issubset(signups_t.columns):
        sg = pl.from_pandas(signups_t, include_index=False).with_columns(
            [
                pl.col("topic_norm").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("topic_norm"),
                pl.col("email").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("email"),
                pl.col("topic").cast(pl.Utf8).str.strip_chars().alias("topic") if "topic" in signups_t.columns else pl.lit(None).cast(pl.Utf8).alias("topic"),
            ]
        ).filter((pl.col("topic_norm") != "") & (pl.col("email") != ""))
        if not sg.is_empty():
            keys.append(sg.select(["topic_norm", "email"]))
            if "topic" in sg.columns:
                topic_labels.append(sg.group_by("topic_norm").agg(pl.col("topic").first().alias("topic")))
            if "signup_id" in sg.columns:
                y_tab_df = sg.group_by(["topic_norm", "email"]).agg(pl.col("signup_id").n_unique().cast(pl.Float64).alias("Y_tab"))
            else:
                y_tab_df = sg.group_by(["topic_norm", "email"]).len().rename({"len": "Y_tab"}).with_columns(pl.col("Y_tab").cast(pl.Float64))

    if not keys:
        return pd.DataFrame(columns=["topic", "topic_norm", "email", "S", "O", "C", "U", "Q", "Y", "owner", "last_open_ts"])

    base = pl.concat(keys, how="vertical_relaxed").unique()
    labels = (
        pl.concat(topic_labels, how="vertical_relaxed")
        .group_by("topic_norm")
        .agg(pl.col("topic").drop_nulls().first().alias("topic"))
        if topic_labels
        else pl.DataFrame(schema={"topic_norm": pl.Utf8, "topic": pl.Utf8})
    )

    out = (
        base.join(s_df, on=["topic_norm", "email"], how="left")
        .join(o_df, on=["topic_norm", "email"], how="left")
        .join(c_df, on=["topic_norm", "email"], how="left")
        .join(u_df, on=["topic_norm", "email"], how="left")
        .join(q_df, on=["topic_norm", "email"], how="left")
        .join(y_ev_df, on=["topic_norm", "email"], how="left")
        .join(y_tab_df, on=["topic_norm", "email"], how="left")
        .join(last_open_df, on=["topic_norm", "email"], how="left")
        .join(labels, on="topic_norm", how="left")
        .with_columns(
            [
                pl.col("topic").fill_null(pl.col("topic_norm")),
                pl.col("S").fill_null(0.0),
                pl.col("O").fill_null(0.0),
                pl.col("C").fill_null(0.0),
                pl.col("U").fill_null(0.0),
                pl.col("Q").fill_null(0.0),
                pl.col("Y_ev").fill_null(0.0),
                pl.col("Y_tab").fill_null(0.0),
            ]
        )
        .with_columns(
            [
                ((pl.col("Y_ev") > 0) | (pl.col("Y_tab") > 0)).cast(pl.Int64).alias("Y"),
                ((pl.col("Y_ev") > 0) | (pl.col("Y_tab") > 0)).alias("owner"),
            ]
        )
    )
    return out.select(
        ["topic", "topic_norm", "email", "S", "O", "C", "U", "Q", "Y_ev", "Y", "owner", "last_open_ts"]
    ).to_pandas()


def _db_fingerprint(paths: Tuple[str, str, str]) -> str:
    """
    Build a stable fingerprint for cache invalidation based on file mtime + size.
    This avoids hashing huge DataFrames on every rerun.
    """
    parts: list[str] = []
    for p in paths:
        try:
            st_ = os.stat(p)
            parts.append(f"{p}:{st_.st_mtime_ns}:{st_.st_size}")
        except FileNotFoundError:
            parts.append(f"{p}:missing")
    return "|".join(parts)


@st.cache_resource(show_spinner=False)
def _cached_customer360_bundle_by_fp(
    fp: str, events_db: str, sends_db: str, campaigns_db: str
):
    """
    Cached bundle for Customer 360 page.
    Heavy transforms live here so reruns don't recompute them.
    NOTE: fp is only used as a cache key.
    """
    events, sends, campaigns, signups = load_all_data(events_db, sends_db, campaigns_db)

    # Normalize dtypes once
    for df_ in (events, sends, campaigns, signups):
        if "campaign" in df_.columns:
            df_["campaign"] = df_["campaign"].astype(str)

    # Ensure email in events (join from sends if needed)
    events_prepared = _prepare_events_with_email(events, sends)

    camp2topic = _build_campaign_topic_map(campaigns)

    # Build topic-aware versions + topic corpus + ownership map
    ev_t, sd_t = _prepare_ev_sd_with_topics(events_prepared, sends, campaigns)
    corpus = _topic_corpus(ev_t, sd_t, signups, campaigns)
    owners_mi = _owners_series(ev_t, signups, campaigns)
    signups_t = _prepare_signups_with_topics(signups, camp2topic)
    topic_totals = _topic_level_totals(ev_t, sd_t, signups_t)
    topic_email_rollup = _topic_email_rollup(ev_t, sd_t, signups_t)

    # Global user aggregates (Recipient Detail tab)
    users_df = build_user_aggregates(events_prepared, sends, signups)
    users_df = compute_eb_rates(users_df) if not users_df.empty else users_df

    return (
        events_prepared,
        sends,
        campaigns,
        signups,
        ev_t,
        sd_t,
        corpus,
        owners_mi,
        users_df,
        camp2topic,
        signups_t,
        topic_totals,
        topic_email_rollup,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_customer360_ui_derivatives_by_fp(
    fp: str, events_db: str, sends_db: str, campaigns_db: str
) -> tuple[list[str], list[str], list[str]]:
    """Small UI derivatives cached by DB fingerprint to speed reruns."""
    (
        _events,
        _sends,
        _campaigns,
        _signups,
        _ev_t,
        _sd_t,
        corpus,
        _owners_mi,
        users_df,
        _camp2topic,
        _signups_t,
        _topic_totals,
        _topic_email_rollup,
    ) = _cached_customer360_bundle_by_fp(fp, events_db, sends_db, campaigns_db)

    emails_sorted = users_df["email"].dropna().astype(str).sort_values().unique().tolist()
    domain_options = sorted([d for d in users_df["domain"].dropna().unique().tolist() if isinstance(d, str)])

    topic_source = corpus.copy()
    if "S_total" in topic_source.columns:
        topic_source = topic_source[pd.to_numeric(topic_source["S_total"], errors="coerce").fillna(0) > 0]
    topic_options = (
        topic_source["topic"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .sort_values()
        .unique()
        .tolist()
    )
    return emails_sorted, domain_options, topic_options


    # ---------------------- Small helpers ----------------------
def _ensure_num_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure columns exist and are numeric (filled with 0.0)."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _recipient_topic_metrics(
    picked_email: str,
    events: pd.DataFrame,
    sends: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-recipient metrics by topic: counts and rates."""
    if not picked_email:
        return pd.DataFrame()

    # Normaliza email y event_ts
    ev = events.copy()
    ev["email"] = ev["email"].astype(str).str.strip().str.lower()
    ts_col = "event_ts" if "event_ts" in ev.columns else ("ts" if "ts" in ev.columns else None)
    if ts_col:
        ev["event_ts"] = pd.to_datetime(ev[ts_col], errors="coerce", utc=True)

    sd = sends.copy()
    # Hallar columna de recipient en sends
    rc = "recipient" if "recipient" in sd.columns else ("email" if "email" in sd.columns else None)
    if rc:
        sd["email"] = sd[rc].astype(str).str.strip().str.lower()
    else:
        sd["email"] = pd.NA
    # msg_id -> campaign desde events
    if "campaign" in ev.columns and "msg_id" in ev.columns:
        msg2camp = (
            ev.loc[ev["campaign"].notna(), ["msg_id", "campaign"]]
            .drop_duplicates("msg_id")
            .set_index("msg_id")["campaign"]
        )
    else:
        msg2camp = pd.Series(dtype=str)

    # Mapa campaign -> topic
    camp2topic = _build_campaign_topic_map(campaigns)

    # ---- Sends por topic para este email ----
    sd_e = sd[(sd["email"] == picked_email)].copy()
    if "msg_id" in sd_e.columns and not msg2camp.empty:
        sd_e["campaign"] = sd_e["msg_id"].map(msg2camp)
    else:
        sd_e["campaign"] = sd_e.get("campaign", None)
    sd_e["topic"] = sd_e["campaign"].map(camp2topic).fillna(
        sd_e["campaign"].map(_extract_topic_from_text).fillna("Other")
    )
    sends_topic = (
        sd_e.dropna(subset=["topic"])
            .groupby("topic")["msg_id"].nunique()
            .rename("N_sends_topic")
    )

    # ---- Eventos opens/clicks/unsubs por topic para este email ----
    ev_e = ev[(ev["email"] == picked_email)].copy()
    ev_e["topic"] = ev_e["campaign"].map(camp2topic).fillna(
        ev_e["campaign"].map(_extract_topic_from_text).fillna("Other")
    )
    # únicos msg por topic y tipo de evento
    def _count(evtype: str) -> pd.Series:
        m = ev_e[ev_e["event_type"].astype(str).str.lower().eq(evtype)]
        return m.dropna(subset=["topic"]).groupby("topic")["msg_id"].nunique()

    opens_t = _count("open").rename("N_opens_topic")
    clicks_t = _count("click").rename("N_clicks_topic")
    unsubs_t = _count("unsubscribe").rename("N_unsubs_topic")

    df = (
        pd.DataFrame(index=pd.Index([], name="topic"))
        .join(sends_topic, how="outer")
        .join(opens_t, how="outer")
        .join(clicks_t, how="outer")
        .join(unsubs_t, how="outer")
        .fillna(0)
        .reset_index()
    )
    if df.empty:
        return df

    # tasas por topic (con protección por 0)
    df["open_rate"] = (df["N_opens_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["ctr"] = (df["N_clicks_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["unsub_rate"] = (df["N_unsubs_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)

    # ordenar por envíos y apertura
    df = df.sort_values(["N_sends_topic", "N_opens_topic"], ascending=[False, False]).reset_index(drop=True)
    return df


def _default_db_paths() -> tuple[str, str, str]:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    return (
        str(data_dir / "email_events.db"),
        str(data_dir / "email_map.db"),
        str(data_dir / "campaigns.db"),
    )


def _as_utc(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)


def _prepare_events_with_email(events: pd.DataFrame, sends: pd.DataFrame) -> pd.DataFrame:
    """Ensure events have an 'email' column (fallback joining sends if needed)."""
    ev = events.copy()
    if "email" not in ev.columns or ev["email"].isna().all():
        if "msg_id" in ev.columns and "msg_id" in sends.columns:
            # Choose recipient-like column in sends
            rc = "recipient" if "recipient" in sends.columns else ("email" if "email" in sends.columns else None)
            if rc:
                ev = ev.merge(sends[["msg_id", rc]].rename(columns={rc: "email"}), on="msg_id", how="left")
    ev["email"] = ev["email"].astype(str).str.strip().str.lower()
    # Timestamp
    ts_col = "event_ts" if "event_ts" in ev.columns else ("ts" if "ts" in ev.columns else None)
    if ts_col:
        ev["event_ts"] = pd.to_datetime(ev[ts_col], errors="coerce", utc=True)
    return ev


def _heatmap_df(ev: pd.DataFrame, email: str, event_type: str) -> pd.DataFrame:
    """Build hour×weekday matrix for a given email and event type."""
    if ev.empty:
        return pd.DataFrame()
    df = ev[
        (ev["email"] == email)
        & (ev["event_type"].astype(str).str.lower() == event_type)
        & (ev["event_ts"].notna())
    ][["event_ts"]].copy()
    if df.empty:
        return pd.DataFrame()
    df["hour"] = df["event_ts"].dt.hour
    df["dow"] = df["event_ts"].dt.dayofweek  # 0=Mon
    mat = df.groupby(["dow", "hour"]).size().reset_index(name="count")
    # Friendly labels
    mat["weekday"] = mat["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    return mat[["weekday", "hour", "count"]]


def _request_nav_to_editor(recipients: list[str], subject: str) -> None:
    st.session_state["mo_recipients"] = recipients
    st.session_state["mo_subject_live"] = subject
    st.session_state.pop("nav", None)
    st.session_state["pending_nav"] = "Email Editor"
    st.rerun()


def _recipient_topic_detail_from_rollup(
    picked_email: str,
    topic_email_rollup: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Build per-recipient topic detail from pre-aggregated topic_email_rollup."""
    cols = [
        "topic",
        "N_sends_topic",
        "N_opens_topic",
        "N_clicks_topic",
        "N_unsubs_topic",
        "N_signups_topic",
        "open_rate",
        "ctr",
        "unsub_rate",
        "p_signup",
        "Registered",
    ]
    if not picked_email or topic_email_rollup is None or topic_email_rollup.empty:
        return pd.DataFrame(columns=cols)

    df = topic_email_rollup[
        topic_email_rollup["email"].astype(str).str.strip().str.lower().eq(str(picked_email).strip().lower())
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Keep previous semantics for Recipient Detail signups (event signups if available).
    signups_src = "Y_ev" if "Y_ev" in df.columns else ("Y" if "Y" in df.columns else None)
    if signups_src is None:
        df["N_signups_topic"] = 0.0
    else:
        df["N_signups_topic"] = pd.to_numeric(df[signups_src], errors="coerce").fillna(0.0)

    df["N_sends_topic"] = pd.to_numeric(df.get("S", 0.0), errors="coerce").fillna(0.0)
    df["N_opens_topic"] = pd.to_numeric(df.get("O", 0.0), errors="coerce").fillna(0.0)
    df["N_clicks_topic"] = pd.to_numeric(df.get("C", 0.0), errors="coerce").fillna(0.0)
    df["N_unsubs_topic"] = pd.to_numeric(df.get("U", 0.0), errors="coerce").fillna(0.0)
    owner_col = df["owner"] if "owner" in df.columns else pd.Series(False, index=df.index)
    df["Registered"] = owner_col.astype(bool)

    df["open_rate"] = (df["N_opens_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["ctr"] = (df["N_clicks_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)
    df["unsub_rate"] = (df["N_unsubs_topic"] / df["N_sends_topic"]).where(df["N_sends_topic"] > 0, other=0.0)

    p0 = corpus[["topic", "p0_signup"]] if {"topic", "p0_signup"}.issubset(corpus.columns) else pd.DataFrame(columns=["topic", "p0_signup"])
    df = df.merge(p0, on="topic", how="left").fillna({"p0_signup": 0.01})
    df["p_signup"] = (df["N_signups_topic"] + float(alpha) * df["p0_signup"]) / (df["N_sends_topic"] + float(alpha))

    out = df[cols].sort_values(["p_signup", "N_sends_topic"], ascending=[False, False]).reset_index(drop=True)
    return out


def render_recipient_insights() -> None:
    """Customer 360º (Recipient Insights) with tabs:
       - Campaign Planning (topic audience + best next campaigns)
       - Recipient Detail (per email)

    Probability: p_signup (Empirical-Bayes) per topic.
    Registered: detected via 'signup' events and/or 'signups' table.
    """
    import re
    import plotly.express as px

    # ---------------------- Internal constants (hidden) ----------------------
    ALPHA = 5.0          # EB prior strength
    TOPN_DEFAULT = 500   # default top N in topic audience

    # ---------------------- Load & harmonize data ----------------------
    st.title("Customer 360º")

    events_db, sends_db, campaigns_db = _default_db_paths()
    try:
        fp = _db_fingerprint((events_db, sends_db, campaigns_db))
        (
            events,
            sends,
            campaigns,
            signups,
            ev_t,
            sd_t,
            corpus,
            owners_mi,
            users_df,
            camp2topic,
            signups_t,
            topic_totals,
            topic_email_rollup,
        ) = _cached_customer360_bundle_by_fp(fp, events_db, sends_db, campaigns_db)
        emails_sorted_cached, domain_options_cached, topic_options_cached = _cached_customer360_ui_derivatives_by_fp(
            fp, events_db, sends_db, campaigns_db
        )
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    if users_df.empty:
        st.info("No recipient data available.")
        return

    # --- persistent audience state (for Campaign Planning) ---
    st.session_state.setdefault("aud_built", False)
    st.session_state.setdefault("aud_base", None)
    st.session_state.setdefault("aud_topic", "")

    # selection controls default values
    st.session_state.setdefault("aud_sel_mode", "Top N")
    st.session_state.setdefault("aud_sel_n", 500)
    st.session_state.setdefault("aud_sel_pct", 20)
    st.session_state.setdefault("aud_sel_target", 50.0)
    st.session_state.setdefault("aud_bands_pick", ["Very High","High"])

    # ========================== Tabs ==========================
    tab_campaign, tab_recipient = st.tabs(["Campaign Planning", "Recipient Detail"])

    # ======================= TAB: Recipient Detail =======================
    with tab_recipient:
        st.markdown("#### Recipient detail")
        # Cachea emails_sorted por fingerprint (evita sort+unique en cada rerun)
        if st.session_state.get("_emails_sorted_fp") != fp:
            st.session_state["_emails_sorted_fp"] = fp
            st.session_state["_emails_sorted"] = emails_sorted_cached

        emails_sorted = st.session_state.get("_emails_sorted", emails_sorted_cached)
        use_select = len(emails_sorted) <= 10000
        if use_select:
            picked = st.selectbox("Find recipient", options=[""] + emails_sorted, index=0)
        else:
            picked = st.text_input("Find recipient (type exact email; case-insensitive)").strip().lower()
            if picked and picked not in set(emails_sorted):
                st.warning("Email not found in current data.")

        if not picked:
            st.info("Pick a recipient to see their 360º view.")
        else:
            row = users_df.loc[users_df["email"] == picked]
            if row.empty:
                st.warning("Recipient not found after filters.")
            else:
                r = row.iloc[0]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Sends", f"{int(r['N_sends']):,}")
                k2.metric("Opens", f"{int(r['N_opens']):,}", delta=f"{r['open_rate_raw']:.1%}")
                k3.metric("Clicks", f"{int(r['N_clicks']):,}", delta=f"{r['click_rate_raw']:.1%}")
                k4.metric("Unsubs", f"{int(r['N_unsubscribes']):,}", delta=f"{r['unsubscribe_rate_raw']:.1%}", delta_color="inverse")
                k5.metric("EB open rate", f"{r['open_rate_eb']:.1%}")

                # ---- Per-topic detail with Registered + p_signup (EB) ----
                st.subheader("Topic interest (per recipient)")
                tdf = _recipient_topic_detail_from_rollup(
                    picked, topic_email_rollup, corpus, alpha=ALPHA
                )

                if tdf.empty:
                    st.info("No topic-level information is available for this recipient.")
                else:
                    # View controls
                    col_tt1, col_tt2 = st.columns([1, 1])
                    top_only = col_tt1.checkbox("Top topics only", value=True, help="Show only the K most relevant topics for this recipient.")
                    top_k = int(col_tt2.number_input("K", min_value=3, max_value=30, value=8, step=1))

                    tdf = tdf.sort_values(["p_signup", "N_sends_topic"], ascending=[False, False])
                    topic_view = tdf.head(top_k) if top_only else tdf

                    # Heatmap of rates (visual summary)
                    heat = topic_view.melt(
                        id_vars=["topic"],
                        value_vars=["open_rate", "ctr", "unsub_rate"],
                        var_name="metric",
                        value_name="value",
                    )
                    metric_labels = {"open_rate": "Open rate", "ctr": "CTR", "unsub_rate": "Unsub rate"}
                    heat["metric"] = heat["metric"].map(metric_labels)
                    counts = topic_view[["topic", "N_sends_topic", "N_opens_topic", "N_clicks_topic", "N_unsubs_topic"]]
                    heat = heat.merge(counts, on="topic", how="left")

                    c1_v, c2_v = st.columns([2, 3], gap="large")
                    with c1_v:
                        st.caption("Rates heatmap (per topic)")
                        fig_hm = px.density_heatmap(
                            heat,
                            x="metric", y="topic", z="value",
                            histfunc="avg", text_auto=True, color_continuous_scale="Blues", title=None,
                        )
                        fig_hm.update_layout(coloraxis_colorbar=dict(title="Rate"))
                        fig_hm.update_traces(
                            hovertemplate=(
                                "Topic: %{y}<br>Metric: %{x}<br>Value: %{z:.1%}"
                                "<br>Sends: %{customdata[0]:,}"
                                "<br>Opens: %{customdata[1]:,}"
                                "<br>Clicks: %{customdata[2]:,}"
                                "<br>Unsubs: %{customdata[3]:,}"
                            ),
                            customdata=heat[["N_sends_topic", "N_opens_topic", "N_clicks_topic", "N_unsubs_topic"]].values,
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)

                    with c2_v:
                        st.caption("Per-topic detail")
                        pretty = topic_view.rename(columns={
                            "N_sends_topic": "Sends",
                            "N_opens_topic": "Opens",
                            "N_clicks_topic": "Clicks",
                            "N_unsubs_topic": "Unsubs",
                            "open_rate": "Open rate",
                            "ctr": "CTR",
                            "unsub_rate": "Unsub rate",
                            "p_signup": "p_signup",
                        })
                        st.dataframe(
                            pretty[["topic", "Registered", "Sends", "Opens", "Clicks", "Unsubs", "Open rate", "CTR", "Unsub rate", "p_signup"]],
                            use_container_width=True, height=360,
                        )

    # ======================= TAB: Campaign Planning =======================
    with tab_campaign:

        st.markdown("#### Campaign Planning")

        # Goal: acquisition excludes existing owners for the chosen topic
        goal = st.radio(
            "Goal",
            ["New acquisition (exclude existing customers)", "All recipients"],
            index=0,
            help=("‘New acquisition’: excludes emails already registered in that "
                "product/topic. ‘All recipients’: includes all eligible recipients."),
            horizontal=False,
        )
        exclude_owners = goal.startswith("New acquisition")

        # --- Topic audience ---
        st.subheader("Topic audience")
        topic_options = topic_options_cached
        if not topic_options:
            st.info("No topics available to build an audience.")
        else:
            chosen_topic = st.selectbox(
                "Topic",
                options=topic_options,
                index=0,
                help="Select the topic/campaign for which you want to build an audience.",
            )

            # How many we ultimately want to keep (used later in selection phase)
            topN_default = 500
            st.session_state.setdefault("aud_sel_n", topN_default)

            # ---- Refinement controls (persisted) ----
            with st.expander("Refine audience (optional)"):
                col_f1, col_f2, col_f3 = st.columns([1.1, 1.1, 1.2])

                col_f1.number_input(
                    "Min. sends in topic",
                    min_value=0,
                    value=st.session_state.get("refine_min_s_topic", 1),  # recommended default 1
                    step=1,
                    key="refine_min_s_topic",
                    help="Minimum number of historical sends in this topic to consider the recipient.",
                )

                col_f2.number_input(
                    "Max. days since last open in topic (0 = disabled)",
                    min_value=0,
                    value=st.session_state.get("refine_max_recency_days", 0),
                    step=10,
                    key="refine_max_recency_days",
                    help="0 = no recency filter. If >0, requires an open within that number of days (in this topic).",
                )

                domain_options = domain_options_cached
                col_f3.multiselect(
                    "Domains (include; empty=all)",
                    options=domain_options,
                    default=st.session_state.get("refine_domains_pick", []),
                    key="refine_domains_pick",
                    help="Restrict the audience to specific email domains.",
                )

            # ---------- PHASE A: Build base audience (runs once when you click the button) ----------
            if st.button("Build audience", key="build_aud"):
                # Read refinements from session (robust on reruns)
                min_s_topic       = int(st.session_state.get("refine_min_s_topic", 1))
                max_recency_days  = int(st.session_state.get("refine_max_recency_days", 0))
                domains_pick      = st.session_state.get("refine_domains_pick", []) or []

                # Slice frames for the chosen topic (normalized compare)
                topic_key = str(chosen_topic).strip().lower()
                topic_df = topic_email_rollup[
                    topic_email_rollup["topic_norm"].astype(str).str.strip().str.lower().eq(topic_key)
                ].copy()
                tstats = topic_totals[
                    topic_totals["topic_norm"].astype(str).str.strip().str.lower().eq(topic_key)
                ]

                # p0 for this topic (fallbacks)
                def _clamp(x, lo, hi):
                    try:
                        return float(max(lo, min(hi, x)))
                    except Exception:
                        return lo

                if tstats.empty:
                    S_tot = O_tot = C_tot = Y_ev_tot = Y_tab_tot = 0.0
                else:
                    row_t = tstats.iloc[0]
                    S_tot = float(row_t.get("S_tot", 0.0))
                    O_tot = float(row_t.get("O_tot", 0.0))
                    C_tot = float(row_t.get("C_tot", 0.0))
                    Y_ev_tot = float(row_t.get("Y_ev_tot", 0.0))
                    Y_tab_tot = float(row_t.get("Y_tab_tot", 0.0))

                Y_tot = max(Y_ev_tot, Y_tab_tot)

                p0_open  = _clamp((O_tot / S_tot) if S_tot > 0 else 0.05, 1e-4, 0.9)
                p0_click = _clamp((C_tot / S_tot) if S_tot > 0 else 0.02, 1e-4, 0.9)
                p0_sc    = _clamp((Y_tot / C_tot) if C_tot > 0 else 0.10, 1e-4, 0.9)  # P(signup|click)

                # Owners in this topic (pre-aggregated)
                if topic_df.empty:
                    own_topic = pd.Series(dtype=bool)
                else:
                    own_topic = topic_df.set_index("email")["owner"].astype(bool)

                # Assemble audience base from pre-aggregated topic-email rollup
                if topic_df.empty:
                    df = pd.DataFrame(columns=["email", "S", "Y", "O", "C", "U", "Q"])
                else:
                    df = topic_df[["email", "S", "Y", "O", "C", "U", "Q"]].copy()
                # Ensure numeric columns exist
                for col in ("S","Y","O","C","U","Q"):
                    if col not in df.columns:
                        df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

                if df.empty or df["S"].sum() == 0:
                    st.warning("No eligible recipients for this topic.")
                else:
                    # EB-smoothed probability model
                    ALPHA_O, ALPHA_C, ALPHA_SC = 2.0, 2.0, 2.0
                    EPS = 1e-9

                    S_ser = df["S"]
                    O_ser = df["O"]
                    C_ser = df["C"]
                    Y_ser = df["Y"]

                    p_open_hat  = (O_ser + ALPHA_O * p0_open)  / (S_ser + ALPHA_O)
                    p_click_hat = (C_ser + ALPHA_C * p0_click) / (S_ser + ALPHA_C)
                    p_c_given_o = (p_click_hat / p_open_hat.clip(lower=EPS)).clip(upper=1.0)
                    p_sc_hat    = (Y_ser + ALPHA_SC * p0_sc) / (C_ser + ALPHA_SC)

                    df["p_signup"] = (p_open_hat * p_c_given_o * p_sc_hat).fillna(0.0)

                    # Eligibility filters (except Top N/%/Target)
                    # 1) min exposure
                    df = df[df["S"] >= float(min_s_topic)]
                    # 2) exclude owners if acquisition
                    if exclude_owners:
                        mask_owner = df["email"].map(own_topic).fillna(False)
                        df = df[~mask_owner]
                    # 3) exclude unsub/complaint
                    if "U" not in df.columns: df["U"] = 0.0
                    if "Q" not in df.columns: df["Q"] = 0.0
                    df = df[(df["U"] == 0.0) & (df["Q"] == 0.0)]
                    # 4) recency within topic
                    if max_recency_days > 0 and not topic_df.empty and "last_open_ts" in topic_df.columns:
                        last_open = topic_df.set_index("email")["last_open_ts"]
                        now_utc = pd.Timestamp.now(tz="UTC")
                        recency_ok = last_open.apply(lambda ts: (now_utc - ts).days <= max_recency_days if pd.notna(ts) else False)
                        df = df[df["email"].isin(recency_ok[recency_ok].index)]
                    # 5) domain filter
                    if domains_pick:
                        df = df[df["email"].str.split("@").str[-1].isin(domains_pick)]

                    if df.empty:
                        st.warning("No eligible recipients after applying filters.")
                    else:
                        # Persist sorted base audience once (deterministic ordering)
                        df_base = df.sort_values(by=["p_signup", "S"], ascending=[False, False]).reset_index(drop=True)
                        st.session_state["aud_base"]  = df_base
                        st.session_state["aud_topic"] = chosen_topic
                        st.session_state["aud_built"] = True
                        st.rerun()

            # ---------- PHASE B: Always-on selection UI (won't reset on radio/slider changes) ----------
            base = st.session_state.get("aud_base")
            if st.session_state.get("aud_built") and isinstance(base, pd.DataFrame) and not base.empty:
                st.markdown(f"**Topic:** {st.session_state.get('aud_topic','')}")

                # Probability bands based on current base distribution
                q = base["p_signup"].quantile([0.50, 0.75, 0.90]).to_dict()
                q50, q75, q90 = float(q.get(0.50, 0.0)), float(q.get(0.75, 0.0)), float(q.get(0.90, 0.0))

                def _band(p: float) -> str:
                    if p >= q90: return "Very High"
                    if p >= q75: return "High"
                    if p >= q50: return "Medium"
                    return "Low"

                dfb = base.copy()
                dfb["prob_band"] = dfb["p_signup"].astype(float).map(_band)

                band_sizes = (
                    dfb.groupby("prob_band")["email"].nunique()
                    .reindex(["Very High","High","Medium","Low"])
                    .fillna(0).astype(int)
                )
                st.caption("Band sizes → " + " | ".join([f"{b}: {band_sizes.get(b,0):,}" for b in ["Very High","High","Medium","Low"]]))

                bands_pick = st.multiselect(
                    "Include probability bands",
                    options=["Very High","High","Medium","Low"],
                    default=st.session_state.get("aud_bands_pick", ["Very High","High"]),
                    key="aud_bands_pick",
                    help="Choose which probability bands to include in the audience."
                )
                if bands_pick:
                    dfb = dfb[dfb["prob_band"].isin(bands_pick)]

                # ======= Interactive selection over the built audience =======
                base = st.session_state.get("aud_base")
                if st.session_state.get("aud_built") and isinstance(base, pd.DataFrame) and not base.empty:
                    st.markdown(f"**Topic:** {st.session_state.get('aud_topic','')}")

                    # Probability bands (ya calculadas arriba en dfb)
                    try:
                        aud_work = dfb.copy()
                    except NameError:
                        aud_work = base.copy()

                    if aud_work is None or aud_work.empty:
                        st.info("No recipients match the current filters. Adjust filters or bands.")
                    else:
                        # Asegurar dtype numérico para ordenar/quantiles (no altera tu lógica)
                        aud_work["p_signup"] = pd.to_numeric(aud_work["p_signup"], errors="coerce").fillna(0.0)

                        st.markdown("**How many to target?**")
                        c1, c2, _sp = st.columns([1.6, 1.2, 0.2])

                        selection_mode = c1.radio(
                            "Selection mode",
                            ["Top N", "Top %", "Expected signups target"],
                            horizontal=True,
                            key="aud_sel_mode",
                            help=("Top N: keep the N highest probabilities. "
                                "Top %: keep the top X percentile by probability. "
                                "Expected signups target: keep the smallest set whose summed p_signup reaches your target.")
                        )

                        # Orden determinista por probabilidad; empates por S
                        df_sorted = aud_work.sort_values(by=["p_signup", "S"], ascending=[False, False]).reset_index(drop=True)

                        # === Widgets con keys separadas y sincronización explícita ===
                        if selection_mode == "Top N":
                            n_value = c2.number_input(
                                "N",
                                min_value=1,
                                value=int(st.session_state.get("aud_sel_n", 500)),
                                step=10,
                                key="aud_sel_n_w",        # <- key de WIDGET distinta
                                format="%d",
                            )
                            # sincroniza estado de negocio
                            st.session_state["aud_sel_n"] = int(n_value)
                            n_keep = max(1, min(st.session_state["aud_sel_n"], len(df_sorted)))
                            df_sel = df_sorted.iloc[:n_keep].reset_index(drop=True)

                        elif selection_mode == "Top %":
                            pct_value = c2.slider(
                                "Percent",
                                min_value=1,
                                max_value=100,
                                value=int(st.session_state.get("aud_sel_pct", 20)),
                                step=1,
                                key="aud_sel_pct_w",      # <- key de WIDGET distinta
                            )
                            st.session_state["aud_sel_pct"] = int(pct_value)
                            pct_keep = st.session_state["aud_sel_pct"]
                            cutoff = float(df_sorted["p_signup"].quantile(1 - pct_keep / 100.0)) if len(df_sorted) else 1.0
                            df_sel = df_sorted[df_sorted["p_signup"] >= cutoff].reset_index(drop=True)

                        else:  # Expected signups target
                            target_value = c2.number_input(
                                "Target signups",
                                min_value=0.5,
                                value=float(st.session_state.get("aud_sel_target", 50.0)),
                                step=1.0,
                                key="aud_sel_target_w",   # <- key de WIDGET distinta
                            )
                            st.session_state["aud_sel_target"] = float(target_value)
                            target = st.session_state["aud_sel_target"]
                            cum = df_sorted["p_signup"].cumsum().values
                            pos = int(np.searchsorted(cum, target, side="left"))
                            pos = max(0, min(pos, len(df_sorted) - 1))
                            df_sel = df_sorted.iloc[:pos + 1].reset_index(drop=True)

                        # KPIs + tabla de la selección actual
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Eligible (base)", f"{len(df_sorted):,}")
                        k2.metric("Selected", f"{len(df_sel):,}")
                        k3.metric("Expected signups", f"{df_sel['p_signup'].sum():.1f}")

                        st.dataframe(
                            df_sel[["email", "S", "Y", "p_signup", "O", "C"]],
                            use_container_width=True,
                            height=360,
                        )

                        # Export + Send to Email Editor for the selection
                        exp_col, mo_col = st.columns([1, 2])
                        topic_slug = re.sub(r"[^a-z0-9]+", "_", str(st.session_state.get("aud_topic", "")).lower()).strip("_") or "topic"
                        csv_name = f"distribution_list_topic_{topic_slug}_selected.csv"

                        exp_col.download_button(
                            "Download selected (CSV)",
                            data=df_sel[["email"]].to_csv(index=False).encode("utf-8"),
                            file_name=csv_name,
                            mime="text/csv",
                        )

                        ed_subject = mo_col.text_input(
                            "Subject to use in Email Editor",
                            value=f"[{(st.session_state.get('aud_topic') or 'Campaign').title()}] New Campaign",
                            key="subject_topic_selected",
                            help="Will be prefilled in the Email Editor.",
                        )

                        if st.button("Send to Email Editor", type="primary", key="send_selected_to_email_editor"):
                            # Email Editor prefill: these are exactly the keys it pop()s on load
                            st.session_state["mo_recipients"] = df_sel["email"].astype(str).tolist()
                            # Email Editor mira primero 'mo_subject_live' y luego 'mo_subject'
                            st.session_state["mo_subject_live"] = ed_subject
                            st.session_state["mo_topic"] = st.session_state.get("aud_topic") or ""
                            st.session_state["nav_redirect"] = "Email Editor"
                            st.rerun()

                        # (Opcional) clear
                        if st.button("Clear audience", key="clear_aud"):
                            st.session_state["aud_built"] = False
                            st.session_state["aud_base"] = None
                            st.session_state["aud_topic"] = ""
                            st.rerun()

            else:
                st.info("Build an audience to see selection controls.")
