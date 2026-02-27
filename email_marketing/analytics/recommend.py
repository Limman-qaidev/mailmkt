"""Client recommendation engine."""

from __future__ import annotations

from typing import List, cast

import pandas as pd
import polars as pl

from . import features, model


def recommend_clients(campaign_id: str, threshold: float = 0.5) -> pd.DataFrame:
    """Return recommended clients for ``campaign_id`` with score >= threshold."""
    feats = features.build_features_for_campaign(campaign_id)
    if feats.empty:
        return pd.DataFrame(columns=["email", "score"])

    clf = model.load_model()
    probs = clf.predict_proba(feats[features.FEATURE_COLUMNS])[:, 1]
    feats_scored = pl.from_pandas(feats, include_index=False).with_columns(
        pl.Series(name="score", values=probs)
    )
    out = feats_scored.filter(pl.col("score") >= float(threshold)).select(["email", "score"])
    return out.to_pandas()


def get_distribution_list(campaign_id: str, threshold: float = 0.5) -> List[str]:
    """Return the list of recommended email addresses."""
    df = recommend_clients(campaign_id, threshold)
    return cast(List[str], df["email"].astype(str).tolist())
