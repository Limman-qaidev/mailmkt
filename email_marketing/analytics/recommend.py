"""Client recommendation engine."""

from __future__ import annotations

from typing import List

import pandas as pd

from . import features, model


def recommend_clients(
        campaign_id: str,
        threshold: float = 0.5
        ) -> pd.DataFrame:
    """Return recommended clients for ``campaign_id``
    with probability >= ``threshold``."""
    feats = features.build_features_for_campaign(campaign_id)
    if feats.empty:
        return pd.DataFrame(columns=["email", "score"])
    clf = model.load_model()
    probs = clf.predict_proba(feats[features.FEATURE_COLUMNS])[:, 1]
    feats = feats.assign(score=probs)
    return feats.loc[feats["score"] >= threshold, ["email", "score"]]


def get_distribution_list(
        campaign_id: str,
        threshold: float = 0.5
        ) -> List[str]:
    """Return a simple list of recommended email addresses."""
    df = recommend_clients(campaign_id, threshold)
    return df["email"].tolist()
