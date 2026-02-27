"""Training and loading of the recommendation model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from . import db, features

MODEL_PATH = Path(__file__).with_name("model.pkl")


def train_model(model_path: Optional[Path | None] = None) -> LogisticRegression:
    """Train the recommendation model from historical campaigns."""
    sends = db.load_send_log()
    campaigns = sends["campaign"].dropna().unique() if (not sends.empty and "campaign" in sends.columns) else []

    frames = [features.build_features_for_campaign(str(c)) for c in campaigns]
    if frames:
        non_empty = [pl.from_pandas(f, include_index=False) for f in frames if not f.empty]
        data = pl.concat(non_empty, how="vertical_relaxed").to_pandas() if non_empty else pd.DataFrame()
    else:
        data = pd.DataFrame()

    if data.empty:
        return LogisticRegression()

    X = data[features.FEATURE_COLUMNS]
    y = data["signed_up"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    path = model_path or MODEL_PATH
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    return clf


def load_model(model_path: Path | None = None) -> LogisticRegression:
    """Load the trained model from disk."""
    path = model_path or MODEL_PATH
    if not path.exists():
        return train_model(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)
