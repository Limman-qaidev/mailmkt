"""Training and loading of the recommendation model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import db, features

MODEL_PATH = Path(__file__).with_name("model.pkl")


def train_model(
        model_path: Optional[Path | None] = None
        ) -> LogisticRegression:
    """Train the recommendation model from historical campaigns."""
    sends = db.load_send_log()
    campaigns = sends["campaign"].dropna().unique() if not sends.empty else []

    frames = [features.build_features_for_campaign(c) for c in campaigns]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        model = LogisticRegression()
        return model

    X = data[features.FEATURE_COLUMNS]
    y = data["signed_up"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    path = model_path or MODEL_PATH
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return model


def load_model(model_path: Path | None = None) -> LogisticRegression:
    """Load the trained model from disk."""
    path = model_path or MODEL_PATH
    if not path.exists():
        return train_model(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)
