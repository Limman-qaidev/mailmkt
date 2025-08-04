"""Weight calibration for campaign metrics."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from . import metrics

WEIGHTS_PATH = Path(__file__).with_name("weights.pkl")


def calibrate_weights(
        data: np.ndarray,
        targets: np.ndarray
        ) -> Tuple[float, float, float]:
    """Calibrate ``alpha``, ``beta`` and ``gamma`` using linear regression."""
    model = LinearRegression()
    model.fit(data, targets)
    alpha, beta = model.coef_
    gamma = float(model.intercept_)
    return float(alpha), float(beta), gamma


def recalculate_weights() -> Tuple[float, float, float]:
    """Recalculate and persist metric weights.

    Returns
    -------
    Tuple[float, float, float]
        The calibrated ``alpha``, ``beta`` and ``gamma`` coefficients.
    """
    m = metrics.compute_campaign_metrics()
    if m.empty:
        coeffs = (0.0, 0.0, 0.0)
    else:
        X = m[["opens", "clicks"]].to_numpy(dtype=float)
        y = m["signups"].to_numpy(dtype=float)
        coeffs = calibrate_weights(X, y)
    with open(WEIGHTS_PATH, "wb") as fh:
        pickle.dump(coeffs, fh)
    return coeffs


def load_weights() -> Tuple[float, float, float]:
    """Load previously calibrated weights from disk."""
    if not WEIGHTS_PATH.exists():
        return 0.0, 0.0, 0.0
    with open(WEIGHTS_PATH, "rb") as fh:
        alpha, beta, gamma = pickle.load(fh)
    return alpha, beta, gamma
