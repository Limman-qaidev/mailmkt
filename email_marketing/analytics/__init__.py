"""Analytics and recommendation subsystem for Mailmkt."""

from . import (
    db,
    metrics,
    calibration,
    features,
    model,
    recommend,
    segmentation,
)

__all__ = [
    "db",
    "metrics",
    "calibration",
    "features",
    "model",
    "recommend",
    "segmentation",
]
