"""Analytics and recommendation subsystem for Mailmkt."""

from . import db, metrics, calibration, features, model, recommend

__all__ = [
    "db",
    "metrics",
    "calibration",
    "features",
    "model",
    "recommend",
]
