"""Streamlit dashboard components.

This package groups together the user interface modules used by the
dashboard. The components are designed to be self-contained and reusable
across different layouts. See the documentation in ``style.py`` for
thematic configuration.
"""

from __future__ import annotations

from . import email_editor
from . import stats_view
from . import campaign_metrics_view
from . import style


__all__ = ["email_editor", "stats_view", "campaign_metrics_view", "style"]
