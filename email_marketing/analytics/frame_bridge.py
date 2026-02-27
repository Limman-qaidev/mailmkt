"""Bridges between pandas and Polars for analytics internals.

Polars is used in analytics for multi-threaded execution and columnar memory
layout. Public APIs and Streamlit integration still return pandas DataFrames.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import polars as pl


def to_pl(df_pd: pd.DataFrame | None) -> pl.DataFrame:
    """Convert pandas to Polars with sensible defaults for None/empty values."""
    if df_pd is None:
        return pl.DataFrame()
    if isinstance(df_pd, pl.DataFrame):
        return df_pd
    return pl.from_pandas(df_pd, include_index=False)


def to_pd(df_pl: pl.DataFrame | pd.DataFrame | None) -> pd.DataFrame:
    """Convert Polars to pandas at integration boundaries."""
    if df_pl is None:
        return pd.DataFrame()
    if isinstance(df_pl, pd.DataFrame):
        return df_pl
    return df_pl.to_pandas()


def assert_schema(df: pl.DataFrame | pd.DataFrame, cols: Iterable[str], dtypes: dict[str, Any] | None = None) -> None:
    """Validate expected columns and optional dtypes."""
    cols = list(cols)
    present = set(df.columns)
    missing = [c for c in cols if c not in present]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not dtypes:
        return
    if isinstance(df, pd.DataFrame):
        for c, t in dtypes.items():
            if c in df.columns and str(df[c].dtype) != str(t):
                raise ValueError(f"Column {c!r} has dtype {df[c].dtype!r}, expected {t!r}")
    else:
        schema = df.schema
        for c, t in dtypes.items():
            if c in schema and schema[c] != t:
                raise ValueError(f"Column {c!r} has dtype {schema[c]!r}, expected {t!r}")


def sort_for_determinism(df: pl.DataFrame | pd.DataFrame, keys: list[str]) -> pl.DataFrame | pd.DataFrame:
    """Apply stable sorting when deterministic ordering is required."""
    if not keys:
        return df
    valid = [k for k in keys if k in df.columns]
    if not valid:
        return df
    if isinstance(df, pd.DataFrame):
        return df.sort_values(valid, kind="stable")
    return df.sort(valid)
