import pandas as pd
import numpy as np
from utils.helpers import get_column_types, compute_quality_score


def profile_dataframe(df):
    """
    Master profiling function.
    Returns a dict with all key profile information.
    """
    col_types = get_column_types(df)
    quality_score = compute_quality_score(df)

    profile = {
        "shape": df.shape,
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "quality_score": quality_score,
        "column_types": col_types,
        "missing_summary": get_missing_summary(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": get_column_profiles(df, col_types),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }
    return profile


def get_missing_summary(df):
    """Return missing value counts and percentages per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return {}
    pct = (missing / len(df) * 100).round(2)
    return {
        col: {"count": int(missing[col]), "percent": float(pct[col])}
        for col in missing.index
    }


def get_column_profiles(df, col_types):
    """Return per-column profile statistics."""
    profiles = {}
    for col in df.columns:
        ctype = col_types.get(col, "other")
        base = {
            "type": ctype,
            "missing_count": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_values": int(df[col].nunique()),
        }

        if ctype == "categorical":
            value_counts = df[col].value_counts()
            base.update({
                "top_5_values": value_counts.head(5).to_dict(),
                "mode": str(df[col].mode()[0]) if not df[col].mode().empty else "N/A",
            })

        elif ctype == "numerical":
            base.update({
                "mean": round(float(df[col].mean()), 3),
                "median": round(float(df[col].median()), 3),
                "std": round(float(df[col].std()), 3),
                "min": round(float(df[col].min()), 3),
                "max": round(float(df[col].max()), 3),
            })

        elif ctype == "datetime":
            base.update({
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
            })

        profiles[col] = base
    return profiles