import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_column_types


def run_univariate_analysis(df):
    """Run univariate analysis on all columns. Returns dict of results + figures."""
    col_types = get_column_types(df)
    results = {}

    for col in df.columns:
        ctype = col_types.get(col, "other")
        if ctype == "categorical":
            results[col] = analyze_categorical(df, col)
        elif ctype == "numerical":
            results[col] = analyze_numerical(df, col)
        elif ctype == "datetime":
            results[col] = analyze_datetime(df, col)

    return results


def analyze_categorical(df, col):
    """Analyze a categorical column."""
    series = df[col].dropna()
    value_counts = series.value_counts()
    value_pct = series.value_counts(normalize=True).round(4) * 100

    # Bar chart
    fig_bar = px.bar(
        x=value_counts.index[:20],
        y=value_counts.values[:20],
        labels={"x": col, "y": "Count"},
        title=f"Value Distribution — {col}",
        color=value_counts.values[:20],
        color_continuous_scale="Blues",
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)

    # Pie chart (top 8)
    fig_pie = px.pie(
        names=value_counts.index[:8],
        values=value_counts.values[:8],
        title=f"Top Categories — {col}",
    )

    return {
        "type": "categorical",
        "column": col,
        "total_count": len(series),
        "unique_count": series.nunique(),
        "mode": str(series.mode()[0]) if not series.mode().empty else "N/A",
        "missing": int(df[col].isnull().sum()),
        "top_values": value_counts.head(10).to_dict(),
        "top_pct": value_pct.head(10).to_dict(),
        "fig_bar": fig_bar,
        "fig_pie": fig_pie,
    }


def analyze_numerical(df, col):
    """Analyze a numerical column."""
    series = df[col].dropna()

    stats = {
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "q1": round(float(series.quantile(0.25)), 4),
        "q3": round(float(series.quantile(0.75)), 4),
        "skewness": round(float(series.skew()), 4),
        "kurtosis": round(float(series.kurtosis()), 4),
    }

    # Histogram
    fig_hist = px.histogram(
        series,
        nbins=30,
        title=f"Distribution — {col}",
        labels={"value": col, "count": "Frequency"},
        color_discrete_sequence=["#4F8EF7"],
    )

    # Box plot
    fig_box = px.box(
        df,
        y=col,
        title=f"Box Plot — {col}",
        color_discrete_sequence=["#4F8EF7"],
    )

    return {
        "type": "numerical",
        "column": col,
        "stats": stats,
        "missing": int(df[col].isnull().sum()),
        "fig_hist": fig_hist,
        "fig_box": fig_box,
    }


def analyze_datetime(df, col):
    """Analyze a datetime column."""
    series = pd.to_datetime(df[col], errors="coerce").dropna()

    fig_timeline = px.histogram(
        series,
        title=f"Date Distribution — {col}",
        labels={"value": col},
        color_discrete_sequence=["#4F8EF7"],
    )

    return {
        "type": "datetime",
        "column": col,
        "min_date": str(series.min()),
        "max_date": str(series.max()),
        "missing": int(df[col].isnull().sum()),
        "fig_timeline": fig_timeline,
    }
