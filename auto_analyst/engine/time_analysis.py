import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_column_types


def detect_datetime_columns(df):
    """Try to detect datetime columns including string-based date columns."""
    datetime_cols = []
    col_types = get_column_types(df)

    for col in df.columns:
        if col_types.get(col) == "datetime":
            datetime_cols.append(col)
        elif col_types.get(col) in ["categorical", "other"]:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().mean() > 0.7:  # 70%+ parsed successfully
                    datetime_cols.append(col)
            except Exception:
                pass

    return datetime_cols


def run_time_analysis(df):
    """
    If datetime columns exist, analyze trends of categorical/numerical
    columns over time. Returns results dict.
    """
    datetime_cols = detect_datetime_columns(df)

    if not datetime_cols:
        return {"available": False, "message": "No datetime columns detected in this dataset."}

    col_types = get_column_types(df)
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    num_cols = [c for c, t in col_types.items() if t == "numerical"]

    results = {
        "available": True,
        "datetime_columns": datetime_cols,
        "trends": [],
    }

    date_col = datetime_cols[0]
    df_time = df.copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
    df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)

    # Determine best time grouping
    date_range = (df_time[date_col].max() - df_time[date_col].min()).days
    if date_range > 730:
        freq, freq_label = "YE", "Yearly"
    elif date_range > 60:
        freq, freq_label = "ME", "Monthly"
    elif date_range > 14:
        freq, freq_label = "W", "Weekly"
    else:
        freq, freq_label = "D", "Daily"

    results["freq_label"] = freq_label

    # Row count over time
    df_time_indexed = df_time.set_index(date_col)
    count_over_time = df_time_indexed.resample(freq).size().reset_index()
    count_over_time.columns = [date_col, "count"]

    fig_count = px.line(
        count_over_time,
        x=date_col,
        y="count",
        title=f"Record Count Over Time ({freq_label})",
        markers=True,
        color_discrete_sequence=["#4F8EF7"],
    )
    results["fig_count"] = fig_count

    # Numerical trends over time
    for num_col in num_cols[:4]:
        try:
            trend = df_time_indexed[[num_col]].resample(freq).mean().reset_index()
            fig = px.line(
                trend,
                x=date_col,
                y=num_col,
                title=f"Average {num_col} Over Time ({freq_label})",
                markers=True,
                color_discrete_sequence=["#F76B4F"],
            )
            results["trends"].append({
                "column": num_col,
                "type": "numerical",
                "fig": fig,
            })
        except Exception:
            pass

    # Categorical distribution over time (top 5 categories)
    for cat_col in cat_cols[:3]:
        try:
            top_cats = df_time[cat_col].value_counts().head(5).index.tolist()
            sub = df_time[df_time[cat_col].isin(top_cats)]
            sub_indexed = sub.set_index(date_col)
            cat_trend = (
                sub_indexed.groupby([pd.Grouper(freq=freq), cat_col])
                .size()
                .reset_index(name="count")
            )
            fig = px.line(
                cat_trend,
                x=date_col,
                y="count",
                color=cat_col,
                title=f"{cat_col} Categories Over Time ({freq_label})",
                markers=True,
            )
            results["trends"].append({
                "column": cat_col,
                "type": "categorical",
                "fig": fig,
            })
        except Exception:
            pass

    return results
