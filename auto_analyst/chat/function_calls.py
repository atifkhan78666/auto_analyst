import pandas as pd
import numpy as np
from utils.helpers import get_column_types


def get_value_counts(df, column, top_n=10):
    """Return top N value counts for a column."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    vc = df[column].value_counts().head(top_n)
    result = f"Top {top_n} values in '{column}':\n"
    for val, count in vc.items():
        pct = round(count / len(df) * 100, 1)
        result += f"  {val}: {count} ({pct}%)\n"
    return result


def filter_data(df, column, value):
    """Return count and percentage of rows matching a filter."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    filtered = df[df[column].astype(str).str.lower() == str(value).lower()]
    count = len(filtered)
    pct = round(count / len(df) * 100, 2)
    return f"Rows where '{column}' = '{value}': {count} ({pct}% of total {len(df)} rows)"


def compare_categories(df, col1, col2):
    """Cross-tabulation between two categorical columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return "One or both columns not found."
    ct = pd.crosstab(df[col1], df[col2])
    return f"Cross-tab of '{col1}' vs '{col2}':\n{ct.to_string()}"


def show_top_n(df, column, n=5, ascending=False):
    """Show top N rows sorted by a numerical column."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    sorted_df = df.sort_values(column, ascending=ascending).head(n)
    return sorted_df.to_string(index=False)


def get_column_stats(df, column):
    """Return descriptive stats for a column."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    col_types = get_column_types(df)
    ctype = col_types.get(column)
    if ctype == "numerical":
        s = df[column].describe()
        return f"Stats for '{column}':\n{s.to_string()}"
    elif ctype == "categorical":
        vc = df[column].value_counts()
        return f"Value counts for '{column}':\n{vc.head(15).to_string()}"
    else:
        return f"Column '{column}' is of type '{ctype}', stats not available."


def get_missing_info(df):
    """Return missing value info for the entire dataframe."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "No missing values found in the dataset!"
    result = "Missing values per column:\n"
    for col, count in missing.items():
        pct = round(count / len(df) * 100, 2)
        result += f"  {col}: {count} missing ({pct}%)\n"
    return result


def get_group_mean(df, cat_col, num_col):
    """Mean of a numerical column grouped by a categorical column."""
    if cat_col not in df.columns or num_col not in df.columns:
        return "One or both columns not found."
    grouped = df.groupby(cat_col)[num_col].mean().round(4).sort_values(ascending=False)
    return f"Average '{num_col}' by '{cat_col}':\n{grouped.to_string()}"


# Registry of available functions for the AI to reference
FUNCTION_REGISTRY = {
    "get_value_counts": get_value_counts,
    "filter_data": filter_data,
    "compare_categories": compare_categories,
    "show_top_n": show_top_n,
    "get_column_stats": get_column_stats,
    "get_missing_info": get_missing_info,
    "get_group_mean": get_group_mean,
}


def run_function(df, func_name, **kwargs):
    """Execute a registered function on the dataframe."""
    func = FUNCTION_REGISTRY.get(func_name)
    if func is None:
        return f"Function '{func_name}' not found."
    try:
        return func(df, **kwargs)
    except Exception as e:
        return f"Error running '{func_name}': {str(e)}"
