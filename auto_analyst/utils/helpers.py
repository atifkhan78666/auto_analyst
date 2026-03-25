import pandas as pd
import numpy as np


def load_file(uploaded_file):
    """Load CSV or Excel file into a DataFrame."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload CSV or Excel."
        return df, None
    except Exception as e:
        return None, str(e)


def get_column_types(df):
    """Classify columns as categorical, numerical, datetime, or other."""
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numerical"
        elif df[col].nunique() <= 50 or pd.api.types.is_object_dtype(df[col]):
            col_types[col] = "categorical"
        else:
            col_types[col] = "other"
    return col_types


def compute_quality_score(df):
    """Return a data quality score out of 100."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_penalty = (missing_cells / total_cells) * 60
    duplicate_penalty = (duplicate_rows / df.shape[0]) * 40

    score = 100 - missing_penalty - duplicate_penalty
    return round(max(score, 0), 1)