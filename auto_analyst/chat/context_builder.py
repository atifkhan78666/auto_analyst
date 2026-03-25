import pandas as pd
import numpy as np
from utils.helpers import get_column_types


def build_data_context(df, profile=None, max_sample_rows=5):
    """
    Convert a DataFrame into a rich text summary for the LLM.
    This is injected as system context so the AI can answer questions
    about the actual data without needing to see every row.
    """
    col_types = get_column_types(df)
    lines = []

    lines.append("=== DATASET OVERVIEW ===")
    lines.append(f"Total Rows: {df.shape[0]}")
    lines.append(f"Total Columns: {df.shape[1]}")
    lines.append(f"Column Names: {', '.join(df.columns.tolist())}")
    lines.append("")

    lines.append("=== COLUMN DETAILS ===")
    for col in df.columns:
        ctype = col_types.get(col, "other")
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1)
        lines.append(f"\nColumn: '{col}' | Type: {ctype} | Missing: {missing} ({missing_pct}%)")

        if ctype == "categorical":
            vc = df[col].value_counts().head(10)
            lines.append(f"  Unique Values: {df[col].nunique()}")
            lines.append(f"  Top Values: {dict(vc)}")
            if not df[col].mode().empty:
                lines.append(f"  Mode (Most Common): {df[col].mode()[0]}")

        elif ctype == "numerical":
            lines.append(f"  Mean: {round(float(df[col].mean()), 4)}")
            lines.append(f"  Median: {round(float(df[col].median()), 4)}")
            lines.append(f"  Std Dev: {round(float(df[col].std()), 4)}")
            lines.append(f"  Min: {round(float(df[col].min()), 4)}")
            lines.append(f"  Max: {round(float(df[col].max()), 4)}")

        elif ctype == "datetime":
            dt = pd.to_datetime(df[col], errors="coerce")
            lines.append(f"  Range: {dt.min()} to {dt.max()}")

    lines.append("")
    lines.append("=== SAMPLE DATA (First 5 Rows) ===")
    sample = df.head(max_sample_rows).to_string(index=False)
    lines.append(sample)

    return "\n".join(lines)


def build_system_prompt(df, profile=None):
    """
    Build the full system prompt for the chat AI.
    """
    data_context = build_data_context(df, profile)

    system_prompt = f"""You are AutoAnalyst AI, an expert data analyst assistant.
The user has uploaded a dataset and wants to chat with their data.
You have full knowledge of the dataset summarized below.

Your job is to:
- Answer questions about the data clearly and accurately
- Provide specific numbers, percentages, and counts when asked
- Suggest interesting patterns or insights when relevant
- Be concise but thorough
- If asked something you cannot determine from the summary, say so honestly

When giving numbers, be precise. When explaining patterns, be clear and use plain language.
Do NOT make up data that isn't in the summary below.

{data_context}

You are ready to answer questions about this dataset."""

    return system_prompt
