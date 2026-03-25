import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import chi2_contingency
from utils.helpers import get_column_types


def run_bivariate_analysis(df):
    """Run bivariate analysis between column pairs. Returns results dict."""
    col_types = get_column_types(df)
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    num_cols = [c for c, t in col_types.items() if t == "numerical"]

    results = {
        "cat_cat": [],
        "num_num": [],
        "cat_num": [],
        "correlation_fig": None,
    }

    # Categorical vs Categorical
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            res = analyze_cat_cat(df, cat_cols[i], cat_cols[j])
            if res:
                results["cat_cat"].append(res)

    # Numerical vs Numerical — correlation heatmap
    if len(num_cols) >= 2:
        results["num_num"] = analyze_num_num(df, num_cols)
        results["correlation_fig"] = plot_correlation_heatmap(df, num_cols)

    # Categorical vs Numerical
    for cat in cat_cols:
        for num in num_cols:
            res = analyze_cat_num(df, cat, num)
            if res:
                results["cat_num"].append(res)

    # Sort cat-cat by Cramer's V descending
    results["cat_cat"] = sorted(
        results["cat_cat"], key=lambda x: x.get("cramers_v", 0), reverse=True
    )

    return results


def analyze_cat_cat(df, col1, col2):
    """Chi-square test + Cramer's V between two categorical columns."""
    try:
        sub = df[[col1, col2]].dropna()
        if sub.shape[0] < 10:
            return None
        contingency = pd.crosstab(sub[col1], sub[col2])
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        cramers_v = round(
            float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))), 4
        )

        # Heatmap of crosstab
        fig = px.imshow(
            contingency,
            title=f"Cross-tab: {col1} vs {col2} (Cramer's V = {cramers_v})",
            color_continuous_scale="Blues",
            text_auto=True,
        )

        return {
            "col1": col1,
            "col2": col2,
            "cramers_v": cramers_v,
            "p_value": round(p, 6),
            "chi2": round(chi2, 4),
            "significant": p < 0.05,
            "fig": fig,
        }
    except Exception:
        return None


def analyze_num_num(df, num_cols):
    """Correlation matrix between numerical columns."""
    corr = df[num_cols].corr().round(4)
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            pairs.append({
                "col1": num_cols[i],
                "col2": num_cols[j],
                "correlation": round(float(corr.iloc[i, j]), 4),
            })
    pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def plot_correlation_heatmap(df, num_cols):
    """Plotly heatmap of correlation matrix."""
    corr = df[num_cols].corr().round(4)
    fig = px.imshow(
        corr,
        title="Numerical Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=True,
    )
    return fig


def analyze_cat_num(df, cat_col, num_col):
    """Box plot of numerical column grouped by categorical column."""
    try:
        sub = df[[cat_col, num_col]].dropna()
        if sub.shape[0] < 10 or sub[cat_col].nunique() > 20:
            return None

        fig = px.box(
            sub,
            x=cat_col,
            y=num_col,
            title=f"{num_col} by {cat_col}",
            color=cat_col,
        )
        fig.update_layout(showlegend=False)

        group_means = sub.groupby(cat_col)[num_col].mean().round(4).to_dict()

        return {
            "cat_col": cat_col,
            "num_col": num_col,
            "group_means": group_means,
            "fig": fig,
        }
    except Exception:
        return None
