from chat.chat_engine import generate_ai_narrative


def build_analysis_summary(profile, bivariate_results=None, cluster_results=None, assoc_results=None):
    """
    Build a plain-text summary of all analysis results
    to pass to the LLM for narrative generation.
    """
    lines = []

    lines.append(f"Dataset: {profile['num_rows']} rows, {profile['num_cols']} columns")
    lines.append(f"Quality Score: {profile['quality_score']} / 100")
    lines.append(f"Duplicate Rows: {profile['duplicate_rows']}")

    # Missing values
    if profile.get("missing_summary"):
        lines.append("Missing Values:")
        for col, info in profile["missing_summary"].items():
            lines.append(f"  - {col}: {info['count']} missing ({info['percent']}%)")
    else:
        lines.append("Missing Values: None")

    # Column types
    type_counts = {}
    for col, ctype in profile["column_types"].items():
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    lines.append(f"Column Types: {type_counts}")

    # Bivariate highlights
    if bivariate_results:
        cat_cat = bivariate_results.get("cat_cat", [])
        if cat_cat:
            top = cat_cat[0]
            lines.append(
                f"Strongest Categorical Relationship: '{top['col1']}' vs '{top['col2']}' "
                f"(Cramer's V = {top['cramers_v']}, significant={top['significant']})"
            )
        num_num = bivariate_results.get("num_num", [])
        if num_num:
            top_corr = num_num[0]
            lines.append(
                f"Strongest Numerical Correlation: '{top_corr['col1']}' vs '{top_corr['col2']}' "
                f"(r = {top_corr['correlation']})"
            )

    # Clustering highlights
    if cluster_results and "error" not in cluster_results:
        n = cluster_results.get("n_clusters")
        counts = cluster_results.get("cluster_counts", {})
        lines.append(f"Clustering: {n} segments identified — sizes: {counts}")

    # Association rules highlights
    if assoc_results and "error" not in assoc_results:
        rules_df = assoc_results.get("rules_df")
        if rules_df is not None and not rules_df.empty:
            top_rule = rules_df.iloc[0]
            lines.append(
                f"Top Association Rule: IF {top_rule['antecedents']} THEN {top_rule['consequents']} "
                f"(lift={top_rule['lift']}, confidence={top_rule['confidence']})"
            )

    return "\n".join(lines)


def generate_report_narrative(df, profile, bivariate_results=None,
                            cluster_results=None, assoc_results=None):
    """
    Full narrative generation entry point.
    Returns AI-written executive summary string.
    """
    analysis_summary = build_analysis_summary(
        profile, bivariate_results, cluster_results, assoc_results
    )
    narrative = generate_ai_narrative(df, profile, analysis_summary, profile_only=False)
    return narrative


def generate_quick_profile_narrative(df, profile):
    """
    Quick narrative for just the profile tab (no full analysis needed).
    """
    return generate_ai_narrative(df, profile, analysis_summary="", profile_only=True)
