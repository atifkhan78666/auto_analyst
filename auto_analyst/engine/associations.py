import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from utils.helpers import get_column_types


def run_association_analysis(df, min_support=0.05, min_confidence=0.3, top_n=20):
    """
    Run association rule mining on categorical columns using Apriori.
    Returns rules dataframe and visualizations.
    """
    col_types = get_column_types(df)
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if len(cat_cols) < 2:
        return {"error": "Need at least 2 categorical columns for association analysis."}

    # Build transactions: each row is a list of "col=value" strings
    df_cat = df[cat_cols].dropna()
    if df_cat.shape[0] < 20:
        return {"error": "Not enough rows for association analysis (need at least 20)."}

    transactions = []
    for _, row in df_cat.iterrows():
        transaction = [f"{col}={row[col]}" for col in cat_cols]
        transactions.append(transaction)

    # Encode
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    try:
        frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return {"error": f"No frequent itemsets found. Try lowering min_support (current: {min_support})."}

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets),
        )

        if rules.empty:
            return {"error": "No strong association rules found. Try adjusting thresholds."}

        rules = rules.sort_values("lift", ascending=False).head(top_n)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(4)

        # Scatter: support vs confidence, size = lift
        fig_scatter = px.scatter(
            rules,
            x="support",
            y="confidence",
            size="lift",
            hover_data=["antecedents", "consequents", "lift"],
            title="Association Rules — Support vs Confidence (size = Lift)",
            color="lift",
            color_continuous_scale="Viridis",
        )

        # Top rules bar chart by lift
        fig_bar = px.bar(
            rules.head(10),
            x="lift",
            y=rules.head(10).apply(
                lambda r: f"{r['antecedents']} → {r['consequents']}", axis=1
            ),
            orientation="h",
            title="Top 10 Association Rules by Lift",
            labels={"x": "Lift", "y": "Rule"},
            color="lift",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})

        return {
            "rules_df": rules,
            "num_rules": len(rules),
            "fig_scatter": fig_scatter,
            "fig_bar": fig_bar,
        }

    except Exception as e:
        return {"error": f"Association analysis failed: {str(e)}"}
