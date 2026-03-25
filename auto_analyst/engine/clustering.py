import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils.helpers import get_column_types


def run_clustering(df, n_clusters=None):
    """
    Encode categorical + numerical columns, run KMeans clustering,
    return cluster labels, summary, and PCA visualization.
    """
    col_types = get_column_types(df)
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    num_cols = [c for c, t in col_types.items() if t == "numerical"]

    if not cat_cols and not num_cols:
        return {"error": "No suitable columns found for clustering."}

    df_clean = df[cat_cols + num_cols].dropna()

    if df_clean.shape[0] < 10:
        return {"error": "Not enough rows for clustering (need at least 10)."}

    # Encode categoricals
    encoded = df_clean.copy()
    for col in cat_cols:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(encoded)

    # Auto-select best k using silhouette score (k=2 to 6)
    if n_clusters is None:
        n_clusters = _find_best_k(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df_clean = df_clean.copy()
    df_clean["Cluster"] = labels.astype(str)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df_viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_viz["Cluster"] = labels.astype(str)

    fig_pca = px.scatter(
        df_viz,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=f"Customer Segments (PCA View) — {n_clusters} Clusters",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    # Cluster summary
    cluster_summary = _build_cluster_summary(df_clean, cat_cols, num_cols)

    # Elbow chart
    fig_elbow = _elbow_chart(X)

    return {
        "n_clusters": n_clusters,
        "cluster_counts": df_clean["Cluster"].value_counts().to_dict(),
        "cluster_summary": cluster_summary,
        "fig_pca": fig_pca,
        "fig_elbow": fig_elbow,
        "df_with_clusters": df_clean,
    }


def _find_best_k(X, k_range=range(2, 7)):
    """Return k with highest silhouette score."""
    best_k, best_score = 2, -1
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            pass
    return best_k


def _build_cluster_summary(df_clean, cat_cols, num_cols):
    """Build a per-cluster summary of top categories and means."""
    summary = {}
    for cluster_id in sorted(df_clean["Cluster"].unique()):
        group = df_clean[df_clean["Cluster"] == cluster_id]
        cluster_info = {"size": len(group), "categorical": {}, "numerical": {}}

        for col in cat_cols:
            top = group[col].value_counts().head(3).to_dict()
            cluster_info["categorical"][col] = top

        for col in num_cols:
            cluster_info["numerical"][col] = round(float(group[col].mean()), 4)

        summary[f"Cluster {cluster_id}"] = cluster_info
    return summary


def _elbow_chart(X, k_range=range(2, 8)):
    """Return plotly elbow chart figure."""
    inertias = []
    ks = []
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        ks.append(k)

    fig = px.line(
        x=ks,
        y=inertias,
        markers=True,
        title="Elbow Chart — Optimal Number of Clusters",
        labels={"x": "Number of Clusters (k)", "y": "Inertia"},
    )
    return fig
