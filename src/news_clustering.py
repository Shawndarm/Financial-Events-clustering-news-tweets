import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from pyclustering.cluster.kmedians import kmedians
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage


################# Clustering evaluation ##############
################# Clustering evaluation ##############
def run_clustering_evaluation(X, k_range=range(2, 11), min_samples=1):
    """
    Evaluates different clustering algorithms using the Silhouette Score.
    min_samples: Minimum number of samples required per cluster for HAC stability.
    """
    X_norm = normalize(X)
    X_list = X.tolist()

    results = {"k": [], "Agglomerative": [], "K-Means": [], "K-Medians": []}

    for k in k_range:
        results["k"].append(k)

        # Method A: Agglomerative Clustering with Stability Logic
        try:
            curr_X_hac = X.copy()
            labels_hac = None
            # Loop until all clusters meet min_samples requirement
            while True:
                if len(curr_X_hac) < k:
                    labels_hac = None
                    break
                model = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                )
                temp_labels = model.fit_predict(curr_X_hac)
                counts = pd.Series(temp_labels).value_counts()
                to_remove = counts[counts < min_samples].index

                if len(to_remove) == 0:
                    labels_hac = temp_labels
                    break
                # Filter out samples from small clusters and retry
                curr_X_hac = curr_X_hac[~np.isin(temp_labels, to_remove)]

            if labels_hac is not None:
                score_hac = silhouette_score(curr_X_hac, labels_hac, metric="cosine")
                results["Agglomerative"].append(score_hac)
            else:
                results["Agglomerative"].append(np.nan)
        except Exception:
            results["Agglomerative"].append(np.nan)

        # Method B: K-Means++ (Standard)
        try:
            model_km = KMeans(
                n_clusters=k, init="k-means++", n_init=10, random_state=42
            )
            labels_km = model_km.fit_predict(X_norm)
            score_km = silhouette_score(X_norm, labels_km)
            results["K-Means"].append(score_km)
        except Exception:
            results["K-Means"].append(np.nan)

        # Method C: K-Medians (Standard)
        try:
            initial_medians = X_list[:k]
            model_kmed = kmedians(X_list, initial_medians)
            model_kmed.process()
            clusters = model_kmed.get_clusters()
            labels_kmed = np.zeros(len(X_list), dtype=int)
            for cluster_idx, cluster in enumerate(clusters):
                for sample_idx in cluster:
                    labels_kmed[sample_idx] = cluster_idx
            results["K-Medians"].append(
                silhouette_score(X, labels_kmed, metric="manhattan")
            )
        except Exception:
            results["K-Medians"].append(np.nan)

    return pd.DataFrame(results)


################# Viualization of Silhouette scores ##############
def plot_clustering_comparison(results_df):
    """
    Génère un graphique Plotly comparant les scores de silhouette.
    """
    fig = go.Figure()

    methods = [
        ("Agglomerative", "#2ecc71", "circle"),  # Vert
        ("K-Means", "#3498db", "square"),  # Bleu
        ("K-Medians", "#e74c3c", "triangle-up"),  # Rouge
    ]

    for column, color, symbol in methods:
        fig.add_trace(
            go.Scatter(
                x=results_df["k"],
                y=results_df[column],
                mode="lines+markers",
                name=column,
                line=dict(color=color, width=2),
                marker=dict(symbol=symbol, size=10),
            )
        )

    fig.update_layout(
        title="<b>Clustering methods comparison</b><br><sup>Silhouette Score</sup>",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Average Silhouette Score",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    fig.show()
    return


################### Centroid calculation for event signatures ##############
def calculate_event_centroids(X, labels):
    """
    Calculates the centroid of each cluster using the MEDIAN vector.
    As per the paper, the median is more robust to outliers within clusters.
    """
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        # Extract all news embeddings belonging to this specific cluster
        cluster_samples = X[labels == label]

        # Compute the median across each of the 300 dimensions
        # axis=0 means we calculate the median for each feature across all articles
        event_signature = np.median(cluster_samples, axis=0)
        centroids[label] = event_signature

    return centroids


################## Visualization of HAC clusters in 2D with Plotly ##############
def visualize_hac_tsne_range(
    X_full, df_features, start_date, end_date, k, perplexity=30, min_samples=1
):
    """
    Filters data, applies stability logic, and shows t-SNE visualization.
    """
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    sub_df = df_features.loc[mask].copy()
    X_sub = X_full[mask.values]

    # HAC Stability Logic
    curr_X = X_sub.copy()
    curr_indices = np.arange(len(sub_df))
    final_labels = None

    while True:
        if len(curr_X) < k:
            print(f"Could not achieve a stable configuration for k={k}.")
            return None
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        temp_labels = model.fit_predict(curr_X)
        counts = pd.Series(temp_labels).value_counts()
        to_remove = counts[counts < min_samples].index

        if len(to_remove) == 0:
            final_labels = temp_labels
            break
        # Remove and update indices to keep metadata in sync
        mask_stable = ~np.isin(temp_labels, to_remove)
        curr_X = curr_X[mask_stable]
        curr_indices = curr_indices[mask_stable]

    # Synchronize metadata with the stable articles
    stable_df = sub_df.iloc[curr_indices].copy()
    stable_df["Cluster"] = final_labels.astype(str)

    # t-SNE on stable articles
    actual_perplexity = min(perplexity, max(1, len(curr_X) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(curr_X)
    stable_df["X"] = X_2d[:, 0]
    stable_df["Y"] = X_2d[:, 1]

    # Plotly
    fig = px.scatter(
        stable_df,
        x="X",
        y="Y",
        color="Cluster",
        hover_data={"headline": True, "date": True, "X": False, "Y": False},
        title=f"Stable Financial Events ({start_date} to {end_date}) | k={k}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color="white"))
    )
    fig.update_layout(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")

    return fig


################## Final HAC ##############


def compute_stable_hac_linkage(
    X_full, df_features, start_date, end_date, k, min_samples=2
):
    """
    Computes the stable HAC linkage matrix Z as described in the paper.
    Returns: Z (Linkage Matrix), stable_headlines (List), and X_stable (Matrix).
    """
    # Temporal filtering
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    X_sub = X_full[mask.values]
    sub_df = df_features.loc[mask].copy()

    # Stable HAC Logic (Stability Condition)
    curr_X = X_sub.copy()
    curr_indices = np.arange(len(sub_df))

    while True:
        if len(curr_X) < k:
            return None, None, None  # Case where we can't form k stable clusters

        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        temp_labels = model.fit_predict(curr_X)
        counts = pd.Series(temp_labels).value_counts()
        to_remove = counts[counts < min_samples].index

        if len(to_remove) == 0:
            break  # Stable configuration achieved

        # Remove articles from small clusters and update indices
        mask_stable = ~np.isin(temp_labels, to_remove)
        curr_X = curr_X[mask_stable]
        curr_indices = curr_indices[mask_stable]

    # Compute the Final Linkage Matrix (Z) on the stable subset
    # Using 'average' and 'cosine' as per the paper
    Z = linkage(curr_X, method="average", metric="cosine")

    # Get the headlines of the final stable articles
    stable_headlines = sub_df.iloc[curr_indices]["headline"].tolist()

    return Z, stable_headlines, curr_X


################## Dendrogram visualization with Plotly ##############
def plot_hac_dendrogram_plotly(Z, leaf_labels, start_date, end_date):
    """
    Visualizes the precomputed linkage matrix Z as an interactive Plotly dendrogram.
    """
    if Z is None:
        print("No stable clustering results to plot.")
        return None

    # ff.create_dendrogram takes data but we pass our precomputed Z via linkagefun
    # We pass a dummy array for X because we already have the matrix Z
    dummy_X = np.zeros((len(leaf_labels), 300))

    fig = ff.create_dendrogram(
        dummy_X,
        orientation="bottom",
        labels=leaf_labels,
        linkagefun=lambda x: Z,  # We force Plotly to use our stable linkage
    )

    fig.update_layout(
        title=f"<b>Stable HAC Dendrogram</b><br><sup>Period: {start_date} to {end_date}</sup>",
        width=1000,
        height=800,
        template="plotly_white",
        xaxis=dict(title="Financial News Articles (Stable Subset)"),
        yaxis=dict(title="Cosine Distance"),
    )

    return fig
