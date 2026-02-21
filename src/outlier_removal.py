from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import cosine
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE


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


#################### Advanced Outlier Removal  ######################
def remove_news_outliers_advanced(X, labels, percentile_threshold=20):
    """
    Advanced Outlier Removal as per Carta et al.

    This function uses two metrics to detect noise:
    1. Per-sample silhouette coefficient (Ambiguity check)
    2. Cosine similarity to the cluster median (Semantic isolation check)
    """
    n_samples = len(X)

    # Safety check: silhouette needs at least 2 clusters
    if n_samples < 2 or len(np.unique(labels)) < 2:
        return X, labels, np.ones(n_samples, dtype=bool)

    # Calculate Initial Centroid
    initial_centroids = calculate_event_centroids(X, labels)

    # Metric 1: Per-sample Silhouette
    # Measures how well each document is separated from other clusters
    sil_scores = silhouette_samples(X, labels, metric="cosine")

    # Metric 2: Cosine Similarity to Centroid
    # Measures how close each document is to the "heart" of its assigned event
    centroid_sims = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_id = labels[i]
        centroid_vec = initial_centroids[cluster_id]
        # Similarity = 1 - distance (Cosine similarity)
        centroid_sims[i] = 1 - cosine(X[i], centroid_vec)

    # Define Cutoff Thresholds (Percentile-based)
    sil_cutoff = np.percentile(sil_scores, percentile_threshold)
    sim_cutoff = np.percentile(centroid_sims, percentile_threshold)

    # Identify and Remove Outliers (Logical OR)
    # As per the paper: "all samples whose scores are below the cutoff threshold
    # in ONE of the two rankings are suppressed."
    is_outlier = (sil_scores < sil_cutoff) | (centroid_sims < sim_cutoff)
    keep_mask = ~is_outlier

    X_clean = X[keep_mask]
    labels_clean = labels[keep_mask]

    print(f"Complete: Removed {np.sum(is_outlier)} articles.")
    print(f"Cutoffs: Silhouette < {sil_cutoff:.3f} | Centroid Sim < {sim_cutoff:.3f}")

    return X_clean, labels_clean, keep_mask


######### Visualization of HAC with Centroids (t-SNE) #########
def visualize_hac_with_centroids(X_clean, clean_df, k, perplexity=10):
    # 1. Calculate final centroids using your previously defined function
    centroids_dict = calculate_event_centroids(
        X_clean, clean_df["Cluster"].astype(int).values
    )

    # Convert dict to array ordered by cluster ID
    # We use sorted keys to ensure the order matches the cluster IDs
    centroid_matrix = np.array(
        [centroids_dict[i] for i in sorted(centroids_dict.keys())]
    )

    # 2. Combine for t-SNE (Articles + Centroids)
    combined_X = np.vstack([X_clean, centroid_matrix])

    # Security: perplexity must be < n_samples
    safe_perplexity = min(perplexity, len(combined_X) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(combined_X)

    # Split the 2D coordinates back into articles and centroids
    articles_2d = X_2d[: len(X_clean)]
    centroids_2d = X_2d[len(X_clean) :]

    fig = go.Figure()

    # Add Articles traces
    for cluster_id in sorted(clean_df["Cluster"].unique()):
        # Convert cluster_id to string for the mask if necessary
        mask = clean_df["Cluster"] == str(cluster_id)

        fig.add_trace(
            go.Scatter(
                x=articles_2d[mask.values, 0],
                y=articles_2d[mask.values, 1],
                mode="markers",
                name=f"Event {cluster_id}",
                marker=dict(size=9, opacity=0.7),
                text=clean_df[mask]["headline"],
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    # Add Centroids (The "X" Signatures)
    fig.add_trace(
        go.Scatter(
            x=centroids_2d[:, 0],
            y=centroids_2d[:, 1],
            mode="markers+text",
            name="EVENT SIGNATURES (MEDIANS)",
            text=[f"SIGNATURE {i}" for i in sorted(centroids_dict.keys())],
            textposition="top center",
            marker=dict(size=18, symbol="x", color="black", line=dict(width=3)),
        )
    )

    fig.update_layout(
        title="<b>Final Event Signatures Map (t-SNE)</b><br><sup>Articles and their Median Signatures</sup>",
        template="plotly_white",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title="Legend",
    )

    return fig
