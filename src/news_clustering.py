import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from pyclustering.cluster.kmedians import kmedians
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE

################# Clustering evaluation ##############
def run_clustering_evaluation(X, k_range=range(2, 11)):
    """
    Evaluates different clustering algorithms using the Silhouette Score.
    This helps determine the optimal number of clusters (k) and the best method.
    """
    # Normalization is required for Euclidean-based methods (like K-Means) 
    # to behave similarly to Cosine similarity on embeddings.
    X_norm = normalize(X)
    X_list = X.tolist() # pyclustering requires list format

    results = { 
        'k': [],
        'Agglomerative': [], 
        'K-Means': [],
        'K-Medians': []
    }
    for k in k_range:
        results['k'].append(k)

        # Method A: Agglomerative Clustering (Cosine Similarity)
        try:
            model_hac = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
            labels_hac = model_hac.fit_predict(X)
            score_hac = silhouette_score(X, labels_hac, metric='cosine')
            results['Agglomerative'].append(score_hac)
        except Exception as e:
            print(f"Agglomerative error at k={k}: {e}")
            results['Agglomerative'].append(np.nan)

        # Method B: K-Means++ (Euclidean on Normalized Data)
        try:
            model_km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels_km = model_km.fit_predict(X_norm)
            score_km = silhouette_score(X_norm, labels_km)
            results['K-Means'].append(score_km)
        except Exception as e:
            print(f"K-Means error at k={k}: {e}")
            results['K-Means'].append(np.nan)

        # Method C: K-Medians (Manhattan Distance)
        try:
            # pyclustering needs initial medians
            initial_medians = X_list[:k] 
            model_kmed = kmedians(X_list, initial_medians)
            model_kmed.process()           
            # Convert pyclustering output to sklearn-style labels
            clusters = model_kmed.get_clusters()
            labels_kmed = np.zeros(len(X_list), dtype=int)
            for cluster_idx, cluster in enumerate(clusters):
                for sample_idx in cluster:
                    labels_kmed[sample_idx] = cluster_idx           
            # Silhouette score using Manhattan (L1) for K-Medians consistency
            results['K-Medians'].append(silhouette_score(X, labels_kmed, metric='manhattan'))
        except Exception as e:
            print(f"K-Medians error at k={k}: {e}")
            results['K-Medians'].append(np.nan)

    res = pd.DataFrame(results)
    return res

################# Viualization of Silhouette scores ##############
def plot_clustering_comparison(results_df):
    """
    Génère un graphique Plotly comparant les scores de silhouette.
    """
    fig = go.Figure()

    methods = [
        ('Agglomerative', '#2ecc71', 'circle'),  # Vert
        ('K-Means', '#3498db', 'square'),        # Bleu
        ('K-Medians', '#e74c3c', 'triangle-up')  # Rouge
    ]

    for column, color, symbol in methods:
        fig.add_trace(go.Scatter(
            x=results_df['k'], 
            y=results_df[column],
            mode='lines+markers',
            name=column,
            line=dict(color=color, width=2),
            marker=dict(symbol=symbol, size=10)
        ))

    fig.update_layout(
        title='<b>Clustering methods comparison</b><br><sup>Silhouette Score</sup>',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Average Silhouette Score',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
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
def visualize_hac_tsne_range(X_full, df_features, start_date, end_date, k, perplexity=30):
    """
    Filtre les données, effectue le clustering et affiche le t-SNE sur une période donnée.
    
    X_full      : Matrice complète des embeddings (N, 300)
    df_features : DataFrame complet (news_features.csv)
    start_date  : Date de début 'YYYY-MM-DD'
    end_date    : Date de fin 'YYYY-MM-DD'
    k           : Nombre de clusters
    """
    # Temporal filtering of the DataFrame to match the date range
    mask = (df_features['date'] >= start_date) & (df_features['date'] <= end_date)
    sub_df = df_features.loc[mask].copy()
    X_sub = X_full[mask.values] 
    
    if len(sub_df) < k:
        print(f"Not enough samples ({len(sub_df)}) to create {k} clusters.")
        return None

    # HAC
    model = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    labels = model.fit_predict(X_sub)
    sub_df['Cluster'] = labels.astype(str)

    # t-SNE 
    # Perplexity should be less than the number of samples.
    actual_perplexity = min(perplexity, max(1, len(sub_df) - 1))
    tsne = TSNE(
        n_components=2, 
        perplexity=actual_perplexity, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    X_2d = tsne.fit_transform(X_sub)
    sub_df['X'] = X_2d[:, 0]
    sub_df['Y'] = X_2d[:, 1]

    # Plotly
    fig = px.scatter(
        sub_df, 
        x='X', 
        y='Y', 
        color='Cluster',
        hover_data={'headline': True, 'date': True, 'X': False, 'Y': False},
        title=f"Financial Events Clusters ({start_date} au {end_date})",
        labels={'Cluster': 'Detected Event'},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Dark24
    ) 
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')))
    fig.update_layout(
        xaxis_title="t-SNE dimension 1",
        yaxis_title="t-SNE dimension 2",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    return fig















