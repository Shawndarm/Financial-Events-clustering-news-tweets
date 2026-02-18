from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

###################### Relevant Words Extraction ######################
def extract_relevant_words_with_scores(clean_df, lexicon, top_n=10):
    tfidf = TfidfVectorizer(vocabulary=lexicon, stop_words='english', lowercase=True)
    tfidf_matrix = tfidf.fit_transform(clean_df['headline'])
    feature_names = tfidf.get_feature_names_out()
    
    results = {}
    for cluster_id in sorted(clean_df['Cluster'].unique()):
        # We get the index corresponding to the clusters
        cluster_indices = clean_df[clean_df['Cluster'] == cluster_id].index
        row_indices = [clean_df.index.get_loc(idx) for idx in cluster_indices]
        
        # Average TF-IDF by cluster
        avg_scores = np.asarray(tfidf_matrix[row_indices].mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        
        # We keep (word, score) pairs for the top words with positive scores
        results[cluster_id] = [(feature_names[i], avg_scores[i]) for i in top_indices if avg_scores[i] > 0]
    return results


###################### Visualization of Relevant Words ######################
def plot_keywords_bar_chart(keywords_results):
    """
    Creates a horizontal bar chart for each cluster's top words with fixed spacing.
    """
    n_clusters = len(keywords_results)
    
    # On augmente légèrement le vertical_spacing (passé de 0.1 à 0.15 ou 0.2 selon n_clusters)
    # Plus il y a de clusters, plus l'espace relatif doit être géré
    v_space = 0.15 if n_clusters > 2 else 0.2

    fig = make_subplots(
        rows=n_clusters, cols=1, 
        subplot_titles=[f"<b>Cluster {c}</b> - Key Financial Terms" for c in keywords_results.keys()],
        vertical_spacing=v_space # Ajusté pour éviter les chevauchements
    )

    for i, (cluster_id, words_data) in enumerate(keywords_results.items()):
        words = [item[0] for item in words_data][::-1] 
        scores = [item[1] for item in words_data][::-1]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=words,
                orientation='h',
                name=f"Cluster {cluster_id}",
                marker=dict(color=px.colors.qualitative.Dark24[i % 24])
            ),
            row=i+1, col=1
        )

    # Ajustements de mise en page pour la clarté
    fig.update_layout(
        height=320 * n_clusters, 
        title_text="<b>Relevant Words Extraction per Event Cluster</b><br><sup>Based on TF-IDF + Personalized Lexicon</sup>",
        title_x=0.5, # Centre le titre principal
        showlegend=False,
        template="plotly_white",
        # Ajout de marges pour que le titre du haut ne soit pas coupé
        margin=dict(t=120, b=50, l=150, r=50) 
    )
    
    # On ne met le titre de l'axe X que sur le dernier graphique pour épurer
    fig.update_xaxes(title_text="Average TF-IDF Score", row=n_clusters, col=1)
    # Optionnel : Forcer les ticks à être lisibles
    fig.update_yaxes(ticksuffix="  ") # Ajoute un petit espace entre le mot et la barre

    return fig