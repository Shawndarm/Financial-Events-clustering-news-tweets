import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cosine


######################  Preprocessing and Tokenisation ######################
def preprocess_tweets_spacy(text, nlp):
    """
    Optimized for Step 6: Social Media Assignment (Carta et al.)
    Includes Lemmatization and Twitter-specific filtering.
    """
    if not isinstance(text, str): 
        return ""
    # On passe en minuscule et on traite avec spaCy
    doc = nlp(text.lower())
    
    tokens = [
        t.text for t in doc 
        if t.is_alpha              # Garde uniquement les lettres
        and not t.is_stop           # Supprime les mots vides
        and not t.like_url         # Supprime les liens (spécifique tweets)
        #and not t.text.startswith('@') # Supprime les mentions
        and len(t.text) > 2        # Ignore les mots trop courts
    ]
    return " ".join(tokens)


############"#############  Financial Filtering and Embedding Calculation ######################"
import numpy as np

def filter_and_embed_tweets(df, text_col, lexicon, w2v_model):
    """
    Étape 6 (Partie 2) : Filtrage financier et calcul de l'embedding.
    Conforme à Carta et al. : 
    1. Suppression des doublons (anti-spam).
    2. Filtre les mots par lexique.
    3. Calcule la moyenne des embeddings.
    """
    # --- AJOUT : Étape de dédoublonnage (spécifiée par les auteurs pour éviter le spam) ---
    # On supprime les tweets ayant le même contenu textuel exact
    df = df.drop_duplicates(subset=['full_content']).copy()
    
    # Utilisation d'un set pour une recherche ultra-rapide
    lexicon_set = set(lexicon)
    embeddings = []
    
    for text in df[text_col]:
        # 1. Extraction des mots du texte nettoyé
        words = str(text).split() 
        
        # 2. Filtrage par le lexique spécialisé (Loughran-McDonald / Personnalisé)
        relevant_words = [w for w in words if w in lexicon_set] 
        
        # 3. Récupération des vecteurs Word2Vec pour ces mots précis
        valid_vecs = [w2v_model[w] for w in relevant_words if w in w2v_model] 
        
        if valid_vecs:
            # 4. Moyenne des embeddings des mots restants (Vecteur du tweet)
            embeddings.append(np.mean(valid_vecs, axis=0))
        else:
            # Si aucun mot financier n'est trouvé, le tweet est considéré comme bruit
            embeddings.append(None)          
            
    df['tweet_embedding'] = embeddings
    
    # Suppression des tweets "bruit" (ceux qui n'ont pas de mots financiers)
    df_clean = df.dropna(subset=['tweet_embedding']).copy()
    
    print(f"Tweets analysés (après dédoublonnage) : {len(df)}")
    print(f"Tweets filtrés (bruit social) : {len(df) - len(df_clean)}")
    print(f"Tweets conservés (signal financier) : {len(df_clean)}")
    
    return df_clean

###################### Assignment to clusters #################################
def assign_tweets_to_events_by_period(tweets_df, news_signatures, start_date, end_date, threshold=0.6):
    """
    Step 6: Assigns tweets to clusters for a specific time window.
    """
    # 1. Filtrage temporel des tweets
    mask = (tweets_df['date'] >= start_date) & (tweets_df['date'] <= end_date)
    period_tweets = tweets_df.loc[mask].copy()
    
    if period_tweets.empty:
        print(f"Aucun tweet trouvé entre le {start_date} et le {end_date}.")
        return pd.DataFrame()

    results = []
    
    # 2. Boucle d'assignation (Similarité Cosinus)
    for idx, row in period_tweets.iterrows():
        tweet_vec = row['tweet_embedding']
        best_sim = -1
        best_cluster = -1
        
        for cluster_id, sig_vec in news_signatures.items():
            # Similarité = 1 - Distance Cosinus
            sim = 1 - cosine(tweet_vec, sig_vec)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id
        
        # 3. Validation par le seuil Delta (Filtrage du bruit)
        if best_sim >= threshold:
            results.append({
                'date': row['date'],
                'full_content': row['full_content'],
                'assigned_event': best_cluster,
                'similarity': best_sim
            })
            
    assigned_df = pd.DataFrame(results)
    
    print(f"--- Résultat pour la période {start_date} au {end_date} ---")
    print(f"Tweets dans la période : {len(period_tweets)}")
    print(f"Tweets assignés aux événements : {len(assigned_df)}")
    
    return assigned_df

################## Vizualization Tweets assignment  ########################

def plot_tweet_assignment_bars(tweets_df, news_signatures, start_date, end_date, threshold=0.55):
    # 1. Filtrage par date
    mask = (tweets_df['date'] >= start_date) & (tweets_df['date'] <= end_date)
    df_period = tweets_df.loc[mask].copy()
    
    # 2. Calcul de la similarité maximale pour chaque tweet
    max_similarities = []
    for _, row in df_period.iterrows():
        tweet_vec = row['tweet_embedding']
        # Calcul de la similarité avec chaque signature d'événement
        sims = [1 - cosine(tweet_vec, sig_vec) for sig_vec in news_signatures.values()]
        max_similarities.append(max(sims) if sims else 0)
    
    df_period['max_similarity'] = max_similarities
    
    # 3. Tri chronologique (important pour l'abscisse)
    df_period = df_period.sort_values(by='date')
    
    # 4. Définition des couleurs (Vert pour assigné, Rouge pour rejeté)
    colors = ['#2ecc71' if sim >= threshold else '#e74c3c' for sim in df_period['max_similarity']]
    
    # 5. Création du Bar Plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(range(len(df_period))), # Index numérique pour l'ordre
        y=df_period['max_similarity'],
        marker_color=colors,
        # On injecte les données pour le survol
        customdata=np.stack((
            df_period['date'].dt.strftime('%Y-%m-%d'), 
            df_period['full_content'],
            df_period['max_similarity']
        ), axis=-1),
        hovertemplate=(
            "<b>Date:</b> %{customdata[0]}<br>" +
            "<b>Similarité:</b> %{customdata[2]:.4f}<br>" +
            "<b>Texte Nettoyé:</b> %{customdata[1]}<extra></extra>"
        )
    ))

    # 6. Ajout de la ligne de seuil (Threshold)
    fig.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="#3498db", 
        line_width=2,
        annotation_text=f"Seuil Delta ({threshold})", 
        annotation_position="top right"
    )

    # Mise en page
    fig.update_layout(
        title=f"<b>Distribution des Assignations de Tweets</b><br><sup>Période : {start_date} au {end_date}</sup>",
        xaxis_title=f"Tweets triés par date (Total: {len(df_period)})",
        yaxis_title="Niveau de Similarité Cosinus",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12),
        height=600
    )

    # Masquer les étiquettes de l'axe X (trop nombreuses) pour privilégier le survol
    fig.update_xaxes(showticklabels=False)

    return fig


