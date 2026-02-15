import pandas as pd
import numpy as np
import spacy
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.feature_extraction.text import CountVectorizer
from datetime import timedelta
from tqdm import tqdm
import os


######################  Preprocessing and Tokenisation ######################
def preprocess_spacy(text, nlp):
    """
    Lowercasing, Tokenization,
    Stopwords removal and alpha-filtering 
    """
    if not isinstance(text, str): return ""
    doc = nlp(text.lower())
    # We keep only alphabetic tokens that are not stopwords and have more than 2 characters
    tokens = [t.text for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 2]
    return " ".join(tokens)
######################  Daily Lexicon generation ######################
def build_daily_lexicon(news_train, prices_map, current_date, output_dir, dtm_output_dir):
    """
    Implements the Marginal Screening formula f(j), saves daily lexicon,
    and exports the Document-Term Matrix (DTM) for visualization.
    """
    N = len(news_train)
    if N < 5: return None  # Skip days with too few articles to avoid noise in the lexicon 
    ## Roland chose 5 because max 6 articles per day.

    # a. Vectorization (Binary 0/1) and Frequency Filtering: max_df=0.9, min_df=10
    vectorizer = CountVectorizer(
        binary=True, 
        max_df=0.90,     
        min_df=10,       
        stop_words='english'
    )   
    try:
        dtm_sparse = vectorizer.fit_transform(news_train['clean'])
        words = vectorizer.get_feature_names_out()
    except ValueError: 
        return None

    # b. Document-Term Matrix (DTM)
    os.makedirs(dtm_output_dir, exist_ok=True)
    dtm_df = pd.DataFrame(dtm_sparse.toarray(), columns=words) # We create a DataFrame where rows are news articles and columns are words
    # We add the dates to make the visualization more informative
    dtm_df.insert(0, 'article_date', news_train['date'].values)
    dtm_path = os.path.join(dtm_output_dir, f"dtm_{current_date}.csv")
    dtm_df.to_csv(dtm_path, index=False)
    # print(f"DTM saved: {dtm_path}") # Optional debug print

    # c. Marginal Screening Formula f(j) : f(j) = (1/N) * sum( X_k(j) * delta_k )
    deltas = news_train['date'].map(prices_map).fillna(0).values
    # Efficient calculation using dot product
    sum_product = np.array(dtm_sparse.T.dot(deltas)).flatten()
    f_j = sum_product / N

    # d. Percentiles (P20 & P80) based Selection
    p20 = np.percentile(f_j, 20)
    p80 = np.percentile(f_j, 80)
    lexicon_df = pd.DataFrame({'word': words, 'score': f_j})
    os.makedirs(output_dir, exist_ok=True)
    lexicon_df.sort_values('score', ascending=False).to_csv( # Save the Daily Lexicon ranking
        os.path.join(output_dir, f"lexicon_{current_date}.csv"), index=False
    )
    
    # Return Top/Bottom words for Sentiment Index calculation
    pos_words = lexicon_df[lexicon_df['score'] >= p80].set_index('word')['score'].to_dict()
    neg_words = lexicon_df[lexicon_df['score'] <= p20].set_index('word')['score'].to_dict()
    return {**pos_words, **neg_words}
######################  Lexicon visualization ######################
def visualize_daily_lexicon(date_str):
    """
    Loads a daily lexicon CSV and plots the f(j) distribution with thresholds.
    """
    FILE_PATH = f'../data/processed/daily_lexicons/lexicon_{date_str}.csv'

    # Load and sort data
    df = pd.read_csv(FILE_PATH)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    # Re-calculate thresholds for the visualization
    p20 = np.percentile(df['score'], 20)
    p80 = np.percentile(df['score'], 80)
    # Categorize words for coloring
    df['type'] = 'Neutral'
    df.loc[df['score'] >= p80, 'type'] = 'Positive'
    df.loc[df['score'] <= p20, 'type'] = 'Negative'

    # Bar Chart
    fig = px.bar(
        df, 
        x='word', 
        y='score',
        color='type',
        color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#bdc3c7'},
        title=f"Lexicon Sentiment Scores (f_j) - {date_str}",
        labels={'score': 'Score f(j)', 'word': 'Financial Terms', 'type': 'Category'},
        hover_data={'score': ':.5f'}
    )
    # Add Horizontal Lines for P80 and P20
    fig.add_hline(y=p80, line_dash="dash", line_color="#27ae60", 
                  annotation_text=f"P80 Threshold ({p80:.5f})", annotation_position="top right")
    fig.add_hline(y=p20, line_dash="dash", line_color="#c0392b", 
                  annotation_text=f"P20 Threshold ({p20:.5f})", annotation_position="bottom right")
    # Styling: Add Range Slider because there are many words
    fig.update_layout(
        xaxis_title="Words (Sorted by Score)",
        yaxis_title="Marginal Screening Score f(j)",
        xaxis_tickangle=-45,
        height=600,
        template="plotly_white"
    )
    fig.show()