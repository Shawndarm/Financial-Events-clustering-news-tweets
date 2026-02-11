import pandas as pd
import numpy as np
import spacy
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