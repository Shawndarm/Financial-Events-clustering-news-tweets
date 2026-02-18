"""
tweet_preprocessing.py — Tweet Preprocessing + Embedding
Implements Step 6 (Tweet Modeling) from Carta et al. (2021).

Adapted for SVB crisis period (March 3-9, 2023).
Uses the SAME GloVe model and lexicons as Roland's news pipeline
to ensure tweets and news share the SAME semantic space.

Author: Maeva — Master 2 MOSEF 2024-2025
"""

import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════
#  TWEET CLEANING
# ═══════════════════════════════════════════════════════════

def clean_tweet(text):
    """
    Tweet-specific cleaning pipeline:
    1. Remove retweet prefixes (RT @user:) and embedded retweet markers
    2. Remove URLs
    3. Convert $CASHTAGS → lowercase (e.g., $SPX → spx)
    4. Remove @mentions
    5. Remove hashtag symbols (keep word)
    6. Remove punctuation and numbers
    7. Lowercase and strip extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<:retweet:[^>]+>', '', text)
    text = re.sub(r'\bRT\b\s*@?\w*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\$([A-Za-z]+)', lambda m: m.group(1).lower(), text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_tweet_spacy(text, nlp):
    """
    spaCy-based preprocessing (mirrors Roland's preprocess_spacy 
    in lexicon_generation.py):
    - Tokenization
    - Stopwords removal  
    - Alpha-only filtering
    - Min length > 2
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""
    doc = nlp(text.lower())
    tokens = [t.text for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 2]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════
#  TWEET EMBEDDING (mirrors feature_engineering.py)
# ═══════════════════════════════════════════════════════════

def compute_tweet_embedding(clean_text, lexicon_set, model, dim=300):
    """
    Core Tweet Embedding Step (mirrors compute_news_embedding):
    1. Filter: Retain only words present in the daily lexicon.
    2. Embedding: Extract GloVe vectors for these specific words.
    3. Average: Compute the mean vector (Tweet-Embedding, 300D).
    
    CRITICAL: Uses the SAME model and lexicon as news articles.
    """
    tokens = clean_text.split()
    valid_vectors = [
        model[w] for w in tokens
        if w in lexicon_set and w in model
    ]
    if not valid_vectors:
        return None
    return np.mean(valid_vectors, axis=0)


# ═══════════════════════════════════════════════════════════
#  LOAD LEXICONS FOR A DATE RANGE
# ═══════════════════════════════════════════════════════════

def load_lexicons_for_period(lexicon_dir, start_date=None, end_date=None):
    """
    Load daily lexicons from Roland's filtered lexicon directory.
    Returns a dict {date_str: set(words)} and the union of all words.
    
    Parameters
    ----------
    lexicon_dir : path to daily_lexicons_filtered/
    start_date, end_date : optional date filters (str 'YYYY-MM-DD')
    """
    available_lexicons = {}
    all_words = set()
    
    for f in sorted(os.listdir(lexicon_dir)):
        if not f.startswith('lexicon_filtered_') or not f.endswith('.csv'):
            continue
        day_str = f.replace('lexicon_filtered_', '').replace('.csv', '')
        
        # Date filter
        if start_date and day_str < start_date:
            continue
        if end_date and day_str > end_date:
            continue
            
        try:
            lex_df = pd.read_csv(os.path.join(lexicon_dir, f))
            words = set(lex_df['word'].tolist())
            available_lexicons[day_str] = words
            all_words.update(words)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    
    print(f"Loaded {len(available_lexicons)} daily lexicons")
    print(f"Union lexicon: {len(all_words)} unique words")
    return available_lexicons, all_words


# ═══════════════════════════════════════════════════════════
#  FULL TWEET EMBEDDING PIPELINE
# ═══════════════════════════════════════════════════════════

def run_tweet_embedding_pipeline(tweets_df, lexicon_set, model):
    """
    Embed all tweets using a single lexicon set (union or daily).
    
    Parameters
    ----------
    tweets_df : DataFrame with 'clean' column (preprocessed text)
    lexicon_set : set of lexicon words to filter by
    model : gensim KeyedVectors (GloVe Dolma 300d)
    
    Returns
    -------
    DataFrame with added columns: 'embedding', 'has_embedding'
    """
    result = tweets_df.copy()
    embeddings = []
    
    for idx, row in tqdm(result.iterrows(), total=len(result), desc="Tweet Embedding"):
        vector = compute_tweet_embedding(row['clean'], lexicon_set, model)
        embeddings.append(vector)
    
    result['embedding'] = embeddings
    result['has_embedding'] = result['embedding'].apply(lambda x: x is not None)
    
    n_ok = result['has_embedding'].sum()
    print(f"\nEmbedded: {n_ok}/{len(result)} tweets ({n_ok/len(result)*100:.1f}%)")
    return result
