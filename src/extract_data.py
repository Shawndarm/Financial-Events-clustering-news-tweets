#################### Main Functions for data extraction ####################
import pandas as pd
import os
from newspaper import Article
from tqdm import tqdm
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


###################### News extraction ######################

def check_density(path):
    """Loads dataset, checks for date coverage and identifies gaps."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])   
    # Identify gaps in time series
    full_range = pd.date_range(df['date'].min(), df['date'].max())
    missing = full_range.difference(df['date'].unique())   
    # Print summary statistics
    print(f"Total records: {len(df)}")
    print(f"Coverage: {df['date'].nunique()} / {len(full_range)} days")
    print(f"Gaps found: {len(missing)} days")

    return df

def scrape_content(df, filename):
    """Scrapes full text and saves valid results incrementally."""
    results = []
    print("Starting extraction process...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            art = Article(row['url'])
            art.download()
            art.parse()           
            # Filter for high-quality content only (>500 chars)
            if len(art.text) > 500:
                results.append({
                    'date': row['date'],
                    'headline': art.title,
                    'body': art.text,
                    'url': row['url'],
                    'source': row['source']
                })
        except:
            continue       
        # Incremental save every 100 successful extractions
        if len(results) % 100 == 0 and results:
            pd.DataFrame(results).to_csv(filename, index=False)
    # Final save
    final_df = pd.DataFrame(results)
    final_df.to_csv(filename, index=False)
    print(f"Complete: {len(final_df)} valid articles extracted.")
    return final_df


###################### Tweets extraction ######################

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def process_social_data(file_path):
    # Load dataset and format dates
    df = pd.read_csv(file_path, low_memory=False)
    df['dt_obj'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
    df['date'] = df['dt_obj'].dt.date
    # Filter for 2023 and S&P 500 keywords
    df_2023 = df[df['dt_obj'].dt.year == 2023].copy()
    if df_2023.empty: return None
    keywords = r'\$SPY|\$SPX|S&P 500|SP500|Stock Market'
    mask = df_2023['description'].str.contains(keywords, case=False, na=False) | \
           df_2023['embed_title'].str.contains(keywords, case=False, na=False)
    df_spy = df_2023[mask].copy()
    # Merge content and calculate sentiment scores
    df_spy['full_content'] = df_spy['embed_title'].fillna('') + " " + df_spy['description'].fillna('')
    df_spy['sentiment'] = df_spy['full_content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    # Simulate engagement metrics using log-normal distribution
    n = len(df_spy)
    intensity = df_spy['sentiment'].abs()
    # Generate likes, retweets, and followers based on sentiment intensity
    df_spy['likes'] = (np.random.lognormal(1.5, 2.0, n) * (1 + intensity)).astype(int).clip(0, 50000)
    df_spy['retweets'] = (df_spy['likes'] * np.random.uniform(0.1, 0.3, n)).astype(int)
    df_spy['followers'] = np.random.lognormal(5.0, 3.0, n).astype(int).clip(0, 1000000)
    # Final column selection and date object conversion
    cols = ['date', 'full_content', 'likes', 'retweets', 'followers', 'sentiment', 'url']
    final_df = df_spy[cols].copy()
    final_df['date'] = pd.to_datetime(final_df['date'])  
    print(f"Extraction complete: {len(final_df)} records for 2023")
    return final_df




























