"""
tweet_assignment.py — Tweet Assignment to news clusters + Alert Generation.
Implements the final pipeline steps from Carta et al. (2021).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def assign_tweets_to_clusters(tweet_vectors: np.ndarray, centroids: dict,
                               similarity_threshold: float = 0.55) -> dict:
    """
    Assign each tweet to the closest news-cluster centroid.
    
    Parameters
    ----------
    tweet_vectors : (n_tweets, dim) array of tweet embeddings
    centroids : {cluster_id: centroid_vector}
    similarity_threshold : minimum cosine similarity to assign (paper uses 0.5 distance = 0.5 similarity)
                          Reference project uses 0.55.
    
    Returns
    -------
    dict with 'assignments' (cluster_id -> list of tweet indices),
              'assigned_clusters' (array of cluster assignments, -1 if unassigned),
              'similarities' (array of max similarities)
    """
    centroid_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[c] for c in centroid_ids])

    sim_matrix = cosine_similarity(tweet_vectors, centroid_matrix)
    best_idx = sim_matrix.argmax(axis=1)
    best_sim = sim_matrix.max(axis=1)

    assigned_clusters = np.full(len(tweet_vectors), -1, dtype=int)
    assignments = {c: [] for c in centroid_ids}

    for i in range(len(tweet_vectors)):
        if best_sim[i] >= similarity_threshold:
            cluster = centroid_ids[best_idx[i]]
            assigned_clusters[i] = cluster
            assignments[cluster].append(i)

    n_assigned = (assigned_clusters != -1).sum()
    print(f"Tweet assignment: {n_assigned}/{len(tweet_vectors)} assigned "
          f"({n_assigned/len(tweet_vectors)*100:.1f}%) | threshold={similarity_threshold}")
    
    for c in centroid_ids:
        print(f"  Cluster {c}: {len(assignments[c])} tweets")

    return {
        'assignments': assignments,
        'assigned_clusters': assigned_clusters,
        'similarities': best_sim
    }


def compute_daily_assignment_ratio(tweet_dates: pd.Series,
                                    assigned_clusters: np.ndarray) -> pd.DataFrame:
    """
    Compute the daily ratio of assigned tweets / total tweets.
    This is the metric used for alert generation.
    """
    df = pd.DataFrame({
        'date': pd.to_datetime(tweet_dates).values,
        'assigned': assigned_clusters != -1
    })
    
    daily = df.groupby('date').agg(
        total=('assigned', 'count'),
        assigned=('assigned', 'sum')
    ).reset_index()
    
    daily['ratio'] = daily['assigned'] / daily['total']
    daily['pct'] = daily['ratio'] * 100
    
    return daily


def generate_alerts(daily_ratios: pd.DataFrame,
                    alert_threshold: float = 0.03) -> pd.DataFrame:
    """
    Generate alerts when the daily assigned ratio exceeds the threshold.
    
    Parameters
    ----------
    daily_ratios : DataFrame from compute_daily_assignment_ratio
    alert_threshold : fraction (0.03 = 3% as in the paper)
    """
    daily_ratios['alert'] = daily_ratios['ratio'] > alert_threshold
    n_alerts = daily_ratios['alert'].sum()
    print(f"Alert generation: {n_alerts}/{len(daily_ratios)} days with alerts "
          f"({n_alerts/len(daily_ratios)*100:.1f}%) | threshold={alert_threshold*100:.1f}%")
    return daily_ratios


# ─────────────────────── GROUND TRUTH ───────────────────────────

def build_ground_truth(prices_df: pd.DataFrame, variation_threshold: float = 0.02,
                       gap_tolerance: int = 3) -> tuple:
    """
    Build event ground truth based on weekly S&P 500 variations > threshold.
    
    Returns (event_days_df, events_list)
    """
    prices = prices_df.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values('date')
    prices['close_7d'] = prices['close'].shift(-7)
    prices['weekly_var'] = abs(
        (prices['close_7d'] - prices['close']) / prices['close']
    )
    prices['is_event'] = prices['weekly_var'] > variation_threshold

    # Aggregate consecutive event days into events
    events = []
    current_start = None
    last_event_day = None

    for _, row in prices.iterrows():
        if row['is_event']:
            if current_start is None:
                current_start = row['date']
            last_event_day = row['date']
        else:
            if current_start is not None:
                gap = (row['date'] - last_event_day).days
                if gap > gap_tolerance:
                    events.append((current_start, last_event_day))
                    current_start = None

    if current_start is not None:
        events.append((current_start, last_event_day))

    n_event_days = prices['is_event'].sum()
    print(f"Ground truth: {n_event_days} event days ({n_event_days/len(prices)*100:.1f}%) "
          f"→ {len(events)} events")
    return prices, events


def evaluate_alerts(alert_dates: list, events: list) -> dict:
    """
    Compute Precision, Recall, F-score.
    """
    alert_dates = pd.to_datetime(alert_dates)
    
    # Recall: for each event, at least one alert in the interval?
    spotted = 0
    for start, end in events:
        if any((start <= a) & (a <= end) for a in alert_dates):
            spotted += 1
    recall = spotted / len(events) if events else 0

    # Precision: for each alert, does it fall within an event?
    hits = sum(1 for a in alert_dates
               if any((start <= a) & (a <= end) for start, end in events))
    precision = hits / len(alert_dates) if len(alert_dates) > 0 else 0

    f_score = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0)

    print(f"Evaluation: Precision={precision:.3f} Recall={recall:.3f} F-score={f_score:.3f}")
    return {'precision': precision, 'recall': recall, 'f_score': f_score}
