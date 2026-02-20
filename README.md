# Event Detection in Finance by Clustering News and Tweets 📈📰

![Python Badge](https://img.shields.io/badge/Python-3.10%2B-blue)
![Data Science Badge](https://img.shields.io/badge/Data_Science-NLP_%7C_Clustering-orange)
![Academic Badge](https://img.shields.io/badge/Université_Paris_1-Panthéon_Sorbonne-maroon)

**Authors:** Roland DUTAUZIET & Maeva N'GUESSAN
**Program:** Master 2 MOSEF - Data Science (Modélisations Statistiques Économiques et Financières) - 2025/2026  
**Context:** Quantitative Finance Project

## 📖 Project Overview

This project is a reproduction and an extension of the academic paper *"Event Detection in Finance by Clustering News and Tweets"* (Carta et al., 2021). The core objective is to build a robust NLP pipeline capable of detecting major financial events by clustering professional news articles, and then validating these events by measuring the "Social Heat" (public attention) through social media platforms (Twitter/Stocktwits).

We applied this methodology to the **S&P 500 index for the year 2023**, a period marked by high volatility, including the Silicon Valley Bank (SVB) collapse and and the ARM IPO.



---

## 🏗️ Project Architecture

The repository is structured to ensure reproducibility and clean code separation.

```text
Financial-Events-clustering-news-tweets
├── data/
│   ├── for_models/
│   │   ├── output/
│   │   │   ├── table_3_tweet_assignment_AI.csv
│   │   │   │   # Tweet-to-cluster assignment results for the AI enthusiasm period (May–July 2023).
│   │   │   │   # Used for quantitative evaluation and representative tweet analysis.
│   │   │   ├── table_3_tweet_assignment_SVB.csv
│   │   │   │   # Tweet assignment results for the Silicon Valley Bank crisis period (March 2023).
│   │   │   │   # Contains cosine similarity scores and assigned event IDs.
│   │   │   ├── clean_news_week_AI.csv
│   │   │   │   # Preprocessed weekly news dataset for the AI event window.
│   │   │   ├── clean_news_week_SVB.csv
│   │   │   │   # Preprocessed weekly news dataset for the SVB crisis window.
│   │   │   ├── final_event_signatures_AI.csv
│   │   │   │   # Median-based centroids (event signatures) after outlier removal (AI period).
│   │   │   ├── final_event_signatures_SVB.csv
│   │   │   │   # Robust cluster centroids after cleaning (SVB period).
│   │   │   ├── news_features.csv
│   │   │   │   # 300-dimensional document embeddings (GloVe) for each news article.
│   │   │   ├── tweets_assigned.csv
│   │   │   │   # Full tweet assignment output across all periods.
│   │   │   ├── tweets_assigned_AI.csv
│   │   │   │   # Tweets assigned to clusters during AI enthusiasm period.
│   │   │   ├── tweets_assigned_SVB.csv
│   │   │   │   # Tweets assigned to clusters during SVB crisis.
│   │   │   └── tweets_features.csv
│   │   │       # 300-dimensional embeddings for tweets (same GloVe model as news).
│   │   │
│   │   └── processed/
│   │       ├── daily_dtm/
│   │       │   # Daily binary Document-Term Matrices used for Marginal Screening.
│   │       ├── daily_lexicons_filtered/
│   │       │   # Daily filtered lexicons (P20/P80 percentile selection).
│   │       ├── daily_lexicons_full/
│   │       │   # Full lexicons before percentile filtering.
│   │       ├── news_2023.csv
│   │       │   # 1,565 financial news articles (GDELT, Yahoo Finance, CNBC).
│   │       ├── news_2023_clean.csv
│   │       │   # Cleaned and preprocessed version of news_2023.csv.
│   │       ├── sp500_2023.csv
│   │       │   # Daily S&P 500 prices (271 trading days). Used for return computation and ground truth.
│   │       └── tweets_2023.csv
│   │           # 2,243 filtered financial tweets mentioning $SPX, $SPY, or S&P 500.
│
├── docs/
│   ├── P10 Event detection in finance using ...
│   │   # Original reference paper (Carta et al., 2021).
│   ├── Rapport_Event_detection_Roland_...
│   │   # Full academic report (methodology, results, evaluation).
│   └── Slides Finance quantitative.pdf
│       # Presentation slides summarizing the project.
│
├── img/
│   ├── 1_lexicon_generation/
│   │   # Marginal Screening plots and percentile threshold visualizations.
│   ├── 2_feature_engineering/
│   │   # Embedding illustrations and vector representations.
│   ├── 3_news_clustering/
│   │   # Silhouette scores, dendrograms, t-SNE visualizations.
│   ├── 4_outlier_removal/
│   │   # Cluster cleaning and centroid repositioning figures.
│   ├── 5_relevant_words_extraction/
│   │   # TF-IDF keyword extraction results per cluster.
│   ├── 6_tweets_assignment/
│   │   # Tweet-to-cluster similarity visualizations.
│   ├── 7_alert_generation/
│   │   # Alert ratio plots and ground truth comparisons.
│   ├── 6_tweets_assignment.zip
│   │   # Archived visual results for tweet assignment.
│   └── 7_alert_generation.zip
│       # Archived visual results for alert generation.
│
├── notebooks/
│   ├── 0_extract_data.ipynb
│   │   # Data ingestion, preprocessing, sentiment computation (VADER).
│   ├── 1_lexicon_generation.ipynb
│   │   # Implementation of Marginal Screening and daily lexicon construction.
│   ├── 2_feature_engineering.ipynb
│   │   # GloVe embedding computation and document vector construction.
│   ├── 3_news_clustering.ipynb
│   │   # Hierarchical clustering (HAC), silhouette maximization, centroid computation.
│   ├── 4_outlier_removal.ipynb
│   │   # Double-criterion outlier removal (silhouette + cosine similarity).
│   ├── 5_relevant_words_extraction.ipynb
│   │   # TF-IDF keyword extraction per cluster.
│   ├── 6_tweet_assignment.ipynb
│   │   # Tweet embedding and cosine similarity-based assignment.
│   └── 7_Alert_generation.ipynb
│       # Alert computation, ground truth construction, Precision/Recall/F-score evaluation.
│
├── src/
│   ├── __pycache__/
│   │   # Compiled Python bytecode (ignored by git).
│   ├── alert_generation.py
│   │   # Computes daily assignment ratio R(d), generates alerts (R(d) > θ),
│   │   # builds S&P 500 ground truth (|weekly return| > 2%), and evaluates Precision/Recall/F-score.
│   ├── extract_data.py
│   │   # Handles dataset loading, cleaning, date alignment,
│   │   # tweet filtering by cashtags, and sentiment scoring (VADER).
│   ├── feature_engineering.py
│   │   # Filters articles using daily lexicons and computes 300D GloVe embeddings.
│   ├── lexicon_generation.py
│   │   # Builds daily binary DTM, computes Marginal Screening f(j),
│   │   # selects positive/negative financial terms via percentile thresholds.
│   ├── news_clustering.py
│   │   # Runs Agglomerative Clustering (cosine + average linkage),
│   │   # compares with K-Means/K-Medians, computes silhouette scores,
│   │   # and calculates median-based centroids (event signatures).
│   ├── outlier_removal.py
│   │   # Applies double filtering: per-sample silhouette + centroid cosine similarity.
│   │   # Removes noisy articles and recalculates centroids.
│   ├── relevant_words_extraction.py
│   │   # Computes TF-IDF within each cluster to extract top representative financial terms.
│   └── tweet_assignment.py
│       # Embeds tweets using the same GloVe model,
│       # assigns them to nearest event centroids via cosine similarity (threshold = 0.5).
│
├── .gitignore
│   # Excludes large data files, virtual environments, and cache folders.
│
├── .python-version
│   # Specifies Python interpreter version for reproducibility.
│
├── LICENSE
│   # Project license.
│
├── README.md
│   # Project documentation and methodological overview.
│
├── pyproject.toml
│   # Project metadata and dependency management (uv-compatible).
│
└── uv.lock
    # Locked dependency versions ensuring reproducible environments.
```

---

## Installation & Setup

We use uv, an extremely fast Python package and project manager written in Rust, to handle our virtual environment and dependencies.

### 1. Install uv
If you haven't installed uv yet, run:

```Bash
# On Windows (PowerShell)
pip install uv
# On macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
### 2. Setup the Virtual Environment
Navigate to the project directory, create and activate the environment:

```Bash
cd Financial-Events-clustering-news-tweets
uv venv
# Windows:
 .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
3. Install Dependencies
Install the required packages (Pandas, Numpy, Scikit-learn, Plotly, SciPy, Gensim):

```Bash
uv sync
```


# Financial Event Detection using News & Tweets (S&P 500 – 2023)

## Abstract

This project reproduces and extends the methodology of Carta et al. (2021) for financial event detection using hierarchical clustering of news articles combined with social media resonance analysis.  

Applied to the S&P 500 over 2023, our pipeline integrates dynamic lexicon generation, document embeddings, hierarchical clustering, tweet assignment via cosine similarity, and alert generation based on social activity.  

The objective is to detect financially significant events in near real-time and evaluate performance against market ground truth derived from S&P 500 weekly variations.

---

# Methodology & Pipeline

Our pipeline follows the 7-step framework proposed by Carta et al. (2021), adapted to the 2023 S&P 500 context.

---

# Step 1 — Lexicon Generation

To reduce textual noise and retain financially meaningful signals, we construct a **dynamic domain-specific lexicon**.

- A **binary Document-Term Matrix (DTM)** is built over a rolling 4-week window.
- We compute the **Marginal Screening score** for each term:

\[
f(j) = \frac{1}{N} \sum_{k=1}^{N} X_k^{(j)} \cdot \delta_k
\]

Where:
- \( X_k^{(j)} \in \{0,1\} \) indicates the presence of term \( j \) in article \( k \)
- \( \delta_k \) is the daily S&P 500 return
- \( N \) is the number of articles in the window

- Terms above the 80th percentile (positive impact) and below the 20th percentile (negative impact) are retained.
- Neutral terms are discarded.

This produces a daily financial lexicon capturing market-relevant vocabulary.

📌 *Insert Screenshot:* Marginal Screening score distribution or lexicon word cloud.

---

# Step 2 — Feature Engineering (Embeddings)

Each news article is transformed into a dense numerical vector.

- Texts are tokenized and cleaned.
- Words not present in the daily lexicon are discarded.
- Pre-trained embeddings (GloVe 300D) are used.
- The document embedding is computed as:

\[
v_a = \frac{1}{|W_a|} \sum_{w \in W_a} \text{Embedding}(w)
\]

Where \( W_a \) is the set of lexicon-filtered words in article \( a \).

This step converts financial text into structured mathematical representations.

📌 *Insert Screenshot:* Example of 300D embeddings table or embedding visualization.

---

# Step 3 — News Clustering

We group news articles into candidate financial events.

Algorithms tested:
- K-Means  
- Agglomerative Clustering (HAC)  
- K-Medians  
- (Optional comparison: DBSCAN / GMM)

The optimal number of clusters \( k \) is selected by maximizing the **Silhouette Score**:

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Where:
- \( a(i) \) = intra-cluster distance
- \( b(i) \) = nearest-cluster distance

Consistent with the original paper, **Hierarchical Agglomerative Clustering (cosine distance + average linkage)** produced the most coherent and interpretable clusters.

📌 *Insert Screenshot:* Silhouette maximization plot or t-SNE cluster visualization.

---

# Step 4 — Relevant Words Extraction & Outlier Removal

## Relevant Words Extraction

- We compute **Average TF-IDF** per cluster.
- The top representative financial terms are extracted.
- This improves cluster interpretability.

## Outlier Removal

We apply a **double filtering criterion**:

1. Per-sample Silhouette score  
2. Cosine similarity to cluster centroid  

Articles below the percentile threshold in either metric are removed.

Clusters lacking strong financial relevance are discarded.

📌 *Insert Screenshot:* Bar chart of key financial terms per cluster.

---

# Step 5 — Event Signatures

For each validated cluster, we compute a robust centroid:

\[
c_k = \text{median}(\{v_a : a \in \text{cluster}_k\})
\]

This centroid represents the **Event Signature**, acting as a mathematical summary of the event and serving as a reference for social media matching.

📌 *Insert Screenshot:* Cleaned clusters with centroid markers.

---

# Step 6 — Tweet Assignment

We measure public attention by linking tweets to event signatures.

Process:

- **De-duplication:** Remove identical tweets.
- **Embedding:** Tweets are embedded using the same 300D model.
- **Cosine Similarity:** Each tweet is compared to event centroids.

\[
\text{sim}(t, c_k) = \frac{t \cdot c_k}{\|t\| \|c_k\|}
\]

- Tweets with similarity ≥ threshold \( \delta \) are assigned to the event.

This quantifies social resonance around detected events.

📌 *Insert Screenshot:* Tweet similarity distribution or assignment plot.

---

# Step 7 — Alert Generation & Evaluation

## Social Heat

We define the daily assignment ratio:

\[
R(d) = \frac{\text{Assigned Tweets}_d}{\text{Total Tweets}_d}
\]

An alert is triggered when:

\[
R(d) > \theta
\]

---

## Ground Truth Construction

To evaluate performance, we define market event intervals using weekly S&P 500 variation:

\[
\Delta_d = \frac{|close(d+7) - close(d)|}{close(d)}
\]

Days satisfying:

\[
\Delta_d > 0.02
\]

are labeled as event days. Consecutive event days are aggregated into event intervals.

---

## Evaluation Metrics

We compute:

- **Precision**
- **Recall**
- **F-Score**

These metrics measure alignment between generated alerts and true market events.

📌 *Insert Screenshot:* Plot showing S&P 500 price, ground truth intervals, and social alerts.

---

# Case Studies (2023)

## Silicon Valley Bank Collapse (March 2023)

- Significant spike in Social Heat
- Clear cluster separation
- Detected prior to major market drawdown

## AI Boom & Nvidia Rally (May–July 2023)

- Technology-focused clusters
- Strong resonance between news and tweets
- Captured momentum shift

## ARM IPO (September 2023)

- Anticipation reflected in clustering
- Immediate post-listing social amplification

---

# Key Findings

## HAC Dominance

Hierarchical Agglomerative Clustering consistently outperformed K-Means and density-based approaches in producing semantically coherent clusters.

## Recall over Precision

In financial risk management, missing a crash (low Recall) is more costly than issuing a false alert (low Precision).  
Our model prioritizes Recall and successfully captures most ground truth events.

## Social Hype Dynamics

- News dissemination is immediate.
- Social Heat may anticipate or slightly lag price reactions.
- This temporal asymmetry may provide exploitable alpha signals.

---

# Acknowledgments

**Original Authors:**  
Carta, S., et al. (2021). *Event Detection in Finance by Clustering News and Tweets.*

**Institution:**  
Université Paris 1 Panthéon-Sorbonne — Master 2 MOSEF (Quantitative Finance)
