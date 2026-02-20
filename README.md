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
Methodology & Pipeline

Our pipeline reproduces the 7-step framework proposed by Carta et al. (2021), adapted to the 2023 S&P 500 context.
Each phase is executed sequentially to detect financially significant events from news and social media data.

Step 1 — Lexicon Generation

To reduce textual noise and retain only financially meaningful signals, we construct a dynamic domain-specific lexicon.

A binary Document-Term Matrix (DTM) is built over a rolling 4-week window.

We apply Marginal Screening to compute a score 
𝑓
(
𝑗
)
f(j) for each term:

𝑓
(
𝑗
)
=
1
𝑁
∑
𝑘
=
1
𝑁
𝑋
𝑘
(
𝑗
)
⋅
𝛿
𝑘
f(j)=
N
1
	​

k=1
∑
N
	​

X
k
(j)
	​

⋅δ
k
	​


Terms above the 80th percentile (positive impact) and below the 20th percentile (negative impact) are retained.

Neutral words are discarded.

This produces a daily financial lexicon capturing market-relevant vocabulary.

📌 Insert Screenshot: Lexicon distribution plot or Marginal Screening score chart.

Step 2 — Feature Engineering (Embeddings)

Each news article is transformed into a dense numerical representation.

Texts are tokenized and cleaned.

Words not present in the daily lexicon are discarded.

Pre-trained word embeddings are used (GloVe 300D in our implementation).

The document embedding is computed as the average of valid word vectors:

𝑣
𝑎
=
1
∣
𝑊
𝑎
∣
∑
𝑤
∈
𝑊
𝑎
Embedding
(
𝑤
)
v
a
	​

=
∣W
a
	​

∣
1
	​

w∈W
a
	​

∑
	​

Embedding(w)

This converts raw financial text into structured mathematical vectors suitable for clustering.

📌 Insert Screenshot: Example of 300D embeddings table or embedding visualization.

Step 3 — News Clustering

We group news articles into candidate financial events.

Algorithms tested:

K-Means

Agglomerative Clustering (HAC)

K-Medians

(Optional comparison: DBSCAN / GMM depending on experiment)

The number of clusters 
𝑘
k is optimized via Silhouette Score maximization:

𝑠
(
𝑖
)
=
𝑏
(
𝑖
)
−
𝑎
(
𝑖
)
max
⁡
(
𝑎
(
𝑖
)
,
𝑏
(
𝑖
)
)
s(i)=
max(a(i),b(i))
b(i)−a(i)
	​


Consistent with the original paper, Hierarchical Agglomerative Clustering (cosine distance + average linkage) achieved the best performance, producing compact and semantically coherent clusters.

📌 Insert Screenshot: Silhouette Score comparison or t-SNE cluster visualization.

Step 4 — Relevant Words Extraction & Outlier Removal

Not all clusters represent true financial events.

Relevant Words Extraction

We compute Average TF-IDF per cluster.

The top representative financial terms are extracted.

This provides interpretability for each detected event.

Outlier Removal

We apply a double filtering criterion:

Per-sample Silhouette score

Cosine similarity to cluster centroid

Articles below the percentile threshold in either metric are removed.

Clusters lacking strong financial relevance are discarded.

📌 Insert Screenshot: Bar chart of top financial keywords per cluster.

Step 5 — Event Signatures

For validated clusters, we compute a robust centroid:

𝑐
𝑘
=
median
(
{
𝑣
𝑎
:
𝑎
∈
cluster
𝑘
}
)
c
k
	​

=median({v
a
	​

:a∈cluster
k
	​

})

This centroid represents the Event Signature — a compact mathematical summary of the event.

It acts as a gravitational anchor for social media resonance detection.

📌 Insert Screenshot: Cleaned clusters with centroid markers.

Step 6 — Tweet Assignment

We measure public attention by linking tweets to event signatures.

Process:

De-duplication: Remove identical tweets (anti-spam).

Embedding: Tweets are embedded using the same 300D model.

Cosine Similarity: Each tweet is compared to event centroids.

Threshold 
𝛿
δ: Tweets with similarity ≥ threshold are assigned to the event.

sim
(
𝑡
,
𝑐
𝑘
)
=
𝑡
⋅
𝑐
𝑘
∥
𝑡
∥
∥
𝑐
𝑘
∥
sim(t,c
k
	​

)=
∥t∥∥c
k
	​

∥
t⋅c
k
	​

	​


This step quantifies social resonance around detected events.

📌 Insert Screenshot: Tweet similarity distribution or assignment visualization.

Step 7 — Alert Generation & Evaluation
Alert Generation

We define Social Heat:

𝑅
(
𝑑
)
=
Assigned Tweets
𝑑
Total Tweets
𝑑
R(d)=
Total Tweets
d
	​

Assigned Tweets
d
	​

	​


An alert is triggered if:

𝑅
(
𝑑
)
>
𝜃
R(d)>θ
Ground Truth Construction

To evaluate performance, we define market event intervals based on weekly S&P 500 variation:

Δ
𝑑
=
∣
𝑐
𝑙
𝑜
𝑠
𝑒
(
𝑑
+
7
)
−
𝑐
𝑙
𝑜
𝑠
𝑒
(
𝑑
)
∣
𝑐
𝑙
𝑜
𝑠
𝑒
(
𝑑
)
Δ
d
	​

=
close(d)
∣close(d+7)−close(d)∣
	​


Days where:

Δ
𝑑
>
0.02
Δ
d
	​

>0.02

are labeled as event days, and consecutive event days are aggregated into intervals.

Evaluation Metrics

We compute:

Precision

Recall

F-Score

These metrics measure the alignment between generated alerts and true market events.

📌 Insert Screenshot: Plotly chart showing S&P 500 price, ground truth intervals, and social alerts.

Case Studies (2023)

Our pipeline successfully detected major market events:

Silicon Valley Bank Collapse (March 2023)

Sharp spike in Social Heat

Strong clustering structure

Detected before major market drawdown

AI Boom & Nvidia Rally (May–July 2023)

Technology-related clusters

Strong resonance between news and tweets

Captured market momentum shift

ARM IPO (September 2023)

Anticipation reflected in clustering

Immediate post-listing social reaction

Key Findings
HAC Dominance

Hierarchical Agglomerative Clustering consistently outperformed K-Means and other methods in generating semantically coherent clusters.

Recall over Precision

In a trading context, missing a crash (low Recall) is more costly than a false alert (low Precision).
Our model prioritizes Recall and successfully captures most ground truth events.

Social Hype Dynamics

The pipeline shows that:

News breaks instantly.

Social Heat sometimes anticipates or slightly lags price impact.

This temporal asymmetry may provide exploitable alpha signals.

Acknowledgments

Original Authors:
Carta, S., et al. (2021). Event Detection in Finance by Clustering News and Tweets.

Institution:
Université Paris 1 Panthéon-Sorbonne — Master 2 MOSEF (Quantitative Finance)
