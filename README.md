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

---
Methodology & Pipeline StepsOur pipeline follows the 7 steps detailed by Carta et al. Below is the breakdown of each phase applied to our 2023 dataset.Step 1: Lexicon GenerationTo filter out noise and focus on financially meaningful text, we generated a domain-specific lexicon. We computed a Document-Term Matrix (DTM) and used Marginal Screening to select words with high predictive correlation to market movements, avoiding standard stop-words.Insert Screenshot: Word cloud or Lexicon distribution chart here.Step 2: Feature Engineering (Embeddings)We transformed raw text into dense mathematical vectors using Word2Vec.Texts were tokenized.Words outside the generated Lexicon were discarded.The remaining valid Word2Vec vectors were averaged to create a single Document Embedding for each news article.Step 3: News ClusteringWe tested four clustering algorithms: K-Means, Agglomerative Clustering (HAC), DBSCAN, and Gaussian Mixture Models (GMM). We optimized the number of clusters (k) by maximizing the Silhouette Score. Consistent with our report, HAC yielded the best results, forming dense and coherent groups of news.Insert Screenshot: Silhouette Score maximization chart or t-SNE plot (Figure 13a).Step 4: Relevant Words Extraction & Outlier RemovalNot all news clusters represent financial events. We computed the Average TF-IDF for each cluster to extract its top relevant words. Clusters lacking strong financial terminology were classified as "Outliers" and discarded.Insert Screenshot: Bar charts of Key Financial Terms (Figure 14).Step 5: Event SignaturesFor the remaining valid clusters, we computed the Centroid (the mathematical average of all document vectors in that cluster). This centroid becomes the "Signature" of the event, acting as a gravitational pull for the upcoming social media analysis.Insert Screenshot: Cleaned Clusters with Centroids (Figure 13b).Step 6: Tweet AssignmentTo gauge public attention, we linked social media data to the professional news.De-duplication: Removed spam and bot-generated identical tweets.Cosine Similarity: Measured the distance between a tweet's embedding and the Event Signatures.Threshold (delta): Tweets with a similarity score above the delta threshold were assigned to the event.Insert Screenshot: Distribution of Tweet Assignments (Figure 15).Step 7: Alert Generation & EvaluationAn alert is generated if the "Social Heat" (the ratio of assigned tweets to the total daily tweets) exceeds a specific threshold.To evaluate the model's accuracy, we calculated a Ground Truth based on the S&P 500 weekly variation:$$ \Delta_d = \frac{|close(d+7) - close(d)|}{close(d)} $$Days with $\Delta_d > 0.02$ were marked as event intervals. We then computed Precision, Recall, and F-Score.Insert Screenshot: Plotly chart showing the S&P 500 price with Ground Truth zones and social alerts.Case Studies (2023)Our pipeline successfully detected major market shifts in 2023:The Silicon Valley Bank (SVB) Collapse (March 2023): Detected a massive spike in Social Heat just prior to the heavy market drawdown.The AI Boom & Nvidia (May 2023): Captured the technological hype translating into market momentum.ARM IPO (September 2023): Tracked the anticipation and immediate aftermath of a major tech listing.Key FindingsHAC Dominance: Hierarchical Agglomerative Clustering outperformed K-means and DBSCAN in creating semantically meaningful clusters.Recall over Precision: In a trading context, missing a crash (low recall) is worse than a false alarm (low precision). Our algorithm successfully captures almost all ground truth events.Social Hype Delay: The pipeline demonstrated that while news breaks instantly, the "Social Heat" sometimes anticipates or slightly lags behind the price impact, providing valuable alpha.AcknowledgmentsOriginal Authors: Carta, S., et al. (2021). Event Detection in Finance by Clustering News and Tweets.Institution: Université Paris 1 Panthéon-Sorbonne (Master MOSEF).
