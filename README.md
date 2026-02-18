# Financial-Events-clustering-news-tweets
Ce travail académique, basé sur un article de recherche, a pour but transformer le flux désordonné des actualités et des réseaux sociaux en signaux exploitables pour la finance. L'idée centrale est de compenser les faiblesses d'une source par les forces d'une autre afin de détecter au mieux les évenenments financiers majeurs.


## Project Structure

```text
PROJET_ROLAND_MAEVA/
├── data/                       # Dataset storage
│   ├── for_models/             # Final inputs for clustering & analysis
│   │   ├── clean_news_week_SVB.csv
│   │   ├── final_event_signatures.csv
│   │   └── news_features.csv
│   ├── processed/              # Cleaned and engineered data
│   │   ├── daily_dtm/          # Daily Document-Term Matrices
│   │   ├── daily_lexicons/     # Filtered and full daily lexicons
│   │   ├── news_2023_clean.csv
│   │   ├── sp500_2023.csv
│   │   └── tweets_2023.csv
│   └── raw/                    # Raw source files (BigQuery results, etc.)
│       └── financial_tweets.csv
├── docs/                       # Project documentation and papers
├── img/                        # Visualizations and exported charts
│   ├── 1_lexicon_generation/
│   ├── 2_feature_engineering/
│   ├── 3_news_clustering/
│   ├── 4_outlier_removal/
│   └── 5_relevant_words_extraction/
├── models/                     # Pre-trained models (Word2Vec/Dolma)
├── notebooks/                  # Interactive Jupyter notebooks (Experiments)
│   ├── 0_extract_data.ipynb
│   ├── 1_lexicon_generation.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_news_clustering.ipynb
│   ├── 4_outlier_removal.ipynb
│   └── 5_relevant_words_extraction.ipynb
├── src/                        # Production-ready source code (.py)
│   ├── extract_data.py
│   ├── feature_engineering.py
│   ├── lexicon_generation.py
│   ├── news_clustering.py
│   ├── outlier_removal.py
│   └── relevant_words_extraction.py
├── .gitignore                  # Git ignore rules
├── .python-version             # Python environment version
├── LICENSE                     # Project license
├── pyproject.toml              # Project dependencies and configuration
├── README.md                   # Project overview
└── uv.lock                     # Lockfile for dependency management (uv)
