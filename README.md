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
 ┣ 📂 data/                 # Data directory (ignored in git if too large)
 ┃ ┣ sp500_2023.csv      # Ground truth S&P 500 price data
 ┃ ┣ news_2023.csv       # Financial news dataset
 ┃ ┗ tweets_2023.csv     # Social media datasets (Twitter/StockTwits)
 ┣ 📂 notebooks/            # Jupyter Notebooks for step-by-step exploration
 ┃ ┣ 📜 01_Lexicon_Gen.ipynb
 ┃ ┣ 📜 02_Clustering.ipynb
 ┃ ┗ 📜 03_Alert_Eval.ipynb
 ┣ 📂 src/                  # Python source code modules
 ┃ ┣ 📜 preprocessing.py    # Text cleaning and Tokenization
 ┃ ┣ 📜 clustering.py       # K-Means, HAC, DBSCAN, GMM wrappers
 ┃ ┣ 📜 assignment.py       # Cosine similarity and threshold filtering
 ┃ ┗ 📜 metrics.py          # Recall, Precision, F-Score calculations
 ┣ 📂 output/               # Generated graphs, charts, and summary tables
 ┃ ┣ 📜 table_3_tweet_assignment.csv
 ┃ ┗ 🖼️ *.png               # (Saved plots for the README)
 ┣ 📜 .gitignore            # Excludes data/, .venv/, and __pycache__/
 ┣ 📜 pyproject.toml        # Dependencies and project metadata (used by uv)
 ┗ 📜 README.md             # This documentation
