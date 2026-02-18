# Étapes 6 & 7 — Tweet Assignment + Alert Generation
## Projet Event Detection — Carta et al. (2021)
### Maeva & Roland — Master 2 MOSEF 2024-2025

---

## 📂 Architecture du projet

```
task67/
│
├── notebooks/                          ← NOTEBOOKS JUPYTER (exécutés)
│   ├── 6_tweet_assignment.ipynb        ← Étape 6 : Nettoyage + embedding + assignation tweets
│   └── 7_alert_generation.ipynb        ← Étape 7 : Ground truth + alertes + évaluation P/R/F
│
├── src/                                ← MODULES PYTHON
│   ├── __init__.py
│   ├── tweet_preprocessing.py          ← Fonctions de nettoyage et embedding des tweets
│   │                                      • clean_tweet() : suppression RT, URLs, mentions, cashtags
│   │                                      • preprocess_tweet_spacy() : tokenisation spaCy
│   │                                      • compute_tweet_embedding() : GloVe 300d (même que news)
│   │                                      • load_lexicons_for_period() : charge les lexiques par dates
│   │                                      • run_tweet_embedding_pipeline() : pipeline complet
│   └── tweet_assignment.py             ← Fonctions d'assignation et d'évaluation
│                                          • assign_tweets_to_clusters() : cosine similarity → seuil 0.5
│                                          • compute_daily_assignment_ratio() : R(d) quotidien
│                                          • generate_alerts() : R(d) > θ → alerte
│                                          • build_ground_truth() : variation hebdo S&P 500 > 2%
│                                          • evaluate_alerts() : Precision, Recall, F-score
│
├── data/
│   ├── processed/                      ← DONNÉES TRAITÉES
│   │   ├── tweets_2023.csv             ← 2,243 tweets financiers (Maeva — Kaggle + VADER)
│   │   ├── sp500_2023.csv              ← 271 jours de trading S&P 500 (Roland — Yahoo Finance)
│   │   └── lexicons_filtered/          ← 43 lexiques quotidiens filtrés P20/P80 (Roland)
│   │       ├── lexicon_filtered_2023-02-03.csv    ← Début : 28j avant la période d'analyse
│   │       ├── ...                                 (26 fichiers février)
│   │       ├── lexicon_filtered_2023-03-03.csv    ← Début période SVB
│   │       ├── ...                                 (17 fichiers mars)
│   │       └── lexicon_filtered_2023-03-17.csv    ← Fin période d'analyse
│   │
│   └── for_models/                     ← DONNÉES POUR LES MODÈLES
│       ├── final_event_signatures.csv  ← Centroïdes des 2 clusters (Roland — médiane, 300D)
│       ├── clean_news_week_SVB.csv     ← 22 articles clustérisés du 3-9 mars (Roland)
│       └── output/                     ← RÉSULTATS INTERMÉDIAIRES (Maeva)
│           ├── tweets_assigned.csv     ← 550 tweets avec cluster assigné + similarité
│           └── daily_assignment_ratios.csv  ← Ratio quotidien d'assignation (10 jours)
│
├── outputs/                            ← RÉSULTATS FINAUX (graphiques + tableaux)
│   ├── tweet_similarity_distribution.png  ← Distribution des similarités cosinus
│   ├── daily_assignment_ratio.png         ← % tweets assignés par jour + date SVB
│   ├── sp500_ground_truth.png             ← S&P 500 avec event days en rouge
│   ├── alert_generation.png               ← Alertes vs événements
│   ├── precision_recall_fscore.png        ← Courbes P/R/F vs seuil
│   ├── evaluation_results.csv             ← Métriques pour chaque seuil (1%-40%)
│   ├── comparison_table.csv               ← Tableau comparatif avec Carta et al.
│   └── alerts_best_threshold.csv          ← Alertes au meilleur seuil
│
└── requirements.txt                    ← Dépendances Python
```

---

## 📊 Résumé des résultats

### Étape 6 — Tweet Assignment

| Métrique | Valeur |
|----------|--------|
| Tweets analysés (période 3-17 mars) | 163 après nettoyage |
| Tweets avec embedding valide | ~550 (après dédoublonnage + spaCy) |
| Taux d'assignation (seuil 0.5) | **100%** |
| Similarité moyenne | ~0.90 |
| Clusters | 2 (cluster 0 : 3 articles, cluster 1 : 19 articles) |

### Étape 7 — Alert Generation

| Seuil | Alertes | Precision | Recall | F-score |
|-------|---------|-----------|--------|---------|
| 1% à 40% | 10/10 jours | 80% | 5.6% | 10.4% |

Les métriques sont **identiques pour tous les seuils** car le ratio d'assignation est à 100% chaque jour.

### Comparaison avec le papier

| Métrique | Carta et al. (2021) | Notre projet |
|----------|-------------------|--------------|
| Articles clustérisés | 8,403 | 22 |
| Tweets | 283,473 | 163 |
| Période | 4 ans (2016-2020) | 10 jours (mars 2023) |
| Recall (seuil 3%) | ~70% | 6% |
| Precision (seuil 3%) | ~55% | 80% |
| F-score (seuil 3%) | ~60% | 10% |
| Nb événements GT | ~25 | 18 |
| % event days | ~15% | 38% |

---

## 🔍 Analyse critique — Pourquoi les résultats diffèrent du papier

### 1. Taux d'assignation à 100% : pourquoi ?

Le papier obtient un taux d'assignation de **15-30%** car ses 283K tweets Stocktwits sont très variés (spam, hors-sujet, langage informel). Le filtre cosinus à 0.5 élimine naturellement les tweets non pertinents.

Dans notre cas, les **193 tweets** sont déjà pré-filtrés sur les cashtags S&P 500 ($SPY, $SPX). Après filtrage par le lexique (mots à impact marché uniquement) et embedding GloVe, leur vocabulaire est tellement concentré sur le domaine financier que **tous** convergent vers les centroïdes avec une similarité > 0.5 (la plupart entre 0.85 et 0.98).

→ **Le filtre de similarité ne discrimine plus rien.**

### 2. Precision élevée (80%) mais Recall faible (6%)

- **Precision = 80%** : 8 des 10 jours d'alerte tombent dans un event day. C'est élevé car 2023 a été volatile (38% d'event days vs 15% dans le papier).
- **Recall = 6%** : nos 10 jours d'alerte ne couvrent que 1 événement sur 18. Normal : on n'a des tweets que sur 10 jours (3-17 mars), mais la ground truth couvre toute l'année 2023 (18 événements).

### 3. Ce qui est respecté vs ce qui est limité

**✅ Méthodologie respectée fidèlement :**
- Nettoyage tweets (RT, URLs, mentions, cashtags) — papier §8
- Même espace sémantique (GloVe Dolma 300d) pour news et tweets — papier §4 + §8
- Filtrage par le lexique quotidien (P20/P80) — papier §3
- Assignation par cosine similarity, seuil 0.5 — papier §8
- Ground truth : variation hebdo S&P 500 > 2%, gap tolerance 3j — papier §10.2
- Évaluation : Precision / Recall / F-score — papier §10.3
- Test multi-seuil (1% à 40%) — papier §10

**⚠️ Limites liées aux données :**
- 163 tweets vs 283K dans le papier (×1,700 de moins)
- 22 articles sur 1 semaine vs 8,403 sur 4 ans
- 2 clusters fixes vs k optimal par silhouette (2-10)
- Tweets pré-filtrés sur le domaine → seuil de similarité inefficace
- Ground truth annuelle vs alertes sur 10 jours seulement
---

## ▶️ Comment reproduire

1. Installer les dépendances : `pip install -r requirements.txt`
2. Installer spaCy : `python -m spacy download en_core_web_sm`
3. Placer le modèle GloVe dans `models/` (non inclus, 4 Go)
4. Exécuter `notebooks/6_tweet_assignment.ipynb`
5. Exécuter `notebooks/7_alert_generation.ipynb`

---

## 📌 Fichiers de Roland utilisés

| Fichier | Description | Produit par |
|---------|------------|-------------|
| `final_event_signatures.csv` | 2 centroïdes × 300D (médiane des clusters) | Roland — Étape 3-4 |
| `clean_news_week_SVB.csv` | 22 articles du 3-9 mars avec labels Cluster (0 ou 1) | Roland — Étape 3 |
| `lexicons_filtered/` | 43 lexiques quotidiens (3 fév → 17 mars) | Roland — Étape 1 |
| `sp500_2023.csv` | Prix S&P 500, 271 jours de trading | Roland — Étape 0 |
