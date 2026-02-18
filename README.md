# Étapes 6 & 7 — Tweet Assignment + Alert Generation
## Maeva — Master 2 MOSEF 2024-2025 — Période SVB (Mars 2023)

---

## 🔴 CE QUE TU DOIS RÉCUPÉRER DE ROLAND

| Fichier Roland | Chemin chez Roland | Statut |
|---|---|---|
| `final_event_signatures.csv` | `data/for_models/` | ✅ Reçu (2 centroïdes × 300D) |
| `clean_news_week_SVB.csv` | `data/for_models/` | ✅ Reçu (22 articles, 2 clusters) |
| `daily_lexicons_filtered/` | `data/processed/` | ⚠️ **À télécharger (voir ci-dessous)** |
| `sp500_2023.csv` | `data/processed/` | ✅ Déjà disponible |
| GloVe Dolma 300d | `models/` | ⚠️ À télécharger (4 Go) |

### 📅 Quels lexiques télécharger ?

**Télécharge les lexiques du 3 février au 17 mars 2023** (43 fichiers).

Pourquoi : le papier utilise une fenêtre glissante de 28 jours pour la génération
du lexique. Donc pour la période d'analyse (3-9 mars), les lexiques sont construits
à partir d'articles des 28 jours précédents. Pour être safe et couvrir aussi la
période post-SVB (10-17 mars), il faut :
- `lexicon_filtered_2023-02-03.csv` → `lexicon_filtered_2023-03-17.csv`

**En pratique :** sélectionne les fichiers de février et mars dans le dossier de Roland.

---

## 📂 Structure à respecter

```
project/
├── notebooks/
│   ├── 6_tweet_assignment.ipynb    ← Étape 6
│   └── 7_alert_generation.ipynb    ← Étape 7
├── src/
│   ├── __init__.py
│   ├── tweet_preprocessing.py      ← Nettoyage + embedding tweets
│   └── tweet_assignment.py         ← Assignment + alertes + évaluation
├── models/
│   └── dolma_300_2024_1.2M.100_combined.txt  ← GloVe (4 Go)
├── data/
│   ├── processed/
│   │   ├── tweets_2023.csv
│   │   ├── sp500_2023.csv
│   │   └── daily_lexicons_filtered/    ← 43 fichiers CSV de Roland
│   │       ├── lexicon_filtered_2023-02-03.csv
│   │       ├── ...
│   │       └── lexicon_filtered_2023-03-17.csv
│   └── for_models/
│       ├── final_event_signatures.csv  ← Centroïdes (Roland)
│       └── clean_news_week_SVB.csv     ← Articles clustérisés (Roland)
└── outputs/                            ← Généré automatiquement
    ├── tweet_similarity_distribution.png
    ├── daily_assignment_ratio.png
    ├── alert_generation.png
    ├── precision_recall_fscore.png
    └── sp500_ground_truth.png
```

---

## ▶️ Exécution

### Étape 1 : Placer les fichiers
1. Mettre les fichiers de Roland aux bons endroits (voir ci-dessus)
2. Mettre le modèle GloVe dans `models/`

### Étape 2 : Notebook 6 (`6_tweet_assignment.ipynb`)
- Charge les centroïdes de `final_event_signatures.csv` → dict {0: vec300d, 1: vec300d}
- Charge les tweets et filtre pour la période 3-17 mars 2023
- Nettoie les tweets (RT, URLs, mentions, cashtags)
- Tokenise avec spaCy (même pipeline que Roland)
- Filtre par le lexique union de la période
- Embed chaque tweet via GloVe (même modèle que les news → espace partagé)
- Assigne chaque tweet au cluster le plus proche (seuil cosine = 0.5)
- **Produit** : `daily_assignment_ratios.csv`

### Étape 3 : Notebook 7 (`7_alert_generation.ipynb`)
- Charge les ratios quotidiens + prix S&P 500
- Construit la ground truth (variation hebdo > 2%)
- Teste 11 seuils d'alerte (1% à 40%)
- Calcule Precision / Recall / F-score pour chaque
- **Produit** : graphiques + `evaluation_results.csv`

---

## 📐 Paramètres du papier respectés

| Paramètre | Valeur | Référence papier |
|---|---|---|
| Distance cosinus seuil | 0.5 | §8 Tweet Assignment |
| Alert thresholds | 1%-40% | §10 (papier teste 1-5%) |
| Ground truth δ | 2% hebdo | §10.2 |
| Gap tolerance | 3 jours | §10.2 |
| Centroïde | Médiane (300D) | §5.3 |
| Embedding | GloVe Dolma 300d = même que news | §4 + §8 |
