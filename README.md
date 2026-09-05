# Olfactory Fingerprints: NLP Models for Fragrance Profiling

This project evaluates classical NLP techniques against advanced deep
semantic models to map perfume similarities, cluster fragrances, and
generate consumer recommendations.

Using data extracted from **Fragrantica**, we explore an edge case in
machine learning: how well do context-driven models (like BERT) handle
non-standard text inputs consisting entirely of comma-separated ingredient
lists (top, middle, and base notes) rather than fluent sentences with
normal grammar.

## Project Overview

- **Dataset Source:** Fragrantica Perfumes Dataset (Kaggle)
- **Baseline Model:** TF-IDF + K-Means Clustering
- **Semantic Model:** Sentence-BERT (`all-mpnet-base-v2`) + K-Means +
  UMAP Visualization
- **Core Finding:** TF-IDF aligns well with rigid retail category
  standards (like commercial gender brackets). BERT generates qualitatively
  smarter recommendations by focusing on overall olfactory character rather
  than superficial keyword matching.

---

## Technical Pipeline

The Jupyter notebook `source/Perfume_project.ipynb` is structured into five
core stages:

### 1. Data Cleaning & Engagement Filtering
* Loads the raw catalog of 24,063 perfume items.
* Drops fragrances with low community interaction using a **75th percentile
  engagement threshold** (keeping only perfumes with $\ge 360$ reviews).
* Reduces dataset sparsity to a high-quality pool of **6,023 perfumes**.
* Combines top, middle, and base note layers into a single `scent_profile`
  column.

### 2. The Baseline Model (TF-IDF)
* Builds a sparse vector matrix counting unique fragrance notes.
* Applies classical $K$-Means clustering where $K=5$ is chosen using the
  Elbow Method.
* Identifies literal word boundaries across groups and keyword traits for
  scent profiles.

### 3. The Semantic Model (Sentence-BERT)
* Embeds word lists into 768-dimensional dense spaces using the sentence
  transformer `all-mpnet-base-v2`.
* Groups vectors with identical $K$-Means ($K=5$) for an unbiased
  comparison against TF-IDF.
* Compresses 768 dimensions into a 2D layout using **UMAP**.

### 4. Qualitative Testing (Case Studies)
* Queries both frameworks with iconic perfumes (*Shalimar*, *Sauvage*,
  *Coco Mademoiselle*) to compare output differences.
* Evaluates handling of product line expansions ("flankers") and marketing
  biases (gender categories).

### 5. Quantitative Validation
* Calculates **Silhouette Scores** to assess cluster compactness.
* Compares clusters against external markers (Gender labels) using
  **Homogeneity, Completeness, and V-Measure** scores.

---

## Summary of Results

| Metric | TF-IDF (LSA Fair Baseline) | BERT (`all-mpnet-base-v2`) | Performance Winner |
| :--- | :---: | :---: | :---: |
| **Silhouette Score** | 0.0363 | **0.0536** | **BERT** (Tighter, cleaner clusters) |
| **Gender V-Measure** | **0.1908** | 0.1468 | **TF-IDF** (Matches literal retail labels) |

### Key Experimental Discoveries
1. **The Short-Text Constraint:** median input length is just 18 tokens.
   Lacking syntax and grammar, BERT's context-attention is partially
   deactivated, letting a simple word-counting baseline stay competitive.
2. **Flanker Blindness:** TF-IDF gets trapped by identical text tokens
   (e.g. *Sauvage* → other Sauvage flankers). BERT evaluates ingredient
   combinations and suggests related, alternative brands.
3. **Gender Smoothing:** TF-IDF tracks retail categories via literal
   keywords (lavender → men, sweet gourmands → women). BERT abstracts these
   into a unified space, focusing on shared chemical profiles.

---

## Repository Visualizations

The execution of the notebook generates these graphics:
* `short_text.png`: histogram and cumulative distribution of the dataset
  token constraints against transformer requirements.
* `reccomendation.png`: horizontal bar chart contrasting ingredient overlap
  percentages for recommendations across both setups.

---

## Repository Structure

```
.
├── data/
│   └── raw/
│       └── fra_cleaned.csv        # raw input dataset (24,063 perfumes)
├── source/
│   └── Perfume_project.ipynb      # main analysis notebook
├── output/
│   └── figures/                   # generated figures (saved on execution)
├── reports/
│   └── nlp-models-for-perfume-analysis.pdf
├── requirements.txt
└── README.md
```

---

## Installation & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/sali-ns/text-mining-and-sentiment-analysis.git
cd text-mining-and-sentiment-analysis

# 2. Set up a virtual environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install required libraries
pip install -r requirements.txt

# 4. Launch the notebook (from the repository root)
jupyter notebook source/Perfume_project.ipynb
```

The notebook reads its data from `data/raw/fra_cleaned.csv` (a relative
path), so run it from the repository root.