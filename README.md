# Olfactory Fingerprints: NLP Models for Fragrance Profiling

This project evaluates and compares classical Natural Language Processing (NLP) techniques with advanced deep semantic models to map perfume similarities, cluster fragrances, and generate consumer recommendations. 

Using data extracted from **Fragrantica**, we explore a unique edge case in machine learning: how well do complex context-driven models (like BERT) handle non-standard text inputs consisting entirely of comma-separated ingredient lists (top, middle, and base notes) rather than fluent sentences with normal grammar.

## Project Overview

- **Dataset Source:** Fragrantica Perfumes Dataset (Kaggle)
- **Baseline Model:** Term Frequency-Inverse Document Frequency (TF-IDF) + K-Means Clustering
- **Semantic Model:** Sentence-BERT (`all-mpnet-base-v2`) + K-Means Clustering + UMAP Visualization
- **Core Finding:** TF-IDF aligns exceptionally well with rigid retail category standards (like commercial gender brackets). However, BERT generates qualitatively smarter recommendations by focusing on the overall olfactory character rather than superficial keyword matching.

---

## Technical Pipeline

The Jupyter notebook `Perfume_project.ipynb` is structured into five core stages:

### 1. Data Cleaning & Engagement Filtering
* Loads the raw catalog of 24,063 perfume items.
* Drops fragrances with low community interaction using a **75th percentile engagement threshold** (keeping only perfumes with $\ge 360$ consumer reviews).
* Reduces dataset sparsity down to a high-quality pool of **6,023 perfumes**.
* Combines individual top, middle, and base note layers into a single cohesive string column called `scent_profile`.

### 2. The Baseline Model (TF-IDF)
* Builds a sparse vector matrix counting unique fragrance notes across the dataset.
* Applies a classical $K$-Means clustering routine where the optimal number of groups ($K=5$) is determined using the Elbow Method.
* Identifies literal word boundaries across groups and surfaces keyword traits for specific scent profiles.

### 3. The Semantic Model (Sentence-BERT)
* Embeds the word lists into fixed, continuous 768-dimensional dense spaces using the sentence transformer `all-mpnet-base-v2`.
* Groups the resulting vectors using an identical $K$-Means framework ($K=5$) for a direct, unbiased comparison against TF-IDF.
* Compresses the 768 dimensions down into a clean 2D layout using **UMAP** for clear visualization.

### 4. Qualitative Testing (Case Studies)
* Queries both frameworks using iconic perfumes (*Shalimar*, *Sauvage*, and *Coco Mademoiselle*) to look closely at output differences.
* Evaluates how the models handle product line expansions ("flankers") and marketing biases (gender categories).

### 5. Quantitative Validation
* Calculates **Silhouette Scores** to assess internal cluster compactness and boundaries.
* Compares clusters against external categorical markers (Gender labels) using **Homogeneity, Completeness, and V-Measure** scores.

---

## Summary of Results

### Clustering Quality Metrics
| Metric | TF-IDF (LSA Fair Baseline) | BERT (`all-mpnet-base-v2`) | Performance Winner |
| :--- | :---: | :---: | :---: |
| **Silhouette Score** | 0.0363 | **0.0536** | **BERT** (Tighter, cleaner clusters) |
| **Gender V-Measure** | **0.1908** | 0.1468 | **TF-IDF** (Matches literal retail labels) |

### Key Experimental Discoveries
1. **The Short-Text Constraint:** The median input length is just 18 tokens. Because these profiles completely lack syntax, grammar, verbs, or structure, BERT's advanced internal context-attention mechanisms are partially deactivated. This allows a simple word-counting baseline to remain highly competitive.
2. **Flanker Blindness:** TF-IDF gets trapped by identical text tokens. For a query like *Sauvage*, its top suggestions are simply different versions of the exact same product because their names overlap. BERT bypasses this, evaluating the core ingredient combinations to suggest related, alternative brands.
3. **Gender Smoothing:** TF-IDF traces traditional retail categories tightly because it registers explicit, literal keywords (e.g., assigning lavender to men and sweet gourmands to women). BERT abstracts these keywords into a unified spatial architecture, focusing on the shared chemical profile rather than artificial commercial boundaries.

---

## Repository Visualizations

The execution of the notebook generates the following graphics for analysis:
* `short_text.png`: A histogram and cumulative distribution graph illustrating the dataset token constraints against transformer requirements.
* `reccomendation.png`: A horizontal bar chart contrasting the exact ingredient overlap percentages between target items and recommendations across both setups.

---

## Installation & Environment Setup

To clone this repository and run the notebook locally, configure your environment using the following steps:

```bash
# 1. Clone the repository
git clone [https://github.com/sali-ns/Text-Mining-and-Sentiment-Analysis.git](https://github.com/sali-ns/Text-Mining-and-Sentiment-Analysis.git)
cd Text-Mining-and-Sentiment-Analysis

# 2. Set up a virtual environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install required libraries
pip install notebook pandas numpy scikit-learn sentence-transformers umap-learn matplotlib seaborn
