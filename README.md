# Meme Context Clustering: NLP-Based Approach to Thematic Grouping of Internet Memes

---

## Overview

Memes look chaotic. They're not.

Beneath the humor, irony, and internet slang lies a surprisingly consistent taxonomy  - patterns of genre, tone, cultural reference, and emotional register that repeat across thousands of posts. This project sets out to **reverse-engineer that hidden structure** using a fully interpretable NLP pipeline, built from scratch with standard Python libraries.

> *"Meme culture is not noise - it's compressed cultural signal. Our job is to decompress it."*

The goal: cluster `5,819` meme descriptions into meaningful, human-interpretable thematic groups - using **no external datasets, no pre-trained models, and no AutoML**.

---

## Dataset Description

The competition dataset comprises 5,819 meme records, each described by three fields.

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Unique identifier for each meme |
| `input` | String | Textual description of the meme → includes caption, scene context, poster intent, and entity mapping |
| `url` | String | Source image link, used as a secondary contextual signal |


The primary analytical signal is the `input` column. Each entry follows an implicit structure of the form: **caption → scene description → poster intent → entity mapping**  providing multi-layered semantic information within a single text field. The `url` column was used supplementarily to extract domain-level metadata that could hint at community context.

---

## Technical Pipeline (Methodology)

###  Pipeline Overview (TL;DR)

The end-to-end pipeline follows a sequential transformation from raw text to labelled clusters:

| Stage | Method | Purpose |
|-------|--------|---------|
| Preprocessing | Rule-based text cleaning | Remove noise, normalize vocabulary |
| Vectorization | TF-IDF (unigrams + bigrams) | Represent term importance relative to corpus |
| Dimensionality Reduction | LSA via TruncatedSVD | Capture latent semantic relationships |
| Normalization | L2 normalization | Ensure scale invariance across vectors |
| Clustering | K-Means (K=7) | Partition memes into thematic groups |
| Validation | Agglomerative Clustering + ARI | Verify cluster stability |
| Interpretation | Per-cluster TF-IDF + manual review | Assign human-readable cluster labels |

---

### a) Text Preprocessing

Raw text descriptions were cleaned through the following sequential steps:

| Step | Operation |
|------|-----------|
| 1 | Stripped the structural `TEXT:` prefix present in every `input` entry |
| 2 | Converted all characters to lowercase |
| 3 | Removed embedded URLs, punctuation, and numeric tokens |
| 4 | Collapsed excess whitespace |
| 5 | Applied a curated list of approximately 60 English stopwords |
| 6 | Applied lightweight suffix-stripping stemming (`-ing`, `-tion`, `-ness`, `-ment`, `-er`, `-ed`, `-ly`, `-es`, `-s`) |
| 7 | Enforced a minimum word length of 3 characters |

### b) Feature Engineering

Two complementary representations were constructed from the cleaned text.

**TF-IDF Vectorization**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 2,000 | Limits vocabulary to high-signal terms |
| `ngram_range` | (1, 2) | Captures unigrams and bigrams for multi-word concepts |
| `min_df` | 2 | Excludes ultra-rare terms unlikely to generalize |
| `max_df` | 0.85 | Down-weights near-universal boilerplate terms |
| `sublinear_tf` | True | Applies log-scaling to term frequency |

Bigrams preserve meaningful multi-word concepts like `"meme poster"`, `"trying convey"`, and `"not related"` that carry structural meaning a unigram model would miss.

**Latent Semantic Analysis (LSA)**

LSA projects the **2,000-dimensional sparse TF-IDF space** into a **dense 50-dimensional semantic space**, capturing latent co-occurrence patterns that pure term matching misses entirely.

| Technique | Input Dimensions | Output Dimensions | Purpose |
|-----------|-----------------|------------------|---------|
| TF-IDF | Raw text | 2,000 (sparse) | Term importance |
| LSA / TruncatedSVD | 2,000 | 50 (dense) | Semantic compression |
| L2 Normalization | 50 | 50 (unit vectors) | Scale invariance |

### c) Optimal Cluster Selection

The optimal number of clusters was determined empirically using two independent metrics evaluated across K = 4 to K = 14:

- **Elbow Method** → Inertia flattens at the inflection point
- **Silhouette Score** → Peaks at the optimal cluster count

**→ K = 7 selected**, yielding clusters large enough for statistical stability and specific enough for thematic coherence.

### d) Clustering

**Primary Method → K-Means**

K-Means clustering was applied to the L2-normalized LSA vectors with K = 7, using 20 random initializations to reduce sensitivity to centroid initialization.

**Validation Method → Agglomerative Clustering (Ward Linkage)**

| Method | Type | Linkage / Init |
|--------|------|----------------|
| K-Means | Centroid-based | n_init = 20, seed = 42 |
| Agglomerative | Hierarchical | Ward linkage |

Using two fundamentally different algorithms (*centroid-based* vs *hierarchical*) and measuring their agreement via **Adjusted Rand Index (ARI)** provides a strong statistical argument that the discovered clusters are **real structure**, not noise.

### e) Visualization

t-SNE was applied to the 50-dimensional LSA vectors to produce a 2-dimensional projection for visual inspection. The resulting scatter plot, saved as `cluster_viz.png`, confirms meaningful spatial separation between cluster regions, corroborating the quantitative validation results.

<img width="600" height="500" alt="cluster_viz" src="https://github.com/user-attachments/assets/2ca5c29c-6fca-433a-9ad9-4623d14e3719" />


### f) Cluster Interpretation

Cluster labels were assigned through a two-step interpretive process:

1. **Keyword extraction** - A per-cluster TF-IDF analysis was run to identify the top 10 most distinctive terms for each group
2. **Manual review** - A minimum of 15 meme entries per cluster were manually read to validate that keyword-derived labels accurately reflected the full range of content within each group

---

## 3. Results (Cluster Summary)

Seven thematic clusters were identified, each with a human-assigned label grounded in keyword analysis and manual content review.

| Cluster | Label | Representative Keywords | Count |
|---------|-------|------------------------|-------|
| 1 | **Relatable Everyday Struggles** | `woman`, `man`, `trying`, `convey`, `look`, `hold`, `right` | 292 |
| 2 | **Relationship & Social Dynamics** | relationship cues, social situations, interpersonal dynamics | 351 |
| 3 | **Surreal & Absurdist Humor** | bizarre imagery, non-sequitur logic, chaotic scenarios | 1908 |
| 4 | **Gaming & Tech Culture** | `game`, `pc`, `console`, `download`, `update` | 1035 |
| 5 | **Political & Social Commentary** | `society`, `government`, `commentary`, `news` | 675 |
| 6 | **Pop Culture & Entertainment** | `movie`, `character`, `scene`, `reference` | 181 |
| 7 | **Animals & Pet Behavior** | `cat`, `dog`, `pet`, `animal` | 1376 |

All cluster labels emerged organically from keyword convergence and content inspection (no labels were pre-defined)

---

## 4. Challenges

| Challenge | Description | Mitigation Strategy |
|-----------|-------------|-------------------|
| **Short Text Lengths** | Many descriptions contain fewer than 15 words, limiting per-meme signal | Bigrams and LSA leverage corpus-wide co-occurrence to compensate |
| **Informal & Slang Language** | Slang, abbreviations, and deliberate misspellings are prevalent | Preprocessing was calibrated to preserve meaningful informal terms |
| **Boilerplate Text patterns** | Repeated structural phrases appear across hundreds of entries | `max_df = 0.85` in TF-IDF down-weights near-universal terms automatically |
| **Cluster Boundary ambiguity** | Some memes are thematically multi-class | Acknowledged as a hard clustering limitation; soft clustering proposed as future work |
| **Absence of Word Embeddings** | Competition rules prohibit pre-trained models | LSA partially compensates by capturing latent semantic relationships across the full vocabulary |

---

## 5. Tools & Libraries

| Category | Library / Tool |
|----------|---------------|
| Language | Python 3.12 |
| Data handling | Pandas, NumPy |
| Vectorization | `TfidfVectorizer`, `CountVectorizer` *(Scikit-learn)* |
| Dimensionality reduction | `TruncatedSVD` *(Scikit-learn)* |
| Clustering | `KMeans`, `AgglomerativeClustering` *(Scikit-learn)* |
| Evaluation | `silhouette_score`, `adjusted_rand_score` *(Scikit-learn)* |
| Visualization | `TSNE` *(Scikit-learn)*, Matplotlib, Seaborn |
| Serialization | Pickle |
| Environment | Jupyter Notebook |

---

## 6. Reproducibility

Every component of this pipeline is fully reproducible:

- All random seeds fixed to `42`
- Notebook executes **top-to-bottom without errors**
- Model artifacts saved as `.pkl` for independent inspection
- No external data downloads or API calls required
- `output.txt` generated **programmatically** from the notebook
---

## 7. Deliverables

| File | Description |
|------|-------------|
| `meme_clustering.ipynb` | Full reproducible pipeline notebook |
| `output.txt` | Cluster listings + per-meme reasoning + methodology + pitch |
| `report.txt` | Approach report |
| `vectorizer.pkl` | Trained TF-IDF vectorizer |
| `cluster_model.pkl` | Trained K-Means model |

---

## 8. Future Directions

The following improvements are proposed conceptually. Here's what an unconstrained v2.0 could look like:

#### **Multimodal Analysis** 
The `url` column points to actual meme images. A future system could combine **OCR** (to extract overlaid text) with **image classifiers** (to identify visual meme templates like the Drake format or Distracted Boyfriend). Visual + textual fusion would dramatically improve cluster fidelity.

#### **Sentiment-Aware Clustering**
An emotional layer (`positive` / `negative` / `sarcastic` / `dark`) would split clusters that share a theme but differ in tone — separating *"motivational memes"* from *"motivational sarcasm"*, for example.

#### **Hierarchical Taxonomy**
A two-level system would mirror how humans naturally categorize memes:
```
Humor    →  [Dark Humor, Self-deprecating, Absurdist]
Culture  →  [Gaming, Anime, Sports, Politics]
Emotion  →  [Wholesome, Relatable, Rage-bait]
```

#### **Temporal Trend Detection**
With timestamps, tracking cluster growth and decay over time could reveal meme lifecycle patterns and predict emerging formats before they go viral.

#### **Community-Context Features**
URL domains (Reddit subreddits, Know Your Meme, Imgur) encode deep community context. Graph-based clustering that weights memes from the same origin community could significantly improve thematic coherence.

#### **Human-in-the-Loop Refinement**
A semi-supervised approach where domain experts label a seed set of **50–100 memes** could bootstrap constrained K-Means or label propagation — combining machine scale with human judgment.

#### **Real-World Applications**
- **Content Moderation** — Automatically flag offensive meme clusters for human review
- **Recommendation Engines** — Suggest memes from contextually related clusters
- **Cultural Analytics** — Track the evolution of internet humor across communities and time

---
