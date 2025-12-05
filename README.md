# NLP-Metacritics: Predicting Game Quality from User Reviews

> **Course**: IST 322 – Natural Language Processing  
> **Repo**: `chen-zhu/NLP-Metacritics`  
> **Main Goal**: Predict whether a video game has a **high or low critic Metascore** using **only player-written Metacritic English reviews**.

---

## Notebooks

- **Main project Colab**  
  https://colab.research.google.com/drive/16pO0hsVSHEsIS4Cp8RBm8cdggkm0FrRA  

- **Fuzzed game-name robustness Colab**  
  https://colab.research.google.com/drive/1_KF4DyKw3LC021H6fTjVVvioL2vtJUNZ?usp=sharing  

(These notebooks are also exported into this repo as `.ipynb` files.)

---

## 1. Project Overview

The video game industry generates **>$180B annually** and has major impact on entertainment, cognition, and social interaction. Platforms like **Metacritic** collect large numbers of **user reviews** and **critic scores** for each game.

This project explores whether **text alone** from user reviews is sufficient to predict how well a game is evaluated by **professional critics**.

### Research Question

> **Can we predict whether a game has a relatively high or low critic Metascore solely from the text of player-written English reviews?**

We focus on critic metascore because user scores are very sparse in the dataset.

### Main Contributions

- Built a **10,488-document corpus** of Metacritic user reviews across **46 games**.
- Created a full **NLP pipeline**: scraping → preprocessing → embeddings → sentiment → topic modeling → supervised learning.
- Designed a **hybrid model** combining TF-IDF, sentiment, and review length that achieves **F1 ≈ 0.87, AUC ≈ 0.94**.
- Ran a **robustness check** by **fuzzing game names** (replacing titles with “the game”) to ensure the model is not just memorizing titles.

---

## 2. Data

### 2.1 Source

- **Site**: [Metacritic](https://www.metacritic.com/game/)  
- **Unit**: individual **user review** for a game  
- **Fields scraped**:
  - `game` – URL-style identifier (e.g., `/game/football-manager-26/`)
  - `avg_user_score` – average user rating for the game
  - `meta_score` – critic metascore
  - `user_score` – numeric rating from that user (often missing)
  - `review` – free-text review body

Scraping was done with **`requests` + `BeautifulSoup`**, using delays and a browser-like `User-Agent` and storing results as a CSV.

### 2.2 Language Filtering

- Detected language using **`langdetect`**.
- Kept **English-only** reviews:
  - **10,488** documents
  - **46** unique games
- Average raw length: **79.15 words** (min ~1–2, max 927).

### 2.3 Final Metadata Schema

For each English review we store:

- `game`, `avg_user_score`, `meta_score`, `user_score`
- `review` (raw text)
- `lang`
- Preprocessed fields (e.g., `review_expanded`, `clean_text`, embeddings, sentiment scores, etc.)

---

## 3. Methods

### 3.1 Text Preprocessing

Implemented in spaCy-based pipeline:

1. **Contraction expansion** with `contractions`  
   - `"can't" → "cannot"`, `"I'm" → "I am"`.
2. **spaCy tokenization** (`en_core_web_sm`)
3. **Cleaning & normalization**
   - lower-casing
   - remove URLs / HTML
   - remove non-alphabetic tokens & tokens with digits
   - remove English stopwords
   - drop tokens with length < 3
4. **POS tagging** and **lemmatization** (spaCy)
5. Join lemmas into `clean_text` for modeling.

**Summary stats (cleaned):**

- Avg. lemma count: **≈35 tokens / review**
- Unique lemmas: **14,787**  
- Lexical diversity: **≈0.89**

### 3.2 Word Embeddings

Using **Gensim** pre-trained models with **average pooling** per review:

- **GloVe**: `glove-wiki-gigaword-50` (50-D)
- **Word2Vec**: Google News (300-D)
- **FastText**: `wiki-news-subwords-300` (300-D)

Coverage for all three models: **≈99.94%** of reviews had at least one in-vocabulary token.

PCA & t-SNE visualizations show linguistically homogeneous critic text with no clear separation by score in 2D; therefore prediction relies on the full high-dimensional space.

### 3.3 Sentiment Analysis

Computed on `review_expanded`:

- **TextBlob**:
  - `polarity` (−1 to +1)
  - `subjectivity` (0 to 1)
- **VADER**:
  - `compound` (−1 to +1)

Key findings:

- Sentiment is **strongly skewed positive** (players review games they like).
- Correlations with scores are **weak but positive**  
  - vs. metascore: polarity ≈ 0.12, compound ≈ 0.10  
  - vs. user score: polarity ≈ 0.21, compound ≈ 0.24

### 3.4 Topic Modeling

We used both **LSA (SVD)** and **LDA**.

- **LSA** on BOW and TF-IDF (Gensim)
  - Best coherence around **k = 6 topics**.
  - Semantic axes: enjoyment/gameplay, combat/difficulty, technical quality, narrative/atmosphere.
- **LDA** on BOW
  - Best coherence with **k = 4 topics** (≈0.41).
  - Themes:  
    1. Gameplay & enjoyment  
    2. Negative experiences & technical issues  
    3. Visuals/art direction/genre  
    4. Combat & mechanics

A **hybrid interpretation** (LSA + LDA + manual reading) yields four clear human-labeled topics: gameplay feelings, technical problems, visuals/atmosphere, and combat/difficulty.

### 3.5 Supervised Learning

We frame prediction as **binary classification**:

- Target: `target_high_meta = 1` if critic `meta_score` ≥ 85, else 0  
  (84.4% high vs 15.6% low – still non-trivial but skewed).

**Train/test scheme**

- 80/20 train–test split, **stratified**.
- **3-fold CV** inside `GridSearchCV`.
- Metrics: **Accuracy, F1 (primary), AUC (ROC)**.
- Models evaluated for each feature set:
  - Logistic Regression
  - SVM (RBF)
  - Random Forest

**Features**

1. **TF-IDF** on `review_expanded`
2. **Embeddings** (Word2Vec, FastText, GloVe document vectors)
3. **Knowledge-driven**: TextBlob/VADER scores + review length
4. **Hybrid**: TF-IDF + sentiment + length

---

## 4. Results (Original Dataset)

### 4.1 TF-IDF Baseline

- **TF-IDF + Random Forest**
  - Accuracy ≈ **0.93**
  - F1 ≈ **0.96**
  - AUC ≈ **0.91**

### 4.2 Embedding-Based Models (Best RF models)

- **Word2Vec + RF** – F1 ≈ **0.83**, AUC ≈ **0.87**  
- **FastText + RF** – F1 ≈ **0.82**, AUC ≈ **0.86**  
- **GloVe + RF** – F1 ≈ **0.79**, AUC ≈ **0.83**

### 4.3 Knowledge-Driven (Sentiment + Length)

- **RF (sentiment + length)** – Accuracy ≈ **0.62**, F1 ≈ **0.71**, AUC ≈ **0.66**  
  → Helpful but weaker than text-based features.

### 4.4 Hybrid (TF-IDF + Sentiment + Length)

- **Hybrid + Random Forest (final model)**  
  - Accuracy ≈ **0.85**  
  - F1 ≈ **0.87**  
  - AUC ≈ **0.94**  

This is our **best overall model**, balancing performance and interpretability.

---

## 5. Robustness Check: Fuzzing Game Names

To ensure our models were not simply memorizing specific game titles, we ran an additional **“name fuzzing” experiment** (see the separate Colab).

### What we did

1. Extracted a cleaned `game_name` from each URL (e.g., `/game/football-manager-26/` → `"football manager 26"`).
2. Created `review_fuzzed` where occurrences of `game_name` in `review` were replaced with a neutral placeholder like **“the game”**.
3. Re-ran the **entire pipeline** (language filtering, preprocessing, sentiment, topic modeling, all supervised models) on this fuzzed corpus.

### How much text changed?

- `df_raw['changed'] = df_raw['review'] != df_raw['review_fuzzed']`
- **≈47%** of reviews changed after fuzzing  
  - `df_raw['changed'].mean() ≈ 0.47`  
  - `df_raw['changed'].sum() = 6,359`
- **≈29%** of reviews contain the placeholder phrase `"the game"` in `review_fuzzed`.

Changed examples show explicit titles being masked; unchanged examples either never mentioned the title or referred to it indirectly.

### Topic modeling after fuzzing

- LSA and LDA topics remain **qualitatively similar**:
  - enjoyment/gameplay, negative/technical issues, visuals/atmosphere, combat/mechanics.
- LDA coherences stay around **0.41–0.42** for 2–4 topics.

### Supervised learning after fuzzing

Using the same `meta_score ≥ 85` threshold:

- **TF-IDF + RF**: still strong (Accuracy ≈ 0.93, F1 ≈ 0.96, AUC ≈ 0.91).
- **Embeddings + RF**: F1 ≈ 0.79–0.82, AUC ≈ 0.83–0.86.
- **Knowledge-driven**: small changes (F1 ≈ 0.70, AUC ≈ 0.65).
- **Hybrid + RF**: Accuracy ≈ 0.84, F1 ≈ 0.86, AUC ≈ 0.92 (slightly below original but very close).

**Conclusion:**  
Masking game titles affects nearly half the corpus but has **minimal impact** on topic structure and model performance. Our models are driven mainly by **adjectives, sentiment, genre terms, and gameplay descriptions**, not the raw game names.

---

## 6. Repository Structure

Suggested layout (you can adjust to match your actual files):

```bash
.
├── notebooks/
│   ├── 01_data_collection_and_preprocessing.ipynb
│   ├── 02_eda_embeddings_sentiment_topics.ipynb
│   └── 03_supervised_models_and_fuzzing.ipynb
├── data/
│   ├── raw/           # raw scraped Metacritic CSV (not committed if large)
│   └── processed/     # English-only & feature-enriched CSVs (sa.csv, etc.)
├── README.md
└── requirements.txt

---

## 7. How to Run Locally

1. Clone the repo
git clone https://github.com/chen-zhu/NLP-Metacritics.git
cd NLP-Metacritics

2. Create enviornment & install dependencies
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Minimal packages (if user build their own requirement.txt):
- pandas, numpy, scikit-learn
- spacy, gensim, langdect, contractions
- textblob, vaderSentiment
- matplotlib, seaborn
- jupyter/notebook

3. Open notebooks
jupyter notebook

Then run the notebooks in order:
1. Data collection & preprocessing
2. EDA, embeddings, sentiment, topic modeling
3. Supervised learning & fuzzed-name robustness check

---

8. Authors & Acknowledgements
- Project team: Chen Zhu, Hla Win Tun, James Hah
- Course: IST 322 – Natural Language Processing, Claremont Graduate University.
- Instructor: Hengwei Zhang
- TA: Yi Zhuang

We gratefully acknowledge:
- Metacritic for providing public access to user reviews and scores.
- The authors of the libraries used: spaCy, scikit-learn, Gensim, TextBlob, VADER, and others.

---
9. References

- Key references and tools used in this project (also listed in the report):
- Metacritic. (2025). Metacritic video game reviews.
- Buijsman, M. (2025). Global games market to hit $189 billion in 2025 as growth shifts to console. Newzoo.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
- Honnibal, M., et al. (2020). spaCy: Industrial-strength natural language processing in Python.
- Loria, S. (2018). TextBlob Documentation.
- Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text.
- Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation.
- Bojanowski, P., et al. (2017). Enriching word vectors with subword information.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation.
- Sesari, E., Hort, M., & Sarro, F. (2022). An empirical study on the fairness of pre-trained word embeddings.

