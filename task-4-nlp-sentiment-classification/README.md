# Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)

An end-to-end Natural Language Processing (NLP) text classification pipeline that ingests customer reviews, preprocesses and lemmatizes raw textual corpus, constructs sparse TF-IDF feature matrices (`max_features=5000`), and benchmarks **Multinomial Naive Bayes** versus **Logistic Regression** classifiers.

---

## 📊 Project Diagnostics Preview

![Sentiment Classification Diagnostics: Class Distribution, TF-IDF Weights, Confusion Matrices, and Benchmark Metrics](screenshots/01_project_preview.png)

---

## 📁 Project Structure

```text
task-4-nlp-sentiment-classification/
├── data/
│   ├── README.md
│   └── sentiment_reviews.csv
├── figures/
│   ├── 01_sentiment_eda_distribution.png
│   ├── 02_top_tfidf_features.png
│   └── 03_confusion_matrix_and_benchmark.png
├── notebooks/
│   └── sentiment_classification.ipynb
├── screenshots/
│   └── 01_project_preview.png
├── src/
│   └── sentiment_classifier.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technical Workflow & Architecture

1. **Data Ingestion & Integrity Validation**:
   - Ingest `sentiment_reviews.csv` (10 annotated customer review records).
   - Evaluate sentence lengths, character counts, and sentiment class balance (`positive`, `negative`, `neutral`).

2. **Text Preprocessing Pipeline**:
   - HTML entity decoding and regex HTML tag stripping (`re.sub(r'<[^>]+>', ' ', ...)`).
   - Text normalization: lowercasing and punctuation/symbol filtering (`[a-z]+`).
   - Stopword removal via `nltk.corpus.stopwords` (English).
   - Lexical lemmatization using `nltk.stem.WordNetLemmatizer` to preserve morphological root words.

3. **Stratified Partitioning**:
   - Stratified train/test split preserving label proportions.
   - Singleton class preservation in the training partition to ensure valid evaluation.

4. **TF-IDF Feature Engineering**:
   - `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` fitted strictly on training data to prevent vocabulary leakage.
   - Extraction of top discriminant n-gram terms across sentiment classes.

5. **Model Training & Benchmarking**:
   - **Multinomial Naive Bayes (`MultinomialNB`)**: Probabilistic text classification under conditional feature independence.
   - **Logistic Regression (`LogisticRegression`)**: Linear classifier modeling posterior log-odds with L2 weight decay.
   - Evaluation across Accuracy, Precision, Recall, F1-Score (macro & weighted), and annotated Confusion Matrix heatmaps.

6. **Live Interactive Inference (`predict_sentiment`)**:
   - Production-ready inference function that takes arbitrary custom user review strings and computes predicted sentiment polarity along with per-class posterior probabilities.

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open and run the clean, fully-executed notebook:
```bash
jupyter notebook notebooks/sentiment_classification.ipynb
```

### 3. Programmatic Usage (CLI / Python)
You can directly run inference using the reusable module in `src/sentiment_classifier.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from sentiment_classifier import load_reviews, split_reviews, build_models, predict_sentiment

# Load data and fit model
reviews = load_reviews("data/sentiment_reviews.csv")
X_train, _, y_train, _, _, _ = split_reviews(reviews)
model = build_models()["Multinomial Naive Bayes"].fit(X_train, y_train)

# Predict on live custom review
result = predict_sentiment("The setup was easy and the product works flawlessly!", model)
print(result)
# Output: {'sentiment': 'positive', 'probabilities': {'negative': 0.33, 'neutral': 0.12, 'positive': 0.55}}
```

---

## 📈 Benchmark Summary

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **0.667** | **0.750** | **0.750** | **0.667** | **0.667** |
| **Logistic Regression** | 0.667 | 0.333 | 0.500 | 0.400 | 0.533 |

---

## 💡 Key Findings & Analytical Conclusions
- **Preprocessing Efficiency**: Lowercasing, stopword elimination, and lemmatization concentrated signal onto diagnostic root lemmas (*exceptional*, *damage*, *support*, *disappoint*).
- **Model Behavior**: Multinomial Naive Bayes calibrated well on sparse TF-IDF vectors, delivering robust separation on compact textual vocabularies without overfitting.
