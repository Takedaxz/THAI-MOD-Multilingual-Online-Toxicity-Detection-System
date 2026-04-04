# C4 Level 3: ML Component Diagram -- v1 Current (TF-IDF + Logistic Regression)

> Zooms into the **Model Service** container from the Level 2 diagram.
> Shows the internal components of the ML pipeline as currently deployed.

## Diagram

```mermaid
C4Component
    title THAI-MOD ML Component Diagram -- v1 Current (TF-IDF + LR)

    Container_Boundary(model_svc, "Model Service (ToxicityModelService)") {

        Component(cache_mgr, "Cache Manager", "joblib + JSON", "Handles load/save of trained pipeline and metadata. Atomic writes via .tmp files. Checks for existing cache on startup.")
        Component(data_loader, "Dataset Loader", "pandas", "Reads 8 CSV datasets, applies per-file cleaning, deduplicates by text content, combines into unified DataFrame.")
        Component(label_mapper, "Label Mapper", "pandas", "Maps raw labels to binary: neg->toxic(1), pos/neu->non-toxic(0). Drops rows with unmappable labels.")
        Component(preprocessor, "Text Preprocessor", "regex, emoji lib", "Cleans raw text: removes URLs, converts emojis to English text descriptions, lowercases ASCII characters. Static method, shared by training and inference.")
        Component(tokenizer, "Thai Tokenizer", "PyThaiNLP newmm", "Word segmentation for Thai text using newmm dictionary-based engine. Produces token list for TF-IDF. Handles Thai+English mixed text.")
        Component(feature_ext, "TF-IDF Vectorizer", "scikit-learn TfidfVectorizer", "Converts tokenized text to TF-IDF sparse vectors. Config: unigram+bigram, min_df=3, max_features=20,000. Uses Thai Tokenizer as custom tokenizer callable.")
        Component(classifier, "Logistic Regression Classifier", "scikit-learn LogisticRegression", "Binary classifier with class_weight='balanced' to handle toxic/non-toxic imbalance. max_iter=1000, random_state=42.")
        Component(pipeline, "scikit-learn Pipeline", "scikit-learn Pipeline", "Chains TF-IDF Vectorizer + LR Classifier into a single fit/predict object. The cached artifact is this entire pipeline.")
        Component(threshold_engine, "Threshold Engine", "Python", "Compares toxic_score (predict_proba[:,1]) against configurable threshold (default 0.4). Produces label + recommendation.")
        Component(evaluator, "Training Evaluator", "scikit-learn metrics", "Computes accuracy, precision, recall, F1, F2, confusion matrix on test split after training. Metrics saved to metadata.")
        Component(splitter, "Data Splitter", "scikit-learn train_test_split", "Stratified 80/20 split with random_state=42. Ensures class ratio preserved in both sets.")

    }

    Container_Ext(api, "FastAPI Backend", "Calls predict() and get_model_info()")
    Container_Ext(datasets, "Training Datasets", "8 CSV files via Git LFS")
    Container_Ext(model_cache, "Model Cache", "models/thai_mod_baseline.joblib + .metadata.json")

    Rel(api, threshold_engine, "predict(text, threshold)", "")
    Rel(api, cache_mgr, "ensure_ready() on startup", "")

    Rel(cache_mgr, model_cache, "Load/save pipeline + metadata", "joblib, JSON")
    Rel(cache_mgr, data_loader, "If no cache: trigger training", "")

    Rel(data_loader, datasets, "Read CSV files", "pandas.read_csv")
    Rel(data_loader, preprocessor, "Clean each text row", "")
    Rel(data_loader, label_mapper, "Map labels to binary", "")

    Rel(splitter, data_loader, "Receives unified DataFrame", "")

    Rel(pipeline, feature_ext, "Step 1: vectorize", "")
    Rel(pipeline, classifier, "Step 2: classify", "")
    Rel(feature_ext, tokenizer, "Custom tokenizer callable", "")

    Rel(threshold_engine, preprocessor, "Clean input text", "")
    Rel(threshold_engine, pipeline, "predict_proba()", "")

    Rel(evaluator, splitter, "Receives test split", "")
    Rel(evaluator, pipeline, "Evaluate predictions", "")
    Rel(evaluator, cache_mgr, "Provide metrics for metadata", "")
```

## Component Descriptions

### Cache Manager
- **Responsibility**: Decides whether to load an existing model or trigger training
- **Startup flow**:
  1. Check if `models/thai_mod_baseline.joblib` AND `.metadata.json` exist
  2. If yes: `joblib.load()` the pipeline, parse metadata JSON -> return bundle with `cache_status: "loaded_from_cache"`
  3. If no (or load fails): trigger full training pipeline -> save with atomic `.tmp` writes -> return bundle with `cache_status: "trained_and_cached"`
- **Atomic write**: Writes to `.tmp` files first, then `Path.replace()` to prevent corruption on crash

### Dataset Loader
- **Responsibility**: Aggregate 8 heterogeneous CSV datasets into one clean DataFrame
- **Per-dataset steps**:
  1. `pd.read_csv(dataset_file)`
  2. Drop rows with NaN in `category` or `texts`
  3. Apply `preprocess_text()` to each row
  4. Remove empty-string rows after preprocessing
  5. Map labels via Label Mapper
  6. Tag each row with `source` = filename
- **After combining**: `drop_duplicates(subset=["texts"], keep="first")` to prevent data leakage
- **Output**: DataFrame with columns `[texts, category, source]` where category is int (0 or 1)

### Label Mapper
- **Responsibility**: Standardize diverse label schemes to binary
- **Mapping**:
  - `pos` -> `neu` (first, collapse positive into neutral)
  - `neg` -> `1` (toxic)
  - `neu` -> `0` (non-toxic)
- **Drops**: Rows where category is NaN after mapping (labels that do not fit the scheme)
- **Rationale**: Different datasets use different label names; this normalizes them all

### Text Preprocessor
- **Responsibility**: Clean raw text identically for training and inference (prevents train-serve skew)
- **Steps** (in order):
  1. Handle NaN input -> return empty string
  2. Cast to string
  3. Remove HTTP/HTTPS URLs via regex
  4. Remove www.* URLs via regex
  5. Convert emojis to English text descriptions via `emoji.demojize(text, language="en")`
  6. Lowercase ASCII characters only (Thai characters unchanged)
- **Implementation**: Static method on `ToxicityModelService`
- **Critical invariant**: Same function used in `_prepare_dataset()` (training) and `predict()` (inference)

### Thai Tokenizer
- **Responsibility**: Word segmentation for Thai text
- **Engine**: PyThaiNLP `word_tokenize` with `newmm` (dictionary-based maximum matching)
- **Post-processing**: Filter out empty strings and whitespace-only tokens
- **Handles**: Thai, English, and mixed Thai-English (code-switching) text
- **Integration**: Passed as `tokenizer` parameter to TfidfVectorizer (replaces default regex tokenizer)

### TF-IDF Vectorizer
- **Responsibility**: Convert tokenized text into numerical feature vectors
- **Configuration**:
  - `tokenizer`: Thai Tokenizer (PyThaiNLP newmm)
  - `token_pattern`: None (disabled, using custom tokenizer)
  - `ngram_range`: (1, 2) -- unigrams and bigrams
  - `min_df`: 3 -- ignore terms appearing in fewer than 3 documents
  - `max_features`: 20,000 -- vocabulary cap
- **Output**: Sparse matrix of TF-IDF weights

### Logistic Regression Classifier
- **Responsibility**: Binary classification (toxic vs non-toxic)
- **Configuration**:
  - `class_weight`: 'balanced' -- automatically adjusts weights inversely proportional to class frequencies, addressing the 74/26 imbalance
  - `max_iter`: 1000 -- sufficient for convergence
  - `random_state`: 42 -- reproducibility
- **Output**: Probability estimates via `predict_proba()` (used by Threshold Engine)

### scikit-learn Pipeline
- **Responsibility**: Chain vectorizer + classifier into a single object
- **Steps**: `[("vect", TfidfVectorizer(...)), ("clf", LogisticRegression(...))]`
- **Benefit**: Single `.fit()` and `.predict_proba()` call. The cached artifact IS this pipeline.

### Threshold Engine
- **Responsibility**: Convert probability to actionable moderation decision
- **Logic**:
  - `toxic_score = pipeline.predict_proba([processed_text])[0][1]`
  - If `toxic_score >= threshold`: label="toxic", recommendation="FLAG_FOR_REVIEW"
  - Else: label="non-toxic", recommendation="ALLOW"
  - `confidence = toxic_score` if predicted toxic, else `1.0 - toxic_score`
- **Default threshold**: 0.4 (lower than standard 0.5 to favor recall)
- **Caller can override**: threshold is a parameter on every predict call

### Training Evaluator
- **Responsibility**: Compute metrics after training for metadata storage
- **Metrics computed**:
  - `accuracy_score`
  - `precision_score`
  - `recall_score`
  - `f1_score`
  - `fbeta_score(beta=2)` (F2 -- recall-weighted)
  - `confusion_matrix`
  - `test_size` (number of test samples)
- **Note**: Evaluation uses threshold-based predictions (not argmax), consistent with inference behavior

### Data Splitter
- **Responsibility**: Create reproducible train/test split
- **Config**: `test_size=0.2`, `random_state=42`, `stratify=y`
- **Stratification**: Ensures toxic/non-toxic ratio (~26/74) is preserved in both splits

## Training Flow (detailed)

```
[Startup: ensure_ready()]
    |
    v
[Cache Manager] -- cache exists? --YES--> joblib.load() --> READY
    |
    NO
    v
[Dataset Loader]
    |-- Read dataset1.csv ... dataset8.csv
    |-- Per file: dropna -> preprocess -> filter empty -> label map -> tag source
    |-- Concat all -> deduplicate by text
    |
    v
[Data Splitter]
    |-- stratified split: 80% train, 20% test
    |
    v
[Pipeline.fit(X_train, y_train)]
    |-- TF-IDF Vectorizer: fit_transform (builds vocabulary, computes IDF)
    |   |-- Thai Tokenizer: segment each text
    |-- LR Classifier: fit (learn weights with balanced class penalty)
    |
    v
[Training Evaluator]
    |-- predict_proba(X_test) with default threshold
    |-- Compute: accuracy, precision, recall, F1, F2, confusion matrix
    |
    v
[Cache Manager]
    |-- Save pipeline to .joblib.tmp -> rename to .joblib
    |-- Save metadata to .json.tmp -> rename to .json
    |
    v
READY (cache_status: "trained_and_cached")
```

## Inference Flow (detailed)

```
[API: predict(text="...", threshold=0.4)]
    |
    v
[Threshold Engine]
    |
    v
[Preprocessor]
    |-- Remove URLs
    |-- Demojize emojis
    |-- Lowercase ASCII
    |
    v
[Pipeline.predict_proba([processed_text])]
    |
    v
[TF-IDF Vectorizer.transform]
    |-- [Thai Tokenizer] segment text
    |-- Look up tokens in learned vocabulary
    |-- Compute TF-IDF weights
    |
    v
[LR Classifier.predict_proba]
    |-- Apply learned weights
    |-- Return [P(non-toxic), P(toxic)]
    |
    v
[Threshold Engine]
    |-- toxic_score = P(toxic)
    |-- toxic_score >= 0.4? -> "toxic" / "FLAG_FOR_REVIEW"
    |--                   else -> "non-toxic" / "ALLOW"
    |-- confidence = max(toxic_score, 1-toxic_score)
    |
    v
[Return PredictionResult]
    {text, processed_text, predicted_label, toxic_score,
     confidence, threshold, recommendation, source_model}
```

## Performance Characteristics (v1)

| Metric | Value |
|---|---|
| Accuracy | 83.0% |
| Precision (toxic) | 78% |
| Recall (toxic) | 77.6% (balanced) / 81% (threshold 0.3) |
| F1 | 79% |
| F2 | ~79% |
| Inference latency (CPU) | ~0.5-1 ms per request |
| Model file size | ~5-20 MB |
| RAM usage | ~50-200 MB |
| Vocabulary size | up to 20,000 features |
