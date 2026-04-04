# C4 Level 4: Code-level Diagram -- Model Service

> Zooms into the **Model Service** component to show class structure, method signatures, and data flow at the code level.
> Based on the actual implementation in `src/thai_mod_api/model_service.py`.

## Class Diagram

```mermaid
classDiagram
    class ToxicityModelService {
        -Path project_root
        -float default_threshold
        -list~Path~ dataset_files
        -Path model_dir
        -Path model_path
        -Path metadata_path
        -dict~str, Any~ bundle

        +__init__(project_root: Path, default_threshold: float)
        +ensure_ready() void
        +predict(text: str, threshold: float|None) PredictionResult
        +get_model_info() dict~str, Any~
        +preprocess_text(text: str) str$
        -_load_or_train() dict~str, Any~
        -_load_metadata() dict~str, Any~
        -_save_bundle(bundle: dict) void
        -_train_bundle() dict~str, Any~
        -_load_full_dataset() DataFrame
        -_prepare_dataset(dataset_file: Path) DataFrame
        -_tokenize_text(text: str) list~str~$
    }

    class PredictionResult {
        <<dataclass>>
        +str text
        +str processed_text
        +str predicted_label
        +float toxic_score
        +float confidence
        +float threshold
        +str recommendation
        +str source_model
    }

    class PredictRequest {
        <<Pydantic BaseModel>>
        +str text
        +float|None threshold
    }

    class BatchPredictRequest {
        <<Pydantic BaseModel>>
        +list~str~ texts
        +float|None threshold
    }

    class PredictionResponse {
        <<Pydantic BaseModel>>
        +str text
        +str processed_text
        +str predicted_label
        +float toxic_score
        +float confidence
        +float threshold
        +str recommendation
        +str source_model
    }

    class BatchPredictionResponse {
        <<Pydantic BaseModel>>
        +list~PredictionResponse~ predictions
    }

    class Pipeline {
        <<scikit-learn>>
        +fit(X, y) Pipeline
        +predict_proba(X) ndarray
        +transform(X) sparse_matrix
    }

    class TfidfVectorizer {
        <<scikit-learn>>
        +tokenizer: callable
        +ngram_range: tuple
        +min_df: int
        +max_features: int
    }

    class LogisticRegression {
        <<scikit-learn>>
        +class_weight: str
        +max_iter: int
        +random_state: int
    }

    ToxicityModelService --> PredictionResult : creates
    ToxicityModelService --> Pipeline : owns (in bundle)
    Pipeline --> TfidfVectorizer : step "vect"
    Pipeline --> LogisticRegression : step "clf"
    TfidfVectorizer ..> ToxicityModelService : uses _tokenize_text()

    note for ToxicityModelService "Central class managing the entire ML lifecycle:\ndata loading, training, caching, and inference.\nAll methods are synchronous (called from async FastAPI handlers)."
```

## Method Detail

### `__init__(project_root, default_threshold=0.4)`

```
Sets up paths and configuration. No model loading happens here.

Attributes initialized:
  project_root       -> root of the repository (2 levels up from main.py)
  default_threshold  -> 0.4 (recall-favoring default)
  dataset_files      -> [datasets/dataset1.csv, ..., datasets/dataset8.csv]
  model_dir          -> models/ (created if not exists)
  model_path         -> models/thai_mod_baseline.joblib
  metadata_path      -> models/thai_mod_baseline.metadata.json
  bundle             -> None (loaded lazily)
```

### `ensure_ready()`

```
Called once during app startup (FastAPI lifespan).
If bundle is None: triggers _load_or_train().
Idempotent: safe to call multiple times.

Flow:
  bundle is None? --YES--> _load_or_train() --> assign to self.bundle
                  --NO---> return (already ready)
```

### `_load_or_train() -> dict`

```
Decision point: load cached model or train from scratch.

Flow:
  model_path exists AND metadata_path exists?
    |
    YES --> try:
    |         joblib.load(model_path) -> pipeline
    |         _load_metadata() -> metadata dict
    |         return bundle with cache_status="loaded_from_cache"
    |       except:
    |         fall through to training
    |
    NO/FAIL --> _train_bundle() -> bundle
                _save_bundle(bundle)
                bundle["cache_status"] = "trained_and_cached"
                return bundle
```

### `_train_bundle() -> dict`

```
Full training pipeline from raw CSV to evaluated model.

Steps:
  1. _load_full_dataset() -> DataFrame [texts, category, source]

  2. train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  3. Build Pipeline:
     Pipeline([
       ("vect", TfidfVectorizer(
           tokenizer=_tokenize_text,    # PyThaiNLP newmm
           token_pattern=None,          # disable default regex
           ngram_range=(1, 2),          # unigrams + bigrams
           min_df=3,                    # ignore rare terms
           max_features=20_000          # vocabulary cap
       )),
       ("clf", LogisticRegression(
           class_weight="balanced",     # handle 74/26 imbalance
           max_iter=1000,
           random_state=42
       ))
     ])

  4. pipeline.fit(X_train, y_train)

  5. Evaluate on test set WITH threshold:
     probabilities = pipeline.predict_proba(X_test)[:, 1]
     predictions = (probabilities >= default_threshold).astype(int)
     Compute: accuracy, precision, recall, F1, F2, confusion_matrix

  6. Return bundle dict:
     {pipeline, model_name, deployment_mode, default_threshold,
      trained_at (UTC ISO), dataset_rows, dataset_sources, metrics}
```

### `_load_full_dataset() -> DataFrame`

```
Aggregates all 8 datasets into one clean DataFrame.

Flow:
  for each dataset_file in dataset_files:
    _prepare_dataset(file) -> per-file DataFrame
  pd.concat(all frames)
  drop_duplicates(subset=["texts"], keep="first")
  reset_index

Output columns: [texts: str, category: int(0|1), source: str]
```

### `_prepare_dataset(dataset_file) -> DataFrame`

```
Cleans and normalizes a single CSV dataset.

Steps:
  1. pd.read_csv(dataset_file)
  2. dropna(subset=["category", "texts"])
  3. Apply preprocess_text() to "texts" column
  4. Remove rows where texts is empty after preprocessing
  5. Label mapping:
     "pos" -> "neu"  (collapse positive into neutral)
     "neg" -> 1      (toxic)
     "neu" -> 0      (non-toxic)
  6. dropna(subset=["category"]) (remove unmappable labels)
  7. Cast category to int
  8. Add source column = filename (e.g., "dataset1.csv")
  9. Return DataFrame[texts, category, source]
```

### `preprocess_text(text) -> str` [static]

```
Text cleaning function. MUST be identical for training and inference.

Steps:
  1. if pd.isna(text): return ""
  2. cleaned = str(text)
  3. Remove HTTP(S) URLs:
     re.sub(r"http[s]?://...", "", cleaned)
  4. Remove www URLs:
     re.sub(r"www\\..*", "", cleaned)
  5. Emoji to text:
     emoji.demojize(cleaned, language="en")
  6. Lowercase ASCII only:
     for each char: lower() if ord(char) < 128, else keep original
  7. Return cleaned string

Example:
  Input:  "ไอ้บ้า 😡 https://t.co/abc THIS IS TOXIC"
  Output: "ไอ้บ้า :enraged_face:  this is toxic"
```

### `_tokenize_text(text) -> list[str]` [static]

```
Thai word segmentation. Used as custom tokenizer for TfidfVectorizer.

Steps:
  1. word_tokenize(str(text), engine="newmm")
     - newmm = dictionary-based maximum matching algorithm
     - Handles Thai, English, and mixed text
  2. Filter: remove empty strings and whitespace-only tokens
  3. Return list of token strings

Example:
  Input:  "โคตร toxic เลย"
  Output: ["โคตร", "toxic", "เลย"]
```

### `predict(text, threshold=None) -> PredictionResult`

```
Single text inference. Main entry point for API.

Steps:
  1. ensure_ready() (idempotent)
  2. effective_threshold = threshold or self.default_threshold
  3. processed_text = preprocess_text(text)
  4. toxic_score = pipeline.predict_proba([processed_text])[0][1]
  5. predicted_toxic = int(toxic_score >= effective_threshold)
  6. confidence = toxic_score if predicted_toxic else (1.0 - toxic_score)
  7. Return PredictionResult:
     - text: original input
     - processed_text: cleaned version
     - predicted_label: "toxic" or "non-toxic"
     - toxic_score: float (0.0 - 1.0), rounded to 4 decimals
     - confidence: float, rounded to 4 decimals
     - threshold: effective threshold used
     - recommendation: "FLAG_FOR_REVIEW" or "ALLOW"
     - source_model: bundle["model_name"]
```

### `get_model_info() -> dict`

```
Returns model metadata for /api/model-info and /api/health.

Returns:
  {model_name, deployment_mode, cache_status, default_threshold,
   trained_at, dataset_rows, dataset_sources, metrics}
```

### `_save_bundle(bundle)`

```
Atomic save of trained model + metadata.

Steps:
  1. Extract metadata dict from bundle (excludes pipeline object)
  2. Write pipeline to .joblib.tmp via joblib.dump()
  3. Write metadata to .json.tmp via json.dump()
  4. Atomic rename: .tmp -> final path (Path.replace)

Why atomic: prevents corrupted files if process crashes mid-write.
```

## Bundle Structure (internal state)

```python
self.bundle = {
    "pipeline":         Pipeline,          # scikit-learn Pipeline (TF-IDF + LR)
    "model_name":       str,               # "TF-IDF + Logistic Regression (Balanced)"
    "deployment_mode":  str,               # "cached_startup_baseline" | "trained_and_cached"
    "default_threshold": float,            # 0.4
    "trained_at":       str,               # ISO 8601 UTC timestamp
    "dataset_rows":     int,               # total rows after dedup
    "dataset_sources":  list[str],         # ["dataset1.csv", ..., "dataset8.csv"]
    "metrics": {
        "accuracy":         float,
        "precision":        float,
        "recall":           float,
        "f1_score":         float,
        "f2_score":         float,
        "confusion_matrix": list[list[int]],  # [[TN, FP], [FN, TP]]
        "test_size":        int
    },
    "cache_status":     str                # "loaded_from_cache" | "trained_and_cached"
}
```

## Sequence Diagram: Single Prediction

```mermaid
sequenceDiagram
    participant Client as Web UI / API Client
    participant API as FastAPI (main.py)
    participant Svc as ToxicityModelService
    participant Pipe as sklearn Pipeline
    participant TFIDF as TfidfVectorizer
    participant Tok as PyThaiNLP Tokenizer
    participant LR as LogisticRegression

    Client->>API: POST /api/predict {text, threshold}
    API->>Svc: predict(text, threshold)
    Svc->>Svc: ensure_ready() [idempotent]
    Svc->>Svc: preprocess_text(text)
    Note over Svc: Remove URLs, demojize,<br/>lowercase ASCII
    Svc->>Pipe: predict_proba([processed_text])
    Pipe->>TFIDF: transform([processed_text])
    TFIDF->>Tok: _tokenize_text(text)
    Tok-->>TFIDF: ["token1", "token2", ...]
    TFIDF-->>Pipe: sparse TF-IDF vector
    Pipe->>LR: predict_proba(tfidf_vector)
    LR-->>Pipe: [[P(non-toxic), P(toxic)]]
    Pipe-->>Svc: [[P(non-toxic), P(toxic)]]
    Svc->>Svc: toxic_score = P(toxic)
    Svc->>Svc: score >= threshold?
    Note over Svc: Yes: FLAG_FOR_REVIEW<br/>No: ALLOW
    Svc-->>API: PredictionResult
    API-->>Client: PredictionResponse JSON
```

## Sequence Diagram: App Startup

```mermaid
sequenceDiagram
    participant App as FastAPI Lifespan
    participant Svc as ToxicityModelService
    participant Cache as Filesystem (models/)
    participant DS as Datasets (CSV)
    participant Pipe as sklearn Pipeline

    App->>Svc: ensure_ready()
    Svc->>Svc: bundle is None? Yes
    Svc->>Svc: _load_or_train()

    alt Cache exists
        Svc->>Cache: joblib.load(thai_mod_baseline.joblib)
        Cache-->>Svc: Pipeline object
        Svc->>Cache: read thai_mod_baseline.metadata.json
        Cache-->>Svc: metadata dict
        Svc->>Svc: bundle.cache_status = "loaded_from_cache"
    else No cache
        Svc->>DS: read dataset1.csv ... dataset8.csv
        DS-->>Svc: raw DataFrames
        Svc->>Svc: _prepare_dataset() x 8
        Note over Svc: Clean, map labels,<br/>deduplicate
        Svc->>Svc: train_test_split(80/20, stratified)
        Svc->>Pipe: Pipeline.fit(X_train, y_train)
        Pipe-->>Svc: trained Pipeline
        Svc->>Svc: evaluate on test set
        Svc->>Cache: save .joblib.tmp + .json.tmp
        Svc->>Cache: atomic rename to final paths
        Svc->>Svc: bundle.cache_status = "trained_and_cached"
    end

    Svc->>App: ready
    Note over App: app.state.model_service = svc
```

## File Dependencies

```
src/thai_mod_api/
    main.py
        imports: model_service.ToxicityModelService
        imports: schemas.PredictRequest, PredictionResponse, ...
        creates: ToxicityModelService(PROJECT_ROOT)
        lifespan: model_service.ensure_ready()

    model_service.py
        imports: joblib, pandas, sklearn.*, pythainlp, emoji, re
        defines: PredictionResult (dataclass)
        defines: ToxicityModelService (main class)
        reads: datasets/dataset*.csv (training)
        reads/writes: models/thai_mod_baseline.* (cache)

    schemas.py
        imports: pydantic.BaseModel
        defines: PredictRequest, BatchPredictRequest
        defines: PredictionResponse, BatchPredictionResponse

    static/
        index.html  -> served at GET /
        admin.html  -> served at GET /admin
        app.js      -> moderator UI logic, calls /api/predict
        admin.js    -> admin UI logic, calls /api/health, /api/model-info
        styles.css  -> shared styles
```
