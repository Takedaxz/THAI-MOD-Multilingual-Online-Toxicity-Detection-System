# THAI-MOD: Multilingual Online Toxicity Detection System

A binary text classifier (Toxic vs Non-toxic) for Thai, English, and code-switched content, built as a decision-support tool for content moderators. The system combines traditional ML baselines with transformer-based models and exposes predictions through a FastAPI web application with a moderator UI.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Scripts](#scripts)
- [Documentation](#documentation)
- [Conventions](#conventions)

---

## Features

- **Multilingual support** -- Thai, English, and code-switched text
- **Dual model backend** -- TF-IDF + Logistic Regression (fast baseline) and WangchanBERTa (high-accuracy transformer)
- **Auto-fallback** -- prefers WangchanBERTa when available, falls back to LR automatically
- **Moderator UI** -- web interface for real-time toxicity analysis with review/flag workflow
- **Admin dashboard** -- model update orchestration (train/promote candidates), monitoring, and drift detection
- **Session authentication** -- login-protected admin and monitoring endpoints
- **Prediction logging** -- JSONL-based prediction logs with drift reporting
- **Model update pipeline** -- train candidate models and promote them with metric-gate checks
- **MLflow integration** -- experiment tracking for training runs

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Moderator Browser                  │
│         (index.html / admin.html / login.html)       │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Application                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Auth (session)│  │ Predict API  │  │ Admin API  │ │
│  └─────────────┘  └──────┬───────┘  └─────┬──────┘ │
│                          │                 │         │
│  ┌───────────────────────▼─────────────────▼──────┐ │
│  │            ToxicityModelService                 │ │
│  │  ┌──────────────────┐  ┌────────────────────┐  │ │
│  │  │ TF-IDF + LR      │  │ WangchanBERTa      │  │ │
│  │  │ (default backend) │  │ (when available)   │  │ │
│  │  └──────────────────┘  └────────────────────┘  │ │
│  │  ┌──────────────────┐  ┌────────────────────┐  │ │
│  │  │ Text Preprocessing│  │ Toxic Keywords     │  │ │
│  │  │ (PyThaiNLP)       │  │ (keyword glossary) │  │ │
│  │  └──────────────────┘  └────────────────────┘  │ │
│  └────────────────────────────────────────────────┘ │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │ MonitoringService │  │ RecentRequestMonitor     │ │
│  │ (drift/logs)      │  │ (windowed metrics)       │ │
│  └──────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### ML Pipeline

1. **Data aggregation** -- 8 source datasets combined into unified train/test splits with binary labels
2. **Preprocessing** -- URL removal, emoji demojization, ASCII lowercasing, Thai tokenization (PyThaiNLP newmm)
3. **Baselines** -- TF-IDF (word + character n-grams) with Logistic Regression, Linear SVC, XGBoost
4. **Transformers** -- Fine-tuned WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa
5. **Evaluation** -- Recall-oriented (catching toxic content is prioritized), F1, accuracy, confusion matrices
6. **Class imbalance** -- Handled via `class_weight='balanced'` and SMOTE/oversampling

---

## Model Performance

All models evaluated on the THAI-MOD test set (n = 6,124):

| Model | Accuracy | Precision | Recall (toxic) | F1 |
|---|---|---|---|---|
| **WangchanBERTa** | **0.891** | 0.800 | 0.775 | **0.787** |
| PhayaThaiBERT | 0.889 | 0.804 | 0.760 | 0.782 |
| XLM-RoBERTa | 0.865 | 0.775 | 0.678 | 0.723 |
| Linear SVC (Balanced) | 0.840 | 0.780 | 0.800 | 0.790 |
| **LR (Balanced)** | 0.830 | 0.780 | **0.810** | 0.790 |

**Deployment decision**: LR (Balanced) is the default backend due to its highest toxic recall (0.81), sub-millisecond latency, and minimal resource requirements. WangchanBERTa is available as an alternative backend with higher overall accuracy and macro F1.

See [`docs/reports/lr-vs-bert-comparison.md`](docs/reports/lr-vs-bert-comparison.md) for the full comparison report.

---

## Project Structure

```
THAI-MOD/
├── model.ipynb                  # Main training pipeline (baselines + fine-tuning)
├── thai-bert.ipynb              # BERT model evaluation and comparison
├── toxicity_detection.ipynb     # Combined full-pipeline notebook
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev/test dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Environment variable template
│
├── datasets/                    # Data assets (Git LFS)
│   ├── dataset[1-8].csv         # 8 source datasets (Thai + English)
│   ├── toxic_keywords.csv       # Thai toxic keyword glossary
│   ├── exploration/             # EDA notebooks, train/test splits
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── keywords/            # Keyword lists (neg/neu/pos/q)
│   └── monitoring/              # Reference profiles for drift detection
│
├── src/thai_mod_api/            # FastAPI application
│   ├── main.py                  # App entry point, routes, lifespan
│   ├── model_service.py         # Model loading, training, prediction
│   ├── text_processing.py       # Shared preprocessing (URLs, emoji, lowercase)
│   ├── monitoring_service.py    # Prediction logging and drift detection
│   ├── monitoring.py            # Recent-request windowed monitor
│   ├── schemas.py               # Pydantic request/response models
│   ├── config.py                # .env file loader
│   └── static/                  # Frontend assets
│       ├── index.html           # Moderator analysis UI
│       ├── admin.html           # Admin dashboard
│       ├── login.html           # Login page
│       ├── app.js               # Moderator UI logic
│       ├── admin.js             # Admin dashboard logic
│       ├── login.js             # Login logic
│       └── styles.css           # Shared styles
│
├── scripts/                     # Operational scripts
│   ├── train_lr_candidate.py    # Train LR candidate model
│   ├── promote_lr_candidate.py  # Promote LR candidate with metric checks
│   ├── train_wangchanberta.py   # Train WangchanBERTa candidate
│   ├── promote_wangchanberta_candidate.py  # Promote BERT candidate
│   ├── export_wangchanberta_artifact.py    # Export BERT artifact
│   ├── build_reference_profile.py          # Build monitoring reference profile
│   └── generate_full_pipeline_notebook.py  # Regenerate combined notebook
│
├── tests/                       # Test suite
│   ├── conftest.py              # Fixtures (mock model bundle)
│   ├── test_api.py              # API endpoint integration tests
│   ├── test_preprocessing.py    # Text preprocessing unit tests
│   ├── test_dataset_prep.py     # Dataset preparation tests
│   ├── test_monitoring_service.py
│   └── test_recent_request_monitor.py
│
├── models/                      # Saved model artifacts (gitignored)
│   ├── candidates/              # Candidate models for promotion
│   ├── reviewed/                # Human-reviewed prediction examples
│   └── monitoring/              # Prediction logs and drift data
│
└── docs/                        # Documentation
    ├── architecture/            # C4 architecture diagrams
    ├── reports/                 # Model comparison reports
    ├── plans/                   # Phase planning documents
    └── progress/                # Progress notes
```

---

## Getting Started

### Prerequisites

- Python 3.11.13
- Conda (recommended)
- Git LFS (for dataset files)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd THAI-MOD-Multilingual-Online-Toxicity-Detection-System

# Pull large dataset files
git lfs pull

# Create and activate conda environment
conda create -n cedt python=3.11.13 -y
conda activate cedt

# Install dependencies
pip install -r requirements.txt

# (Optional) Install dev/test dependencies
pip install -r requirements-dev.txt
```

### Configure Environment

```bash
# Copy the example env file and fill in required values
cp .env.example .env
```

Edit `.env` and set the required variables:

```bash
THAI_MOD_MODEL_BACKEND=auto          # auto | lr | bert
THAI_MOD_AUTH_USERNAME=<your-username>
THAI_MOD_AUTH_PASSWORD=<your-password>
THAI_MOD_SESSION_SECRET=<random-secret>

# Optional MLflow config
THAI_MOD_MLFLOW_EXPERIMENT=thai-mod
# THAI_MOD_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### Run the Application

```bash
conda activate cedt
uvicorn src.thai_mod_api.main:app --reload
```

Open in your browser:

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/` | Moderator analysis UI |
| `http://127.0.0.1:8000/admin` | Admin dashboard (login required) |
| `http://127.0.0.1:8000/docs` | Interactive API documentation (Swagger) |

### Verify Health

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Word + Character TF-IDF + Logistic Regression (Balanced)",
  "deployment_mode": "lr",
  "cache_status": "loaded_from_cache"
}
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `THAI_MOD_MODEL_BACKEND` | No | `auto` | Model selection: `auto` (prefer BERT, fallback LR), `lr` (force LR), `bert` (require BERT, fail if missing) |
| `THAI_MOD_AUTH_USERNAME` | Yes | -- | Admin login username |
| `THAI_MOD_AUTH_PASSWORD` | Yes | -- | Admin login password |
| `THAI_MOD_SESSION_SECRET` | Yes | -- | Secret key for session cookies |
| `THAI_MOD_PROTECT_ANALYZER` | No | `false` | If `true`, requires authentication for the predict endpoint |
| `THAI_MOD_MLFLOW_EXPERIMENT` | No | `thai-mod` | MLflow experiment name for training runs |
| `THAI_MOD_MLFLOW_TRACKING_URI` | No | local `./mlruns` | MLflow tracking server URI |

---

## API Reference

### Public Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | System health check (status, model info, cache status) |
| `GET` | `/api/model-info` | Detailed model metadata |
| `POST` | `/api/predict` | Classify a single text |
| `POST` | `/api/batch-predict` | Classify up to 100 texts |
| `GET` | `/api/monitoring/summary` | Prediction log summary |
| `GET` | `/api/monitoring/drift` | Drift detection report |

### Authenticated Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Session login |
| `POST` | `/api/auth/logout` | Session logout |
| `GET` | `/api/auth/me` | Current auth status |
| `GET` | `/api/monitoring` | Windowed monitoring summary |
| `GET` | `/api/monitoring/events` | Recent prediction events |
| `POST` | `/api/monitoring/reset` | Clear monitoring window |
| `GET` | `/api/admin/overview` | Admin overview (health + model info) |
| `GET` | `/api/admin/model-update/status` | Model update job status and candidates |
| `POST` | `/api/admin/model-update/train-candidate` | Train a new BERT candidate |
| `POST` | `/api/admin/model-update/promote-candidate` | Promote BERT candidate to production |
| `POST` | `/api/admin/model-update/train-lr-candidate` | Train a new LR candidate |
| `POST` | `/api/admin/model-update/promote-lr-candidate` | Promote LR candidate to production |
| `GET` | `/api/reviewed-examples/summary` | Count of human-reviewed examples |
| `POST` | `/api/reviewed-examples` | Save a reviewed example |

### Prediction Request/Response

**`POST /api/predict`**

```json
{
  "text": "ไอดอก อย่ามาทำตัว",
  "threshold": 0.4
}
```

```json
{
  "request_id": "abc123",
  "text": "ไอดอก อย่ามาทำตัว",
  "processed_text": "ไอดอก อย่ามาทำตัว",
  "predicted_label": "toxic",
  "toxic_score": 0.82,
  "confidence": 0.82,
  "threshold": 0.4,
  "recommendation": "FLAG_FOR_REVIEW",
  "source_model": "Word + Character TF-IDF + Logistic Regression (Balanced)"
}
```

The `recommendation` field follows a decision policy:
- `toxic_score >= threshold` --> `FLAG_FOR_REVIEW`
- `toxic_score < threshold` --> `ALLOW`

---

## Testing

```bash
conda activate cedt
pip install -r requirements.txt -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src/thai_mod_api --cov-report=term-missing
```

Tests use a mock model bundle (see [`tests/conftest.py`](tests/conftest.py)) -- no real datasets or model training occurs.

---

## Scripts

| Script | Description |
|---|---|
| `scripts/train_lr_candidate.py` | Train an LR candidate model, save to `models/candidates/lr_candidate/` |
| `scripts/promote_lr_candidate.py` | Promote LR candidate to production with metric-gate checks |
| `scripts/train_wangchanberta.py` | Fine-tune WangchanBERTa, save to `models/candidates/wangchanberta_candidate/` |
| `scripts/promote_wangchanberta_candidate.py` | Promote BERT candidate with metric-gate checks |
| `scripts/export_wangchanberta_artifact.py` | Export WangchanBERTa artifact for deployment |
| `scripts/build_reference_profile.py` | Build monitoring reference profile from a batch of predictions |
| `scripts/generate_full_pipeline_notebook.py` | Regenerate `toxicity_detection.ipynb` |

Example:

```bash
# Train a new LR candidate (overwrites existing)
python scripts/train_lr_candidate.py --force

# Promote candidate to production (checks metrics before promoting)
python scripts/promote_lr_candidate.py
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture/`](docs/architecture/) | C4 architecture diagrams (context, container, component, code levels) |
| [`docs/architecture/design-decisions.md`](docs/architecture/design-decisions.md) | 10 key design decisions with rationale |
| [`docs/reports/lr-vs-bert-comparison.md`](docs/reports/lr-vs-bert-comparison.md) | LR vs WangchanBERTa comparison report |
| [`docs/auth-flow.md`](docs/auth-flow.md) | Authentication flow documentation |
| [`docs/monitoring-and-drift.md`](docs/monitoring-and-drift.md) | Monitoring and drift detection design |
| [`docs/model-update-pipeline.md`](docs/model-update-pipeline.md) | Model update and promotion pipeline |
| [`docs/testing-and-maintainability.md`](docs/testing-and-maintainability.md) | Testing strategy |
| [`docs/evaluation-and-responsible-ml.md`](docs/evaluation-and-responsible-ml.md) | Evaluation methodology and responsible ML |
| [`docs/requirements.md`](docs/requirements.md) | Project requirements |
| [`AGENTS.md`](AGENTS.md) | AI agent guidance for working with this repository |

