# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

THAI-MOD is a multilingual (Thai + English) online toxicity detection system — a binary text classifier (Toxic vs Non-toxic) built as a decision-support tool for content moderators. This is a research/ML project driven primarily through Jupyter notebooks.

## Development Environment

- **Python**: 3.11.13 (conda environment: `cedt`)
- **Install dependencies**: `pip install -r requirements.txt`
- **Key frameworks**: PyTorch, Hugging Face Transformers, scikit-learn, PyThaiNLP

## Project Structure

- `model.ipynb` — Main training pipeline: dataset aggregation, preprocessing, TF-IDF baselines (Logistic Regression, Linear SVC), and transformer fine-tuning
- `thai-bert.ipynb` — BERT model evaluation and comparison (WangchanBERTa, XLM-RoBERTa, PhayaThaiBERT)
- `toxicity_detection.ipynb` — Combined full-pipeline notebook with baseline and transformer comparison plus model caching
- `datasets/` — All data assets (Git LFS tracked via `.gitattributes`)
  - `dataset[1-8].csv` — Individual source datasets (10 sources total, Thai + English)
  - `toxic_keywords.csv` — Thai toxic keyword glossary
  - `exploration/` — EDA notebooks, train/test splits, keyword lists
- `models/` — Saved model artifacts (gitignored)
- `src/thai_mod_api/` — Phase 2 FastAPI app, model service, schemas, and static moderator UI
- `scripts/generate_full_pipeline_notebook.py` — Regenerates `toxicity_detection.ipynb`

## Architecture & ML Pipeline

1. **Data aggregation**: Combines 10 datasets into unified train/test splits with binary toxic/non-toxic labels
2. **Preprocessing**: Text cleaning, Thai tokenization (PyThaiNLP), emoji handling
3. **Baselines**: TF-IDF vectorization → Logistic Regression / Linear SVC / XGBoost
4. **Transformers**: Fine-tuning WangchanBERTa (selected model — best Thai language support with multilingual capability)
5. **Evaluation**: Prioritizes recall (catching toxic content is more important than precision), uses F1, accuracy, confusion matrices
6. **Class imbalance**: Handled via imbalanced-learn (SMOTE/resampling techniques)

## Current System Status

- The repository now includes a working Phase 2 app with:
  - FastAPI backend
  - moderator UI
  - cached baseline deployment path
- The deployed app currently uses `TF-IDF + Logistic Regression (Balanced)` for inference
- The app exposes:
  - `GET /api/health`
  - `GET /api/model-info`
  - `POST /api/predict`
  - `POST /api/batch-predict`
- The app currently does not yet include:
  - authentication
  - deployed BERT inference
  - automated tests
  - full monitoring/dashboard backend

## Manual App Check

- Run:
  - `conda activate cedt`
  - `pip install -r requirements.txt`
  - `uvicorn src.thai_mod_api.main:app --reload`
- Open:
  - `http://127.0.0.1:8000/`
  - `http://127.0.0.1:8000/docs`
- Verify `/api/health` returns `status`, `model_loaded`, and `cache_status`
- Test sample comments in Thai, English, and code-switching

## Conventions

- Notebooks use `cedt` kernel (Python 3.11.13)
- Dataset files are tracked with Git LFS — run `git lfs pull` after cloning
- Model artifacts go in `models/` (gitignored, not committed)
