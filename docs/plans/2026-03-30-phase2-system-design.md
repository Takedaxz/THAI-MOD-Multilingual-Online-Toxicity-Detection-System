# Phase 2 System Design

This Phase 2 implementation turns THAI-MOD from a notebook-first ML project into a small working system with two parts:

1. A `FastAPI` backend for inference and model metadata.
2. A lightweight moderator-facing web UI served by the same app.

The design follows `docs/proposal.txt` and `docs/progress/progress2.txt`, especially the documented inference flow:

1. Receive a Thai, English, or code-switched comment.
2. Apply the same preprocessing used in training.
3. Generate a toxicity score.
4. Convert the score into a moderation recommendation for a human reviewer.

Because the repository does not currently contain a saved fine-tuned transformer artifact that can be deployed directly, the system uses a practical deployment baseline:

- Train a TF-IDF + Logistic Regression classifier from the 8 project datasets.
- Cache the trained artifact in `models/` so later runs start quickly.
- Keep the app structure ready for a future WangchanBERTa deployment by isolating model logic inside a dedicated service.

## Backend

The backend exposes:

- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`
- `POST /api/batch-predict`

It also serves the frontend at `/`.

## Frontend

The frontend is a simple moderator console with:

- text box for single-message analysis
- threshold control
- sample toxic / non-toxic inputs
- score, label, confidence, and recommendation cards
- recent prediction history

## Decision Policy

The system is framed as decision support, not automatic punishment:

- If score is above threshold: `FLAG_FOR_REVIEW`
- Otherwise: `ALLOW`

This matches the project’s emphasis on recall and human-in-the-loop moderation.

