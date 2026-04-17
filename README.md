Docs: https://docs.google.com/document/d/1SDprscNm5qjwhVCb6OyxcEs274uK9q7H2_kECSaunxk/edit?usp=sharing

## Phase 2 System

This repository now includes a small FastAPI-based system for the THAI-MOD project:

- Backend API: `src/thai_mod_api/main.py`
- Moderator UI: served at `/`

### Current Capabilities

The current app can:

- serve a moderator web UI
- serve FastAPI Swagger docs
- train or load a cached baseline toxicity model
- classify Thai, English, and code-switched text
- return `predicted_label`, `toxic_score`, `confidence`, `threshold`, and moderation recommendation
- support both single prediction and batch prediction
- expose model metadata and cache status

Current deployed model:

- `TF-IDF + Logistic Regression (Balanced)`

Current known gaps:

- basic demo authentication is implemented (single demo account)
- no deployed BERT inference in the web app yet
- no formal monitoring dashboard backend yet

### Run

```bash
conda activate cedt
pip install -r requirements.txt
uvicorn src.thai_mod_api.main:app --reload
```

Required auth environment variables:

```bash
export THAI_MOD_AUTH_USERNAME="moderator"
export THAI_MOD_AUTH_PASSWORD="thai-mod-demo-2026"
export THAI_MOD_SESSION_SECRET="set-a-random-secret-for-your-demo"
# optional: protect analyzer page "/" and prediction APIs as well
export THAI_MOD_PROTECT_ANALYZER="false"
```

The app now refuses to start if these auth values are missing.
If a project-root `.env` file exists, the app loads it automatically on startup.

Then open:

- API docs: `http://127.0.0.1:8000/docs`
- Moderator UI: `http://127.0.0.1:8000/`
- Login page: `http://127.0.0.1:8000/login`

### Quick Manual Test

1. Start the app with `uvicorn src.thai_mod_api.main:app --reload`
2. Open `http://127.0.0.1:8000/api/health`
3. Confirm the response shows:
   - `status: ok`
   - `model_loaded: true`
   - `cache_status`
4. Open `http://127.0.0.1:8000/`
5. Test example inputs such as:
   - `ขอบคุณมากครับ ช่วยได้เยอะเลย`
   - `มึงมันแย่มาก คนแบบนี้ไม่ควรอยู่ที่นี่`
   - `this comment is abusive and hateful`
   - `โคตร toxic เลย report มันไป`
6. Restart the server once and check that startup reuses the cache instead of retraining
7. Open `http://127.0.0.1:8000/admin` while logged out and confirm redirect to `/login`
8. Login with demo credentials and confirm `/admin` is accessible
9. Logout from admin page and confirm `/admin` is protected again

### API Endpoints

- `GET /`
- `GET /login`
- `GET /admin` (protected)
- `GET /api/auth/me`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/health`
- `GET /api/model-info`
- `GET /api/admin/overview` (protected)
- `POST /api/predict`
- `POST /api/batch-predict`

### Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v --cov=src/thai_mod_api --cov-report=term-missing
```

See `docs/testing-and-maintainability.md` for full details.

See `docs/auth-flow.md` for presenter-friendly login/access-control flow.

### Notes

- On first startup, the app trains a lightweight baseline model from the project datasets and caches it in `models/thai_mod_baseline.joblib` with metadata in `models/thai_mod_baseline.metadata.json`.
- The current deployment path is a practical baseline for the system demo.
- The model service is structured so a future fine-tuned WangchanBERTa artifact can replace the baseline cleanly.
