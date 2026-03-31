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

- no authentication yet
- no deployed BERT inference in the web app yet
- no formal monitoring dashboard backend yet
- no automated tests yet

### Run

```bash
conda activate cedt
pip install -r requirements.txt
uvicorn src.thai_mod_api.main:app --reload
```

Then open:

- API docs: `http://127.0.0.1:8000/docs`
- Moderator UI: `http://127.0.0.1:8000/`

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

### API Endpoints

- `GET /`
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`
- `POST /api/batch-predict`

### Notes

- On first startup, the app trains a lightweight baseline model from the project datasets and caches it in `models/thai_mod_baseline.joblib` with metadata in `models/thai_mod_baseline.metadata.json`.
- The current deployment path is a practical baseline for the system demo.
- The model service is structured so a future fine-tuned WangchanBERTa artifact can replace the baseline cleanly.
