from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_service import ToxicityModelService
from .monitoring_service import MonitoringService
from .schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    PredictRequest,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
model_service = ToxicityModelService(PROJECT_ROOT)
monitoring_service = MonitoringService(PROJECT_ROOT, model_service.get_monitoring_baseline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.ensure_ready()
    app.state.model_service = model_service
    app.state.monitoring_service = monitoring_service
    yield


app = FastAPI(
    title="THAI-MOD API",
    description="Phase 2 toxicity detection API and moderator UI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
async def admin():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/health")
async def health():
    info = model_service.get_model_info()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": info["model_name"],
        "deployment_mode": info["deployment_mode"],
        "cache_status": info["cache_status"],
    }


@app.get("/api/model-info")
async def model_info():
    return model_service.get_model_info()


@app.get("/api/monitoring/summary")
async def monitoring_summary():
    return monitoring_service.get_summary()


@app.get("/api/monitoring/drift")
async def monitoring_drift():
    return monitoring_service.get_drift_report()


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest):
    result = model_service.predict(payload.text, payload.threshold)
    monitoring_service.log_prediction(result)
    return result.__dict__


@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(payload: BatchPredictRequest):
    predictions = []
    for text in payload.texts:
        if not text.strip():
            continue
        result = model_service.predict(text, payload.threshold)
        monitoring_service.log_prediction(result)
        predictions.append(result.__dict__)
    return {"predictions": predictions}
