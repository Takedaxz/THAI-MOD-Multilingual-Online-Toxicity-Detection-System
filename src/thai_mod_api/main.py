from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_service import ToxicityModelService
from .schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    PredictRequest,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
model_service = ToxicityModelService(PROJECT_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.ensure_ready()
    app.state.model_service = model_service
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


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest):
    result = model_service.predict(payload.text, payload.threshold)
    return result.__dict__


@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(payload: BatchPredictRequest):
    predictions = [
        model_service.predict(text, payload.threshold).__dict__
        for text in payload.texts
        if text.strip()
    ]
    return {"predictions": predictions}
