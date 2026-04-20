from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .model_service import PredictionResult, ToxicityModelService
from .monitoring import RecentRequestMonitor
from .monitoring_service import MonitoringService
from .schemas import (
    AuthStatusResponse,
    BatchPredictRequest,
    BatchPredictionResponse,
    LoginRequest,
    PredictRequest,
    PredictionResponse,
    ReviewedExampleRequest,
    ReviewedExampleResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
model_service = ToxicityModelService(PROJECT_ROOT)
monitoring_service = MonitoringService(PROJECT_ROOT, model_service.get_monitoring_baseline)
request_monitor = RecentRequestMonitor(PROJECT_ROOT)
MODEL_UPDATE_LOG_DIR = PROJECT_ROOT / "models" / "model_update_jobs"
MODEL_UPDATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
REVIEWED_EXAMPLES_PATH = PROJECT_ROOT / "models" / "reviewed" / "reviewed_comments.csv"
REVIEWED_EXAMPLES_FIELDS = [
    "request_id",
    "texts",
    "category",
    "source",
    "predicted_label",
    "toxicity_score",
    "source_model",
    "reviewed_at",
]
reviewed_examples_lock = Lock()
model_update_process: subprocess.Popen | None = None
model_update_job: dict[str, Any] = {
    "status": "idle",
    "kind": None,
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "log_path": None,
    "command": None,
}


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, value.strip().strip("'\""))


_load_dotenv(PROJECT_ROOT / ".env")

PROTECT_ANALYZER = os.getenv("THAI_MOD_PROTECT_ANALYZER", "false").strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


AUTH_USERNAME = _required_env("THAI_MOD_AUTH_USERNAME")
AUTH_PASSWORD = _required_env("THAI_MOD_AUTH_PASSWORD")
SESSION_SECRET = _required_env("THAI_MOD_SESSION_SECRET")


def _session_user(request: Request) -> str | None:
    return request.session.get("username")


def _is_authenticated(request: Request) -> bool:
    return bool(_session_user(request))


def _is_safe_next_path(path: str | None) -> bool:
    if not path:
        return False
    return path.startswith("/") and not path.startswith("//")


def _admin_redirect_to_login(next_path: str = "/admin") -> RedirectResponse:
    return RedirectResponse(url=f"/login?next={next_path}", status_code=303)


def _require_api_auth(request: Request) -> None:
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")


def _record_prediction(result: PredictionResult) -> None:
    try:
        monitoring_service.log_prediction(result)
    except OSError:
        pass

    try:
        request_monitor.record_prediction(result)
    except OSError:
        pass


def _script_command(kind: str) -> list[str]:
    if kind == "train-bert-candidate":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_wangchanberta.py"),
            "--candidate",
            "--force",
        ]
    if kind == "promote-bert-candidate":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "promote_wangchanberta_candidate.py"),
        ]
    if kind == "train-lr-candidate":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lr_candidate.py"),
            "--force",
        ]
    if kind == "promote-lr-candidate":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "promote_lr_candidate.py"),
        ]
    raise ValueError(f"Unknown model update job kind: {kind}")


def _refresh_model_update_job() -> dict[str, Any]:
    global model_update_process

    if model_update_process is not None:
        returncode = model_update_process.poll()
        if returncode is not None:
            model_update_job["returncode"] = returncode
            model_update_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            model_update_job["status"] = "completed" if returncode == 0 else "failed"
            model_update_process = None

            # If failed, check whether it was a metric-check rejection (not a crash)
            if returncode != 0:
                log_rel = model_update_job.get("log_path")
                if log_rel:
                    log_path = PROJECT_ROOT / log_rel
                    try:
                        with open(log_path, "r", encoding="utf-8") as lf:
                            log_data = json.load(lf)
                        if log_data.get("promoted") is False and log_data.get("reason"):
                            model_update_job["promotion_rejected"] = True
                            model_update_job["promotion_details"] = {
                                "reason": log_data["reason"],
                                "checks": log_data.get("checks", {}),
                            }
                    except Exception:
                        model_update_job.setdefault("promotion_rejected", False)

    return model_update_job


def _start_model_update_job(kind: str) -> dict[str, Any]:
    global model_update_process

    current = _refresh_model_update_job()
    if current["status"] == "running":
        raise HTTPException(status_code=409, detail="A model update job is already running")

    command = _script_command(kind)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = MODEL_UPDATE_LOG_DIR / f"{kind}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        model_update_process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    model_update_job.update(
        {
            "status": "running",
            "kind": kind,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "returncode": None,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            "command": " ".join(command),
            "pid": model_update_process.pid,
        }
    )
    return model_update_job


def _reviewed_examples_count() -> int:
    if not REVIEWED_EXAMPLES_PATH.exists():
        return 0

    with reviewed_examples_lock:
        with open(REVIEWED_EXAMPLES_PATH, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return sum(1 for _row in reader)


def _project_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _append_reviewed_example(payload: ReviewedExampleRequest) -> dict[str, Any]:
    REVIEWED_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "request_id": payload.request_id,
        "texts": payload.text,
        "category": payload.reviewed_label,
        "source": "reviewed_traffic",
        "predicted_label": payload.predicted_label or "",
        "toxicity_score": "" if payload.toxicity_score is None else payload.toxicity_score,
        "source_model": payload.source_model or "",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    with reviewed_examples_lock:
        write_header = not REVIEWED_EXAMPLES_PATH.exists() or REVIEWED_EXAMPLES_PATH.stat().st_size == 0
        with open(REVIEWED_EXAMPLES_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REVIEWED_EXAMPLES_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    return {
        "status": "saved",
        "reviewed_count": _reviewed_examples_count(),
        "path": _project_display_path(REVIEWED_EXAMPLES_PATH),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.ensure_ready()
    app.state.model_service = model_service
    request_monitor.ensure_reference_profile(model_service)
    app.state.request_monitor = request_monitor
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
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="thai_mod_session",
    same_site="lax",
    https_only=False,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index(request: Request):
    if PROTECT_ANALYZER and not _is_authenticated(request):
        return RedirectResponse(url="/login?next=/", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
async def login_page(request: Request):
    if _is_authenticated(request):
        return RedirectResponse(url="/admin", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/admin")
async def admin(request: Request):
    if not _is_authenticated(request):
        return _admin_redirect_to_login()
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/auth/me", response_model=AuthStatusResponse)
async def auth_me(request: Request):
    username = _session_user(request)
    return {
        "authenticated": bool(username),
        "username": username,
        "protect_analyzer": PROTECT_ANALYZER,
    }


@app.post("/api/auth/login")
async def auth_login(payload: LoginRequest, request: Request):
    if payload.username != AUTH_USERNAME or payload.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    request.session["username"] = payload.username
    requested_next = payload.next_path if _is_safe_next_path(payload.next_path) else "/admin"
    return {"ok": True, "next_path": requested_next, "username": payload.username}


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


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


@app.get("/api/monitoring")
async def monitoring_summary(request: Request):
    _require_api_auth(request)
    return request_monitor.build_monitoring_summary(model_service)


@app.get("/api/monitoring/events")
async def monitoring_events(request: Request, limit: int = 20):
    _require_api_auth(request)
    return request_monitor.recent_events(limit)


@app.post("/api/monitoring/reset")
async def reset_monitoring_window(request: Request):
    _require_api_auth(request)
    cleared = request_monitor.clear()
    return {
        "status": "ok",
        "cleared_requests": cleared,
        "window_capacity": request_monitor.recent_window_size,
    }


@app.get("/api/monitoring/summary")
async def monitoring_log_summary():
    return monitoring_service.get_summary()


@app.get("/api/monitoring/drift")
async def monitoring_drift():
    return monitoring_service.get_drift_report()


@app.get("/api/admin/model-update/status")
async def model_update_status(request: Request):
    _require_api_auth(request)
    job = _refresh_model_update_job().copy()
    
    # Ensure promotion_rejected is correctly set even after server reload
    if job.get("status") == "failed" and job.get("log_path") and not job.get("promotion_rejected"):
        log_path = PROJECT_ROOT / job["log_path"]
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                log_data = json.load(lf)
            if log_data.get("promoted") is False and log_data.get("reason"):
                job["promotion_rejected"] = True
                job["promotion_details"] = {
                    "reason": log_data["reason"],
                    "checks": log_data.get("checks", {}),
                }
        except Exception:
            pass
    
    candidates = {}
    for kind, path, filename in [
        ("lr", "lr_candidate", "thai_mod_baseline.metadata.json"),
        ("bert", "wangchanberta_candidate", "metadata.json"),
    ]:
        meta_path = PROJECT_ROOT / "models" / "candidates" / path / filename
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    candidates[kind] = {
                        "trained_at": data.get("trained_at"),
                        "metrics": data.get("metrics"),
                    }
            except Exception:
                pass
                
    job["candidates"] = candidates
    return job


@app.post("/api/admin/model-update/train-candidate")
async def train_model_candidate(request: Request):
    _require_api_auth(request)
    return _start_model_update_job("train-bert-candidate")


@app.post("/api/admin/model-update/promote-candidate")
async def promote_model_candidate(request: Request):
    _require_api_auth(request)
    return _start_model_update_job("promote-bert-candidate")


@app.post("/api/admin/model-update/train-lr-candidate")
async def train_lr_model_candidate(request: Request):
    _require_api_auth(request)
    return _start_model_update_job("train-lr-candidate")


@app.post("/api/admin/model-update/promote-lr-candidate")
async def promote_lr_model_candidate(request: Request):
    _require_api_auth(request)
    return _start_model_update_job("promote-lr-candidate")


@app.get("/api/reviewed-examples/summary")
async def reviewed_examples_summary(request: Request):
    _require_api_auth(request)
    return {
        "path": _project_display_path(REVIEWED_EXAMPLES_PATH),
        "reviewed_count": _reviewed_examples_count(),
    }


@app.post("/api/reviewed-examples", response_model=ReviewedExampleResponse)
async def save_reviewed_example(payload: ReviewedExampleRequest, request: Request):
    _require_api_auth(request)
    return _append_reviewed_example(payload)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest, request: Request):
    if PROTECT_ANALYZER:
        _require_api_auth(request)
    result = model_service.predict(payload.text, payload.threshold)
    _record_prediction(result)
    return result.__dict__


@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(payload: BatchPredictRequest, request: Request):
    if PROTECT_ANALYZER:
        _require_api_auth(request)

    predictions = []
    for text in payload.texts:
        if not text.strip():
            continue
        result = model_service.predict(text, payload.threshold)
        _record_prediction(result)
        predictions.append(result.__dict__)
    return {"predictions": predictions}


@app.get("/api/admin/overview")
async def admin_overview(request: Request) -> dict[str, Any]:
    _require_api_auth(request)
    info = model_service.get_model_info()
    return {
        "health": {
            "status": "ok",
            "model_loaded": True,
            "model_name": info["model_name"],
            "deployment_mode": info["deployment_mode"],
            "cache_status": info["cache_status"],
        },
        "model_info": info,
    }
