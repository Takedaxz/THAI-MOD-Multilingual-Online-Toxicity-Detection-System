from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .model_service import ToxicityModelService
from .schemas import (
    AuthStatusResponse,
    BatchPredictRequest,
    BatchPredictionResponse,
    LoginRequest,
    PredictRequest,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
model_service = ToxicityModelService(PROJECT_ROOT)


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


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictRequest, request: Request):
    if PROTECT_ANALYZER:
        _require_api_auth(request)
    result = model_service.predict(payload.text, payload.threshold)
    return result.__dict__


@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(payload: BatchPredictRequest, request: Request):
    if PROTECT_ANALYZER:
        _require_api_auth(request)
    predictions = [
        model_service.predict(text, payload.threshold).__dict__
        for text in payload.texts
        if text.strip()
    ]
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
