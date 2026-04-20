"""
Shared pytest fixtures for THAI-MOD tests.

The key challenge: ToxicityModelService.ensure_ready() loads or trains
from disk — which requires the 8 dataset CSVs and is too slow for CI.

Strategy (Option A from the implementation plan):
  - Pre-set model_service.bundle on the module-level instance in main.py
    before TestClient is created.
  - When lifespan runs ensure_ready(), it sees bundle != None and skips loading.
  - predict() and get_model_info() use the injected bundle transparently.

The mock pipeline returns a fixed toxic_score so assertions are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def _make_mock_pipeline(toxic_score: float = 0.75) -> MagicMock:
    """Return a mock sklearn pipeline whose predict_proba is deterministic."""
    pipeline = MagicMock()
    # predict_proba([text]) returns shape (1, 2): [non-toxic prob, toxic prob]
    pipeline.predict_proba.return_value = np.array(
        [[1.0 - toxic_score, toxic_score]]
    )
    return pipeline


# Realistic mock bundle mirroring the actual bundle structure in model_service.py.
# Values are taken from the baseline results reported in progress1.txt §5.
MOCK_BUNDLE: dict = {
    "pipeline": _make_mock_pipeline(toxic_score=0.75),
    "model_name": "TF-IDF + Logistic Regression (Balanced)",
    "deployment_mode": "cached_startup_baseline",
    "default_threshold": 0.4,
    "trained_at": "2026-01-01T00:00:00+00:00",
    "dataset_rows": 233931,
    "dataset_sources": ["dataset1.csv", "dataset2.csv"],
    "metrics": {
        "accuracy": 0.8287,
        "precision": 0.6420,
        "recall": 0.7758,
        "f1_score": 0.7026,
        "f2_score": 0.7448,
        "confusion_matrix": [[12000, 3000], [2000, 6000]],
        "test_size": 23000,
    },
    "cache_status": "loaded_from_cache",
}


@pytest.fixture(scope="session")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """
    Provide a FastAPI TestClient with the model bundle pre-injected.

    scope="session" — one client for the whole test run (fast and stateless).
    The mock bundle is reset to None after the session so nothing leaks.
    """
    import src.thai_mod_api.main as main_module  # noqa: PLC0415

    # Inject bundle BEFORE TestClient enters context so ensure_ready() is a no-op
    main_module.model_service.bundle = MOCK_BUNDLE
    monitoring_dir = tmp_path_factory.mktemp("monitoring")
    main_module.monitoring_service.monitoring_dir = monitoring_dir
    main_module.monitoring_service.log_path = monitoring_dir / "prediction_logs.jsonl"
    main_module.request_monitor.project_root = monitoring_dir
    main_module.request_monitor.log_path = monitoring_dir / "monitoring_recent_requests.jsonl"
    main_module.request_monitor.log_path.write_text("", encoding="utf-8")
    main_module.REVIEWED_EXAMPLES_PATH = monitoring_dir / "reviewed_comments.csv"

    with TestClient(main_module.app) as client:
        yield client

    # Cleanup — restore default state
    main_module.model_service.bundle = None
