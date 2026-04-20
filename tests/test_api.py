"""
Integration tests for the THAI-MOD FastAPI endpoints.

Uses FastAPI TestClient backed by a mock model bundle (see conftest.py).
No real datasets are loaded and no model training occurs — the mock
pipeline returns a deterministic toxic_score of 0.75.

Endpoints under test:
  GET  /api/health        → status, model_loaded, cache_status
  POST /api/predict       → full PredictionResponse

Coverage rationale (progress2.txt §7 Deployment):
  - Latency target <200 ms per request → response must be well-formed
  - Decision policy: score >= threshold → FLAG_FOR_REVIEW, else ALLOW
  - Threshold default = 0.4 (from model_service.py)
  - Recall-oriented: mock score 0.75 > 0.4 → "toxic" / FLAG_FOR_REVIEW
"""

from __future__ import annotations

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, api_client: TestClient):
        response = api_client.get("/api/health")
        assert response.status_code == 200

    def test_status_field_is_ok(self, api_client: TestClient):
        data = api_client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_model_loaded_is_true(self, api_client: TestClient):
        """model_loaded must always be true when the bundle is set."""
        data = api_client.get("/api/health").json()
        assert data["model_loaded"] is True

    def test_cache_status_field_present(self, api_client: TestClient):
        """cache_status distinguishes trained_and_cached vs loaded_from_cache."""
        data = api_client.get("/api/health").json()
        assert "cache_status" in data

    def test_model_name_field_present(self, api_client: TestClient):
        data = api_client.get("/api/health").json()
        assert "model_name" in data

    def test_deployment_mode_field_present(self, api_client: TestClient):
        data = api_client.get("/api/health").json()
        assert "deployment_mode" in data


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Validate response schema, field ranges, and decision-policy logic."""

    # --- Basic contract ---

    def test_returns_200_for_valid_text(self, api_client: TestClient):
        response = api_client.post("/api/predict", json={"text": "สวัสดีครับ"})
        assert response.status_code == 200

    def test_response_contains_all_required_fields(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        required = {
            "request_id",
            "text",
            "processed_text",
            "predicted_label",
            "toxic_score",
            "confidence",
            "threshold",
            "recommendation",
            "source_model",
        }
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )

    def test_request_id_is_non_empty(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert data["request_id"]

    # --- Value ranges ---

    def test_toxic_score_between_0_and_1(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert 0.0 <= data["toxic_score"] <= 1.0

    def test_confidence_between_0_and_1(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    # --- Decision policy (progress2.txt §2.2 Inference Pipeline) ---

    def test_predicted_label_is_valid_value(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert data["predicted_label"] in ("toxic", "non-toxic")

    def test_recommendation_is_valid_value(self, api_client: TestClient):
        """Only two recommendations exist: FLAG_FOR_REVIEW or ALLOW."""
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert data["recommendation"] in ("FLAG_FOR_REVIEW", "ALLOW")

    def test_high_score_yields_toxic_label(self, api_client: TestClient):
        """Mock pipeline returns 0.75 > default threshold 0.4 → toxic."""
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert data["predicted_label"] == "toxic"
        assert data["recommendation"] == "FLAG_FOR_REVIEW"

    def test_low_threshold_still_flags_high_score(self, api_client: TestClient):
        """Score 0.75 >= threshold 0.1 → still toxic."""
        data = api_client.post(
            "/api/predict", json={"text": "test", "threshold": 0.1}
        ).json()
        assert data["predicted_label"] == "toxic"

    def test_very_high_threshold_non_toxic(self, api_client: TestClient):
        """Score 0.75 < threshold 0.95 → non-toxic."""
        data = api_client.post(
            "/api/predict", json={"text": "test", "threshold": 0.95}
        ).json()
        assert data["predicted_label"] == "non-toxic"
        assert data["recommendation"] == "ALLOW"

    def test_custom_threshold_reflected_in_response(self, api_client: TestClient):
        data = api_client.post(
            "/api/predict", json={"text": "test", "threshold": 0.3}
        ).json()
        assert abs(data["threshold"] - 0.3) < 0.01

    def test_source_model_is_non_empty(self, api_client: TestClient):
        data = api_client.post("/api/predict", json={"text": "test"}).json()
        assert data["source_model"] and data["source_model"] != ""

    # --- Input validation (Pydantic, schemas.py) ---

    def test_empty_text_returns_422(self, api_client: TestClient):
        """min_length=1 on PredictRequest.text must reject empty strings."""
        response = api_client.post("/api/predict", json={"text": ""})
        assert response.status_code == 422

    def test_missing_text_field_returns_422(self, api_client: TestClient):
        response = api_client.post("/api/predict", json={})
        assert response.status_code == 422

    def test_threshold_above_1_returns_422(self, api_client: TestClient):
        """threshold field has le=1.0 constraint."""
        response = api_client.post(
            "/api/predict", json={"text": "test", "threshold": 1.5}
        )
        assert response.status_code == 422

    def test_threshold_below_0_returns_422(self, api_client: TestClient):
        """threshold field has ge=0.0 constraint."""
        response = api_client.post(
            "/api/predict", json={"text": "test", "threshold": -0.1}
        )
        assert response.status_code == 422

    # --- Multilingual behaviour ---

    def test_thai_only_text_processed(self, api_client: TestClient):
        data = api_client.post(
            "/api/predict", json={"text": "ขอบคุณมากครับ ช่วยได้เยอะเลย"}
        ).json()
        assert data["predicted_label"] in ("toxic", "non-toxic")

    def test_english_only_text_processed(self, api_client: TestClient):
        data = api_client.post(
            "/api/predict", json={"text": "this comment is abusive and hateful"}
        ).json()
        assert data["predicted_label"] in ("toxic", "non-toxic")

    def test_code_switched_text_processed(self, api_client: TestClient):
        """Thai-English code-switching (progress2.txt §6.1 §3)."""
        data = api_client.post(
            "/api/predict", json={"text": "โคตร toxic เลย report มันไป"}
        ).json()
        assert data["predicted_label"] in ("toxic", "non-toxic")

    def test_original_text_preserved_in_response(self, api_client: TestClient):
        original = "สวัสดีครับ Hello"
        data = api_client.post("/api/predict", json={"text": original}).json()
        assert data["text"] == original


class TestAuthEndpoints:
    """Exercise the demo session-auth flow added for the moderator UI."""

    def test_me_reports_unauthenticated_after_logout(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        data = api_client.get("/api/auth/me").json()
        assert data == {
            "authenticated": False,
            "username": None,
            "protect_analyzer": False,
        }

    def test_admin_redirects_when_logged_out(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.get("/admin", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login?next=/admin"

    def test_login_rejects_bad_credentials(self, api_client: TestClient):
        response = api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_accepts_demo_credentials_and_sanitizes_next(self, api_client: TestClient):
        response = api_client.post(
            "/api/auth/login",
            json={
                "username": "moderator",
                "password": "thai-mod-demo-2026",
                "next_path": "//evil.example/path",
            },
        )
        assert response.status_code == 200
        assert response.json()["next_path"] == "/admin"

    def test_admin_overview_requires_login(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.get("/api/admin/overview")
        assert response.status_code == 401

    def test_admin_overview_returns_model_info_after_login(self, api_client: TestClient):
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )
        data = api_client.get("/api/admin/overview").json()
        assert data["health"]["status"] == "ok"
        assert data["model_info"]["model_name"]

    def test_model_update_status_requires_login(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.get("/api/admin/model-update/status")
        assert response.status_code == 401

    def test_model_update_train_endpoint_starts_script_job(self, api_client: TestClient, monkeypatch):
        import src.thai_mod_api.main as main_module

        def fake_start(kind: str):
            return {
                "status": "running",
                "kind": kind,
                "log_path": "models/model_update_jobs/fake.log",
            }

        monkeypatch.setattr(main_module, "_start_model_update_job", fake_start)
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )

        data = api_client.post("/api/admin/model-update/train-candidate").json()

        assert data["status"] == "running"
        assert data["kind"] == "train-bert-candidate"

    def test_model_update_promote_endpoint_starts_script_job(self, api_client: TestClient, monkeypatch):
        import src.thai_mod_api.main as main_module

        def fake_start(kind: str):
            return {
                "status": "running",
                "kind": kind,
                "log_path": "models/model_update_jobs/fake.log",
            }

        monkeypatch.setattr(main_module, "_start_model_update_job", fake_start)
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )

        data = api_client.post("/api/admin/model-update/promote-candidate").json()

        assert data["status"] == "running"
        assert data["kind"] == "promote-bert-candidate"

    def test_model_update_lr_train_endpoint_starts_script_job(self, api_client: TestClient, monkeypatch):
        import src.thai_mod_api.main as main_module

        def fake_start(kind: str):
            return {
                "status": "running",
                "kind": kind,
                "log_path": "models/model_update_jobs/fake.log",
            }

        monkeypatch.setattr(main_module, "_start_model_update_job", fake_start)
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )

        data = api_client.post("/api/admin/model-update/train-lr-candidate").json()

        assert data["status"] == "running"
        assert data["kind"] == "train-lr-candidate"

    def test_model_update_lr_promote_endpoint_starts_script_job(self, api_client: TestClient, monkeypatch):
        import src.thai_mod_api.main as main_module

        def fake_start(kind: str):
            return {
                "status": "running",
                "kind": kind,
                "log_path": "models/model_update_jobs/fake.log",
            }

        monkeypatch.setattr(main_module, "_start_model_update_job", fake_start)
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )

        data = api_client.post("/api/admin/model-update/promote-lr-candidate").json()

        assert data["status"] == "running"
        assert data["kind"] == "promote-lr-candidate"

    def test_protected_predict_requires_auth_when_enabled(self, api_client: TestClient):
        import src.thai_mod_api.main as main_module

        api_client.post("/api/auth/logout")
        original = main_module.PROTECT_ANALYZER
        main_module.PROTECT_ANALYZER = True
        try:
            response = api_client.post("/api/predict", json={"text": "test"})
        finally:
            main_module.PROTECT_ANALYZER = original

        assert response.status_code == 401


class TestReviewedExamples:
    def test_reviewed_examples_summary_requires_login(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.get("/api/reviewed-examples/summary")
        assert response.status_code == 401

    def test_save_reviewed_example_requires_login(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.post(
            "/api/reviewed-examples",
            json={
                "request_id": "abc",
                "text": "example",
                "reviewed_label": "neg",
            },
        )
        assert response.status_code == 401

    def test_save_reviewed_example_updates_summary(self, api_client: TestClient):
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )
        prediction = api_client.post("/api/predict", json={"text": "อี loser"}).json()

        response = api_client.post(
            "/api/reviewed-examples",
            json={
                "request_id": prediction["request_id"],
                "text": prediction["text"],
                "reviewed_label": "neg",
                "predicted_label": prediction["predicted_label"],
                "toxicity_score": prediction["toxic_score"],
                "source_model": prediction["source_model"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"
        assert data["reviewed_count"] >= 1

        summary = api_client.get("/api/reviewed-examples/summary").json()
        assert summary["reviewed_count"] == data["reviewed_count"]


class TestMonitoringEndpoints:
    """Cover monitoring endpoints without relying on persistent local logs."""

    def test_monitoring_summary_shape(self, api_client: TestClient):
        data = api_client.get("/api/monitoring/summary").json()
        assert "total_predictions" in data
        assert "language_distribution" in data
        assert "log_path" in data

    def test_monitoring_drift_report_shape(self, api_client: TestClient):
        for text in ["hello", "สวัสดี", "ไทย toxic", "123", "another comment"]:
            api_client.post("/api/predict", json={"text": text})

        data = api_client.get("/api/monitoring/drift").json()
        assert data["status"] in ("ok", "warning", "insufficient_data")
        assert "checks" in data

    def test_monitoring_events_requires_auth(self, api_client: TestClient):
        api_client.post("/api/auth/logout")
        response = api_client.get("/api/monitoring/events")
        assert response.status_code == 401

    def test_monitoring_events_shows_recent_metadata_after_login(self, api_client: TestClient):
        api_client.post(
            "/api/auth/login",
            json={"username": "moderator", "password": "thai-mod-demo-2026"},
        )
        api_client.post("/api/predict", json={"text": "ไทย toxic"})

        data = api_client.get("/api/monitoring/events?limit=1").json()

        assert data["returned_count"] == 1
        assert data["events"][0]["language_bucket"] == "mixed_script"
        assert "toxicity_score" in data["events"][0]
        assert "text" not in data["events"][0]
