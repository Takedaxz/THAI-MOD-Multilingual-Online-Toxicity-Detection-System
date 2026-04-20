from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.thai_mod_api.model_service import PredictionResult
from src.thai_mod_api.monitoring import (
    DEFAULT_REFERENCE_BATCH_PATH,
    RecentRequestMonitor,
    _build_language_mix_bucket_table,
    _load_batch_file,
    _public_batch_summary,
    build_reference_profile_artifact,
    classify_drift_status,
    default_recent_request_log_path,
    default_reference_batch_path,
    default_reference_profile_path,
    load_reference_profile_artifact,
)


class FakePipeline:
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            toxic_probability = 0.8 if "toxic" in text else 0.2
            rows.append([1.0 - toxic_probability, toxic_probability])
        return np.array(rows)


class FakeService:
    def __init__(self) -> None:
        self.bundle = {
            "pipeline": FakePipeline(),
            "default_threshold": 0.4,
        }

    def ensure_ready(self) -> None:
        return None

    def preprocess_text(self, text: str) -> str:
        return str(text).lower().strip()

    def get_model_info(self) -> dict:
        return {
            "model_name": "fake-model",
            "deployment_mode": "test",
            "default_threshold": 0.4,
        }


def _write_reference_batch(project_root, rows: list[tuple[str, str]] | None = None) -> None:
    batch_path = default_reference_batch_path(project_root)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        ("สวัสดี", "thai"),
        ("hello toxic", "english"),
        ("ไทย hello", "mixed"),
        ("123", "other"),
    ]
    pd.DataFrame(rows, columns=["texts", "source"]).to_csv(batch_path, index=False)


def _write_reference_profile(project_root, payload: dict) -> None:
    profile_path = default_reference_profile_path(project_root)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload), encoding="utf-8")


def _profile_payload(**overrides) -> dict:
    payload = {
        "profile_name": "test_profile",
        "profile_version": "v1",
        "sample_count": 4,
        "prediction_count": 4,
        "language_mix": {
            "thai_only": 0.25,
            "english_only": 0.25,
            "mixed_script": 0.25,
            "other": 0.25,
        },
        "language_bucket_counts": {
            "thai_only": 1,
            "english_only": 1,
            "mixed_script": 1,
            "other": 1,
        },
        "average_text_length": 5.0,
        "toxic_ratio": 0.25,
        "average_toxicity_score": 0.35,
    }
    payload.update(overrides)
    return payload


def _prediction(
    text: str,
    processed_text: str | None = None,
    label: str = "non-toxic",
    score: float = 0.2,
) -> PredictionResult:
    return PredictionResult(
        request_id="test-request-id",
        text=text,
        processed_text=processed_text or text.lower(),
        predicted_label=label,
        toxic_score=score,
        confidence=max(score, 1.0 - score),
        threshold=0.4,
        recommendation="FLAG_FOR_REVIEW" if label == "toxic" else "ALLOW",
        source_model="fake-model",
    )


class TestReferenceProfileHelpers:
    def test_default_paths_are_project_relative(self, tmp_path):
        assert default_reference_batch_path(tmp_path) == tmp_path / DEFAULT_REFERENCE_BATCH_PATH
        assert default_reference_profile_path(tmp_path).name == "reference_profile.json"
        assert default_recent_request_log_path(tmp_path).name == "monitoring_recent_requests.jsonl"

    def test_load_batch_file_accepts_comment_column_and_adds_source(self, tmp_path):
        batch_path = tmp_path / "batch.csv"
        batch_path.write_text("comment\nhello\n\nสวัสดี\n", encoding="utf-8")

        frame = _load_batch_file(batch_path)

        assert frame.to_dict("records") == [
            {"texts": "hello", "source": "batch.csv"},
            {"texts": "สวัสดี", "source": "batch.csv"},
        ]

    def test_load_batch_file_rejects_missing_text_column(self, tmp_path):
        batch_path = tmp_path / "bad.csv"
        batch_path.write_text("body\nhello\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must contain"):
            _load_batch_file(batch_path)

    def test_build_and_load_reference_profile_artifact(self, tmp_path):
        service = FakeService()
        frame = pd.DataFrame(
            {
                "texts": ["สวัสดี 😊", "hello toxic", "ไทย hello", "123"],
                "source": ["a.csv", "a.csv", "b.csv", "b.csv"],
            }
        )

        artifact = build_reference_profile_artifact(
            service,
            frame,
            profile_name="profile",
            profile_version="v1",
            generation_details={"seed": 42},
            reference_batch_path=DEFAULT_REFERENCE_BATCH_PATH,
        )

        assert artifact["sample_count"] == 4
        assert artifact["language_bucket_counts"]["thai_only"] == 1
        assert artifact["language_bucket_counts"]["mixed_script"] == 1
        assert artifact["source_counts"] == {"a.csv": 2, "b.csv": 2}
        assert artifact["reference_batch_path"] == "datasets/monitoring/reference_batch.csv"

        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_reference_profile_artifact(profile_path)
        assert loaded["profile_name"] == "profile"
        assert loaded["sample_count"] == 4
        assert loaded["language_mix"]["other"] == 0.25

    def test_load_reference_profile_rejects_missing_required_keys(self, tmp_path):
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps({"profile_name": "bad"}), encoding="utf-8")

        with pytest.raises(ValueError, match="missing required keys"):
            load_reference_profile_artifact(profile_path)

    def test_public_summary_keeps_profile_metadata_when_present(self):
        public = _public_batch_summary(_profile_payload())

        assert public["profile_name"] == "test_profile"
        assert public["profile_version"] == "v1"
        assert public["sample_count"] == 4

    def test_language_mix_psi_and_status_thresholds(self):
        rows, psi = _build_language_mix_bucket_table(
            {"thai_only": 10, "english_only": 0, "mixed_script": 0, "other": 0},
            {"thai_only": 0, "english_only": 10, "mixed_script": 0, "other": 0},
        )

        assert len(rows) == 4
        assert rows[0]["label"] == "Thai only"
        assert psi > 0.35
        assert classify_drift_status(0.05) == "healthy"
        assert classify_drift_status(0.15) == "observe"
        assert classify_drift_status(0.25) == "warning"
        assert classify_drift_status(0.5) == "degraded"


class TestRecentRequestMonitor:
    def test_event_uses_raw_text_for_language_profile(self, tmp_path):
        monitor = RecentRequestMonitor(tmp_path)
        result = _prediction("สวัสดี 😊", processed_text="สวัสดี :smiling_face:")

        event = monitor._build_event(result)

        assert event["language_bucket"] == "thai_only"
        assert event["english_char_ratio"] == 0.0
        assert event["text_length"] == len("สวัสดี 😊")

    def test_record_loads_recent_window_and_ignores_malformed_lines(self, tmp_path):
        monitor = RecentRequestMonitor(tmp_path, recent_window_size=2)
        monitor.record_prediction(_prediction("hello", label="toxic", score=0.8))
        monitor.record_prediction(_prediction("สวัสดี"))
        monitor.record_prediction(_prediction("ไทย hello"))
        with open(monitor.log_path, "a", encoding="utf-8") as file:
            file.write("not-json\n")

        events = monitor._load_recent_events()

        assert len(events) == 1
        assert events[0]["language_bucket"] == "mixed_script"

    def test_clear_counts_and_truncates_log(self, tmp_path):
        monitor = RecentRequestMonitor(tmp_path)
        assert monitor.clear() == 0

        monitor.record_prediction(_prediction("hello"))
        monitor.record_prediction(_prediction("สวัสดี"))

        assert monitor.total_logged_requests() == 2
        assert monitor.clear() == 2
        assert monitor.total_logged_requests() == 0

    def test_recent_events_returns_newest_first_without_raw_text(self, tmp_path):
        monitor = RecentRequestMonitor(tmp_path)
        monitor.record_prediction(_prediction("hello"))
        monitor.record_prediction(_prediction("สวัสดี", label="toxic", score=0.9))

        payload = monitor.recent_events(limit=1)

        assert payload["log_path"] == "models/monitoring_recent_requests.jsonl"
        assert payload["total_logged_requests"] == 2
        assert payload["returned_count"] == 1
        assert payload["events"][0]["language_bucket"] == "thai_only"
        assert payload["events"][0]["predicted_label"] == "toxic"
        assert "text" not in payload["events"][0]

    def test_ensure_reference_profile_loads_json_profile(self, tmp_path):
        _write_reference_profile(tmp_path, _profile_payload(profile_name="json_profile"))
        monitor = RecentRequestMonitor(tmp_path)

        monitor.ensure_reference_profile(FakeService())

        assert monitor._reference_profile_cache["profile_name"] == "json_profile"

    def test_ensure_reference_profile_falls_back_when_json_is_invalid(self, tmp_path):
        _write_reference_profile(tmp_path, {"profile_name": "broken"})
        _write_reference_batch(tmp_path)
        monitor = RecentRequestMonitor(tmp_path)

        monitor.ensure_reference_profile(FakeService())

        assert monitor._reference_profile_cache["profile_name"] == "reference_batch_fallback"
        assert monitor._reference_profile_cache["sample_count"] == 4

    def test_collecting_summary_includes_recent_metrics_before_drift(self, tmp_path):
        _write_reference_profile(tmp_path, _profile_payload())
        monitor = RecentRequestMonitor(tmp_path, min_recent_requests=2, full_confidence_requests=3)
        monitor.record_prediction(_prediction("hello toxic", label="toxic", score=0.8))

        summary = monitor.build_monitoring_summary(FakeService())

        assert summary["available"] is False
        assert summary["monitoring_window"]["phase"] == "collecting_data"
        assert summary["recent_live_requests"]["prediction_count"] == 1
        assert summary["drift_analysis"]["psi"] is None

    def test_provisional_and_normal_summaries_compute_drift(self, tmp_path):
        _write_reference_profile(
            tmp_path,
            _profile_payload(
                language_bucket_counts={
                    "thai_only": 4,
                    "english_only": 0,
                    "mixed_script": 0,
                    "other": 0,
                },
                language_mix={
                    "thai_only": 1.0,
                    "english_only": 0.0,
                    "mixed_script": 0.0,
                    "other": 0.0,
                },
            ),
        )
        monitor = RecentRequestMonitor(tmp_path, min_recent_requests=2, full_confidence_requests=3)
        monitor.record_prediction(_prediction("hello toxic", label="toxic", score=0.8))
        monitor.record_prediction(_prediction("another toxic", label="toxic", score=0.8))

        provisional = monitor.build_monitoring_summary(FakeService())
        assert provisional["available"] is True
        assert provisional["drift_analysis"]["confidence"] == "provisional"
        assert provisional["drift_analysis"]["bucket_table"]

        monitor.record_prediction(_prediction("more toxic", label="toxic", score=0.8))
        normal = monitor.build_monitoring_summary(FakeService())
        assert normal["monitoring_window"]["phase"] == "normal"
        assert normal["drift_analysis"]["status"] == "degraded"

    def test_monitoring_phase_boundaries(self, tmp_path):
        monitor = RecentRequestMonitor(tmp_path, min_recent_requests=2, full_confidence_requests=4)

        assert monitor._monitoring_phase(1) == "collecting_data"
        assert monitor._monitoring_phase(2) == "provisional"
        assert monitor._monitoring_phase(4) == "normal"
