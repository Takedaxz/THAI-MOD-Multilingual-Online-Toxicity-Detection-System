from __future__ import annotations

from types import SimpleNamespace

from src.thai_mod_api.monitoring_service import MonitoringService


def _service(tmp_path, baseline=None, min_records_for_drift: int = 2) -> MonitoringService:
    return MonitoringService(
        project_root=tmp_path,
        baseline_provider=lambda: baseline
        or {
            "toxic_rate": 0.5,
            "average_text_length": 10.0,
            "language_distribution": {
                "thai": 0.25,
                "english": 0.25,
                "mixed": 0.25,
                "other": 0.25,
            },
            "uncertain_prediction_rate": 0.5,
        },
        min_records_for_drift=min_records_for_drift,
    )


def _prediction(text: str, score: float, label: str = "toxic") -> SimpleNamespace:
    return SimpleNamespace(
        processed_text=text,
        toxic_score=score,
        predicted_label=label,
        confidence=max(score, 1.0 - score),
        threshold=0.4,
        source_model="mock",
    )


class TestMonitoringService:
    def test_empty_summary_has_zero_counts(self, tmp_path):
        summary = _service(tmp_path).get_summary()
        assert summary["total_predictions"] == 0
        assert summary["language_distribution"] == {
            "thai": 0.0,
            "english": 0.0,
            "mixed": 0.0,
            "other": 0.0,
        }

    def test_log_prediction_updates_summary(self, tmp_path):
        service = _service(tmp_path)
        service.log_prediction(_prediction("hello", 0.75))
        service.log_prediction(_prediction("สวัสดี", 0.25, "non-toxic"))

        summary = service.get_summary()
        assert summary["total_predictions"] == 2
        assert summary["label_counts"] == {"toxic": 1, "non-toxic": 1}
        assert summary["language_distribution"]["english"] == 0.5
        assert summary["language_distribution"]["thai"] == 0.5

    def test_drift_report_waits_for_minimum_records(self, tmp_path):
        service = _service(tmp_path, min_records_for_drift=3)
        service.log_prediction(_prediction("hello", 0.75))

        report = service.get_drift_report()
        assert report["status"] == "insufficient_data"
        assert report["minimum_required"] == 3

    def test_drift_report_flags_changed_distribution(self, tmp_path):
        service = _service(
            tmp_path,
            baseline={
                "toxic_rate": 0.0,
                "average_text_length": 5.0,
                "language_distribution": {
                    "thai": 1.0,
                    "english": 0.0,
                    "mixed": 0.0,
                    "other": 0.0,
                },
                "uncertain_prediction_rate": 0.0,
            },
        )
        service.log_prediction(_prediction("hello world", 0.9))
        service.log_prediction(_prediction("another English comment", 0.9))

        report = service.get_drift_report()
        assert report["status"] == "warning"
        assert report["checks"]["language_distribution"]["drift_detected"] is True

    def test_detect_language_type_classifies_common_inputs(self):
        assert MonitoringService.detect_language_type("สวัสดี") == "thai"
        assert MonitoringService.detect_language_type("hello") == "english"
        assert MonitoringService.detect_language_type("ไทย hello") == "mixed"
        assert MonitoringService.detect_language_type("123") == "other"

    def test_invalid_log_lines_are_ignored(self, tmp_path):
        service = _service(tmp_path)
        service.log_path.write_text('{"toxic_score": 0.5}\nnot json\n', encoding="utf-8")

        assert service._read_records() == [{"toxic_score": 0.5}]

    def test_rate_check_handles_missing_baseline(self):
        check = MonitoringService._rate_check(None, 0.8, 0.2, "test")
        assert check["drift_detected"] is False
        assert check["difference"] is None

    def test_average_check_handles_zero_baseline(self):
        check = MonitoringService._average_check(0, 50.0, 0.5, 20.0, "test")
        assert check["drift_detected"] is False
        assert check["relative_difference"] is None
