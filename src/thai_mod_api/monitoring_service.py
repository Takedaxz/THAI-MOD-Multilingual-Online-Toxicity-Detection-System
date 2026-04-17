from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class MonitoringService:
    def __init__(
        self,
        project_root: Path,
        baseline_provider: Callable[[], dict[str, Any]],
        recent_window: int = 200,
        min_records_for_drift: int = 5,
    ) -> None:
        self.baseline_provider = baseline_provider
        self.recent_window = recent_window
        self.min_records_for_drift = min_records_for_drift
        self.monitoring_dir = project_root / "models" / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.monitoring_dir / "prediction_logs.jsonl"

    def log_prediction(self, prediction: Any) -> None:
        processed_text = str(getattr(prediction, "processed_text", ""))
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "toxic_score": float(prediction.toxic_score),
            "predicted_label": str(prediction.predicted_label),
            "confidence": float(prediction.confidence),
            "threshold": float(prediction.threshold),
            "text_length": len(processed_text),
            "language_type": self.detect_language_type(processed_text),
            "source_model": str(prediction.source_model),
        }

        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_summary(self) -> dict[str, Any]:
        records = self._read_records()
        summary = self._summarize(records)
        summary["log_path"] = str(self.log_path)
        return summary

    def get_drift_report(self) -> dict[str, Any]:
        records = self._read_records()[-self.recent_window :]
        if len(records) < self.min_records_for_drift:
            return {
                "status": "insufficient_data",
                "record_count": len(records),
                "minimum_required": self.min_records_for_drift,
                "checks": {},
                "recommendation": "Collect more prediction metadata before evaluating drift.",
            }

        baseline = self.baseline_provider()
        current = self._summarize(records)
        checks = {
            "toxic_rate": self._rate_check(
                baseline.get("toxic_rate"),
                current["toxic_rate"],
                threshold=0.20,
                technique="absolute rate threshold",
            ),
            "average_text_length": self._average_check(
                baseline.get("average_text_length"),
                current["average_text_length"],
                relative_threshold=0.50,
                absolute_threshold=20.0,
                technique="KS-style length distribution proxy",
            ),
            "language_distribution": self._distribution_check(
                baseline.get("language_distribution", {}),
                current["language_distribution"],
                threshold=0.30,
                technique="chi-square-style distribution proxy",
            ),
            "uncertain_prediction_rate": self._rate_check(
                baseline.get("uncertain_prediction_rate"),
                current["uncertain_prediction_rate"],
                threshold=0.20,
                technique="confidence band threshold",
            ),
        }
        drift_detected = any(check["drift_detected"] for check in checks.values())

        return {
            "status": "warning" if drift_detected else "ok",
            "record_count": len(records),
            "baseline": baseline,
            "current": current,
            "checks": checks,
            "recommendation": (
                "Review recent samples with moderators and consider offline retraining."
                if drift_detected
                else "No drift warning from the current metadata window."
            ),
        }

    @staticmethod
    def detect_language_type(text: str) -> str:
        has_thai = any("\u0e00" <= char <= "\u0e7f" for char in text)
        has_english = any(("a" <= char.lower() <= "z") for char in text)

        if has_thai and has_english:
            return "mixed"
        if has_thai:
            return "thai"
        if has_english:
            return "english"
        return "other"

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {
                "total_predictions": 0,
                "toxic_rate": 0.0,
                "average_toxic_score": 0.0,
                "average_confidence": 0.0,
                "average_text_length": 0.0,
                "uncertain_prediction_rate": 0.0,
                "language_distribution": self._normalized_language_counts([]),
                "label_counts": {},
            }

        toxic_scores = [float(record["toxic_score"]) for record in records]
        confidences = [float(record["confidence"]) for record in records]
        text_lengths = [int(record["text_length"]) for record in records]
        labels = [str(record["predicted_label"]) for record in records]
        languages = [str(record["language_type"]) for record in records]
        uncertain = [0.4 <= score <= 0.6 for score in toxic_scores]

        return {
            "total_predictions": len(records),
            "toxic_rate": labels.count("toxic") / len(records),
            "average_toxic_score": sum(toxic_scores) / len(records),
            "average_confidence": sum(confidences) / len(records),
            "average_text_length": sum(text_lengths) / len(records),
            "uncertain_prediction_rate": sum(uncertain) / len(records),
            "language_distribution": self._normalized_language_counts(languages),
            "label_counts": dict(Counter(labels)),
        }

    @staticmethod
    def _normalized_language_counts(languages: list[str]) -> dict[str, float]:
        total = max(len(languages), 1)
        counts = Counter(languages)
        return {
            language: counts.get(language, 0) / total
            for language in ["thai", "english", "mixed", "other"]
        }

    @staticmethod
    def _rate_check(
        baseline: Any,
        current: float,
        threshold: float,
        technique: str,
    ) -> dict[str, Any]:
        if baseline is None:
            return {
                "baseline": None,
                "current": current,
                "difference": None,
                "threshold": threshold,
                "technique": technique,
                "drift_detected": False,
            }

        difference = abs(float(current) - float(baseline))
        return {
            "baseline": float(baseline),
            "current": float(current),
            "difference": difference,
            "threshold": threshold,
            "technique": technique,
            "drift_detected": difference > threshold,
        }

    @staticmethod
    def _average_check(
        baseline: Any,
        current: float,
        relative_threshold: float,
        absolute_threshold: float,
        technique: str,
    ) -> dict[str, Any]:
        if baseline in {None, 0}:
            return {
                "baseline": baseline,
                "current": current,
                "relative_difference": None,
                "absolute_difference": None,
                "technique": technique,
                "drift_detected": False,
            }

        absolute_difference = abs(float(current) - float(baseline))
        relative_difference = absolute_difference / float(baseline)
        return {
            "baseline": float(baseline),
            "current": float(current),
            "relative_difference": relative_difference,
            "absolute_difference": absolute_difference,
            "relative_threshold": relative_threshold,
            "absolute_threshold": absolute_threshold,
            "technique": technique,
            "drift_detected": (
                relative_difference > relative_threshold
                and absolute_difference > absolute_threshold
            ),
        }

    @staticmethod
    def _distribution_check(
        baseline: dict[str, float],
        current: dict[str, float],
        threshold: float,
        technique: str,
    ) -> dict[str, Any]:
        languages = ["thai", "english", "mixed", "other"]
        differences = {
            language: abs(float(current.get(language, 0.0)) - float(baseline.get(language, 0.0)))
            for language in languages
        }
        max_difference = max(differences.values()) if differences else 0.0
        return {
            "baseline": {language: float(baseline.get(language, 0.0)) for language in languages},
            "current": {language: float(current.get(language, 0.0)) for language in languages},
            "differences": differences,
            "max_difference": max_difference,
            "threshold": threshold,
            "technique": technique,
            "drift_detected": max_difference > threshold,
        }
