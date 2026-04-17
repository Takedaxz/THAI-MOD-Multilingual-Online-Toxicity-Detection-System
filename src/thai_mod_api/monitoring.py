from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from .model_service import PredictionResult, ToxicityModelService


DEFAULT_REFERENCE_BATCH_PATH = Path("datasets") / "monitoring" / "reference_batch.csv"
DEFAULT_REFERENCE_PROFILE_PATH = Path("datasets") / "monitoring" / "reference_profile.json"
DEFAULT_RECENT_REQUEST_LOG_PATH = Path("models") / "monitoring_recent_requests.jsonl"
RECENT_REQUEST_WINDOW_SIZE = 100
MIN_RECENT_REQUESTS = 20
FULL_CONFIDENCE_REQUESTS = 50
LANGUAGE_BUCKETS = ("thai_only", "english_only", "mixed_script", "other")
LANGUAGE_BUCKET_LABELS = {
    "thai_only": "Thai only",
    "english_only": "English only",
    "mixed_script": "Mixed script",
    "other": "Other",
}
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")
ENGLISH_PATTERN = re.compile(r"[A-Za-z]")


def default_reference_batch_path(project_root: Path) -> Path:
    return project_root / DEFAULT_REFERENCE_BATCH_PATH


def default_reference_profile_path(project_root: Path) -> Path:
    return project_root / DEFAULT_REFERENCE_PROFILE_PATH


def default_recent_request_log_path(project_root: Path) -> Path:
    return project_root / DEFAULT_RECENT_REQUEST_LOG_PATH


def _load_batch_file(batch_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(batch_path).copy()
    text_column = next((column for column in ("texts", "text", "comment") if column in frame.columns), None)

    if text_column is None:
        raise ValueError(
            f"Batch file `{batch_path}` must contain one of these columns: texts, text, comment."
        )

    frame = frame.dropna(subset=[text_column]).copy()
    frame[text_column] = frame[text_column].astype(str)
    frame = frame[frame[text_column].str.strip() != ""].copy()
    frame = frame.rename(columns={text_column: "texts"})

    if "source" not in frame.columns:
        frame["source"] = batch_path.name

    return frame[["texts", "source"]].reset_index(drop=True)


def _language_bucket(has_thai: bool, has_english: bool) -> str:
    if has_thai and has_english:
        return "mixed_script"
    if has_thai:
        return "thai_only"
    if has_english:
        return "english_only"
    return "other"


def _text_character_profile(text: str) -> dict[str, Any]:
    thai_chars = 0
    english_chars = 0

    for char in text:
        if THAI_PATTERN.match(char):
            thai_chars += 1
        elif ENGLISH_PATTERN.match(char):
            english_chars += 1

    tracked_chars = thai_chars + english_chars
    thai_char_ratio = thai_chars / tracked_chars if tracked_chars else 0.0
    english_char_ratio = english_chars / tracked_chars if tracked_chars else 0.0

    return {
        "thai_char_ratio": thai_char_ratio,
        "english_char_ratio": english_char_ratio,
        "language_bucket": _language_bucket(thai_chars > 0, english_chars > 0),
    }


def _language_bucket_counts(buckets: list[str]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in LANGUAGE_BUCKETS}

    for bucket in buckets:
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _language_mix(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {
        bucket: count / total
        for bucket, count in counts.items()
    }


def _score_processed_texts(service: ToxicityModelService, processed_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    service.ensure_ready()
    assert service.bundle is not None

    probabilities = service.bundle["pipeline"].predict_proba(processed_texts)[:, 1]
    threshold = float(service.bundle["default_threshold"])
    predicted_labels = (probabilities >= threshold).astype(int)
    return probabilities, predicted_labels


def _summarize_reference_batch(service: ToxicityModelService, batch_frame: pd.DataFrame) -> dict[str, Any]:
    raw_texts = batch_frame["texts"].astype(str).tolist()
    processed_texts = batch_frame["texts"].astype(str).map(service.preprocess_text).tolist()
    probabilities, predicted_labels = _score_processed_texts(service, processed_texts)
    return _summarize_processed_values(
        language_buckets=[_text_character_profile(text)["language_bucket"] for text in raw_texts],
        toxic_scores=probabilities,
        predicted_labels=predicted_labels,
        text_lengths=np.array([len(text) for text in raw_texts], dtype=float),
    )


def _summarize_recent_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    text_lengths = np.array([float(event["text_length"]) for event in events], dtype=float)
    toxic_scores = np.array([float(event["toxicity_score"]) for event in events], dtype=float)
    predicted_labels = np.array([1 if event["predicted_label"] == "toxic" else 0 for event in events], dtype=int)
    language_buckets = [str(event["language_bucket"]) for event in events]
    return _summarize_processed_values(
        language_buckets=language_buckets,
        toxic_scores=toxic_scores,
        predicted_labels=predicted_labels,
        text_lengths=text_lengths,
    )


def _summarize_processed_values(
    *,
    language_buckets: list[str],
    toxic_scores: np.ndarray,
    predicted_labels: np.ndarray,
    text_lengths: np.ndarray,
) -> dict[str, Any]:
    language_bucket_counts = _language_bucket_counts(language_buckets)
    language_mix = _language_mix(language_bucket_counts)
    return {
        "prediction_count": int(len(text_lengths)),
        "toxic_ratio": round(float(predicted_labels.mean()), 4),
        "average_toxicity_score": round(float(toxic_scores.mean()), 4),
        "average_text_length": round(float(text_lengths.mean()), 2),
        "language_mix": language_mix,
        "_language_bucket_counts": language_bucket_counts,
        "_language_mix": language_mix,
    }


def _build_language_mix_bucket_table(
    reference_counts: dict[str, int],
    current_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], float]:
    smoothing = 0.5
    reference_total = sum(reference_counts.values())
    current_total = sum(current_counts.values())
    denominator_adjustment = smoothing * len(LANGUAGE_BUCKETS)
    psi = 0.0
    rows: list[dict[str, Any]] = []

    for bucket in LANGUAGE_BUCKETS:
        reference_count = int(reference_counts.get(bucket, 0))
        current_count = int(current_counts.get(bucket, 0))
        reference_ratio = reference_count / max(reference_total, 1)
        current_ratio = current_count / max(current_total, 1)
        safe_reference = (reference_count + smoothing) / (max(reference_total, 0) + denominator_adjustment)
        safe_current = (current_count + smoothing) / (max(current_total, 0) + denominator_adjustment)
        psi += (safe_current - safe_reference) * np.log(safe_current / safe_reference)
        rows.append(
            {
                "bucket": bucket,
                "label": LANGUAGE_BUCKET_LABELS[bucket],
                "reference_ratio": round(reference_ratio, 4),
                "current_ratio": round(current_ratio, 4),
            }
        )

    return rows, round(float(psi), 4)


def classify_drift_status(psi: float) -> str:
    if psi < 0.1:
        return "healthy"
    if psi < 0.2:
        return "observe"
    if psi < 0.35:
        return "warning"
    return "degraded"


def _public_batch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    public_summary = {
        "prediction_count": summary["prediction_count"],
        "toxic_ratio": summary["toxic_ratio"],
        "average_toxicity_score": summary["average_toxicity_score"],
        "average_text_length": summary["average_text_length"],
        "language_mix": {
            bucket: round(float(ratio), 4)
            for bucket, ratio in summary["language_mix"].items()
        },
    }
    if "profile_name" in summary:
        public_summary["profile_name"] = summary["profile_name"]
    if "profile_version" in summary:
        public_summary["profile_version"] = summary["profile_version"]
    if "sample_count" in summary:
        public_summary["sample_count"] = int(summary["sample_count"])
    return public_summary


def build_reference_profile_artifact(
    service: ToxicityModelService,
    batch_frame: pd.DataFrame,
    *,
    profile_name: str,
    profile_version: str,
    generation_details: dict[str, Any],
    reference_batch_path: Path,
) -> dict[str, Any]:
    service.ensure_ready()
    summary = _summarize_reference_batch(service, batch_frame)
    model_info = service.get_model_info()
    source_counts = (
        batch_frame["source"].astype(str).value_counts().sort_index().to_dict()
        if "source" in batch_frame.columns
        else {}
    )
    language_mix = {
        bucket: round(float(ratio), 4)
        for bucket, ratio in summary["language_mix"].items()
    }
    return {
        "profile_name": profile_name,
        "profile_version": profile_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(summary["prediction_count"]),
        "prediction_count": int(summary["prediction_count"]),
        "language_mix": language_mix,
        "psi_reference_distribution": language_mix,
        "language_bucket_counts": {
            bucket: int(count)
            for bucket, count in summary["_language_bucket_counts"].items()
        },
        "average_text_length": summary["average_text_length"],
        "toxic_ratio": summary["toxic_ratio"],
        "average_toxicity_score": summary["average_toxicity_score"],
        "source_counts": source_counts,
        "reference_batch_path": str(reference_batch_path.as_posix()),
        "generation_details": generation_details,
        "model_context": {
            "model_name": model_info["model_name"],
            "deployment_mode": model_info["deployment_mode"],
            "default_threshold": model_info["default_threshold"],
        },
    }


def load_reference_profile_artifact(profile_path: Path) -> dict[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    required_keys = [
        "profile_name",
        "profile_version",
        "sample_count",
        "language_mix",
        "language_bucket_counts",
        "average_text_length",
        "toxic_ratio",
        "average_toxicity_score",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f"Reference profile `{profile_path}` is missing required keys: {', '.join(missing)}"
        )

    return {
        **payload,
        "prediction_count": int(payload.get("prediction_count", payload["sample_count"])),
        "sample_count": int(payload["sample_count"]),
        "language_mix": {
            bucket: float(payload["language_mix"].get(bucket, 0.0))
            for bucket in LANGUAGE_BUCKETS
        },
        "language_bucket_counts": {
            bucket: int(payload["language_bucket_counts"].get(bucket, 0))
            for bucket in LANGUAGE_BUCKETS
        },
        "average_text_length": round(float(payload["average_text_length"]), 2),
        "toxic_ratio": round(float(payload["toxic_ratio"]), 4),
        "average_toxicity_score": round(float(payload["average_toxicity_score"]), 4),
    }


class RecentRequestMonitor:
    def __init__(
        self,
        project_root: Path,
        *,
        recent_window_size: int = RECENT_REQUEST_WINDOW_SIZE,
        min_recent_requests: int = MIN_RECENT_REQUESTS,
        full_confidence_requests: int = FULL_CONFIDENCE_REQUESTS,
    ) -> None:
        self.project_root = project_root
        self.recent_window_size = recent_window_size
        self.min_recent_requests = min_recent_requests
        self.full_confidence_requests = full_confidence_requests
        self.reference_batch_path = default_reference_batch_path(project_root)
        self.reference_profile_path = default_reference_profile_path(project_root)
        self.log_path = default_recent_request_log_path(project_root)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._reference_profile_cache: dict[str, Any] | None = None

    def ensure_reference_profile(self, service: ToxicityModelService) -> None:
        if self._reference_profile_cache is None:
            if self.reference_profile_path.exists():
                try:
                    self._reference_profile_cache = load_reference_profile_artifact(self.reference_profile_path)
                    return
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

            reference_frame = _load_batch_file(self.reference_batch_path)
            summary = _summarize_reference_batch(service, reference_frame)
            self._reference_profile_cache = {
                **summary,
                "profile_name": "reference_batch_fallback",
                "profile_version": "fallback",
                "sample_count": int(summary["prediction_count"]),
                "language_bucket_counts": {
                    bucket: int(count)
                    for bucket, count in summary["_language_bucket_counts"].items()
                },
            }

    def _build_event(self, result: PredictionResult) -> dict[str, Any]:
        profile = _text_character_profile(result.text)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_length": len(result.text),
            "english_char_ratio": round(float(profile["english_char_ratio"]), 4),
            "language_bucket": profile["language_bucket"],
            "toxicity_score": round(float(result.toxic_score), 4),
            "predicted_label": result.predicted_label,
        }

    def record_prediction(self, result: PredictionResult) -> None:
        event = self._build_event(result)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _load_recent_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        recent_lines: deque[str] = deque(maxlen=self.recent_window_size)
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        recent_lines.append(line)

        events: list[dict[str, Any]] = []
        for line in recent_lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def clear(self) -> int:
        with self._lock:
            if not self.log_path.exists():
                return 0
            with open(self.log_path, "r", encoding="utf-8") as file:
                cleared = sum(1 for line in file if line.strip())
            self.log_path.write_text("", encoding="utf-8")
        return cleared

    def total_logged_requests(self) -> int:
        if not self.log_path.exists():
            return 0
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as file:
                return sum(1 for line in file if line.strip())

    def _monitoring_phase(self, request_count: int) -> str:
        if request_count < self.min_recent_requests:
            return "collecting_data"
        if request_count < self.full_confidence_requests:
            return "provisional"
        return "normal"

    def build_monitoring_summary(self, service: ToxicityModelService) -> dict[str, Any]:
        self.ensure_reference_profile(service)
        assert self._reference_profile_cache is not None

        events = self._load_recent_events()
        generated_at = datetime.now(timezone.utc).isoformat()
        phase = self._monitoring_phase(len(events))
        monitoring_window = {
            "prediction_count": len(events),
            "capacity": self.recent_window_size,
            "min_required": self.min_recent_requests,
            "full_confidence_required": self.full_confidence_requests,
            "total_logged_requests": self.total_logged_requests(),
            "phase": phase,
        }
        recent_summary = _summarize_recent_events(events) if events else None

        if phase == "collecting_data":
            return {
                "available": False,
                "generated_at": generated_at,
                "report_kind": "recent_live_request_monitoring",
                "reference_profile": _public_batch_summary(self._reference_profile_cache),
                "recent_live_requests": _public_batch_summary(recent_summary) if recent_summary else None,
                "monitoring_window": monitoring_window,
                "drift_analysis": {
                    "primary_feature": "language_mix",
                    "method": "population_stability_index",
                    "psi": None,
                    "status": "collecting_data",
                    "confidence": "collecting_data",
                    "bucket_table": [],
                },
                "message": (
                    f"Collect at least {self.min_recent_requests} recent requests before showing "
                    "a language-mix drift verdict."
                ),
            }

        assert recent_summary is not None
        bucket_table, psi = _build_language_mix_bucket_table(
            self._reference_profile_cache["language_bucket_counts"],
            recent_summary["_language_bucket_counts"],
        )
        status = classify_drift_status(psi)
        message = (
            f"Provisional signal from {self.min_recent_requests}-{self.full_confidence_requests - 1} "
            "recent requests."
            if phase == "provisional"
            else "Drift status from the current recent-request window."
        )

        return {
            "available": True,
            "generated_at": generated_at,
            "report_kind": "recent_live_request_monitoring",
            "reference_profile": _public_batch_summary(self._reference_profile_cache),
            "recent_live_requests": _public_batch_summary(recent_summary),
            "monitoring_window": monitoring_window,
            "drift_analysis": {
                "primary_feature": "language_mix",
                "method": "population_stability_index",
                "psi": psi,
                "status": status,
                "confidence": phase,
                "bucket_table": bucket_table,
            },
            "message": message,
        }
