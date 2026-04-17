from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModelForSequenceClassification, CamembertTokenizer

from .config import load_env_file
from .text_processing import preprocess_text

PREPROCESSING_VERSION = "shared_preprocess_text_v1"
LR_MODEL_CONFIG_VERSION = "lr_word_char_tfidf_v2"
LR_MODEL_NAME = "Word + Character TF-IDF + Logistic Regression (Balanced)"
LR_WORD_MAX_FEATURES = 30_000
LR_CHAR_MAX_FEATURES = 30_000
TOXIC_KEYWORD_MIN_SCORE = 0.65
TOXIC_KEYWORD_MAX_CHARS = 20


@dataclass
class PredictionResult:
    text: str
    processed_text: str
    predicted_label: str
    toxic_score: float
    confidence: float
    threshold: float
    recommendation: str
    source_model: str


class ToxicityModelService:
    def __init__(self, project_root: Path, default_threshold: float = 0.4) -> None:
        self.project_root = project_root
        self.default_threshold = default_threshold
        load_env_file(project_root / ".env")
        self.model_backend = self._load_model_backend()
        self.logger = logging.getLogger("uvicorn.error")
        self.dataset_files = [project_root / "datasets" / f"dataset{i}.csv" for i in range(1, 9)]
        self.model_dir = project_root / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.transformer_dir = self.model_dir / "wangchanberta_finetuned"
        self.transformer_metadata_path = self.transformer_dir / "metadata.json"
        self.model_path = self.model_dir / "thai_mod_baseline.joblib"
        self.metadata_path = self.model_dir / "thai_mod_baseline.metadata.json"
        self.toxic_keywords_path = project_root / "datasets" / "toxic_keywords.csv"
        self.toxic_keywords = self._load_toxic_keywords()
        self.bundle: dict[str, Any] | None = None
        self.monitoring_baseline: dict[str, Any] | None = None

    def ensure_ready(self) -> None:
        if self.bundle is None:
            self.logger.info("THAI-MOD model backend setting: %s", self.model_backend)
            self.bundle = self._load_or_train()
            self.logger.info(
                "THAI-MOD loaded model: %s (%s, cache_status=%s)",
                self.bundle["model_name"],
                self.bundle["deployment_mode"],
                self.bundle.get("cache_status", "unknown"),
            )

    def _load_or_train(self) -> dict[str, Any]:
        if self.model_backend in {"auto", "bert"}:
            transformer_bundle = self._load_transformer_bundle()
            if transformer_bundle is not None:
                transformer_bundle["selected_backend"] = self.model_backend
                return transformer_bundle
            if self.model_backend == "bert":
                raise RuntimeError(
                    "THAI_MOD_MODEL_BACKEND=bert was requested, but no valid WangchanBERTa artifact was found "
                    f"at {self.transformer_dir}."
                )

        if self.model_backend == "lr":
            lr_bundle = self._load_or_train_lr_bundle()
            lr_bundle["selected_backend"] = self.model_backend
            return lr_bundle

        lr_bundle = self._load_or_train_lr_bundle()
        lr_bundle["selected_backend"] = self.model_backend
        return lr_bundle

    @staticmethod
    def _load_model_backend() -> str:
        backend = os.getenv("THAI_MOD_MODEL_BACKEND", "auto").strip().lower()
        if backend not in {"auto", "lr", "bert"}:
            raise ValueError(
                "THAI_MOD_MODEL_BACKEND must be one of: auto, lr, bert."
            )
        return backend

    def _load_or_train_lr_bundle(self) -> dict[str, Any]:
        if self.model_path.exists() and self.metadata_path.exists():
            try:
                metadata = self._load_metadata()
                if metadata.get("model_config_version") != LR_MODEL_CONFIG_VERSION:
                    self.logger.info(
                        "Cached LR model config is stale (%s); retraining with %s.",
                        metadata.get("model_config_version", "unknown"),
                        LR_MODEL_CONFIG_VERSION,
                    )
                else:
                    pipeline = joblib.load(self.model_path)
                    self.logger.info("Loaded LR model from cache; MLflow training run was not created.")
                    return {
                        "backend": "lr",
                        "pipeline": pipeline,
                        "model_name": LR_MODEL_NAME,
                        "deployment_mode": metadata["deployment_mode"],
                        "default_threshold": metadata["default_threshold"],
                        "trained_at": metadata["trained_at"],
                        "dataset_rows": metadata["dataset_rows"],
                        "dataset_sources": metadata["dataset_sources"],
                        "metrics": metadata["metrics"],
                        "preprocessing_version": metadata.get("preprocessing_version", PREPROCESSING_VERSION),
                        "model_config_version": metadata.get("model_config_version", LR_MODEL_CONFIG_VERSION),
                        "monitoring_baseline": metadata.get("monitoring_baseline"),
                        "cache_status": "loaded_from_cache",
                    }
            except Exception:
                self.logger.exception("Failed to load cached LR model; retraining LR baseline.")

        bundle = self._train_bundle()
        self._save_bundle(bundle)
        bundle["cache_status"] = "trained_and_cached"
        return bundle

    def _load_transformer_bundle(self) -> dict[str, Any] | None:
        if not self.transformer_dir.exists() or not self.transformer_metadata_path.exists():
            return None

        try:
            with open(self.transformer_metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            tokenizer = CamembertTokenizer.from_pretrained(self.transformer_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.transformer_dir)
            device = self._detect_device()
            model.to(device)
            model.eval()

            return {
                "backend": "transformer",
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "max_length": int(metadata.get("max_length", self._default_max_length(device))),
                "model_name": metadata.get("model_name", "WangchanBERTa"),
                "model_id": metadata.get("model_id", "airesearch/wangchanberta-base-att-spm-uncased"),
                "deployment_mode": metadata.get("deployment_mode", "transformer_finetuned"),
                "default_threshold": float(metadata.get("threshold", self.default_threshold)),
                "trained_at": metadata["trained_at"],
                "dataset_rows": int(metadata["dataset_rows"]),
                "dataset_sources": metadata["dataset_sources"],
                "metrics": metadata["metrics"],
                "preprocessing_version": metadata.get("preprocessing_version", PREPROCESSING_VERSION),
                "monitoring_baseline": metadata.get("monitoring_baseline"),
                "cache_status": "loaded_from_transformer_artifact",
            }
        except Exception:
            return None

    def _load_metadata(self) -> dict[str, Any]:
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_toxic_keywords(self) -> set[str]:
        if not self.toxic_keywords_path.exists():
            return set()

        try:
            df = pd.read_csv(self.toxic_keywords_path, usecols=["keyword"])
        except Exception:
            self.logger.exception("Failed to load toxic keyword glossary.")
            return set()

        keywords = set()
        for keyword in df["keyword"].dropna():
            processed = self.preprocess_text(str(keyword)).strip()
            if processed:
                keywords.add(processed)
        return keywords

    def _save_bundle(self, bundle: dict[str, Any]) -> None:
        metadata = {
            "model_name": bundle["model_name"],
            "deployment_mode": bundle["deployment_mode"],
            "default_threshold": bundle["default_threshold"],
            "trained_at": bundle["trained_at"],
            "dataset_rows": bundle["dataset_rows"],
            "dataset_sources": bundle["dataset_sources"],
            "metrics": bundle["metrics"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_config_version": LR_MODEL_CONFIG_VERSION,
            "monitoring_baseline": bundle.get("monitoring_baseline"),
        }

        tmp_model_path = self.model_path.with_suffix(".joblib.tmp")
        tmp_metadata_path = self.metadata_path.with_suffix(".json.tmp")

        joblib.dump(bundle["pipeline"], tmp_model_path)
        with open(tmp_metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        tmp_model_path.replace(self.model_path)
        tmp_metadata_path.replace(self.metadata_path)

    def _train_bundle(self) -> dict[str, Any]:
        df = self._load_full_dataset()
        X = df["texts"]
        y = df["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word_tfidf",
                                TfidfVectorizer(
                                    tokenizer=self._tokenize_text,
                                    token_pattern=None,
                                    ngram_range=(1, 2),
                                    min_df=3,
                                    max_features=LR_WORD_MAX_FEATURES,
                                ),
                            ),
                            (
                                "char_tfidf",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=3,
                                    max_features=LR_CHAR_MAX_FEATURES,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.default_threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "f2_score": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "test_size": int(len(y_test)),
            "uncertain_rate": float(((probabilities >= 0.4) & (probabilities <= 0.6)).mean()),
        }

        bundle = {
            "backend": "lr",
            "pipeline": pipeline,
            "model_name": LR_MODEL_NAME,
            "deployment_mode": "cached_startup_baseline",
            "default_threshold": self.default_threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset_rows": int(len(df)),
            "dataset_sources": sorted(df["source"].unique().tolist()),
            "metrics": metrics,
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_config_version": LR_MODEL_CONFIG_VERSION,
            "monitoring_baseline": self._build_monitoring_baseline(
                df,
                "training_dataset",
                metrics["uncertain_rate"],
            ),
        }
        self._log_lr_experiment(bundle)
        return bundle

    def _log_lr_experiment(self, bundle: dict[str, Any]) -> None:
        try:
            import mlflow
        except ImportError:
            self.logger.info("MLflow is not installed; skipping LR experiment logging.")
            return

        try:
            tracking_uri = os.getenv("THAI_MOD_MLFLOW_TRACKING_URI")
            experiment_name = os.getenv("THAI_MOD_MLFLOW_EXPERIMENT", "thai-mod")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(
                run_name="thai-mod-lr-baseline",
                nested=mlflow.active_run() is not None,
            ):
                mlflow.log_params(
                    {
                        "model_name": bundle["model_name"],
                        "model_type": "word_char_tfidf_logistic_regression",
                        "dataset_rows": bundle["dataset_rows"],
                        "dataset_sources": ",".join(bundle["dataset_sources"]),
                        "preprocessing_version": PREPROCESSING_VERSION,
                        "model_config_version": LR_MODEL_CONFIG_VERSION,
                        "random_state": 42,
                        "threshold": bundle["default_threshold"],
                        "word_tfidf_ngram_range": "(1, 2)",
                        "word_tfidf_min_df": 3,
                        "word_tfidf_max_features": LR_WORD_MAX_FEATURES,
                        "char_tfidf_analyzer": "char_wb",
                        "char_tfidf_ngram_range": "(3, 5)",
                        "char_tfidf_min_df": 3,
                        "char_tfidf_max_features": LR_CHAR_MAX_FEATURES,
                        "classifier_class_weight": "balanced",
                    }
                )
                mlflow.log_metrics(
                    {
                        "accuracy": bundle["metrics"]["accuracy"],
                        "precision": bundle["metrics"]["precision"],
                        "recall": bundle["metrics"]["recall"],
                        "f1": bundle["metrics"]["f1_score"],
                        "f2": bundle["metrics"]["f2_score"],
                        "test_size": bundle["metrics"]["test_size"],
                        "uncertain_rate": bundle["metrics"].get("uncertain_rate", 0.0),
                    }
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    metadata_path = Path(tmp_dir) / "lr_training_metadata.json"
                    confusion_path = Path(tmp_dir) / "lr_confusion_matrix.json"
                    with open(metadata_path, "w", encoding="utf-8") as file:
                        json.dump(
                            {
                                "model_name": bundle["model_name"],
                                "deployment_mode": bundle["deployment_mode"],
                                "dataset_rows": bundle["dataset_rows"],
                                "dataset_sources": bundle["dataset_sources"],
                                "preprocessing_version": PREPROCESSING_VERSION,
                                "model_config_version": LR_MODEL_CONFIG_VERSION,
                                "metrics": bundle["metrics"],
                            },
                            file,
                            ensure_ascii=False,
                            indent=2,
                        )
                    with open(confusion_path, "w", encoding="utf-8") as file:
                        json.dump(bundle["metrics"]["confusion_matrix"], file, indent=2)
                    mlflow.log_artifact(str(metadata_path))
                    mlflow.log_artifact(str(confusion_path))
                mlflow.log_param("artifact_path", str(self.model_path))
            self.logger.info(
                "Logged LR training run to MLflow experiment '%s'%s.",
                experiment_name,
                f" at {tracking_uri}" if tracking_uri else "",
            )
        except Exception:
            self.logger.exception("Failed to log LR training run to MLflow.")
            return

    def _load_full_dataset(self) -> pd.DataFrame:
        frames = [self._prepare_dataset(path) for path in self.dataset_files]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["texts"], keep="first").reset_index(drop=True)
        return combined

    def _prepare_dataset(self, dataset_file: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_file).copy()
        df = df.dropna(subset=["category", "texts"])
        if df.empty:
            return pd.DataFrame(columns=["texts", "category", "source"])
        df["texts"] = df["texts"].apply(self.preprocess_text)
        df = df[df["texts"].str.strip() != ""].copy()
        df["category"] = df["category"].replace({"pos": "neu"})
        df["category"] = df["category"].map({"neg": 1, "neu": 0})
        df = df.dropna(subset=["category"]).copy()
        df["category"] = df["category"].astype(int)
        df["source"] = dataset_file.name
        return df[["texts", "category", "source"]]

    @staticmethod
    def preprocess_text(text: str) -> str:
        return preprocess_text(text)

    def get_monitoring_baseline(self) -> dict[str, Any]:
        if self.monitoring_baseline is not None:
            return self.monitoring_baseline

        self.ensure_ready()
        assert self.bundle is not None

        if self.bundle.get("monitoring_baseline"):
            self.monitoring_baseline = self.bundle["monitoring_baseline"]
            return self.monitoring_baseline

        try:
            df = self._load_monitoring_baseline_sample()
            self.monitoring_baseline = self._build_monitoring_baseline(
                df,
                "training_dataset_sample",
                self.bundle.get("metrics", {}).get("uncertain_rate", 0.15),
            )
        except Exception:
            metrics = self.bundle.get("metrics", {})
            self.monitoring_baseline = {
                "source": "model_metadata_fallback",
                "dataset_rows": int(self.bundle.get("dataset_rows", 0)),
                "toxic_rate": None,
                "average_text_length": None,
                "language_distribution": {},
                "uncertain_prediction_rate": float(metrics.get("uncertain_rate", 0.15)),
                "preprocessing_version": PREPROCESSING_VERSION,
            }

        return self.monitoring_baseline

    def _load_monitoring_baseline_sample(self, rows_per_file: int = 5000) -> pd.DataFrame:
        frames = []
        for dataset_file in self.dataset_files:
            df = pd.read_csv(dataset_file, usecols=["category", "texts"], nrows=rows_per_file).copy()
            df = df.dropna(subset=["category", "texts"])
            df["texts"] = df["texts"].apply(self.preprocess_text)
            df = df[df["texts"].str.strip() != ""].copy()
            df["category"] = df["category"].replace({"pos": "neu"})
            df["category"] = df["category"].map({"neg": 1, "neu": 0})
            df = df.dropna(subset=["category"]).copy()
            df["category"] = df["category"].astype(int)
            frames.append(df[["texts", "category"]])
        return pd.concat(frames, ignore_index=True)

    def _build_monitoring_baseline(
        self,
        df: pd.DataFrame,
        source: str,
        uncertain_prediction_rate: float,
    ) -> dict[str, Any]:
        languages = [self._detect_language_type(text) for text in df["texts"]]
        return {
            "source": source,
            "dataset_rows": int(len(df)),
            "toxic_rate": float(df["category"].mean()),
            "average_text_length": float(df["texts"].str.len().mean()),
            "language_distribution": self._language_distribution(languages),
            "uncertain_prediction_rate": float(uncertain_prediction_rate),
            "preprocessing_version": PREPROCESSING_VERSION,
        }

    @staticmethod
    def _detect_language_type(text: str) -> str:
        has_thai = any("\u0e00" <= char <= "\u0e7f" for char in str(text))
        has_english = any("a" <= char.lower() <= "z" for char in str(text))

        if has_thai and has_english:
            return "mixed"
        if has_thai:
            return "thai"
        if has_english:
            return "english"
        return "other"

    @staticmethod
    def _language_distribution(languages: list[str]) -> dict[str, float]:
        total = max(len(languages), 1)
        return {
            language: languages.count(language) / total
            for language in ["thai", "english", "mixed", "other"]
        }

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        tokens = word_tokenize(str(text), engine="newmm")
        return [token for token in tokens if token and not token.isspace()]

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _default_max_length(device: torch.device) -> int:
        if device.type == "cuda":
            return 128
        if device.type == "mps":
            return 96
        return 64

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        self.ensure_ready()
        assert self.bundle is not None

        effective_threshold = self.bundle["default_threshold"] if threshold is None else threshold
        processed_text = self.preprocess_text(text)

        if self.bundle.get("backend") == "transformer":
            toxic_score = self._predict_transformer_score(processed_text)
        else:
            model = self.bundle["pipeline"]
            toxic_score = float(model.predict_proba([processed_text])[0][1])

        toxic_score = self._apply_toxic_keyword_boost(processed_text, toxic_score)
        predicted_toxic = int(toxic_score >= effective_threshold)
        confidence = toxic_score if predicted_toxic else 1.0 - toxic_score

        return PredictionResult(
            text=text,
            processed_text=processed_text,
            predicted_label="toxic" if predicted_toxic else "non-toxic",
            toxic_score=round(toxic_score, 4),
            confidence=round(confidence, 4),
            threshold=round(float(effective_threshold), 4),
            recommendation="FLAG_FOR_REVIEW" if predicted_toxic else "ALLOW",
            source_model=self.bundle["model_name"],
        )

    def _apply_toxic_keyword_boost(self, processed_text: str, toxic_score: float) -> float:
        normalized = processed_text.strip()
        if not normalized or not self.toxic_keywords:
            return toxic_score

        tokens = self._tokenize_text(normalized)
        token_matches = [token for token in tokens if token in self.toxic_keywords]

        is_exact_keyword = normalized in self.toxic_keywords
        is_short_keyword_phrase = (
            len(normalized) <= TOXIC_KEYWORD_MAX_CHARS
            and bool(tokens)
            and bool(token_matches)
            and len(token_matches) == len(tokens)
        )

        if is_exact_keyword or is_short_keyword_phrase:
            return max(toxic_score, TOXIC_KEYWORD_MIN_SCORE)

        return toxic_score

    def _predict_transformer_score(self, processed_text: str) -> float:
        assert self.bundle is not None

        tokenizer = self.bundle["tokenizer"]
        model = self.bundle["model"]
        device = self.bundle["device"]

        encoded = tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.bundle["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)

        return float(probabilities[0, 1].item())

    def get_model_info(self) -> dict[str, Any]:
        self.ensure_ready()
        assert self.bundle is not None
        return {
            "model_name": self.bundle["model_name"],
            "selected_backend": self.bundle.get("selected_backend", self.model_backend),
            "deployment_mode": self.bundle["deployment_mode"],
            "cache_status": self.bundle.get("cache_status", "unknown"),
            "default_threshold": self.bundle["default_threshold"],
            "trained_at": self.bundle["trained_at"],
            "dataset_rows": self.bundle["dataset_rows"],
            "dataset_sources": self.bundle["dataset_sources"],
            "metrics": self.bundle["metrics"],
            "preprocessing_version": self.bundle.get("preprocessing_version", PREPROCESSING_VERSION),
            "monitoring_baseline": self.bundle.get("monitoring_baseline"),
        }
