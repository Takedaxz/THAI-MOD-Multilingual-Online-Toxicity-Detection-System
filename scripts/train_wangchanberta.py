from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    CamembertTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import preprocess_text

DATASET_FILES = [ROOT / "datasets" / f"dataset{i}.csv" for i in range(1, 9)]
MODEL_DIR = ROOT / "models" / "wangchanberta_finetuned"
MODEL_ID = "airesearch/wangchanberta-base-att-spm-uncased"
RANDOM_STATE = 42
TEST_SIZE = 0.2
PREPROCESSING_VERSION = "shared_preprocess_text_v1"


@dataclass
class RuntimeConfig:
    device: torch.device
    max_length: int
    batch_size: int


class ToxicityDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: CamembertTokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def detect_runtime_config() -> RuntimeConfig:
    if torch.cuda.is_available():
        return RuntimeConfig(device=torch.device("cuda"), max_length=128, batch_size=16)
    if torch.backends.mps.is_available():
        return RuntimeConfig(device=torch.device("mps"), max_length=96, batch_size=16)
    return RuntimeConfig(device=torch.device("cpu"), max_length=64, batch_size=8)


def load_full_dataset() -> pd.DataFrame:
    frames = []
    for dataset_file in DATASET_FILES:
        df = pd.read_csv(dataset_file).copy()
        df = df.dropna(subset=["category", "texts"])
        df["texts"] = df["texts"].apply(preprocess_text)
        df = df[df["texts"].str.strip() != ""].copy()
        df["category"] = df["category"].replace({"pos": "neu"})
        df["category"] = df["category"].map({"neg": 1, "neu": 0})
        df = df.dropna(subset=["category"]).copy()
        df["category"] = df["category"].astype(int)
        df["source"] = dataset_file.name
        frames.append(df[["texts", "category", "source"]])

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["texts"], keep="first").reset_index(drop=True)


def get_label_weights(labels: pd.Series) -> torch.Tensor:
    counts = np.bincount(labels.to_numpy(), minlength=2)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float)


def evaluate(model, dataloader: DataLoader, device: torch.device, threshold: float) -> dict[str, np.ndarray | float]:
    model.eval()
    total_loss = 0.0
    preds: list[int] = []
    probs: list[float] = []
    labels_all: list[int] = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            toxic_probs = torch.softmax(logits, dim=1)[:, 1]
            batch_preds = (toxic_probs >= threshold).long()

            probs.extend(toxic_probs.cpu().numpy().tolist())
            preds.extend(batch_preds.cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "predictions": np.array(preds),
        "probabilities": np.array(probs),
        "labels": np.array(labels_all),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "test_size": int(len(y_true)),
    }


def train_epoch(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    class_weights: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def log_mlflow_run(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    df: pd.DataFrame,
    metrics: dict[str, float | list[list[int]]],
    history: list[dict[str, float | int]],
    metadata: dict[str, object],
) -> None:
    try:
        import mlflow
    except ImportError:
        return

    try:
        with mlflow.start_run(
            run_name="thai-mod-wangchanberta",
            nested=mlflow.active_run() is not None,
        ):
            mlflow.log_params(
                {
                    "model_name": metadata["model_name"],
                    "model_type": "wangchanberta_transformer",
                    "model_id": MODEL_ID,
                    "dataset_rows": int(len(df)),
                    "dataset_sources": ",".join(sorted(df["source"].unique().tolist())),
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "random_state": RANDOM_STATE,
                    "threshold": args.threshold,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "warmup_steps": args.warmup_steps,
                    "max_length": runtime.max_length,
                    "batch_size": runtime.batch_size,
                    "device": str(runtime.device),
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1_score"]),
                    "f2": float(metrics["f2_score"]),
                    "test_size": float(metrics["test_size"]),
                }
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                metadata_path = Path(tmp_dir) / "wangchanberta_training_metadata.json"
                history_path = Path(tmp_dir) / "wangchanberta_history.json"
                confusion_path = Path(tmp_dir) / "wangchanberta_confusion_matrix.json"

                with open(metadata_path, "w", encoding="utf-8") as file:
                    json.dump(metadata, file, ensure_ascii=False, indent=2)
                with open(history_path, "w", encoding="utf-8") as file:
                    json.dump(history, file, ensure_ascii=False, indent=2)
                with open(confusion_path, "w", encoding="utf-8") as file:
                    json.dump(metrics["confusion_matrix"], file, indent=2)

                mlflow.log_artifact(str(metadata_path))
                mlflow.log_artifact(str(history_path))
                mlflow.log_artifact(str(confusion_path))
            mlflow.log_param("artifact_dir", str(MODEL_DIR))
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and export WangchanBERTa for THAI-MOD.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Retrain even if a saved artifact exists.")
    args = parser.parse_args()

    metadata_path = MODEL_DIR / "metadata.json"
    if MODEL_DIR.exists() and metadata_path.exists() and not args.force:
        print(f"Artifact already exists at {MODEL_DIR}. Use --force to retrain.")
        return

    runtime = detect_runtime_config()
    print(f"Device: {runtime.device}")
    print(f"MAX_LENGTH: {runtime.max_length}")
    print(f"BATCH_SIZE: {runtime.batch_size}")

    df = load_full_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["texts"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    tokenizer = CamembertTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model.to(runtime.device)

    train_dataset = ToxicityDataset(X_train.tolist(), y_train.tolist(), tokenizer, runtime.max_length)
    test_dataset = ToxicityDataset(X_test.tolist(), y_test.tolist(), tokenizer, runtime.max_length)
    train_loader = DataLoader(train_dataset, batch_size=runtime.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=runtime.batch_size, shuffle=False)

    class_weights = get_label_weights(y_train)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 10 if total_steps else 0),
        num_training_steps=total_steps,
    )

    best_state_dict = None
    best_f2 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, runtime.device, class_weights)
        eval_output = evaluate(model, test_loader, runtime.device, args.threshold)
        epoch_f2 = fbeta_score(eval_output["labels"], eval_output["predictions"], beta=2, zero_division=0)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "eval_loss": float(eval_output["loss"]),
                "f2": float(epoch_f2),
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | eval_loss={eval_output['loss']:.4f} | f2={epoch_f2:.4f}"
        )

        if epoch_f2 > best_f2:
            best_f2 = epoch_f2
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(runtime.device)

    final_eval = evaluate(model, test_loader, runtime.device, args.threshold)
    metrics = compute_metrics(final_eval["labels"], final_eval["predictions"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metadata = {
        "model_name": "WangchanBERTa",
        "model_id": MODEL_ID,
        "deployment_mode": "transformer_finetuned",
        "threshold": args.threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_sources": sorted(df["source"].unique().tolist()),
        "max_length": runtime.max_length,
        "preprocessing_version": PREPROCESSING_VERSION,
        "metrics": metrics,
        "history": history,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    log_mlflow_run(args, runtime, df, metrics, history, metadata)

    print(f"Saved WangchanBERTa artifact to {MODEL_DIR}")
    print(json.dumps(metrics, indent=2))

    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    elif runtime.device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
