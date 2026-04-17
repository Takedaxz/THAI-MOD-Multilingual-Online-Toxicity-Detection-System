from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import preprocess_text

NOTEBOOK_ARTIFACT_DIR = ROOT / "models" / "transformers" / "wangchanberta"
APP_ARTIFACT_DIR = ROOT / "models" / "wangchanberta_finetuned"
DATASET_FILES = [ROOT / "datasets" / f"dataset{i}.csv" for i in range(1, 9)]


def load_dataset_summary() -> tuple[int, list[str]]:
    frames = []
    for dataset_file in DATASET_FILES:
        df = pd.read_csv(dataset_file).copy()
        df = df.dropna(subset=["category", "texts"])
        df["texts"] = df["texts"].apply(preprocess_text)
        df = df[df["texts"].str.strip() != ""].copy()
        df["category"] = df["category"].replace({"pos": "neu"})
        df["category"] = df["category"].map({"neg": 1, "neu": 0})
        df = df.dropna(subset=["category"]).copy()
        df["source"] = dataset_file.name
        frames.append(df[["texts", "source"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["texts"], keep="first").reset_index(drop=True)
    return int(len(combined)), sorted(combined["source"].unique().tolist())


def normalize_metrics(notebook_metrics: dict) -> dict:
    return {
        "accuracy": float(notebook_metrics.get("Accuracy", 0.0)),
        "precision": float(notebook_metrics.get("Precision", 0.0)),
        "recall": float(notebook_metrics.get("Recall", 0.0)),
        "f1_score": float(notebook_metrics.get("F1-Score", 0.0)),
        "f2_score": float(notebook_metrics.get("F2-Score", 0.0)),
        "confusion_matrix": notebook_metrics.get("confusion_matrix", [[0, 0], [0, 0]]),
        "test_size": int(sum(sum(row) for row in notebook_metrics.get("confusion_matrix", [[0, 0], [0, 0]]))),
    }


def validate_source(source_dir: Path) -> None:
    metadata_path = source_dir / "metadata.json"
    if not source_dir.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Notebook artifact not found at {source_dir}. Run the WangchanBERTa cells in toxicity_detection.ipynb first."
        )

    model_files = ["model.safetensors", "pytorch_model.bin"]
    if not any((source_dir / filename).exists() for filename in model_files):
        raise FileNotFoundError(f"No model weights found in {source_dir}. Expected model.safetensors or pytorch_model.bin.")


def export_artifact(source_dir: Path, target_dir: Path, force: bool) -> None:
    validate_source(source_dir)

    if target_dir.exists():
        if not force:
            raise FileExistsError(f"{target_dir} already exists. Re-run with --force to replace it.")
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)

    notebook_metadata_path = source_dir / "metadata.json"
    with open(notebook_metadata_path, "r", encoding="utf-8") as file:
        notebook_metadata = json.load(file)

    dataset_rows, dataset_sources = load_dataset_summary()
    metrics = normalize_metrics(notebook_metadata.get("metrics", {}))
    app_metadata = {
        "model_name": "WangchanBERTa",
        "model_id": notebook_metadata.get("model_id", "airesearch/wangchanberta-base-att-spm-uncased"),
        "deployment_mode": "transformer_finetuned",
        "threshold": float(notebook_metadata.get("threshold", notebook_metadata.get("metrics", {}).get("Threshold", 0.4))),
        "trained_at": notebook_metadata.get("trained_at", datetime.now(timezone.utc).isoformat()),
        "dataset_rows": dataset_rows,
        "dataset_sources": dataset_sources,
        "max_length": int(notebook_metadata.get("max_length", 96)),
        "metrics": metrics,
        "history": notebook_metadata.get("history", []),
    }

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(app_metadata, file, ensure_ascii=False, indent=2)

    print(f"Exported WangchanBERTa artifact to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export notebook-trained WangchanBERTa weights for the FastAPI app.")
    parser.add_argument("--source", type=Path, default=NOTEBOOK_ARTIFACT_DIR)
    parser.add_argument("--target", type=Path, default=APP_ARTIFACT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    export_artifact(args.source, args.target, args.force)


if __name__ == "__main__":
    main()
