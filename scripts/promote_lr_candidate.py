from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIR = ROOT / "models" / "candidates" / "lr_candidate"
DEFAULT_MODEL_PATH = ROOT / "models" / "thai_mod_baseline.joblib"
DEFAULT_METADATA_PATH = ROOT / "models" / "thai_mod_baseline.metadata.json"
DEFAULT_ARCHIVE_DIR = ROOT / "models" / "archive"
CANDIDATE_MODEL_NAME = "thai_mod_baseline.joblib"
CANDIDATE_METADATA_NAME = "thai_mod_baseline.metadata.json"


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_candidate(candidate_dir: Path) -> tuple[Path, Path]:
    model_path = candidate_dir / CANDIDATE_MODEL_NAME
    metadata_path = candidate_dir / CANDIDATE_METADATA_NAME

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing candidate model artifact: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing candidate metadata: {metadata_path}")

    return model_path, metadata_path


def metric(metadata: dict[str, Any], name: str) -> float:
    return float(metadata.get("metrics", {}).get(name, 0.0))


def promotion_report(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return {
            "passes": True,
            "reason": "No deployed LR metadata exists; candidate can be promoted.",
            "checks": {},
        }

    checks = {
        "recall": {
            "current": metric(current, "recall"),
            "candidate": metric(candidate, "recall"),
            "passes": metric(candidate, "recall") >= metric(current, "recall"),
        },
        "f2_score": {
            "current": metric(current, "f2_score"),
            "candidate": metric(candidate, "f2_score"),
            "passes": metric(candidate, "f2_score") >= metric(current, "f2_score"),
        },
        "precision": {
            "current": metric(current, "precision"),
            "candidate": metric(candidate, "precision"),
            "passes": metric(candidate, "precision") >= metric(current, "precision"),
        },
    }
    passes = checks["recall"]["passes"] and checks["f2_score"]["passes"]
    reason = (
        "LR candidate keeps safety metrics non-degraded."
        if passes
        else "LR candidate did not preserve recall and F2 against the deployed LR model."
    )
    return {"passes": passes, "reason": reason, "checks": checks}


def backup_deployed(model_path: Path, metadata_path: Path, archive_dir: Path) -> Path | None:
    if not model_path.exists() and not metadata_path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = archive_dir / f"lr_baseline_{timestamp}"
    backup_dir.mkdir(parents=True)

    if model_path.exists():
        shutil.copy2(model_path, backup_dir / model_path.name)
    if metadata_path.exists():
        shutil.copy2(metadata_path, backup_dir / metadata_path.name)
    return backup_dir


def promote(
    candidate_dir: Path,
    model_path: Path,
    metadata_path: Path,
    archive_dir: Path,
    force: bool,
) -> dict[str, Any]:
    candidate_model_path, candidate_metadata_path = validate_candidate(candidate_dir)
    candidate_metadata = load_metadata(candidate_metadata_path)
    current_metadata = load_metadata(metadata_path) if metadata_path.exists() else None
    report = promotion_report(current_metadata, candidate_metadata)

    if not report["passes"] and not force:
        return {
            "promoted": False,
            "forced": False,
            "candidate": str(candidate_dir),
            "model_path": str(model_path),
            **report,
        }

    backup_dir = backup_deployed(model_path, metadata_path, archive_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_model_path, model_path)
    shutil.copy2(candidate_metadata_path, metadata_path)

    return {
        "promoted": True,
        "forced": force and not report["passes"],
        "candidate": str(candidate_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "backup": str(backup_dir) if backup_dir else None,
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a TF-IDF + Logistic Regression candidate into the FastAPI LR cache."
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--force", action="store_true", help="Promote even if metric checks fail.")
    args = parser.parse_args()

    report = promote(
        args.candidate_dir,
        args.model_path,
        args.metadata_path,
        args.archive_dir,
        args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
