from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIR = ROOT / "models" / "candidates" / "wangchanberta_candidate"
DEFAULT_DEPLOYED_DIR = ROOT / "models" / "wangchanberta_finetuned"
DEFAULT_ARCHIVE_DIR = ROOT / "models" / "archive"
REQUIRED_FILES = (
    "metadata.json",
    "config.json",
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def load_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_artifact(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {path}")

    missing = [filename for filename in REQUIRED_FILES if not (path / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Artifact at {path} is missing required files: {', '.join(missing)}")

    if not any((path / filename).exists() for filename in WEIGHT_FILES):
        raise FileNotFoundError(
            f"Artifact at {path} is missing model weights: expected one of {', '.join(WEIGHT_FILES)}"
        )


def metric(metadata: dict[str, Any], name: str) -> float:
    return float(metadata.get("metrics", {}).get(name, 0.0))


def promotion_report(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return {
            "passes": True,
            "reason": "No deployed model metadata exists; candidate can be promoted.",
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
        "Candidate keeps safety metrics non-degraded."
        if passes
        else "Candidate did not preserve recall and F2 against the deployed model."
    )
    return {"passes": passes, "reason": reason, "checks": checks}


def backup_deployed(deployed_dir: Path, archive_dir: Path) -> Path | None:
    if not deployed_dir.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = archive_dir / f"{deployed_dir.name}_{timestamp}"
    shutil.copytree(deployed_dir, backup_dir)
    return backup_dir


def promote(candidate_dir: Path, deployed_dir: Path, archive_dir: Path, force: bool) -> dict[str, Any]:
    validate_artifact(candidate_dir)
    candidate_metadata = load_metadata(candidate_dir)
    current_metadata = load_metadata(deployed_dir) if (deployed_dir / "metadata.json").exists() else None
    report = promotion_report(current_metadata, candidate_metadata)

    if not report["passes"] and not force:
        return {
            "promoted": False,
            "forced": False,
            "candidate": str(candidate_dir),
            "deployed": str(deployed_dir),
            **report,
        }

    backup_dir = backup_deployed(deployed_dir, archive_dir)
    if deployed_dir.exists():
        shutil.rmtree(deployed_dir)
    shutil.copytree(candidate_dir, deployed_dir)

    return {
        "promoted": True,
        "forced": force and not report["passes"],
        "candidate": str(candidate_dir),
        "deployed": str(deployed_dir),
        "backup": str(backup_dir) if backup_dir else None,
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a retrained WangchanBERTa candidate into the FastAPI deployment artifact."
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--deployed-dir", type=Path, default=DEFAULT_DEPLOYED_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--force", action="store_true", help="Promote even if metric checks fail.")
    args = parser.parse_args()

    report = promote(args.candidate_dir, args.deployed_dir, args.archive_dir, args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
