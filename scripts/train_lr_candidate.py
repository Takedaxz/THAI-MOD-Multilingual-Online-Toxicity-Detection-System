from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.model_service import ToxicityModelService


DEFAULT_CANDIDATE_DIR = ROOT / "models" / "candidates" / "lr_candidate"


def train_candidate(output_dir: Path, force: bool) -> dict:
    model_path = output_dir / "thai_mod_baseline.joblib"
    metadata_path = output_dir / "thai_mod_baseline.metadata.json"

    if model_path.exists() and metadata_path.exists() and not force:
        return {
            "status": "skipped",
            "reason": f"Candidate already exists at {output_dir}. Re-run with --force to replace it.",
            "candidate_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    service = ToxicityModelService(ROOT)
    service.model_path = model_path
    service.metadata_path = metadata_path
    bundle = service._train_bundle()
    service._save_bundle(bundle)

    return {
        "status": "trained",
        "candidate_dir": str(output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": bundle["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a fast TF-IDF + Logistic Regression candidate for THAI-MOD."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing LR candidate artifact.")
    args = parser.parse_args()

    report = train_candidate(args.output_dir, args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
