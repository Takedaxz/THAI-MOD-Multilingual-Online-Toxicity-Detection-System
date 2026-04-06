from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thai_mod_api.model_service import ToxicityModelService
from src.thai_mod_api.monitoring import (
    build_reference_profile_artifact,
    default_reference_batch_path,
    default_reference_profile_path,
)


PROFILE_NAME = "thai_mod_all_sources_reference_v2"
PROFILE_VERSION = "v2"
RANDOM_SEED = 42
MAX_SAMPLES_PER_DATASET = 100
DATASET_FILES = [f"dataset{i}.csv" for i in range(1, 9)]


def _load_source_frame(dataset_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / dataset_name
    frame = pd.read_csv(path).copy()
    frame = frame.dropna(subset=["texts"]).copy()
    frame["texts"] = frame["texts"].astype(str)
    frame = frame[frame["texts"].str.strip() != ""].copy()
    frame["processed_text"] = frame["texts"].map(ToxicityModelService.preprocess_text)
    frame = frame[frame["processed_text"].str.strip() != ""].copy()
    frame["source"] = dataset_name
    return frame[["texts", "processed_text", "source"]]


def build_reference_batch() -> tuple[pd.DataFrame, dict[str, int]]:
    frames = [_load_source_frame(dataset_name) for dataset_name in DATASET_FILES]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["processed_text"], keep="first").reset_index(drop=True)

    sampled_parts = []
    source_counts: dict[str, int] = {}

    for index, dataset_name in enumerate(DATASET_FILES):
        source_frame = combined[combined["source"] == dataset_name].copy()
        take_count = min(MAX_SAMPLES_PER_DATASET, len(source_frame))
        sampled = source_frame.sample(n=take_count, random_state=RANDOM_SEED + index)
        sampled_parts.append(sampled[["texts", "source"]])
        source_counts[dataset_name] = int(take_count)

    reference_batch = pd.concat(sampled_parts, ignore_index=True)
    reference_batch = reference_batch.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return reference_batch, source_counts


def main() -> None:
    reference_batch, source_counts = build_reference_batch()
    service = ToxicityModelService(PROJECT_ROOT)

    reference_batch_path = default_reference_batch_path(PROJECT_ROOT)
    reference_profile_path = default_reference_profile_path(PROJECT_ROOT)
    reference_batch_path.parent.mkdir(parents=True, exist_ok=True)

    generation_details = {
        "seed": RANDOM_SEED,
        "deduplicate_on": "preprocessed_text",
        "sampling_strategy": "all_source_capped_holdout",
        "max_samples_per_dataset": MAX_SAMPLES_PER_DATASET,
        "source_datasets": DATASET_FILES,
        "source_counts": source_counts,
        "rationale": (
            "No historical production traffic exists in the repository. "
            "This fixed baseline is generated from all eight project datasets with a per-source cap "
            "to prevent the largest datasets from dominating the reference profile. "
            "The final language mix is the observed result of that reproducible sampling process, "
            "not a manually fixed Thai/English ratio."
        ),
    }
    reference_profile = build_reference_profile_artifact(
        service,
        reference_batch,
        profile_name=PROFILE_NAME,
        profile_version=PROFILE_VERSION,
        generation_details=generation_details,
        reference_batch_path=reference_batch_path,
    )

    reference_batch.to_csv(reference_batch_path, index=False, encoding="utf-8")
    with open(reference_profile_path, "w", encoding="utf-8") as file:
        json.dump(reference_profile, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "reference_batch_path": str(reference_batch_path),
                "reference_profile_path": str(reference_profile_path),
                "sample_count": reference_profile["sample_count"],
                "language_mix": reference_profile["language_mix"],
                "source_counts": reference_profile["source_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
