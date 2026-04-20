# Model Update and Retraining Pipeline

THAI-MOD uses a human-in-the-loop update pipeline. Drift monitoring can trigger investigation, but it does not automatically retrain or promote a model from unlabeled traffic.

## Pipeline

1. **Detect shift**
   - The admin monitoring page compares recent live requests against the fixed reference profile.
   - Drift is based on language-mix PSI plus supporting metrics.

2. **Review and label examples**
   - Moderators inspect shifted examples outside the monitoring log.
   - The monitoring log intentionally does not store raw text.
   - Only reviewed and labeled examples should be added back to training data.
   - In the moderator UI, use the human review buttons after prediction to save a reviewed label.

3. **Merge reviewed data**
   - Approved labels are saved into:

```text
models/reviewed/reviewed_comments.csv
```

   - The reviewed dataset uses this format:

```csv
request_id,texts,category,source,predicted_label,toxicity_score,source_model,reviewed_at
abc123,"example toxic comment",neg,reviewed_traffic,toxic,0.87,WangchanBERTa,2026-04-18T...
```

   - Labels are normalized by the training code:
     - `neg` -> toxic class `1`
     - `neu` / `pos` -> non-toxic class `0`

4. **Train a candidate artifact**

For a fast live demo, train the LR candidate:

```bash
conda activate cedt
python scripts/train_lr_candidate.py --force
```

This writes the candidate model to:

```text
models/candidates/lr_candidate/
```

The LR candidate path trains the word+character TF-IDF + Logistic Regression pipeline and is practical to run during a presentation on CPU.

For the heavier WangchanBERTa path:

```bash
conda activate cedt
python scripts/train_wangchanberta.py --candidate --force
```

This writes the candidate model to:

```text
models/candidates/wangchanberta_candidate/
```

The script:

- loads `datasets/dataset1.csv` to `datasets/dataset8.csv`
- includes `models/reviewed/reviewed_comments.csv` when it exists
- applies the same shared preprocessing used by inference
- fine-tunes `airesearch/wangchanberta-base-att-spm-uncased`
- evaluates on a stratified held-out test split
- stores metrics in `metadata.json`
- logs metrics/artifacts to MLflow when MLflow is installed

5. **Compare and promote**

For the LR candidate:

```bash
python scripts/promote_lr_candidate.py
```

This promotes into the LR cache files:

```text
models/thai_mod_baseline.joblib
models/thai_mod_baseline.metadata.json
```

For the WangchanBERTa candidate:

```bash
python scripts/promote_wangchanberta_candidate.py
```

The promotion gate compares the candidate against the deployed artifact at:

```text
models/wangchanberta_finetuned/
```

By default, both promotion scripts require:

- candidate recall >= deployed recall
- candidate F2 >= deployed F2

Precision is reported as a supporting metric because lowering false positives matters for moderator workload, but toxic recall and F2 are the safety-oriented gate.

If the candidate passes, the script:

- backs up the current deployed artifact under `models/archive/`
- replaces `models/wangchanberta_finetuned/` with the candidate
- prints a JSON promotion report

6. **Deploy by restart**

The FastAPI app loads the deployed artifact from `models/wangchanberta_finetuned/` at startup when:

```bash
export THAI_MOD_MODEL_BACKEND=bert
```

Restart the app and verify:

```bash
curl http://127.0.0.1:8000/api/model-info
```

Expected fields:

- `selected_backend: "bert"`
- `deployment_mode: "transformer_finetuned"`
- `cache_status: "loaded_from_transformer_artifact"`

## Why This Is Not Auto-Retraining

Recent production traffic is unlabeled, so it cannot prove recall/F2 degradation by itself. Drift is only an early warning. Promotion requires a reviewed labeled dataset and held-out evaluation before the deployed model changes.

## Key Files

- `scripts/train_lr_candidate.py`
- `scripts/promote_lr_candidate.py`
- `scripts/train_wangchanberta.py`
- `scripts/promote_wangchanberta_candidate.py`
- `src/thai_mod_api/model_service.py`
- `docs/monitoring-and-drift.md`
- `models/candidates/lr_candidate/`
- `models/candidates/wangchanberta_candidate/`
- `models/thai_mod_baseline.joblib`
- `models/wangchanberta_finetuned/`
