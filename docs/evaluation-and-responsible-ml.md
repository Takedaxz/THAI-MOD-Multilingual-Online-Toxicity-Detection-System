# Evaluation, Reproducibility, and Responsible ML

**THAI-MOD — Multilingual Online Toxicity Detection System**
THAI-MOD Team | April 2026

---

## Table of Contents

1. [Dataset Sources & Tracking](#1-dataset-sources--tracking)
2. [Reproducible Experiment Pipeline](#2-reproducible-experiment-pipeline)
3. [Model Comparison](#3-model-comparison)
4. [Evaluation Metric Rationale](#4-evaluation-metric-rationale)
5. [Threshold Tuning](#5-threshold-tuning)
6. [Deployed Baseline vs WangchanBERTa](#6-deployed-baseline-vs-wangchanberta)
7. [Robustness, Bias & Limitations](#7-robustness-bias--limitations)
8. [Privacy, Ethics & Responsible Use](#8-privacy-ethics--responsible-use)

---

## 1. Dataset Sources & Tracking

### 1.1 Source Datasets

| # | Dataset | Domain | Language | Size (rows) |
|---|---|---|---|---|
| 1 | Wisesight Sentiment | Sentiment | Thai | 23,545 |
| 2 | Thai Toxicity Tweet Corpus | Toxicity | Thai | 2,160 |
| 3 | HateThaiSent | Hate speech | Thai | 4,953 |
| 4 | Thai Sentiment Analysis Dataset | Sentiment | Thai | 341 |
| 5 | Thai Cyberbullying LGBT | Hate speech / cyberbullying | Thai | 16,749 |
| 6 | Jigsaw Toxic Comment Challenge | Toxicity | English | 159,571 |
| 7 | Hate Speech Dataset for Social Media | Hate speech | English | 1,829 |
| 8 | Hate Speech and Offensive Language | Hate speech | English | 24,783 |

All source files are stored as CSV under `datasets/` and tracked via **Git LFS** (`.gitattributes`). Run `git lfs pull` after cloning to restore them.

### 1.2 Combined Dataset Statistics

After aggregation, deduplication (`drop_duplicates` on `processed_text`), and label mapping:

| Statistic | Value |
|---|---|
| Raw rows (all sources) | ~233,931 |
| After deduplication | ~30,620 |
| Class distribution — non-toxic | 73.9% |
| Class distribution — toxic | 26.1% |
| Test set size | 6,124 |

### 1.3 Label Mapping

Source datasets use varied label schemes (sentiment polarity, multi-class toxicity, binary hate speech). All labels are unified into a single binary target:

| Source label | Mapped to |
|---|---|
| Negative sentiment / toxic / hate | **Toxic (1)** |
| Positive or neutral sentiment / non-hate | **Non-toxic (0)** |

> **Known risk:** Mapping negative sentiment to toxic introduces label noise. Strongly negative but non-harmful opinions (e.g., harsh product reviews) may be mislabeled toxic. This is discussed further in §7.

---

## 2. Reproducible Experiment Pipeline

### 2.1 Pipeline Steps

```
Raw CSVs (Git LFS)
    ↓
Data aggregation & deduplication
    ↓
Text preprocessing (shared across all models)
    ↓
Stratified 80/20 train/test split (random_state=42)
    ↓
Feature extraction
    ├── Traditional ML: TF-IDF (PyThaiNLP tokenizer, unigram+bigram)
    └── Transformer:    BERT tokenizer per model (max_length 64–128)
    ↓
Model training
    ├── scikit-learn Pipeline (LR, LinearSVC, XGBoost)
    └── PyTorch + HuggingFace Transformers (WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa)
    ↓
Evaluation (Recall, F2, F1, Accuracy, Precision, Confusion Matrix)
    ↓
MLflow experiment logging
```

### 2.2 Text Preprocessing (shared, train and inference identical)

1. Remove URLs via regex (`http://`, `https://`, `www.*`)
2. Convert emojis to text descriptions (`emoji.demojize()`)
3. Lowercase ASCII characters only (Thai characters are unchanged)
4. Deduplicate on `processed_text` to prevent data leakage

Using the same preprocessing at both training and inference time prevents train-serve skew.

### 2.3 Reproducibility Measures

| Measure | Detail |
|---|---|
| Fixed random seed | `random_state=42` throughout |
| Stratified split | `train_test_split(..., stratify=y)` |
| Dataset versioning | Git LFS — exact byte-for-byte files |
| Experiment tracking | MLflow logs hyperparameters, metrics, artifacts, dataset rows, source list, preprocessing version |
| Reference profile | `scripts/build_reference_profile.py` is deterministic (seed 42, 100 samples/dataset) |

Logged MLflow fields per run include: model type, dataset row count, dataset sources, preprocessing version, random seed, threshold, hyperparameters (learning rate, epochs, batch size, TF-IDF settings), and all evaluation metrics.

### 2.4 Training Environments

| Stage | Environment |
|---|---|
| Traditional ML | Any CPU (scikit-learn) |
| Transformer fine-tuning | Apple MPS (M-series) / CUDA GPU / CPU fallback |
| Inference service | FastAPI on local CPU; GPU recommended for WangchanBERTa |

---

## 3. Model Comparison

### 3.1 Full Results — Test Set (n = 6,124)

| Model | Accuracy | Precision | Recall (toxic) | F1 |
|---|---|---|---|---|
| **WangchanBERTa** | **0.891** | 0.800 | 0.775 | **0.787** |
| PhayaThaiBERT | 0.889 | 0.804 | 0.760 | 0.782 |
| XLM-RoBERTa | 0.865 | 0.775 | 0.678 | 0.723 |
| Linear SVC (Balanced) | 0.840 | 0.780 | 0.800 | 0.790 |
| **Logistic Regression (Balanced)** | 0.830 | 0.780 | **0.810** | 0.790 |
| Linear SVC (Oversampled) | 0.830 | 0.790 | 0.790 | 0.790 |
| LogReg (Threshold 0.3) | 0.820 | 0.770 | 0.810 | 0.780 |
| LogReg (Threshold 0.4) | 0.840 | 0.790 | 0.790 | 0.790 |
| LogReg (Threshold 0.5, default) | 0.840 | 0.810 | 0.740 | 0.760 |

### 3.2 Head-to-Head: LR Baseline vs WangchanBERTa

| Metric | TF-IDF + LR (Balanced) | WangchanBERTa | Delta |
|---|---|---|---|
| Accuracy | 0.830 | **0.891** | +6.1 pp |
| Precision | 0.780 | 0.800 | +2.0 pp |
| Recall (toxic) | **0.810** | 0.775 | −3.5 pp |
| F1-score | 0.790 | 0.787 | ~0 |
| Macro F1 | ~0.79 | **0.857** | +6.7 pp |

### 3.3 Key Observations

- **Overall accuracy**: WangchanBERTa leads by +6 pp, driven by far better non-toxic classification.
- **Toxic recall**: Logistic Regression (Balanced) achieves *higher* raw toxic recall (0.81 vs 0.775) because `class_weight='balanced'` aggressively upweights the minority class and TF-IDF is sensitive to known toxic keywords.
- **Macro F1**: WangchanBERTa's macro F1 (0.857) is significantly higher, showing more balanced performance across both classes. LR's high toxic recall comes at the cost of more false positives on non-toxic content.
- **Context understanding**: WangchanBERTa handles sarcasm, slang, code-switching, and ambiguous language substantially better than TF-IDF (qualitative observation from error analysis).
- **Thai specificity**: XLM-RoBERTa, though multilingual, underperforms both Thai-specific models, confirming that domain-specific pre-training on Thai social media text matters.

### 3.4 Resource & Deployment Comparison

| Dimension | TF-IDF + LR | WangchanBERTa |
|---|---|---|
| Model size | ~20 MB | ~500 MB |
| RAM at inference | ~200 MB | ~1.5 GB |
| Latency — CPU | **< 1 ms** | 45–70 ms |
| Latency — GPU (T4) | N/A | 3–5 ms |
| Cold-start | < 1 s | 6–15 s |
| GPU required? | No | Recommended |
| Interpretability | High (TF-IDF weights) | Low (attention) |

---

## 4. Evaluation Metric Rationale

### 4.1 Why Not Accuracy?

The combined dataset has a 73.9% / 26.1% non-toxic / toxic split. A trivial classifier that always predicts "non-toxic" achieves ~74% accuracy. Accuracy alone is therefore a misleading indicator of model quality on this task.

### 4.2 Primary Metric — Recall (Toxic Class)

**Recall** measures the fraction of truly toxic comments that the model correctly flags:

```
Recall = TP / (TP + FN)
```

A False Negative (FN) — toxic content predicted as safe — means a harmful comment reaches users. This has high potential impact: psychological harm to targets, damage to community trust, and regulatory risk for platforms. Minimizing FN is the top priority.

A False Positive (FP) — safe content flagged as toxic — causes moderators to review a harmless post. This is a workload cost but does not cause direct user harm (human-in-the-loop review corrects it).

**Asymmetry of harm → optimize recall.**

### 4.3 Supporting Metric — F2-Score

F2-score is a weighted harmonic mean of precision and recall that gives recall **twice** the weight of precision:

```
F2 = (1 + 4) × (Precision × Recall) / (4 × Precision + Recall)
```

F2 is used instead of F1 because it directly encodes the project's preference: catching toxic content matters more than avoiding false alarms. F2 is reported as:

- **Macro F2** during epoch-level BERT training (treats classes equally, highlights minority-class weakness)
- **Weighted F2** in final evaluation (accounts for class imbalance, reflects real-traffic performance)

### 4.4 Metric Summary

| Metric | Role | Reason |
|---|---|---|
| Recall (toxic) | **Primary** | Minimize False Negatives — toxic content reaching users |
| F2-score | **Primary supporting** | Recall-weighted combined score |
| F1-score | Secondary | Balanced precision/recall reference |
| Accuracy | Tertiary | Overall correctness; misleading under imbalance |
| Precision | Tertiary | False alarm rate for moderators |

---

## 5. Threshold Tuning

The decision threshold converts a model's continuous toxicity probability score into a binary label. The default threshold is 0.5 (predict toxic if score ≥ 0.5).

Lowering the threshold shifts the precision–recall tradeoff toward higher recall at the cost of more false positives.

### 5.1 Logistic Regression Threshold Sweep

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.3 | 0.820 | 0.770 | **0.810** | 0.780 |
| 0.4 | 0.840 | 0.790 | 0.790 | 0.790 |
| 0.5 (default) | 0.840 | 0.810 | 0.740 | 0.760 |

**Selected threshold for the deployed LR baseline: 0.4**

At threshold 0.4, the model achieves 0.79 recall with an acceptable precision of 0.79 and accuracy of 0.84. Threshold 0.3 gains marginally higher recall but at the cost of lower precision and accuracy.

### 5.2 WangchanBERTa Threshold

The default threshold of 0.5 is used for WangchanBERTa in the current configuration. Lowering it to 0.4 would increase recall further. Full threshold sweep for WangchanBERTa is tracked as a future improvement.

### 5.3 Operational Implication

The threshold is a runtime parameter — it can be adjusted without retraining the model. This makes it the primary lever for moderators to balance workload (precision) against safety (recall) in response to operational needs.

---

## 6. Deployed Baseline vs WangchanBERTa

### 6.1 The Two-Model Story

THAI-MOD maintains a distinction between the **selected research model** and the **deployed prototype model**:

| | TF-IDF + LR (Balanced) | WangchanBERTa |
|---|---|---|
| Status | **Deployed in Phase 2 prototype** | **Selected research model** |
| Role | Current serving model | Target production model |
| Why chosen for deployment | No GPU dependency; instant startup; no exported BERT artifact at Phase 2 | Best overall accuracy and context understanding |
| Toxic recall | 0.810 | 0.775 |
| Accuracy | 0.830 | 0.891 |
| Inference latency (CPU) | < 1 ms | 45–70 ms |

### 6.2 Why LR Is Deployed Now

1. **No exported artifact at demo time**: WangchanBERTa fine-tuning produces a model artifact in `models/` (gitignored). At the time of Phase 2 API development, a validated exported artifact was not available.
2. **Demo reliability**: LR has instant cold-start (< 1 s), no GPU dependency, and requires only ~200 MB RAM — viable on any demo machine.
3. **Acceptable baseline performance**: 83% accuracy and 81% toxic recall are acceptable for a prototype with human-in-the-loop review.
4. **API contract is stable**: The `POST /api/predict` and `POST /api/batch-predict` interfaces are model-agnostic. Swapping LR for WangchanBERTa requires only changes to `src/thai_mod_api/model_service.py`.

### 6.3 Promotion Path for WangchanBERTa

WangchanBERTa replaces the LR baseline when:

1. A fine-tuned artifact is exported via `scripts/export_wangchanberta_artifact.py` and validated on the held-out test set.
2. The deployment environment provides GPU access **or** CPU latency of 50–70 ms is acceptable for the moderation workflow.
3. Available RAM budget is ≥ 1.5 GB for the model service.
4. The transformer loading and inference path in `model_service.py` is implemented and tested.

### 6.4 Hybrid Option

For resource-constrained environments, a hybrid serving strategy is viable:

- **LR** handles synchronous real-time scoring (< 1 ms response to the poster)
- **WangchanBERTa** runs asynchronous batch review of flagged content (higher-accuracy secondary pass before moderator action)

This delivers LR speed for immediate response and BERT accuracy for final moderation decisions.

---

## 7. Robustness, Bias & Limitations

### 7.1 Failure Cases

The following categories of text consistently challenge all models evaluated:

| Failure pattern | Example (Thai) | Why it fails |
|---|---|---|
| Implicit toxicity / contextual hate | "เพื่อนชาวพม่าจะเข้ามาทำไม ประเทศไทยจะไม่เหลืออะไรแล้ว" | No direct slur; toxicity requires social context |
| Negative opinion (non-hate) | "terra ไม่ผ่านด้านการออกแบบ ดูมันขาดเสน่ห์มากๆเลย" | Negative sentiment triggers toxic prediction; product criticism ≠ hate |
| Sarcasm / irony | Praise-framed insults | Surface form appears positive; intent is negative |
| New slang | Emerging terms not in training data | OOV for TF-IDF; rare subword for BERT |
| Dog whistles | Encoded references known only within communities | Looks benign to the model |
| Evasion techniques | "อ_เ_ี้ย", character substitution | Keyword filters and vocabulary-based models are bypassed |
| Emoji as subtext | 🐍 = traitor; 💀 = mocking | Emoji-to-text mapping captures literal meaning, not cultural subtext |
| Code-switching ambiguity | "โคตร toxic เลย report มันไป" | Mixed-language structure varies from monolingual training distributions |

### 7.2 Bias Sources

**Label noise from sentiment mapping.** Datasets 1 and 4 are sentiment datasets. Mapping negative sentiment → toxic conflates harsh-but-legitimate opinion with genuinely harmful content. This introduces systematic false positives for critical reviews, complaints, and political commentary.

**Domain imbalance.** Datasets skew toward Twitter-style short text. Longer forum posts, formal complaints, or news comments may be under-represented, limiting generalization.

**Cultural and temporal specificity.** Training data reflects Thai internet language at a specific point in time. Slang evolves rapidly; the model does not automatically adapt to new expressions.

**Group representation.** Dataset 5 (LGBT cyberbullying) focuses on a specific target group. Coverage of other marginalized groups (ethnic minorities, political groups, religious groups) may be uneven, producing higher error rates for those communities.

**Confidence miscalibration.** Predictions in the 0.4–0.6 score range are unreliable. These uncertain predictions are flagged for human review rather than auto-acted upon.

### 7.3 Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No ground-truth labels at inference | Cannot measure live recall | Human-in-the-loop review; periodic sample labeling |
| Static model — no online learning | Degrades as language evolves | Drift monitoring; scheduled retraining |
| Binary classification only | Cannot distinguish hate speech from spam, threats, self-harm | Scope is intentionally limited to prototype; future work |
| No user context | A phrase may be banter among friends or a targeted attack | Model scores text alone; moderator provides context |
| English-heavy at scale | Jigsaw dataset dominates pre-dedup size | Stratified dedup caps per-source dominance |
| Mild overfitting (WangchanBERTa) | Val loss increased at epoch 3 | Optimal epoch count may be 2; further tuning needed |

### 7.4 Robustness Testing Gaps

The following evaluations were not performed within the scope of this project and represent future work:

- Adversarial robustness (deliberate evasion inputs)
- Cross-domain transfer (forum posts, news comments, private messaging)
- Temporal drift evaluation (model performance on text from a later time period)
- Per-group fairness metrics (false positive / false negative rates broken down by target group)

---

## 8. Privacy, Ethics & Responsible Use

### 8.1 Data Privacy

**All training data is publicly available.** No proprietary platform data, private user data, or scraped data requiring consent was used. Sources include Hugging Face Datasets, GitHub repositories, and Kaggle competitions, all released for research use.

**No Personally Identifiable Information (PII).** User metadata (usernames, timestamps, location, user ID) was stripped before use. Only text content and binary labels are retained.

**Data minimization.** The system is designed to process text content only, consistent with the Thai Personal Data Protection Act (PDPA) principle of collecting only what is necessary.

**No raw text logging at inference.** The monitoring service stores only prediction metadata (`timestamp`, `text_length`, `language_bucket`, `toxicity_score`, `predicted_label`). Raw comment text is never written to disk.

### 8.2 Human-in-the-Loop Design

THAI-MOD is explicitly a **decision-support tool**, not an automated enforcement system.

```
User posts comment
        ↓
System scores toxicity (0–1)
        ↓
Score above threshold → flag for review
        ↓
Human moderator reviews flagged content
        ↓
Moderator makes final moderation decision
```

The model does **not** auto-remove, auto-block, or auto-ban. Every flagged item receives human review. This design:

- Absorbs model errors (false positives corrected before action is taken)
- Preserves freedom of expression (no automated censorship)
- Maintains platform accountability (a human is responsible for every enforcement decision)
- Satisfies due process expectations for content removal

### 8.3 Error Impact Analysis

| Error type | Definition | Impact | Severity |
|---|---|---|---|
| False Negative | Toxic predicted as non-toxic | Harmful content reaches users; psychological harm to targets; platform credibility risk | **High** |
| False Positive | Non-toxic predicted as toxic | Innocent user flagged; moderator workload increases; free expression risk | **Medium** |

The asymmetry drives the recall-first optimization strategy (§4).

### 8.4 Low-Confidence Handling

Predictions with toxicity score in the range 0.4–0.6 are treated as uncertain and routed to human review regardless of threshold. This is especially important for Thai content with high ambiguity (slang, sarcasm, banter vs. genuine hostility).

### 8.5 Explainability

**TF-IDF + LR (deployed baseline):** Fully interpretable. TF-IDF feature weights can be inspected directly to see which words drove a prediction. This supports moderator trust and regulatory auditability.

**WangchanBERTa (target model):** Attention weights are not reliably interpretable as explanations. If WangchanBERTa is promoted to production, attention visualization or gradient-based saliency maps should be added to provide moderators with evidence for predictions.

### 8.6 Fairness Considerations

- The system does not use user profile data, demographic information, or account history. Predictions are based solely on text content, reducing profile-based discrimination risk.
- The toxic keyword glossary (`datasets/toxic_keywords.csv`) is used as a supplementary signal, not a block-list. Keyword matching alone is not sufficient for a toxicity decision.
- Training data coverage of different target communities is uneven (§7.2). Groups not well-represented in training data may experience higher false negative rates. Periodic bias audits using per-group evaluation metrics are recommended.

### 8.7 Responsible Use Boundaries

This system is designed for:

- Content moderation **decision support** on Thai/English social media and online communities
- Prototype and research evaluation

This system is **not appropriate** for:

- Fully automated enforcement without human review
- Law enforcement or legal evidence use
- High-stakes decisions (employment screening, visa processing, insurance)
- Languages or domains not covered by the training data
- Real-time audio or video content (text only)

### 8.8 Monitoring and Degradation Policy

Production drift is monitored via Population Stability Index (PSI) on the language-mix distribution (Thai-only, English-only, mixed-script, other). Four drift states guide operational response:

| State | PSI | Action |
|---|---|---|
| Healthy | < 0.10 | Routine review continues |
| Observe | 0.10–0.20 | Monitor the next window |
| Warning | 0.20–0.35 | Increase manual review; inspect shifted samples |
| Degraded | ≥ 0.35 | Treat traffic as degraded; collect examples for retraining |

**Retraining is a human-initiated, offline workflow** — the system does not auto-retrain from unlabeled monitoring traffic. Moderator-reviewed examples from shifted traffic are collected, merged with existing training data, and a candidate model is retrained and evaluated before promotion.

---

## Appendix: Reproducibility Checklist

| Item | Status |
|---|---|
| All datasets public and cited | Done |
| Git LFS tracking for dataset files | Done |
| Fixed random seed (42) throughout | Done |
| Stratified train/test split | Done |
| Identical preprocessing at train and inference | Done |
| MLflow experiment tracking | Done |
| Hyperparameters logged per run | Done |
| Reference profile generation is deterministic | Done |
| Test set held out (not used during training) | Done |
| Deduplication before split (no leakage) | Done |

---

*Sources: `docs/progress/progress1.txt`, `docs/progress/progress2.txt`, `docs/reports/lr-vs-bert-comparison.md`, `docs/monitoring-and-drift.md`, `model.ipynb`, `thai-bert.ipynb`*
