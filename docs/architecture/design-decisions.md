# Design Decisions and Trade-offs

> Documents the key architectural and engineering decisions made in THAI-MOD,
> the alternatives considered, and the rationale behind each choice.

## Table of Contents

1. [Decision 1: Deploy LR Baseline Instead of WangchanBERTa](#decision-1-deploy-lr-baseline-instead-of-wangchanberta)
2. [Decision 2: Recall-Oriented Optimization](#decision-2-recall-oriented-optimization)
3. [Decision 3: Human-in-the-Loop Decision Support](#decision-3-human-in-the-loop-decision-support)
4. [Decision 4: Monolithic Model Service](#decision-4-monolithic-model-service)
5. [Decision 5: Train-on-First-Startup with Caching](#decision-5-train-on-first-startup-with-caching)
6. [Decision 6: Shared Preprocessing for Training and Inference](#decision-6-shared-preprocessing-for-training-and-inference)
7. [Decision 7: Threshold 0.4 as Default](#decision-7-threshold-04-as-default)
8. [Decision 8: PyThaiNLP newmm for Tokenization](#decision-8-pythainlp-newmm-for-tokenization)
9. [Decision 9: Static Frontend with No Build Step](#decision-9-static-frontend-with-no-build-step)
10. [Decision 10: Binary Classification Instead of Multi-class](#decision-10-binary-classification-instead-of-multi-class)

---

## Decision 1: Deploy LR Baseline Instead of WangchanBERTa

### Context
WangchanBERTa was selected as the main research model (accuracy 89.1%, macro F1 85.7%). However, the deployed web app uses TF-IDF + Logistic Regression (Balanced) instead.

### Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **A: Deploy WangchanBERTa** | Best accuracy (+6pp), better context understanding, handles sarcasm/slang | Requires GPU or accepts ~50-70ms CPU latency, ~500MB model, 6-15s cold start, PyTorch dependency |
| **B: Deploy LR Baseline** (chosen) | Sub-millisecond inference, ~20MB model, <1s startup, minimal dependencies, easy to demo | Lower accuracy, no contextual understanding, struggles with sarcasm |
| **C: Deploy both behind A/B split** | Compare in production | Adds complexity for a demo/prototype system |

### Decision
Option B: Deploy LR baseline for the Phase 2 demo.

### Rationale
1. **No saved BERT artifact**: The fine-tuned WangchanBERTa was trained in notebooks but no deployable artifact was exported at the time of Phase 2 development
2. **Demo reliability**: LR starts instantly and runs on any hardware without GPU. For a live course presentation, reliability matters more than model quality
3. **Resource constraints**: The deployment environment may not have GPU access. LR works identically on any machine
4. **Structured for upgrade**: The `ToxicityModelService` class isolates all model logic. Swapping to BERT requires changing the internal pipeline without touching API contracts or UI code
5. **Acceptable baseline performance**: LR (balanced) achieves 83% accuracy and 77.6% toxic recall, sufficient for demonstrating the system concept

### Risk and Mitigation
- **Risk**: Lower accuracy means more misclassifications in demo
- **Mitigation**: Human-in-the-loop design means wrong predictions are caught by the moderator. The system explicitly presents confidence scores so the moderator can judge reliability

### How to Upgrade
Place fine-tuned WangchanBERTa weights in `models/wangchanberta_finetuned/`. The model service detects the artifact at startup and switches automatically. No code changes needed if the fallback mechanism described in the v2 architecture is implemented.

---

## Decision 2: Recall-Oriented Optimization

### Context
In toxicity detection, two types of errors have different impacts:
- **False Negative** (toxic missed): Harmful content reaches users. Impact: **HIGH**
- **False Positive** (non-toxic flagged): Clean content gets reviewed unnecessarily. Impact: **MEDIUM**

### Decision
Optimize for recall (sensitivity) of the toxic class, accepting higher false positive rate.

### Evidence
- Primary metric: Recall (toxic class)
- Supporting metric: F2-score (weights recall 2x over precision)
- Threshold set to 0.4 (below default 0.5) to catch more toxic content
- `class_weight='balanced'` used in LR to upweight the minority toxic class

### Trade-off
| Metric | Default LR (threshold 0.5) | Balanced LR (threshold 0.4) |
|---|---|---|
| Accuracy | 84% | 83% |
| Precision | 81% | 78% |
| Recall | 74% | ~79-81% |
| F1 | 76% | 79% |

We accept 3pp lower precision for ~5-7pp higher recall. In practice: more non-toxic comments get flagged for review, but fewer toxic comments slip through.

### Rationale
- Letting toxic content through (FN) harms users and damages platform trust
- Flagging non-toxic content (FP) costs moderator time but causes no direct harm
- The moderator reviews all flagged content anyway (decision-support model), so FP is low-cost
- This aligns with the project's stated goal: "Optimize Recall to reduce False Negatives"

---

## Decision 3: Human-in-the-Loop Decision Support

### Context
The system could either automatically enforce moderation actions (auto-remove, auto-block) or flag content for human review.

### Decision
THAI-MOD is a **decision-support tool**, not an automated enforcer.

### How It Works
- Score above threshold: `FLAG_FOR_REVIEW` (moderator decides)
- Score below threshold: `ALLOW` (but moderator can still review)
- No automatic removal, blocking, or punishment

### Rationale
1. **Thai language ambiguity**: Sarcasm, slang, banter between friends, and cultural context make automated decisions unreliable. Example: "ไอ้สัส" can be friendly teasing or genuine abuse depending on context
2. **Legal and ethical safety**: Auto-enforcement risks censoring legitimate speech. A human moderator understands context that ML cannot
3. **Error tolerance**: With ~80% recall and ~78% precision, ~20% of toxic content and ~22% of flagged content will be wrong. Auto-enforcement at this error rate is not acceptable
4. **PDPA compliance**: The system avoids storing user data and avoids automated decision-making about individuals, reducing regulatory risk
5. **Course requirement alignment**: The project scope specifies a prototype/demo system, not production enforcement

---

## Decision 4: Monolithic Model Service

### Context
The ML pipeline could be structured as:
- A: Single class handling everything (chosen)
- B: Separate microservices for preprocessing, feature extraction, inference
- C: Separate modules within one package

### Decision
Single `ToxicityModelService` class handles data loading, preprocessing, training, caching, and inference.

### Rationale
1. **Prototype scope**: For a course project demo, a single class is simpler to understand, test, and present
2. **No network overhead**: Everything runs in-process, no serialization/deserialization between services
3. **Consistent preprocessing**: Having preprocessing as a static method on the same class that trains and predicts guarantees train-serve consistency
4. **Easy to refactor later**: The class has clear method boundaries. Splitting into modules later is straightforward

### Trade-off
- **Pro**: Simple, fast, easy to demo and explain
- **Con**: Single class with ~250 lines doing data loading + training + inference. Harder to test components in isolation. Not suitable for production at scale

### Future direction
For v2, consider splitting into:
- `preprocessor.py` (text cleaning)
- `data_loader.py` (dataset aggregation)
- `model_service.py` (inference only)
- `trainer.py` (training pipeline)

---

## Decision 5: Train-on-First-Startup with Caching

### Context
The model needs to be available when the app starts. Options:
- A: Commit trained model to git
- B: Train on first startup, cache locally (chosen)
- C: Download from external model registry

### Decision
Train from raw datasets on first startup, then cache the artifact for subsequent starts.

### Rationale
1. **No binary blobs in git**: Model artifacts are 5-500MB; committing them bloats the repository
2. **Reproducibility**: Anyone can clone the repo and get a working model without external dependencies
3. **Self-contained**: No need for model registry infrastructure, S3 buckets, or download scripts
4. **Fast after first run**: Cache load takes <1s. First-time training takes 30-60s (acceptable for setup)

### Implementation Detail
- Atomic writes via `.tmp` files prevent corrupted cache if process crashes during save
- Metadata JSON stores training timestamp, dataset info, and evaluation metrics alongside the model

### Trade-off
- **Pro**: Zero external dependencies, reproducible, self-documenting (metadata captures training conditions)
- **Con**: First startup is slow (~30-60s). If datasets change, user must manually delete cache to retrain

---

## Decision 6: Shared Preprocessing for Training and Inference

### Context
Train-serve skew (preprocessing differs between training and inference) is a common source of silent ML bugs.

### Decision
`preprocess_text()` is a single static method on `ToxicityModelService`, called both during `_prepare_dataset()` (training) and `predict()` (inference).

### What It Prevents
```
Training:   "ไอ้บ้า 😡 URL" -> preprocess -> "ไอ้บ้า :enraged_face: " -> train
Inference:  "ไอ้บ้า 😡 URL" -> preprocess -> "ไอ้บ้า :enraged_face: " -> predict

If preprocessing differed:
Training:   "ไอ้บ้า 😡 URL" -> clean_v1 -> "ไอ้บ้า  "         -> train (emoji removed)
Inference:  "ไอ้บ้า 😡 URL" -> clean_v2 -> "ไอ้บ้า :enraged_face: " -> predict (emoji as text)
--> Model sees different input distribution at inference -> degraded performance
```

### Rationale
- Single source of truth for text cleaning
- Static method: no instance state, purely functional, easy to test
- Same function regardless of which model (LR or BERT) is used downstream

---

## Decision 7: Threshold 0.4 as Default

### Context
Binary classifiers typically use 0.5 as the decision boundary. For toxic content detection, this can be adjusted.

### Threshold Comparison (from experiments)

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.3 | 82% | 77% | 81% | 78% |
| **0.4** (chosen) | **84%** | **79%** | **79%** | **79%** |
| 0.5 | 84% | 81% | 74% | 76% |

### Decision
Default threshold = 0.4, providing a balanced improvement over 0.5 with +5pp recall at -2pp precision.

### Rationale
1. **Recall improvement**: 0.4 catches ~5% more toxic content than 0.5
2. **Not as aggressive as 0.3**: Threshold 0.3 has diminishing returns (only +2pp recall vs 0.4 but -2pp precision and -2pp accuracy)
3. **User-configurable**: The threshold is exposed in the API and UI slider. Moderators can adjust per their risk tolerance
4. **Supports recall-oriented goal**: Aligns with Decision 2

---

## Decision 8: PyThaiNLP newmm for Tokenization

### Context
Thai text has no spaces between words, requiring a tokenizer. Options:
- A: PyThaiNLP newmm (dictionary-based maximum matching) (chosen)
- B: PyThaiNLP attacut (deep learning-based)
- C: ICU word break (rule-based)
- D: No Thai tokenizer (let TF-IDF use character n-grams)

### Decision
PyThaiNLP `word_tokenize` with `engine="newmm"`.

### Rationale
1. **Speed**: newmm is dictionary-based and very fast (~0.1ms per text), suitable for real-time inference
2. **Good enough quality**: For TF-IDF features, exact segmentation quality matters less than consistency. newmm handles common words and slang well
3. **No GPU needed**: Unlike attacut (neural model), newmm runs on CPU with negligible overhead
4. **Established in Thai NLP**: Widely used, well-tested, minimal surprises

### Trade-off
- **Pro**: Fast, no GPU, reliable, widely used
- **Con**: Struggles with novel slang and intentional misspellings (e.g., "ไอสัส" vs "ไอ้สัตว์"). Neural tokenizers handle these better
- **Note**: For the BERT path (v2), tokenization is handled by SentencePiece (subword), so newmm is only relevant for the LR baseline/fallback

---

## Decision 9: Static Frontend with No Build Step

### Context
The UI could be built with React/Vue/Svelte (with build tooling) or vanilla HTML/CSS/JS.

### Decision
Vanilla HTML/CSS/JS served as static files by FastAPI.

### Rationale
1. **Zero build dependencies**: No Node.js, no npm, no webpack. `uvicorn` serves everything
2. **Demo simplicity**: For a course project, a single-page moderator console does not need a SPA framework
3. **Easy to modify**: Any team member can edit HTML/JS directly without learning a framework
4. **Fast to load**: No bundle size concerns, no hydration, instant load

### Trade-off
- **Pro**: Simple, zero tooling, works everywhere
- **Con**: Hard to scale if UI becomes complex. No component reuse, no state management, no type safety

---

## Decision 10: Binary Classification Instead of Multi-class

### Context
Toxicity could be classified as:
- A: Binary (toxic / non-toxic) (chosen)
- B: Multi-class (hate speech, harassment, threat, spam, clean)
- C: Multi-label (text can be both hateful AND threatening)

### Decision
Binary classification: toxic (1) vs non-toxic (0).

### Rationale
1. **Dataset compatibility**: The 8 source datasets use different label schemes. Mapping them all to binary is straightforward; mapping to a shared multi-class taxonomy would require re-annotation
2. **Scope management**: Binary classification is achievable within the 2-3 month project timeline. Multi-class would require more data, more evaluation, more UI complexity
3. **Sufficient for use case**: The moderator's primary need is "should I look at this?" -- a binary flag answers that. The moderator determines the specific type of violation during review
4. **Higher model performance**: Binary classification is easier to learn, resulting in higher recall and F1 than equivalent multi-class setups

### Trade-off
- **Pro**: Simpler, higher metrics, compatible with all datasets, faster development
- **Con**: Moderator gets no information about WHY content is toxic. All toxic content is treated the same regardless of severity

---

## Decision Summary Table

| # | Decision | Key Trade-off | Reversibility |
|---|---|---|---|
| 1 | Deploy LR instead of BERT | Accuracy vs reliability/simplicity | Easy (model swap) |
| 2 | Recall-oriented optimization | Recall vs precision | Easy (adjust threshold) |
| 3 | Human-in-the-loop | Safety vs automation | Policy change only |
| 4 | Monolithic model service | Simplicity vs modularity | Refactor needed |
| 5 | Train-on-first-startup | Self-contained vs startup speed | Add model registry |
| 6 | Shared preprocessing | Consistency vs flexibility | Should keep |
| 7 | Threshold 0.4 | Recall vs precision balance | User-configurable |
| 8 | PyThaiNLP newmm | Speed vs segmentation quality | Swap tokenizer |
| 9 | Static frontend | Simplicity vs scalability | Rewrite in framework |
| 10 | Binary classification | Simplicity vs granularity | Major rework |
