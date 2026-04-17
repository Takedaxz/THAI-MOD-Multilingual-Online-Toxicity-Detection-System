# TF-IDF + Logistic Regression vs WangchanBERTa

**Model Comparison Report for THAI-MOD -- Multilingual Online Toxicity Detection System**

THAI-MOD Team | April 2026

---

## Abstract

This report compares the two primary model approaches evaluated in the THAI-MOD project: a traditional TF-IDF + Logistic Regression (Balanced) baseline and a fine-tuned WangchanBERTa transformer model. We analyze classification quality, inference latency, resource requirements, and practical deployment considerations. The comparison informs the decision to deploy the LR baseline in Phase 2 while retaining WangchanBERTa as the target production model.

---

## 1. Introduction

THAI-MOD is a multilingual toxicity detection system designed to classify Thai, English, and code-switched text as toxic or non-toxic. The system serves as a decision-support tool for content moderators.

Two model families were evaluated:

1. **Traditional ML**: TF-IDF vectorization with classical classifiers (Logistic Regression, Linear SVC, XGBoost)
2. **Transformer**: Fine-tuned pre-trained language models (WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa)

This report focuses on the two leading candidates: **TF-IDF + Logistic Regression (Balanced)** (deployed baseline) and **WangchanBERTa** (selected research model).

---

## 2. Experimental Setup

### 2.1 Dataset

- 8 source datasets (5 Thai, 3 English), covering sentiment, toxicity, hate speech, and cyberbullying
- Combined size: ~233,931 rows pre-deduplication, ~30,620 post-deduplication
- Binary labels: toxic (1) / non-toxic (0)
- Class distribution: 73.9% non-toxic, 26.1% toxic
- Split: 80/20 stratified (random_state=42)

### 2.2 Preprocessing (shared)

Both models use identical text preprocessing:

1. Remove HTTP/HTTPS and www URLs via regex
2. Convert emojis to English text descriptions (emoji library)
3. Lowercase ASCII characters (Thai characters unchanged)

### 2.3 TF-IDF + Logistic Regression Configuration

- Tokenizer: PyThaiNLP `word_tokenize` (newmm engine)
- TF-IDF: unigram + bigram, min_df=3, max_features=20,000
- Classifier: Logistic Regression with `class_weight='balanced'`, max_iter=1000
- Threshold: 0.4 (tuned for recall)

### 2.4 WangchanBERTa Configuration

- Base model: `airesearch/wangchanberta-base-att-spm-uncased` (RoBERTa-base, 125M params)
- Tokenizer: CamembertTokenizer (SentencePiece)
- Fine-tuning: AdamW optimizer, lr=2e-5, linear warmup, 3 epochs
- Gradient clipping: max_norm=1.0
- max_length: 128 (CUDA), 96 (MPS), 64 (CPU)
- Training device: Apple MPS (M-series)

---

## 3. Classification Quality

### 3.1 Full Model Comparison

All models tested on the THAI-MOD test set (n = 6,124):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **WangchanBERTa** | **0.891** | 0.800 | 0.775 | **0.787** |
| PhayaThaiBERT | 0.889 | 0.804 | 0.760 | 0.782 |
| XLM-RoBERTa | 0.865 | 0.775 | 0.678 | 0.723 |
| Linear SVC (Balanced) | 0.840 | 0.780 | 0.800 | 0.790 |
| **Logistic Regression (Balanced)** | 0.830 | 0.780 | **0.810** | 0.790 |
| Linear SVC (Oversampled) | 0.830 | 0.790 | 0.790 | 0.790 |
| LogReg (Threshold 0.3) | 0.820 | 0.770 | 0.810 | 0.780 |
| LogReg (Threshold 0.4) | 0.840 | 0.790 | 0.790 | 0.790 |
| LogReg (Threshold 0.5) | 0.840 | 0.810 | 0.740 | 0.760 |

### 3.2 Head-to-Head: LR Baseline vs WangchanBERTa

| Metric | TF-IDF + LR (Balanced) | WangchanBERTa | Delta |
|---|---|---|---|
| Accuracy | 0.830 | **0.891** | +6.1pp |
| Precision | 0.780 | 0.800 | +2.0pp |
| Recall (toxic) | **0.810** | 0.775 | -3.5pp |
| F1-score | 0.790 | 0.787 | ~0 |
| Macro F1 | ~0.79 | **0.857** | +6.7pp |

### 3.3 Key Observations

1. **Accuracy**: WangchanBERTa leads by +6.1 percentage points (89.1% vs 83.0%). Substantial improvement driven by better non-toxic classification.

2. **Toxic Recall**: LR (Balanced) actually achieves higher raw toxic recall (0.81 vs 0.775). `class_weight='balanced'` aggressively upweights the toxic class, and TF-IDF representation is biased toward toxic keywords.

3. **Macro F1**: WangchanBERTa's macro F1 (0.857) is significantly higher than LR (~0.79), indicating more balanced performance across both classes. LR achieves high toxic recall at the cost of more false positives on non-toxic content.

4. **Precision**: WangchanBERTa is slightly more precise (0.80 vs 0.78), meaning fewer false alarms for moderators.

5. **Context understanding**: WangchanBERTa handles sarcasm, slang, code-switching, and ambiguous language significantly better than TF-IDF (qualitative observation from error analysis).

---

## 4. Inference Performance

### 4.1 Latency Comparison

Latency measurements are from external benchmarks on architecturally equivalent models (RoBERTa-base for WangchanBERTa; scikit-learn TF-IDF+LR pipeline). No latency profiling was performed within the THAI-MOD notebooks.

| Model | Hardware | Latency (ms) | Throughput (req/s) |
|---|---|---|---|
| TF-IDF + LR | CPU (any) | 0.5-1 | ~1,000 |
| WangchanBERTa | CPU (8-core) | 45-70 | 15-20 |
| WangchanBERTa | Apple MPS (M2) | 8-15 | 70-120 |
| WangchanBERTa | GPU (T4) | 3-5 | 200-300 |

### 4.2 Speed Ratio

| WangchanBERTa Hardware | Slowdown vs LR (CPU) |
|---|---|
| CPU (8-core) | 50-100x slower |
| Apple MPS | 10-20x slower |
| GPU (T4) | 3-7x slower |

### 4.3 Latency Analysis

- LR latency (<1ms) is imperceptible to the user
- BERT on GPU (3-5ms) is also imperceptible
- BERT on CPU (45-70ms) is noticeable but acceptable for async moderation queues
- BERT on CPU is **not suitable** for high-throughput synchronous pipelines (>100 req/s)

---

## 5. Resource Requirements

| Dimension | TF-IDF + LR | WangchanBERTa |
|---|---|---|
| Model file size | 5-20 MB | ~500 MB |
| RAM at inference | 50-200 MB | 1.3-1.5 GB |
| Cold-start time | <1 s | 6-15 s |
| GPU required? | No | Recommended |
| Python dependencies | scikit-learn, PyThaiNLP | + PyTorch, Transformers |
| Parameters | ~20K features | ~125M parameters |

### 5.1 Resource Analysis

1. **Model size**: WangchanBERTa is ~25-100x larger on disk. Impacts container image size and download time.

2. **RAM**: WangchanBERTa requires ~7-10x more RAM. A free-tier cloud instance (512MB-1GB) cannot run BERT but handles LR easily.

3. **Cold-start**: WangchanBERTa takes 6-15 seconds to load weights. Makes serverless deployment impractical without model warming. LR loads in under 1 second.

4. **GPU dependency**: WangchanBERTa without GPU is viable but 50-100x slower. For production throughput, GPU is effectively required.

---

## 6. Qualitative Comparison

### 6.1 Strengths of TF-IDF + LR

- Extremely fast inference on any hardware
- No GPU dependency; runs on minimal resources
- High toxic recall when using `class_weight='balanced'`
- Interpretable: TF-IDF feature weights can be inspected directly
- Instant cold-start; suitable for serverless deployment
- Simple to debug and maintain

### 6.2 Weaknesses of TF-IDF + LR

- Bag-of-words representation: no understanding of word order or context
- Struggles with sarcasm, irony, and implicit toxicity
- Cannot handle novel slang or misspellings not in the training vocabulary
- Code-switching handled at token level only (no cross-language context)
- Higher false positive rate due to keyword sensitivity

### 6.3 Strengths of WangchanBERTa

- Bidirectional contextual understanding of full sentences
- Pre-trained on Thai social media text: understands Thai-specific patterns
- Subword tokenization handles unknown words and misspellings gracefully
- Better at sarcasm, implicit toxicity, and ambiguous language
- Code-switching handled naturally through shared subword vocabulary
- Significantly higher accuracy and macro F1

### 6.4 Weaknesses of WangchanBERTa

- 50-100x slower on CPU; GPU recommended for production
- 500MB model size; 1.3-1.5GB RAM required
- 6-15 second cold-start; unsuitable for serverless
- Less interpretable than TF-IDF (black-box attention mechanism)
- Mild overfitting observed (val loss increased at epoch 3)
- Requires PyTorch + Transformers dependencies (~2GB installed)

---

## 7. Deployment Recommendation

### 7.1 Current Decision: Deploy LR Baseline

For the Phase 2 prototype and course demonstration, **TF-IDF + Logistic Regression (Balanced)** is deployed:

1. No exported BERT artifact available at time of Phase 2 development
2. Demo reliability: instant startup, no GPU dependency
3. Acceptable baseline performance (83% accuracy, 81% toxic recall)
4. Human-in-the-loop design absorbs model errors

### 7.2 Future Target: WangchanBERTa

WangchanBERTa should replace the LR baseline when:

1. A fine-tuned artifact is exported and validated
2. Deployment environment has GPU access (or CPU latency of 50-70ms is acceptable)
3. RAM budget allows 1.5GB+ for the model service

The current Phase 2 model service does not yet auto-load BERT artifacts. Replacing the LR baseline with WangchanBERTa will require implementing transformer loading and inference in `src/thai_mod_api/model_service.py`, while keeping the existing API contract unchanged.

### 7.3 Hybrid Option

A potential middle ground for resource-constrained environments:

- Use LR for real-time synchronous moderation (sub-millisecond response)
- Use WangchanBERTa for async batch review (background processing of flagged content)
- This provides LR speed for immediate response and BERT accuracy for final decisions

---

## 8. Summary

| Criterion | TF-IDF + LR | WangchanBERTa |
|---|---|---|
| Accuracy | 83.0% | **89.1%** |
| Toxic Recall | **81.0%** | 77.5% |
| Macro F1 | ~79% | **85.7%** |
| Inference (CPU) | **<1ms** | 45-70ms |
| Inference (GPU) | N/A | **3-5ms** |
| Model Size | **~20MB** | ~500MB |
| RAM Usage | **~200MB** | ~1.5GB |
| Context Understanding | None (bag-of-words) | **Full (attention)** |
| Sarcasm/Slang | Weak | **Strong** |
| Code-switching | Moderate | **Strong** |
| Interpretability | **High** | Low |
| Deployment Simplicity | **Trivial** | Complex |

**Bottom line**: WangchanBERTa is the better model for production toxicity detection, offering +6pp accuracy and significantly better context understanding. TF-IDF + LR is the better choice for prototyping, demos, and resource-constrained environments, with competitive toxic recall and near-zero infrastructure requirements.

---

## Data Sources and Limitations

- Classification metrics are from the THAI-MOD project notebooks (`model.ipynb`, `thai-bert.ipynb`)
- Latency and resource figures are from external benchmarks, not measured within THAI-MOD:
  - Teyssier (2021), "BERT inference cost/performance analysis CPU vs GPU"
  - Jakovljevic (2023), "BERT inference throughput deathmatch"
  - Nature Scientific Reports (2025), Table 13: BERT/RoBERTa inference benchmarks
- WangchanBERTa toxic recall is estimated from macro F1 (per-class recall not saved in notebook)
- Val loss increase at epoch 3 suggests mild overfitting; 3 epochs may not be optimal
- All comparisons use the same test set (n = 6,124) and preprocessing pipeline
