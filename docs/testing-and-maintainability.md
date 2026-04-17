# Testing & Maintainability — THAI-MOD

## Overview

This document describes the automated test suite and CI pipeline introduced
in the `feat/testing-ci-pipeline` branch of THAI-MOD.  It satisfies the
project requirements for:

- **Testing & Maintainability** (requirements.md §6)
- **MLOps & Deployment: basic automation** (requirements.md §5)
- **Responsible ML: transparency through testability** (requirements.md §9)

---

## How to Run Tests Locally

```bash
# 1. Activate the conda environment
conda activate cedt

# 2. Install production + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Run all tests with verbose output
pytest tests/ -v

# 4. Run with coverage report
pytest tests/ -v --cov=src/thai_mod_api --cov-report=term-missing
```

Expected output: **all tests pass**, coverage ≥ 70 % on `src/thai_mod_api`.

---

## Test Structure

```text
tests/
├── __init__.py
├── conftest.py            # Shared fixtures (mock API client)
├── test_preprocessing.py  # Unit: preprocess_text, _tokenize_text
├── test_dataset_prep.py   # Unit: _prepare_dataset, _load_full_dataset
├── test_api.py            # Integration: API contract, auth, monitoring endpoints
└── test_monitoring_service.py  # Unit: MonitoringService summary and drift helpers
```

### `conftest.py` — Mock Model Bundle

The `api_client` fixture (session-scoped) pre-injects a mock `bundle` into
the module-level `model_service` object before `TestClient` starts.

Because `ensure_ready()` checks `if self.bundle is None`, setting the bundle
first makes it a no-op.  **No dataset files and no training are required.**

The mock pipeline always returns `toxic_score = 0.75`, making test assertions
deterministic regardless of the environment.

---

## Test Inventory

### `test_preprocessing.py` — 20 tests

Tests every cleaning rule from **progress1.txt §3.1** and **progress2.txt §2.2**:

| Class | What it tests |
|-------|--------------|
| `TestUrlRemoval` | http, https, www URLs removed; URL-only text → "" |
| `TestEmojiHandling` | emoji → `:colon_notation:` via `emoji.demojize()` |
| `TestCaseNormalisation` | English lowercased; Thai preserved (ord > 128) |
| `TestEdgeCases` | NaN, empty string, whitespace, code-switching, None |
| `TestTokenizeText` | PyThaiNLP newmm output type, no whitespace tokens |

### `test_dataset_prep.py` — 18 tests

Tests the label mapping and data-cleaning logic in `_prepare_dataset()`:

| Class | What it tests |
|-------|--------------|
| `TestLabelMapping` | `neg→1`, `neu→0`, `pos→0`, unknown→dropped, dtype=int |
| `TestNaNHandling` | NaN texts, NaN category, whitespace-only, URL-only |
| `TestOutputStructure` | Exactly 3 columns: `texts`, `category`, `source` |
| `TestDeduplication` | `_load_full_dataset` dedup on preprocessed `texts` |

**Why in-memory DataFrames?**
The real dataset CSVs (~300 MB total) are managed via Git LFS and are not
pulled in CI.  Writing rows to `tmp_path` (pytest's built-in temp-dir
fixture) gives equivalent coverage without requiring large files.

### `test_api.py` — 20 tests

Integration tests for the two primary endpoints:

| Class | Endpoint | What it tests |
|-------|----------|--------------|
| `TestHealthEndpoint` | `GET /api/health` | HTTP 200, `status`, `model_loaded`, `cache_status` |
| `TestPredictEndpoint` | `POST /api/predict` | Fields, value ranges, decision policy, validation |

Decision policy assertions (from **progress2.txt §2.2**):
- `mock_score (0.75) >= default_threshold (0.4)` → `toxic` / `FLAG_FOR_REVIEW`
- `mock_score (0.75) < threshold (0.95)` → `non-toxic` / `ALLOW`
- Custom threshold is reflected in the response

Pydantic validation assertions:
- Empty `text` → HTTP 422
- Missing `text` → HTTP 422
- `threshold > 1.0` → HTTP 422
- `threshold < 0.0` → HTTP 422

---

## CI Pipeline (`.github/workflows/ci.yml`)

```text
Trigger: push to any branch  OR  pull request to main

Steps:
  1. Checkout  (lfs: false — datasets not needed)
  2. Setup Python 3.11  with pip cache
  3. pip install -r requirements.txt
  4. pip install -r requirements-dev.txt
  5. Cache PyThaiNLP dictionaries  (~/pythainlp-data)
  6. Pre-download PyThaiNLP data  (warms up the cache)
  7. pytest tests/ --cov=src/thai_mod_api --cov-fail-under=70
```

Every push automatically triggers this workflow.  A **green checkmark**
appears in the GitHub Actions tab.  Pull requests to `main` are blocked from
merging if tests fail.

---

## Testing Strategy & Design Decisions

### Why mock the model — not train it in CI?

Training the TF-IDF + Logistic Regression model from the 8 datasets takes
**2–5 minutes** and requires the large Git-LFS CSV files.  The tests exercise
the *API contract* and *decision logic*, not the ML training itself.  The real
model is validated through the notebooks (`model.ipynb`).

This approach is consistent with best practice: unit/integration tests should
be fast and deterministic.

### Why use `scope="session"` for the TestClient fixture?

A single `TestClient` is created once and reused across all API tests.
This reflects how the app behaves in production (one server process,
model loaded once) and keeps the test suite fast.

### Why write dataset tests with `tmp_path` instead of real CSV files?

- Keeps tests self-contained and reproducible without LFS
- Isolates the logic under test from accidental dataset changes
- Avoids encoding issues from large multilingual CSV files in test output

### Why `--cov-fail-under=70`?

70 % is a practical baseline for a prototype.  The uncovered code is
primarily the transformer fine-tuning paths (GPU-dependent) and the
`_train_bundle` heavy path.  These are validated through notebook runs,
not automated tests.

---

## Known Technical Debt

| Item | Description | Priority |
|------|-------------|----------|
| No authentication tests | `/` and `/admin` have no auth; no tests cover it | Medium |
| `_train_bundle` untested | Training from CSV not covered (requires LFS + time) | Low |
| BERT inference untested | WangchanBERTa is notebook-only; no API test | Medium |
| No negative-scenario batch tests | `POST /api/batch-predict` is tested by schema only | Low |
| No performance / latency test | Progress2 targets <200 ms latency but no automated check | Low |
| Coverage threshold 70 % | Aim for ≥ 80 % once BERT service is integrated | Medium |

---

## Presentation Talking Points

1. **"We run automated tests on every commit"** — preprocessing,
   label mapping, API contract, auth, and monitoring tests covering documented
   design decisions from our progress reports.

2. **"Tests are fast and CI-friendly"** — mock bundle strategy means no
   training occurs; full suite completes in <60 seconds locally.

3. **"Our CI blocks merges if tests break"** — GitHub Actions runs on every
   push; green checkmark required before merging to `main`.

4. **"Tests document our ML decisions"** — threshold logic, recall-first
   recommendations, label mapping, and preprocessing rules are all
   explicitly verified by test assertions.

5. **"We identified our technical debt honestly"** — the table above reflects
   known gaps and the rationale for each.
