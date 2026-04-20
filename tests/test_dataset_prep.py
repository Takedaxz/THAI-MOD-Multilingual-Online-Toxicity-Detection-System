"""
Unit tests for ToxicityModelService._prepare_dataset() and _load_full_dataset().

These tests validate the label mapping, NaN handling, deduplication, and output
structure described in:
  - progress1.txt  §3.1  "Label Mapping", "Handling Missing Data",
                         "Data De-duplication"
  - progress2.txt  §2.2  "Data Processing"
  - model_service.py     _prepare_dataset / _load_full_dataset

Tests use in-memory DataFrames written to pytest's tmp_path fixture, so they
run without the large Git-LFS dataset CSVs.  This keeps the suite fast and
CI-compatible.

Label mapping logic under test (from _prepare_dataset source):
    df["category"] = df["category"].replace({"pos": "neu"})
    df["category"] = df["category"].map({"neg": 1, "neu": 0})
    → neg  → 1  (Toxic)
    → neu  → 0  (Non-toxic)
    → pos  → neu → 0  (Non-toxic)
    → anything else → NaN → dropped
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.thai_mod_api.model_service import ToxicityModelService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(tmp_path: Path) -> ToxicityModelService:
    """Return a ToxicityModelService whose project_root is tmp_path."""
    return ToxicityModelService(project_root=tmp_path)


def _csv(tmp_path: Path, rows: list[dict], name: str = "dataset_test.csv") -> Path:
    """Write rows to a CSV file inside tmp_path and return the path."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------------------------


class TestLabelMapping:
    """Each source label must map to the correct binary integer."""

    def test_neg_maps_to_1(self, tmp_path):
        path = _csv(tmp_path, [{"texts": "ไอ้เหี้ย แม่ง", "category": "neg"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 1
        assert result.iloc[0]["category"] == 1

    def test_neu_maps_to_0(self, tmp_path):
        path = _csv(tmp_path, [{"texts": "สวัสดีครับ ขอบคุณ", "category": "neu"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 1
        assert result.iloc[0]["category"] == 0

    def test_pos_remapped_to_neu_then_maps_to_0(self, tmp_path):
        """pos → neu → 0 (two-step mapping in _prepare_dataset)."""
        path = _csv(tmp_path, [{"texts": "ดีมาก ขอบคุณ", "category": "pos"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 1
        assert result.iloc[0]["category"] == 0

    def test_unknown_label_row_dropped(self, tmp_path):
        """Labels outside {neg, neu, pos} become NaN and are dropped."""
        path = _csv(tmp_path, [{"texts": "ข้อความ", "category": "xyz"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 0

    def test_mixed_labels_mapped_correctly(self, tmp_path):
        rows = [
            {"texts": "ด่า fuck you", "category": "neg"},
            {"texts": "ขอบคุณมาก", "category": "neu"},
            {"texts": "ดีมาก", "category": "pos"},
        ]
        path = _csv(tmp_path, rows)
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 3
        # Check both toxic and non-toxic rows exist
        assert 1 in result["category"].values
        assert 0 in result["category"].values

    def test_category_dtype_is_integer(self, tmp_path):
        """category must be stored as int, not float (after NaN dropna)."""
        rows = [
            {"texts": "ไม่ดีเลย", "category": "neg"},
            {"texts": "โอเค", "category": "neu"},
        ]
        path = _csv(tmp_path, rows)
        result = _service(tmp_path)._prepare_dataset(path)
        assert result["category"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# NaN / Invalid Row Handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """Rows with NaN texts or category must be dropped (progress1 §3.1)."""

    def test_nan_texts_row_is_dropped(self, tmp_path):
        rows = [
            {"texts": None, "category": "neg"},
            {"texts": "ข้อความปกติ", "category": "neu"},
        ]
        path = _csv(tmp_path, rows)
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 1
        assert result.iloc[0]["category"] == 0

    def test_nan_category_row_is_dropped(self, tmp_path):
        rows = [
            {"texts": "ดี", "category": None},
            {"texts": "ข้อความ", "category": "neu"},
        ]
        path = _csv(tmp_path, rows)
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 1

    def test_whitespace_only_text_dropped_after_preprocess(self, tmp_path):
        """Text that becomes empty after strip must be excluded."""
        path = _csv(tmp_path, [{"texts": "   ", "category": "neg"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 0

    def test_url_only_text_dropped_after_preprocess(self, tmp_path):
        """URL-only text reduces to "" after preprocess → dropped."""
        path = _csv(tmp_path, [{"texts": "http://spam.com", "category": "neg"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 0

    def test_all_nan_returns_empty_dataframe(self, tmp_path):
        rows = [
            {"texts": None, "category": None},
            {"texts": None, "category": "neg"},
        ]
        path = _csv(tmp_path, rows)
        result = _service(tmp_path)._prepare_dataset(path)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Output Structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """_prepare_dataset must return exactly {texts, category, source}."""

    def test_exactly_three_output_columns(self, tmp_path):
        path = _csv(tmp_path, [{"texts": "ข้อความ", "category": "neu"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert set(result.columns) == {"texts", "category", "source"}

    def test_source_column_equals_csv_filename(self, tmp_path):
        path = _csv(tmp_path, [{"texts": "ข้อความ", "category": "neg"}], "dataset1.csv")
        result = _service(tmp_path)._prepare_dataset(path)
        assert result.iloc[0]["source"] == "dataset1.csv"

    def test_texts_column_contains_preprocessed_text(self, tmp_path):
        """texts column must hold preprocessed (URL-stripped, lowercase) text."""
        path = _csv(tmp_path, [{"texts": "Hello World", "category": "neu"}])
        result = _service(tmp_path)._prepare_dataset(path)
        assert result.iloc[0]["texts"] == "hello world"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Duplicate texts (after preprocessing) must be removed (progress1 §3.1)."""

    def test_duplicate_texts_reduced_to_one(self, tmp_path):
        rows = [
            {"texts": "ข้อความซ้ำ", "category": "neg"},
            {"texts": "ข้อความซ้ำ", "category": "neg"},
            {"texts": "ข้อความใหม่", "category": "neu"},
        ]
        path = _csv(tmp_path, rows, "dataset1.csv")
        svc = _service(tmp_path)
        svc.dataset_files = [path]
        result = svc._load_full_dataset()
        # No duplicate texts should remain
        assert result["texts"].duplicated().sum() == 0

    def test_dedup_preserves_unique_rows(self, tmp_path):
        rows = [
            {"texts": "ข้อความ A", "category": "neg"},
            {"texts": "ข้อความ B", "category": "neu"},
        ]
        path = _csv(tmp_path, rows, "dataset1.csv")
        svc = _service(tmp_path)
        svc.dataset_files = [path]
        result = svc._load_full_dataset()
        assert len(result) == 2

    def test_load_full_dataset_includes_reviewed_examples(self, tmp_path):
        path = _csv(tmp_path, [{"texts": "base text", "category": "neu"}], "dataset1.csv")
        reviewed_path = tmp_path / "models" / "reviewed" / "reviewed_comments.csv"
        reviewed_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "request_id": "abc",
                    "texts": "new toxic slang",
                    "category": "neg",
                    "source": "reviewed_traffic",
                }
            ]
        ).to_csv(reviewed_path, index=False)

        svc = _service(tmp_path)
        svc.dataset_files = [path]
        result = svc._load_full_dataset()

        assert len(result) == 2
        assert "reviewed_traffic" in result["source"].values
        reviewed = result[result["source"] == "reviewed_traffic"].iloc[0]
        assert reviewed["category"] == 1
