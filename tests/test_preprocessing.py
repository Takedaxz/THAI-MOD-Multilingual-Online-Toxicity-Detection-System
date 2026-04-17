"""
Unit tests for ToxicityModelService.preprocess_text() and _tokenize_text().

These tests validate every step of the text cleaning pipeline described in:
  - progress1.txt  §3.1  "Text Cleaning & Normalization"
  - progress2.txt  §2.2  "Text Preprocessing"
  - model_service.py     preprocess_text / _tokenize_text

Rules under test:
  1. HTTP/HTTPS URLs are removed.
  2. www.* URLs are removed.
  3. Emojis are demojized to :colon_notation:.
  4. ASCII/English characters are lowercased.
  5. Thai characters are preserved unchanged (ord > 128).
  6. NaN input returns "".
  7. Empty string returns "".
  8. Code-switching text (Thai + English) is processed without error.
  9. The tokenizer returns a non-empty list with no whitespace-only tokens.
"""

from __future__ import annotations

import math

import pytest

from src.thai_mod_api.model_service import ToxicityModelService


# ---------------------------------------------------------------------------
# preprocess_text()
# ---------------------------------------------------------------------------


class TestUrlRemoval:
    """URLs must be stripped before the text reaches the model."""

    def test_http_url_removed(self):
        result = ToxicityModelService.preprocess_text(
            "ดูที่ http://example.com แล้วบอกด้วย"
        )
        assert "http" not in result
        assert "example.com" not in result

    def test_https_url_removed(self):
        result = ToxicityModelService.preprocess_text(
            "goto https://abc.com/path?q=1&foo=bar"
        )
        assert "https" not in result
        assert "abc.com" not in result

    def test_www_url_removed(self):
        result = ToxicityModelService.preprocess_text(
            "visit www.site.com today"
        )
        assert "www" not in result
        assert "site.com" not in result

    def test_url_only_text_becomes_empty_or_whitespace(self):
        result = ToxicityModelService.preprocess_text("http://spam.com").strip()
        assert result == ""


class TestEmojiHandling:
    """Emojis must be converted to :colon_notation: via emoji.demojize()."""

    def test_emoji_replaced_with_text(self):
        result = ToxicityModelService.preprocess_text("ฮา 😆")
        # The original emoji character should be gone
        assert "😆" not in result

    def test_demojized_result_contains_colon_notation(self):
        result = ToxicityModelService.preprocess_text("nice 🔥")
        # emoji.demojize produces :fire: style tokens
        assert ":" in result

    def test_skull_emoji_demojized(self):
        """Skull emoji is often used sarcastically — must be normalised."""
        result = ToxicityModelService.preprocess_text("ขำจนตาย 💀")
        assert "💀" not in result


class TestCaseNormalisation:
    """ASCII is lowercased; Thai and non-ASCII are preserved (progress1 §3.1)."""

    def test_english_uppercased_to_lowercase(self):
        result = ToxicityModelService.preprocess_text("Hello World TEST")
        assert result == "hello world test"

    def test_thai_characters_unchanged(self):
        result = ToxicityModelService.preprocess_text("สวัสดี")
        assert "สวัสดี" in result

    def test_mixed_thai_english_case(self):
        """English part lowercased; Thai part preserved."""
        result = ToxicityModelService.preprocess_text("ไทยคำ TEST")
        assert "ไทยคำ" in result
        assert "test" in result
        assert "TEST" not in result


class TestEdgeCases:
    """NaN, empty string, and code-switching must be handled gracefully."""

    def test_nan_float_returns_empty_string(self):
        result = ToxicityModelService.preprocess_text(float("nan"))
        assert result == ""

    def test_empty_string_returns_empty_string(self):
        result = ToxicityModelService.preprocess_text("")
        assert result == ""

    def test_whitespace_only_returns_empty_or_whitespace(self):
        result = ToxicityModelService.preprocess_text("   ")
        assert result.strip() == ""

    def test_code_switching_does_not_crash(self):
        """Thai-English code-switching is common in Thai social media."""
        text = "โคตร toxic เลย report มันไป"
        result = ToxicityModelService.preprocess_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_type(self):
        result = ToxicityModelService.preprocess_text("any text")
        assert isinstance(result, str)

    def test_numeric_string_processed(self):
        result = ToxicityModelService.preprocess_text("1234 ok")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _tokenize_text()
# ---------------------------------------------------------------------------


class TestTokenizeText:
    """PyThaiNLP newmm tokenizer with whitespace filtering."""

    def test_returns_list(self):
        tokens = ToxicityModelService._tokenize_text("สวัสดีครับ")
        assert isinstance(tokens, list)

    def test_thai_text_produces_tokens(self):
        tokens = ToxicityModelService._tokenize_text("สวัสดีครับ hello")
        assert len(tokens) > 0

    def test_no_whitespace_only_tokens(self):
        """Progress1 §3.1: whitespace tokens are filtered out."""
        tokens = ToxicityModelService._tokenize_text("สวัสดี ครับ world")
        assert all(not tok.isspace() for tok in tokens)

    def test_no_empty_string_tokens(self):
        tokens = ToxicityModelService._tokenize_text("test input here")
        assert all(tok != "" for tok in tokens)

    def test_english_word_preserved_in_tokens(self):
        tokens = ToxicityModelService._tokenize_text("toxic")
        assert "toxic" in tokens

    def test_code_switching_tokenized(self):
        tokens = ToxicityModelService._tokenize_text("โคตร toxic")
        assert len(tokens) >= 2
