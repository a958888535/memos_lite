"""Tests for the shared CJK-aware tokenizer."""
from __future__ import annotations

import pytest
from memos_lite.memos_tokenize import cjk_aware_tokens, expand_cjk


class TestExpandCjk:
    def test_expand_cjk_bigrams(self) -> None:
        result = expand_cjk("研究", sizes=(2,))
        assert "研究" in result
        assert len(result) >= 1

    def test_expand_cjk_trigrams(self) -> None:
        result = expand_cjk("机器学习", sizes=(2, 3))
        assert "机器学" in result
        assert "机器学习" in result

    def test_expand_cjk_mixed_size(self) -> None:
        result = expand_cjk("深度学习", sizes=(2, 3))
        assert "深度学" in result
        assert "深度学习" in result

    def test_expand_cjk_empty(self) -> None:
        assert expand_cjk("") == set()
        # Non-CJK text still gets n-gram expansion (expected behavior)
        assert len(expand_cjk("hello")) > 0


class TestCjkAwareTokens:
    def test_ascii_tokens(self) -> None:
        tokens = cjk_aware_tokens("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_cjk_tokens(self) -> None:
        tokens = cjk_aware_tokens("研究量化金融")
        assert "研究" in tokens
        assert "量化" in tokens
        assert "金融" in tokens

    def test_mixed_content(self) -> None:
        tokens = cjk_aware_tokens("DID method in 金融量化 analysis")
        assert "did" in tokens
        assert "method" in tokens
        assert "in" in tokens
        assert "金融" in tokens

    def test_cjk_ngram_expansion(self) -> None:
        tokens = cjk_aware_tokens("机器学习")
        assert "机器学习" in tokens
        assert "机器学" in tokens

    def test_lowercase_normalization(self) -> None:
        tokens = cjk_aware_tokens("PYTHON JavaScript")
        assert "python" in tokens
        assert "javascript" in tokens

    def test_short_tokens_not_filtered(self) -> None:
        # cjk_aware_tokens returns all regex-matched tokens including short ones
        tokens = cjk_aware_tokens("a bc def")
        assert "a" in tokens
        assert "bc" in tokens
        assert "def" in tokens

    def test_empty_string(self) -> None:
        assert cjk_aware_tokens("") == set()
        assert cjk_aware_tokens("   ") == set()
