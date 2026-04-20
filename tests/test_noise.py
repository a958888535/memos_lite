"""Tests for noise filtering."""
from __future__ import annotations

import pytest
from memos_lite.noise import (
    has_memory_cue,
    should_auto_retrieve,
    should_store_text,
)


class TestShouldStoreText:
    def test_short_greeting_rejected(self) -> None:
        assert should_store_text("hi") is False
        assert should_store_text("hello") is False
        assert should_store_text("你好") is False

    def test_emoji_only_rejected(self) -> None:
        assert should_store_text("😀👍🎉") is False

    def test_slash_command_rejected(self) -> None:
        assert should_store_text("/help") is False
        assert should_store_text("/status") is False

    def test_tool_result_rejected(self) -> None:
        assert should_store_text("Here is the file content", source="tool_result") is False

    def test_long_meaningful_text_accepted(self) -> None:
        text = "The user discussed how to implement a difference-in-differences analysis using panel data with fixed effects"
        assert should_store_text(text) is True

    def test_min_chars_threshold(self) -> None:
        assert should_store_text("short text", min_chars=5) is True
        assert should_store_text("short", min_chars=10) is False

    def test_memory_cue_overrides_min_chars(self) -> None:
        assert should_store_text("remember that", min_chars=1000) is True

    def test_empty_rejected(self) -> None:
        assert should_store_text("") is False
        assert should_store_text("   ") is False

    def test_exit_code_rejected(self) -> None:
        long_text = "exit code: 1\n" + "\n".join(["error at line " + str(i) for i in range(100)])
        assert should_store_text(long_text) is False


class TestShouldAutoRetrieve:
    def test_short_query_rejected(self) -> None:
        assert should_auto_retrieve("hi") is False
        assert should_auto_retrieve("hello world") is False

    def test_memory_cue_accepted(self) -> None:
        assert should_auto_retrieve("remember what we discussed before") is True
        assert should_auto_retrieve("as I said earlier") is True

    def test_meaningful_long_query_accepted(self) -> None:
        assert should_auto_retrieve("What did we decide about the DID analysis approach?") is True

    def test_greeting_rejected(self) -> None:
        assert should_auto_retrieve("hi") is False
        assert should_auto_retrieve("hello there") is False


class TestHasMemoryCue:
    def test_positive_cases(self) -> None:
        assert has_memory_cue("Remember what we discussed") is True
        assert has_memory_cue("as I said before") is True
        assert has_memory_cue("之前讨论的方案") is True

    def test_negative_cases(self) -> None:
        assert has_memory_cue("I don't remember the address") is False
        assert has_memory_cue("Please remind me") is False
