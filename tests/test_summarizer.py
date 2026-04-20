"""Tests for extractive_digest role-alternating sampling."""
from __future__ import annotations

import pytest
from memos_lite.summarizer import extractive_digest


class TestExtractiveDigest:
    def test_empty_input(self) -> None:
        assert extractive_digest([]) == ""

    def test_single_user_message(self) -> None:
        messages = [{"role": "user", "content": "Hello world"}]
        digest = extractive_digest(messages)
        assert "user: Hello world" in digest

    def test_single_assistant_message(self) -> None:
        messages = [{"role": "assistant", "content": "How can I help?"}]
        digest = extractive_digest(messages)
        assert "assistant: How can I help?" in digest

    def test_role_alternating_preserves_first_and_last(self) -> None:
        # Create 5 messages per role
        messages = (
            [{"role": "user", "content": f"Question {i}"} for i in range(5)]
            + [{"role": "assistant", "content": f"Answer {i}"} for i in range(5)]
        )
        digest = extractive_digest(messages)
        # First and last user messages must appear
        assert "user: Question 0" in digest
        assert "user: Question 4" in digest
        # First and last assistant messages must appear
        assert "assistant: Answer 0" in digest
        assert "assistant: Answer 4" in digest

    def test_limit_truncation(self) -> None:
        messages = [{"role": "user", "content": "x" * 300}, {"role": "assistant", "content": "y" * 300}]
        digest = extractive_digest(messages, limit=100)
        assert len(digest) <= 100

    def test_non_user_assistant_skipped(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "tool", "content": "Tool output"},
            {"role": "user", "content": "Real user message"},
        ]
        digest = extractive_digest(messages)
        assert "system" not in digest
        assert "tool" not in digest
        assert "Real user message" in digest
