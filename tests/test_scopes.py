"""Tests for scope inference and term matching."""
from __future__ import annotations

import pytest
from memos_lite.scopes import _has_any_term, filter_memories_for_query, infer_scope


class TestHasAnyTerm:
    def test_english_word_boundary(self) -> None:
        # "did" should NOT match "didn't"
        assert not _has_any_term("I didn't finish the task", ("did",))

    def test_english_word_boundary_in_sentence(self) -> None:
        # "did" should NOT match "candidate"
        assert not _has_any_term("The candidate performed well", ("did",))

    def test_english_word_boundary_exact(self) -> None:
        # "did" SHOULD match standalone "did"
        assert _has_any_term("What did you do?", ("did",))

    def test_cjk_substring_match(self) -> None:
        # CJK terms should match regardless of word boundaries
        assert _has_any_term("讨论双重差分方法", ("双重差分",))

    def test_cjk_partial_in_longer_text(self) -> None:
        assert _has_any_term("这篇论文使用了面板数据固定效应模型", ("面板数据",))

    def test_health_terms(self) -> None:
        assert _has_any_term("My blood pressure reading was high", ("blood pressure",))

    def test_quant_terms(self) -> None:
        # _has_any_term is case-sensitive; lowercase input works
        assert _has_any_term("Running a did regression analysis", ("did",))

    def test_empty_terms(self) -> None:
        assert not _has_any_term("hello world", ())

    def test_empty_text(self) -> None:
        assert not _has_any_term("", ("did",))


class TestInferScope:
    def test_explicit_scope_preserved(self) -> None:
        assert infer_scope(scope="project:myproj") == "project:myproj"

    def test_profile_from_metadata(self) -> None:
        assert infer_scope(metadata={"profile_name": "researcher"}) == "profile:researcher"

    def test_project_from_metadata(self) -> None:
        assert infer_scope(metadata={"project_id": "backend"}) == "project:backend"

    def test_health_domain(self) -> None:
        assert infer_scope(content="My blood pressure is elevated") == "domain:health"

    def test_research_domain(self) -> None:
        assert infer_scope(content="Running a hypothesis test on the dataset") == "domain:research"

    def test_paper_domain(self) -> None:
        assert infer_scope(content="Reviewing a manuscript for the journal") == "domain:paper"

    def test_quant_domain(self) -> None:
        assert infer_scope(content="Quant trading strategy analysis") == "domain:quant"

    def test_news_domain(self) -> None:
        assert infer_scope(content="Breaking news headline today") == "domain:news"

    def test_global_fallback(self) -> None:
        assert infer_scope(content="Just a random conversation") == "global"


class TestFilterMemoriesForQuery:
    def test_global_memory_always_included(self) -> None:
        memories = [{"scope": "global", "id": "1"}, {"scope": "domain:health", "id": "2"}]
        # Generic query with no domain hint: only global memories pass
        result = filter_memories_for_query(memories, query="random")
        assert len(result) == 1
        assert result[0]["scope"] == "global"

    def test_explicit_scope_filter(self) -> None:
        memories = [
            {"scope": "global", "id": "1"},
            {"scope": "project:backend", "id": "2"},
            {"scope": "domain:health", "id": "3"},
        ]
        result = filter_memories_for_query(memories, query="random", explicit_scope="project:backend")
        assert all(m["scope"] in {"global", "project:backend"} for m in result)

    def test_domain_hint_expands_scopes(self) -> None:
        memories = [
            {"scope": "global", "id": "1"},
            {"scope": "domain:health", "id": "2"},
        ]
        result = filter_memories_for_query(memories, query="blood pressure reading")
        scopes = {m["scope"] for m in result}
        assert "domain:health" in scopes
