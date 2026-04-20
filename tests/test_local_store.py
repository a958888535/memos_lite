"""Tests for LocalStore: dedup, read/write, scope isolation."""
from __future__ import annotations

import pytest
from memos_lite.local_store import LocalStore


class TestDedup:
    def test_exact_duplicate_returns_existing(self, tmp_store: LocalStore) -> None:
        m1 = tmp_store.remember(content="Deploy the backend to production", scope="global", l0="deploy backend")
        m2 = tmp_store.remember(content="Deploy the backend to production", scope="global", l0="deploy backend")
        assert m2["id"] == m1["id"]

    def test_near_duplicate_same_scope(self, tmp_store: LocalStore) -> None:
        m1 = tmp_store.remember(
            content="Research on panel data regression with fixed effects",
            scope="global",
            l0="固定效应面板数据回归",
        )
        m2 = tmp_store.remember(
            content="Fixed effects panel data regression analysis",
            scope="global",
            l0="固定效应面板数据回归",
        )
        assert m2["id"] == m1["id"]

    def test_different_scope_not_duplicated(self, tmp_store: LocalStore) -> None:
        m1 = tmp_store.remember(content="Build a quant model", scope="domain:quant", l0="量化模型")
        m2 = tmp_store.remember(content="Build a quant model", scope="global", l0="量化模型")
        assert m2["id"] != m1["id"]

    def test_far_different_content_not_duplicated(self, tmp_store: LocalStore) -> None:
        m1 = tmp_store.remember(content="Python async tutorial", scope="global")
        m2 = tmp_store.remember(content="Hugging Face NLP guide", scope="global")
        assert m2["id"] != m1["id"]

    def test_merge_tags_on_dedup(self, tmp_store: LocalStore) -> None:
        m1 = tmp_store.remember(content="CI pipeline setup", scope="global", tags=["devops"])
        m2 = tmp_store.remember(content="CI pipeline setup", scope="global", tags=["python"])
        merged = tmp_store.get(m1["id"])
        assert merged is not None
        assert "devops" in merged["tags"]
        assert "python" in merged["tags"]


class TestScopeIsolation:
    def test_iter_memories_filters_scope(self, store_with_memories: LocalStore) -> None:
        memories = store_with_memories.iter_memories(scope="project:backend")
        assert all(m["scope"] in {"project:backend", "global"} for m in memories)

    def test_iter_memories_filters_has_embedding(self, tmp_store: LocalStore) -> None:
        tmp_store.remember(content="Without embedding", scope="global")
        memories = tmp_store.iter_memories(has_embedding=True)
        assert len(memories) == 0

    def test_iter_memories_filters_tags(self, store_with_memories: LocalStore) -> None:
        memories = store_with_memories.iter_memories(tags=["ml"])
        assert all("ml" in m["tags"] for m in memories)


class TestCRUD:
    def test_get_existing_memory(self, store_with_memories: LocalStore) -> None:
        all_memories = store_with_memories.iter_memories(scope="global")
        first = all_memories[0]
        retrieved = store_with_memories.get(first["id"])
        assert retrieved is not None
        assert retrieved["id"] == first["id"]

    def test_get_nonexistent_returns_none(self, tmp_store: LocalStore) -> None:
        assert tmp_store.get("nonexistent_id") is None

    def test_forget(self, store_with_memories: LocalStore) -> None:
        all_memories = store_with_memories.iter_memories(scope="global")
        mid = all_memories[0]["id"]
        assert store_with_memories.forget(mid) is True
        assert store_with_memories.get(mid) is None

    def test_count(self, store_with_memories: LocalStore) -> None:
        counts = store_with_memories.counts()
        assert counts["memory_count"] >= 5


class TestMinHashLSHHelper:
    def test_lsh_helper_lazy_build(self, tmp_store: LocalStore) -> None:
        # Index should not be built until first query
        helper = tmp_store.MinHashHelper(tmp_store._conn)
        assert helper._built is False
        helper.ensure_index()
        assert helper._built is True

    def test_lsh_helper_scope_filtering(self, tmp_store: LocalStore) -> None:
        tmp_store.remember(content="Quant finance model", scope="domain:quant")
        tmp_store.remember(content="Python async tutorial", scope="global")
        helper = tmp_store.MinHashHelper(tmp_store._conn)
        helper.ensure_index()
        candidates = helper.query("量化金融", "domain:quant")
        assert all(
            tmp_store._conn.execute(
                "SELECT scope FROM memories WHERE id = ?", (mid,)
            ).fetchone()[0]
            == "domain:quant"
            for mid in candidates
        )

    def test_lsh_helper_invalidate(self, tmp_store: LocalStore) -> None:
        helper = tmp_store.MinHashHelper(tmp_store._conn)
        helper.ensure_index()
        assert helper._built is True
        helper.invalidate()
        assert helper._built is False

    def test_lsh_helper_query_after_insert(self, tmp_store: LocalStore) -> None:
        # Insert happens inside remember(), helper is lazily rebuilt
        m1 = tmp_store.remember(content="Deploy the backend to production", scope="global")
        # trigger dedup path which rebuilds index
        m2 = tmp_store.remember(content="Deploy the backend to production", scope="global")
        assert m2["id"] == m1["id"]  # dedup detected via MinHash path
