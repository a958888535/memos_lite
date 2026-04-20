from __future__ import annotations

import difflib
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from .embedding import cosine_similarity, same_embedding_space
from .noise import should_auto_retrieve
from .scopes import filter_memories_for_query


from .memos_tokenize import cjk_aware_tokens as _tokens


def _lexical_score(query: str, item: dict) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    haystack = " ".join(
        part
        for part in [
            item.get("content"),
            item.get("summary"),
            item.get("l0"),
        ]
        if part
    )
    overlap = query_tokens & _tokens(haystack)
    return len(overlap) / len(query_tokens)


def _recency_decay(updated_at: str, half_life_days: float) -> float:
    try:
        updated = datetime.fromisoformat(updated_at)
    except Exception:
        return 0.0
    age_days = max((datetime.now(timezone.utc) - updated).total_seconds() / 86400.0, 0.0)
    if half_life_days <= 0:
        return 0.0
    return math.pow(0.5, age_days / half_life_days)


class RetrievalEngine:
    def __init__(
        self,
        store,
        *,
        config: Dict[str, Any] | None = None,
        embedding_provider=None,
    ):
        self._store = store
        self._config = {
            "mode": "hybrid",
            "vector_weight": 0.65,
            "fts_weight": 0.35,
            "candidate_pool_size": 24,
            "top_k": 8,
            "min_score": 0.30,
            "hard_min_score": 0.35,
            "max_chars": 6000,
            "length_norm_anchor": 500,
            "mmr_diversity_threshold": 0.85,
            "recency_half_life_days": 30,
            "recency_weight": 0.10,
            "importance_weight": 0.10,
            "diagnostics": True,
        }
        if config:
            self._config.update(config)
        self._embedding_provider = embedding_provider
        self.last_diagnostics: Dict[str, Any] = {}

    def _new_diagnostics(self) -> Dict[str, Any]:
        return {
            "fts_candidate_count": 0,
            "vector_candidate_count": 0,
            "fused_candidate_count": 0,
            "final_result_count": 0,
            "dropped_by_min_score": 0,
            "dropped_by_scope": 0,
            "dropped_by_noise": 0,
            "embedding_error": None,
        }

    def retrieve(
        self,
        query: str,
        *,
        scope: str | None = None,
        tags: List[str] | None = None,
        manual: bool = False,
        top_k: int | None = None,
        allow_vector: bool = True,
    ) -> List[dict]:
        diagnostics = self._new_diagnostics()
        if not should_auto_retrieve(query, manual=manual):
            diagnostics["dropped_by_noise"] = 1
            self.last_diagnostics = diagnostics
            return []

        mode = self._config.get("mode", "hybrid")
        effective_mode = "fts" if not allow_vector and mode == "vector" else mode
        pool = int(self._config.get("candidate_pool_size", 24))
        lexical_candidates = self._store.search(query, limit=pool, scope=scope, tags=tags)
        lexical_candidates = filter_memories_for_query(
            lexical_candidates,
            query=query,
            explicit_scope=scope,
        )
        diagnostics["fts_candidate_count"] = len(lexical_candidates)

        candidates: Dict[str, dict] = {}
        for item in lexical_candidates:
            candidate = dict(item)
            candidate["_fts_score"] = _lexical_score(query, item)
            candidate["_vector_score"] = 0.0
            candidates[item["id"]] = candidate

        if allow_vector and effective_mode in {"hybrid", "vector"} and self._embedding_provider is not None:
            try:
                query_vector = self._embedding_provider.embed_texts([query])[0]
                query_model = getattr(self._embedding_provider, "model", None)
                query_dim = getattr(self._embedding_provider, "embedding_dim", None) or len(query_vector)
                eligible_items: List[dict] = []
                for item in self._store.iter_memories(scope=scope, tags=tags, has_embedding=True):
                    if not filter_memories_for_query([item], query=query, explicit_scope=scope):
                        diagnostics["dropped_by_scope"] += 1
                        continue
                    eligible_items.append(item)

                if query_model is None:
                    compatible_models = {
                        item.get("embedding_model")
                        for item in eligible_items
                        if item.get("embedding_model") and item.get("embedding_dim") == query_dim
                    }
                    if len(compatible_models) == 1:
                        query_model = next(iter(compatible_models))

                vector_rows = []
                for item in eligible_items:
                    embedding = item.get("embedding") or []
                    if not embedding:
                        continue
                    if not same_embedding_space(
                        item.get("embedding_model"),
                        item.get("embedding_dim"),
                        query_model,
                        query_dim,
                    ):
                        continue
                    score = cosine_similarity(query_vector, embedding)
                    vector_rows.append((score, item))
                vector_rows.sort(key=lambda pair: pair[0], reverse=True)
                diagnostics["vector_candidate_count"] = len(vector_rows)
                for score, item in vector_rows[:pool]:
                    candidate = candidates.get(item["id"], dict(item))
                    candidate["_fts_score"] = candidate.get("_fts_score", 0.0)
                    candidate["_vector_score"] = max(score, candidate.get("_vector_score", 0.0))
                    candidates[item["id"]] = candidate
            except Exception as exc:
                diagnostics["embedding_error"] = str(exc)

        fused: List[dict] = []
        for candidate in candidates.values():
            fts_score = candidate.get("_fts_score", 0.0)
            vector_score = candidate.get("_vector_score", 0.0)
            if effective_mode == "fts":
                score = fts_score
            elif effective_mode == "vector":
                score = vector_score
            else:
                score = (
                    float(self._config.get("vector_weight", 0.65)) * vector_score
                    + float(self._config.get("fts_weight", 0.35)) * fts_score
                )
            updated_at = candidate.get("updated_at") or candidate.get("created_at") or ""
            score += float(self._config.get("recency_weight", 0.10)) * _recency_decay(
                updated_at,
                float(self._config.get("recency_half_life_days", 30)),
            )
            score += float(self._config.get("importance_weight", 0.10)) * float(
                candidate.get("importance") or 0.0
            )
            anchor = max(int(self._config.get("length_norm_anchor", 500)), 1)
            content_length = len(candidate.get("content") or "")
            length_factor = min(anchor / max(content_length, 1), 1.0)
            score *= 0.75 + (0.25 * length_factor)
            candidate["_score"] = score
            if score >= float(self._config.get("min_score", 0.30)):
                fused.append(candidate)
            else:
                diagnostics["dropped_by_min_score"] += 1

        fused.sort(key=lambda item: item["_score"], reverse=True)
        diagnostics["fused_candidate_count"] = len(fused)
        diversified: List[dict] = []
        threshold = float(self._config.get("mmr_diversity_threshold", 0.85))
        for candidate in fused:
            text = candidate.get("content") or candidate.get("l0") or ""
            if any(
                difflib.SequenceMatcher(None, text, existing.get("content") or existing.get("l0") or "").ratio() >= threshold
                for existing in diversified
            ):
                continue
            diversified.append(candidate)
            if len(diversified) >= int(top_k or self._config.get("top_k", 8)):
                break

        for item in diversified:
            item["score"] = round(float(item.pop("_score", 0.0)), 4)
            item.pop("_fts_score", None)
            item.pop("_vector_score", None)

        diagnostics["final_result_count"] = len(diversified)
        self.last_diagnostics = diagnostics
        return diversified
