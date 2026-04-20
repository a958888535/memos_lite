from __future__ import annotations

import re
import threading
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, MutableMapping, Set


_TOKEN_ALIASES = {
    "流程": {"workflow"},
    "工作流": {"workflow"},
    "事件研究": {"event", "study", "event_study"},
    "文献综述": {"literature", "review", "literature_review"},
    "论文": {"paper"},
    "研究": {"research"},
}


def infer_workflow_key(content: str, *, tags: Iterable[str] | None = None, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    if metadata.get("workflow_key"):
        return str(metadata["workflow_key"])
    for tag in tags or []:
        if str(tag).startswith("workflow:"):
            return str(tag).split(":", 1)[1]
    lowered = str(content or "").lower()
    if "event study" in lowered:
        return "finance_event_study"
    if "literature review matrix" in lowered:
        return "paper_literature_review_matrix"
    return ""


class SkillHintEmitter:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self._enabled = bool(config.get("enabled", True))
        self._min_evidence_count = int(config.get("min_evidence_count", 3))
        self._min_confidence = float(config.get("min_confidence", 0.70))
        self._once_per_session = bool(config.get("once_per_session_per_workflow", True))
        self._max_memory_ids = int(config.get("max_memory_ids", 2))
        self._max_sessions = max(int(config.get("max_sessions", 256)), 1)
        self._sensitive_domains = {str(item) for item in config.get("sensitive_domains", ["health", "credentials", "secrets"])}
        self._emitted: MutableMapping[str, Set[str]] = OrderedDict()
        self._lock = threading.RLock()

    def _tokenize(self, text: str) -> Set[str]:
        tokens = set()
        for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", str(text or "").lower()):
            if token.isascii():
                if len(token) >= 3:
                    tokens.add(token)
            elif len(token) >= 2:
                tokens.add(token)
                for size in (2, 3):
                    if len(token) < size:
                        continue
                    for index in range(len(token) - size + 1):
                        piece = token[index : index + size]
                        tokens.add(piece)
                        tokens.update(_TOKEN_ALIASES.get(piece, set()))
                tokens.update(_TOKEN_ALIASES.get(token, set()))
        return tokens

    def _match_score(self, query: str, workflow_key: str) -> int:
        query_tokens = self._tokenize(query)
        workflow_tokens = self._tokenize(workflow_key.replace("_", " "))
        return len(query_tokens & workflow_tokens)

    def _candidate_sort_key(self, candidate: dict) -> tuple[float, float, float, str]:
        return (
            float(candidate["match_score"]),
            float(candidate["evidence_count"]),
            float(candidate["confidence"]),
            float(candidate["access_score"]),
            str(candidate["workflow_key"]),
        )

    def _session_markers(self, session_id: str) -> Set[str]:
        with self._lock:
            key = str(session_id or "")
            markers = self._emitted.pop(key, set())
            self._emitted[key] = markers
            while len(self._emitted) > self._max_sessions:
                self._emitted.popitem(last=False)
            return markers

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._emitted.pop(str(session_id or ""), None)

    def maybe_emit(self, query: str, memories: List[dict], *, session_id: str) -> str:
        if not self._enabled:
            return ""
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for memory in memories:
            workflow_key = str(memory.get("workflow_key") or "").strip()
            if not workflow_key:
                continue
            domain = str(memory.get("domain") or "").strip().lower()
            if domain in self._sensitive_domains or memory.get("scope") == "domain:health":
                continue
            grouped[workflow_key].append(memory)

        candidates: List[dict] = []
        for workflow_key, items in grouped.items():
            if len(items) < self._min_evidence_count:
                continue
            confidence = sum(float(item.get("confidence") or 0.0) for item in items) / len(items)
            if confidence < self._min_confidence:
                continue
            candidates.append(
                {
                    "workflow_key": workflow_key,
                    "memories": items,
                    "confidence": confidence,
                    "evidence_count": len(items),
                    "access_score": sum(float(item.get("access_count") or 0.0) for item in items),
                    "match_score": self._match_score(query, workflow_key),
                }
            )

        if not candidates:
            return ""
        matched = [candidate for candidate in candidates if candidate["match_score"] > 0]
        if not matched:
            return ""
        candidates = matched
        candidates.sort(key=self._candidate_sort_key, reverse=True)
        best = candidates[0]
        best_workflow = str(best["workflow_key"])
        best_memories = list(best["memories"])
        if self._once_per_session:
            markers = self._session_markers(session_id)
            if best_workflow in markers:
                return ""
            markers.add(best_workflow)
        memory_ids = ",".join(item["id"] for item in best_memories[: self._max_memory_ids])
        return (
            f'[memos_lite_hint skill_update_possible=true workflow_key="{best_workflow}" '
            f'evidence_count={len(best_memories)} memory_ids="{memory_ids}"]'
        )
