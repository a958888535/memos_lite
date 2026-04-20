from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "backend": "local",
    "capture": {
        "enabled": True,
        "include_user": True,
        "include_assistant": True,
        "include_tool_results": False,
        "exclude_system": True,
        "min_chars": 40,
    },
    "embedding": {
        "enabled": False,
        "provider": "siliconflow",
        "model": "BAAI/bge-m3",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "batch_size": 16,
        "timeout_seconds": 60,
        "encoding_format": "float",
    },
    "summarizer": {
        "enabled": False,
        "provider": "openai_compatible",
        "model": "",
        "base_url": "",
        "api_key_env": "",
        "threshold_chars": 1200,
        "purpose": "memory_summarize",
    },
    "retrieval": {
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
    },
    "skill_hint": {
        "enabled": True,
        "min_evidence_count": 3,
        "min_confidence": 0.70,
        "once_per_session_per_workflow": True,
        "max_memory_ids": 2,
        "sensitive_domains": ["health", "credentials", "secrets"],
        "format": "single_metadata_line",
    },
    "scopes": {
        "default": "global",
        "health_default": "domain:health",
        "quant_default": "domain:quant",
        "paper_default": "domain:paper",
    },
    "safety": {
        "redact_secrets": True,
        "forbid_coding_plan_for_memory": True,
        "disable_fallback_to_agent_model": True,
    },
}


TOOL_SCHEMAS = [
    {
        "name": "memos_search",
        "description": "Search local memos_lite memory by query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {"type": "integer", "description": "Maximum results to return."},
                "scope": {"type": "string", "description": "Optional scope filter."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filters.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memos_get",
        "description": "Get one memory by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "memos_remember",
        "description": "Persist an explicit memory in local memos_lite storage.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content to store."},
                "scope": {"type": "string", "description": "Optional explicit scope."},
                "domain": {"type": "string", "description": "Optional domain label."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags.",
                },
                "source": {"type": "string", "description": "Optional source label."},
                "metadata": {"type": "object", "description": "Optional metadata object."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memos_forget",
        "description": "Delete a local memory by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id to delete."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "memos_timeline",
        "description": "List recent local memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum results to return."},
                "scope": {"type": "string", "description": "Optional scope filter."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filters.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "memos_status",
        "description": "Show memos_lite provider status and diagnostics.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def merge_config(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return merge_config(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})
