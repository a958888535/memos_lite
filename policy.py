from __future__ import annotations


DISABLE_FALLBACK_TO_AGENT_MODEL = True
MEMORY_MODEL_PURPOSES = {
    "embedding",
    "memory_summarize",
    "memory_dedup",
    "memory_recall_filter",
}
BLOCKED_BASE_URL_FRAGMENTS = (
    "/coding/",
    "/api/coding/paas",
    "api.kimi.com/coding",
)


def ensure_safe_base_url(base_url: str, *, purpose: str) -> str:
    normalized = str(base_url or "").strip()
    lowered = normalized.lower()
    if purpose not in MEMORY_MODEL_PURPOSES:
        raise RuntimeError(f"Unsupported memory model purpose: {purpose}")
    for fragment in BLOCKED_BASE_URL_FRAGMENTS:
        if fragment in lowered:
            raise RuntimeError(
                f"Blocked memory model endpoint for {purpose}: {base_url}"
            )
    return normalized
