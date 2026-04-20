from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import httpx

from .policy import ensure_safe_base_url


# Patterns to strip from content before extracting l0 or building digest.
_NOISE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Markdown links: [text](url) → text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # HTML/Markdown images: ![alt](url)
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    # URLs standalone
    (re.compile(r"https?://\S+"), ""),
    # System markers like [CONTEXT COMPACTION ...] or [System note: ...]
    (re.compile(r"\[(?:CONTEXT\s+COMPACTION|System\s+note?)[^\]]*\]", re.IGNORECASE), ""),
    # Feishu-style @mentions: @_user_1  or @_all
    (re.compile(r"@_\w+"), ""),
    # Tool call artifacts: <function_calls>...</function_calls>
    (re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL), ""),
    # XML/HTML tags
    (re.compile(r"<[^>]+>"), ""),
    # Leading/trailing markdown bold/italic
    (re.compile(r"^[*_]+|[*_]+$"), ""),
]


def _strip_noise(text: str) -> str:
    """Remove markdown URLs, system markers, and other non-content noise."""
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    # Collapse whitespace
    return " ".join(text.split())


def extract_l0(text: str, *, limit: int = 160) -> str:
    """Extract the first substantive sentence as a compact label (l0).

    Strips markdown URLs, system markers, and other noise before extracting.
    """
    cleaned = _strip_noise(str(text or ""))
    if not cleaned:
        return ""
    for separator in (". ", "。", "\n"):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0]
            break
    result = cleaned[:limit].strip()
    # If result looks like pure noise (no CJK or alpha chars), return empty
    if result and not re.search(r"[\u4e00-\u9fff a-zA-Z]", result):
        return ""
    return result


def _clean_summary(raw: str) -> str:
    """Strip markdown formatting that glm-4.7-flash tends to add.

    The model often wraps its answer in **Summary:** headers and bullet
    lists.  We want a single clean sentence for memory storage.
    """
    if not raw:
        return ""
    # Remove markdown bold markers
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", raw)
    # Remove leading label lines like "Summary:", "Sentence:", "Structured Fields:"
    lines = []
    skip_labels = {"summary", "summary:", "sentence", "sentence:", "structured fields", "structured fields:"}
    for line in cleaned.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.rstrip(":").lower() in skip_labels:
            continue
        # Skip bullet lines from structured fields section
        if s.startswith("* ") or s.startswith("- "):
            continue
        lines.append(s)
    # Take the first substantive line (the actual summary sentence)
    for line in lines:
        if len(line) >= 10:
            return line.strip()
    # Fallback: return the longest line
    if lines:
        return max(lines, key=len).strip()
    return cleaned.strip()[:200]


def _tfidf_top(message: str, all_messages: list[str], top_k: int = 12) -> str:
    """Return a compact version of *message* keeping only the most distinctive tokens.

    Uses a simple TF-IDF heuristic: tokens that appear in fewer messages across
    the conversation are more distinctive.  We keep sentences that score highest.
    """
    if len(message) <= 120:
        return message

    # Token frequency across all messages (simple word-level)
    doc_freq: dict[str, int] = {}
    n_docs = len(all_messages)
    for doc in all_messages:
        seen = set(doc.lower().split())
        for tok in seen:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    # Score each sentence by average IDF of its tokens
    sentences = re.split(r"(?<=[.。!?！？])\s+|\n", message)
    best = message[:120]
    best_score = -1.0
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        toks = s.lower().split()
        if not toks:
            continue
        score = sum(
            (n_docs / max(doc_freq.get(t, 1), 1))
            for t in toks
        ) / len(toks)
        if score > best_score:
            best_score = score
            best = s[:120]
    return best


def extractive_digest(messages: List[Dict[str, Any]], *, limit: int = 600) -> str:
    """Build a compact digest preserving high-information-density turns.

    Strategy:
    1. Always include the first user message (the question/topic).
    2. Score every message by TF-IDF distinctiveness.
    3. Pick the top-N scoring messages, interleaving user/assistant roles.
    4. Always include the last assistant message (the conclusion).
    5. Strip noise (URLs, system markers) from each selected turn.
    """
    if not messages:
        return ""

    # Partition by role, preserving order
    by_role: Dict[str, List[str]] = {}
    all_texts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned = _strip_noise(content)
        if not cleaned or len(cleaned) < 5:
            continue
        by_role.setdefault(role, []).append(cleaned)
        all_texts.append(cleaned)

    if not all_texts:
        return ""

    max_turns = 8  # total turns to sample
    sampled: List[str] = []

    for role in ("user", "assistant"):
        turns = by_role.get(role, [])
        if not turns:
            continue

        if len(turns) <= 3:
            # Few turns: keep them all, truncated
            for t in turns:
                sampled.append(f"{role}: {_tfidf_top(t, all_texts)}")
        else:
            # Always first + last, then top distinctive middles
            picked = {0, len(turns) - 1}
            # Score middle turns
            scored: list[tuple[float, int]] = []
            for idx in range(1, len(turns) - 1):
                t = turns[idx]
                toks = set(t.lower().split())
                if not toks:
                    continue
                # IDF-based distinctiveness
                n = len(all_texts)
                score = sum(
                    n / max(sum(1 for d in all_texts if tok in d.lower()), 1)
                    for tok in toks
                ) / max(len(toks), 1)
                scored.append((score, idx))
            # Pick top distinctive middles
            scored.sort(reverse=True)
            for _, idx in scored[:max(max_turns // 2 - 2, 1)]:
                picked.add(idx)
            for idx in sorted(picked):
                sampled.append(f"{role}: {_tfidf_top(turns[idx], all_texts)}")

    digest = " ".join(sampled)
    return digest[:limit].strip()


class OpenAICompatibleSummarizer:
    """LLM-backed summarizer using OpenAI-compatible chat completions API.

    For glm-4.7-flash, thinking is disabled so the summary lands directly
    in ``message.content``.  A post-processing step strips any markdown
    formatting the model adds despite the one-sentence prompt.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.base_url = (
            ensure_safe_base_url(config.get("base_url", ""), purpose="memory_summarize")
            .rstrip("/")
        )
        self.api_key_env = config.get("api_key_env", "")
        try:
            self.timeout_seconds = float(config.get("timeout_seconds", 60))
        except (TypeError, ValueError):
            self.timeout_seconds = 60.0
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                headers={
                    "Authorization": f"Bearer {os.getenv(self.api_key_env, '').strip()}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def summarize(self, text: str) -> Dict[str, Any]:
        if not self.base_url or not self.model or not os.getenv(self.api_key_env, "").strip():
            raise RuntimeError("Summarizer is enabled but not fully configured")
        try:
            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Summarize this conversation fragment into one compact sentence "
                                "for episodic memory storage. Output ONLY the summary sentence, "
                                "nothing else.\n\n"
                                f"{text}"
                            ),
                        }
                    ],
                    # glm-4.7-flash enables thinking by default which puts
                    # output into reasoning_content instead of content.
                    "thinking": {"type": "disabled"},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Summarizer HTTP error {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Summarizer request failed: {exc}") from exc

        choices = payload.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()

        # Post-process: strip markdown the model may add despite instructions
        content = _clean_summary(content)

        return {
            "summary": content,
            "l1_json": json.dumps({"summary": content}, ensure_ascii=False) if content else None,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
