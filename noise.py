from __future__ import annotations

import re

from .redaction import contains_secret


# ---- Tunable thresholds (magic numbers centralised here) ----
_DEFAULT_MIN_STORE_CHARS = 40   # minimum text length for storage
_MAX_NEWLINES_TOOL_OUTPUT = 60  # reject tool output with more line breaks
_MIN_RETRIEVE_CHARS = 12        # minimum query length for auto-retrieval

_ACKS = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "好的",
    "继续",
    "收到",
    "got it",
    "sure",
}
_GREETINGS = {"hi", "hello", "hey", "你好", "嗨"}
_MEMORY_CUES = (
    "remember",
    "previously",
    "last time",
    "earlier",
    "as i said",
    "before",
    "之前",
    "上次",
    "我以前说过",
    "记得",
    "记住",
    "按之前",
    "继续上次",
)
_EMOJI_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_SLASH_COMMAND_RE = re.compile(r"^/[a-zA-Z0-9_-]+$")


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def has_memory_cue(text: str) -> bool:
    lowered = _normalize(text)
    return any(cue in lowered for cue in _MEMORY_CUES)


def should_store_text(text: str, *, source: str = "conversation", min_chars: int = _DEFAULT_MIN_STORE_CHARS) -> bool:
    stripped = str(text or "").strip()
    normalized = _normalize(stripped)
    if not stripped:
        return False
    if contains_secret(stripped):
        return False
    if normalized in _GREETINGS or normalized in _ACKS:
        return False
    if _EMOJI_ONLY_RE.match(stripped):
        return False
    if _SLASH_COMMAND_RE.match(stripped):
        return False
    if source == "tool_result":
        return False
    if "i can't help with that" in normalized or "cannot assist" in normalized:
        return False
    if "exit code:" in normalized or stripped.count("\n") > _MAX_NEWLINES_TOOL_OUTPUT:
        return False
    if len(stripped) < min_chars and not has_memory_cue(stripped):
        return False
    return True


def should_auto_retrieve(query: str, *, manual: bool = False) -> bool:
    if manual:
        return True
    stripped = str(query or "").strip()
    normalized = _normalize(stripped)
    if not stripped:
        return False
    if has_memory_cue(stripped):
        return True
    if normalized in _GREETINGS or normalized in _ACKS:
        return False
    if _EMOJI_ONLY_RE.match(stripped):
        return False
    if _SLASH_COMMAND_RE.match(stripped):
        return False
    if len(stripped) < _MIN_RETRIEVE_CHARS:
        return False
    return True
