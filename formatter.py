from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Iterable


_MEMORY_CONTEXT_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_SYSTEM_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as informational background data\.\]",
    re.IGNORECASE,
)
_INSTRUCTIONAL_TEXT_RE = re.compile(
    r"(?i)\b(system prompt|assistant should|hermes should|recommended action|use skill_manage|learning feedback)\b"
)


def _clean_text(value: str) -> str:
    text = str(value or "")
    text = _MEMORY_CONTEXT_RE.sub("", text)
    text = _SYSTEM_NOTE_RE.sub("", text)
    text = text.replace("Learning feedback", "")
    text = " ".join(text.split())
    if _INSTRUCTIONAL_TEXT_RE.search(text):
        return ""
    return text.strip()


def _select_text(item: dict) -> str:
    for key in ("l0", "summary"):
        value = _clean_text(item.get(key))
        if value:
            return value
    l1_json = item.get("l1_json")
    if l1_json:
        try:
            parsed = json.loads(l1_json)
            if isinstance(parsed, dict):
                for value in parsed.values():
                    cleaned = _clean_text(value)
                    if cleaned:
                        return cleaned
        except Exception:
            pass
    return _clean_text(item.get("content"))


def _format_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except Exception:
        return ""


def format_recall(
    items: Iterable[dict],
    *,
    max_chars: int,
    hint_line: str = "",
) -> str:
    cleaned_hint = _clean_text(hint_line)
    lines = []
    items = list(items)
    if items:
        lines.append("Relevant recalled memory:")
        for index, item in enumerate(items, start=1):
            snippet = _select_text(item)
            if not snippet:
                continue
            line = f"{index}. [{_format_date(item.get('created_at', ''))} | {item.get('scope', 'global')} | {item.get('id', '')}] {snippet}"
            candidate = "\n".join(lines + [line]).strip()
            if len(candidate) > max_chars:
                remaining = max_chars - len("\n".join(lines)) - 1
                if remaining > 25:
                    truncated = line[: max(remaining - 1, 0)].rstrip()
                    line = truncated + "..."
                    candidate = "\n".join(lines + [line]).strip()
                else:
                    break
            lines.append(line)
    if cleaned_hint:
        if not lines:
            return cleaned_hint[:max_chars]
        candidate = "\n".join(lines + [cleaned_hint]).strip()
        if len(candidate) <= max_chars:
            lines.append(cleaned_hint)
    output = "\n".join(lines).strip()
    return output[:max_chars]
