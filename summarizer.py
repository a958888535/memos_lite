from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from .policy import ensure_safe_base_url


def extract_l0(text: str, *, limit: int = 160) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    for separator in (". ", "。", "\n"):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0]
            break
    return cleaned[:limit].strip()


def extractive_digest(messages: List[Dict[str, Any]], *, limit: int = 400) -> str:
    parts = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        parts.append(f"{role}: {content}")
    digest = " ".join(parts)
    return digest[:limit].strip()


class OpenAICompatibleSummarizer:
    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.base_url = ensure_safe_base_url(config.get("base_url", ""), purpose="memory_summarize").rstrip("/")
        self.api_key_env = config.get("api_key_env", "")
        self.timeout_seconds = float(config.get("timeout_seconds", 60))

    def summarize(self, text: str) -> Dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not self.base_url or not self.model or not api_key:
            raise RuntimeError("Summarizer is enabled but not fully configured")
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Summarize this conversation fragment into a compact episodic memory "
                            "with one sentence and optional structured fields.\n\n"
                            f"{text}"
                        ),
                    }
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
        return {
            "summary": content,
            "l1_json": json.dumps({"summary": content}, ensure_ascii=False) if content else None,
        }
