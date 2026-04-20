from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import httpx

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
    """Build a compact digest by alternately sampling from user and assistant turns.

    Instead of naively concatenating all messages (which biases toward the start
    of the conversation), we take the *first* and *last* utterance of each role
    and at most ``max_middle`` evenly-spaced middle samples.  This preserves the
    opening question, the concluding answer, and a representative middle.
    """
    if not messages:
        return ""

    # Partition messages by role, preserving order within each role.
    by_role: Dict[str, List[str]] = {}
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        by_role.setdefault(role, []).append(content)

    max_middle = 2  # max intermediate samples per role
    sampled: List[str] = []
    for role in ("user", "assistant"):
        turns = by_role.get(role, [])
        if not turns:
            continue
        indices: list[int] = []
        if len(turns) == 1:
            indices = [0]
        elif len(turns) == 2:
            indices = [0, 1]
        else:
            # first, last, and up to max_middle evenly-spaced middle indices
            indices = [0, len(turns) - 1]
            step = max((len(turns) - 1) // (max_middle + 1), 1)
            for i in range(1, max_middle + 1):
                idx = min(i * step, len(turns) - 1)
                if idx not in indices:
                    indices.append(idx)
            indices.sort()
        for idx in indices:
            sampled.append(f"{role}: {turns[idx]}")

    digest = " ".join(sampled)
    return digest[:limit].strip()


class OpenAICompatibleSummarizer:
    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.base_url = (
            ensure_safe_base_url(config.get("base_url", ""), purpose="memory_summarize")
            .rstrip("/")
        )
        self.api_key_env = config.get("api_key_env", "")
        self.timeout_seconds = float(config.get("timeout_seconds", 60))
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
                                "Summarize this conversation fragment into a compact episodic memory "
                                "with one sentence and optional structured fields.\n\n"
                                f"{text}"
                            ),
                        }
                    ],
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
        return {
            "summary": content,
            "l1_json": json.dumps({"summary": content}, ensure_ascii=False) if content else None,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

