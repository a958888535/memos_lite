from __future__ import annotations

import math
import os
from typing import Iterable, List

import requests

from .policy import ensure_safe_base_url


def same_embedding_space(
    model_a: str | None,
    dim_a: int | None,
    model_b: str | None,
    dim_b: int | None,
) -> bool:
    return bool(model_a and model_b and dim_a and dim_b and model_a == model_b and dim_a == dim_b)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_vec = list(left)
    right_vec = list(right)
    if not left_vec or not right_vec or len(left_vec) != len(right_vec):
        return 0.0
    dot = sum(a * b for a, b in zip(left_vec, right_vec))
    left_norm = math.sqrt(sum(a * a for a in left_vec))
    right_norm = math.sqrt(sum(b * b for b in right_vec))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class SiliconFlowEmbeddingProvider:
    def __init__(self, config: dict):
        self.model = config.get("model", "BAAI/bge-m3")
        self.base_url = ensure_safe_base_url(
            config.get("base_url", "https://api.siliconflow.cn/v1"),
            purpose="embedding",
        ).rstrip("/")
        self.api_key_env = config.get("api_key_env", "SILICONFLOW_API_KEY")
        self.batch_size = int(config.get("batch_size", 16))
        self.timeout_seconds = float(config.get("timeout_seconds", 60))
        self.encoding_format = config.get("encoding_format", "float")
        self.embedding_dim: int | None = None

    def _headers(self) -> dict:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing embedding API key in {self.api_key_env}")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": self.encoding_format,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or []
            for item in data:
                embedding = item.get("embedding") or []
                if self.embedding_dim is None:
                    self.embedding_dim = len(embedding)
                results.append([float(value) for value in embedding])
        return results
