"""Shared CJK-aware tokenization utilities.

Centralizes the bigram/trigram expansion logic previously duplicated across
retrieval.py, __init__.py, and skill_hint.py.
"""
from __future__ import annotations

import re

# Matches words (ASCII alphanumeric + underscore) or contiguous CJK runs.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def expand_cjk(token: str, sizes: tuple[int, ...] = (2, 3)) -> set[str]:
    """Return the original token plus its CJK n-gram pieces."""
    pieces: set[str] = {token}
    for size in sizes:
        if len(token) < size:
            continue
        for i in range(len(token) - size + 1):
            pieces.add(token[i : i + size])
    return pieces


def cjk_aware_tokens(text: str) -> set[str]:
    """Tokenize *text* with CJK n-gram expansion.

    ASCII tokens are kept as-is.  Each CJK run is expanded into bigrams
    and trigrams so that substring queries can still match.
    """
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(expand_cjk(token))
        else:
            tokens.add(token)
    return tokens
