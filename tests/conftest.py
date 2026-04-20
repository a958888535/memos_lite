"""Shared pytest fixtures for memos_lite tests."""
from __future__ import annotations

from pathlib import Path

import pytest

# Defer importing LocalStore until after pytest has set up, avoiding the
# datasketch.aio / scipy / numpy version-conflict crash at import time.
# Tests that need LocalStore must access it via the `tmp_store` fixture
# (which calls this lazily).


@pytest.fixture
def tmp_store(tmp_path: Path):
    """A LocalStore backed by a temporary directory, auto-closed on teardown."""
    # Import here to avoid the datasketch.aio / scipy / numpy conflict.
    from memos_lite.local_store import LocalStore

    store = LocalStore(tmp_path / "store")
    yield store
    store.close()


@pytest.fixture
def store_with_memories(tmp_store):
    """A LocalStore pre-populated with 5 distinct memories across scopes."""
    tmp_store.remember(content="The user discussed Python async/await patterns", scope="global", tags=["python"])
    tmp_store.remember(content="Fixed the null pointer exception in the auth module", scope="global", tags=["bugfix"])
    tmp_store.remember(
        content="Research paper on difference-in-differences methodology",
        scope="domain:research",
        tags=["paper", "did"],
    )
    tmp_store.remember(content="Set up a CI pipeline for the backend service", scope="project:backend", tags=["devops"])
    tmp_store.remember(content="Bookmarked a tutorial on Hugging Face transformers", scope="global", tags=["ml"])
    return tmp_store
