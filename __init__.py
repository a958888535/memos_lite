from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .embedding import SiliconFlowEmbeddingProvider
from .formatter import format_recall
from .local_store import LocalStore
from .noise import should_store_text
from .policy import DISABLE_FALLBACK_TO_AGENT_MODEL
from .redaction import contains_secret, redact_text
from .retrieval import RetrievalEngine
from .schemas import TOOL_SCHEMAS, load_config
from .scopes import infer_scope
from .skill_hint import SkillHintEmitter, infer_workflow_key
from .summarizer import OpenAICompatibleSummarizer, extract_l0, extractive_digest

logger = logging.getLogger(__name__)

_SENTINEL = object()
_READ_ONLY_TOOLS = {"memos_search", "memos_get", "memos_timeline", "memos_status"}
_WRITE_TOOLS = {"memos_remember", "memos_forget"}
_SHUTDOWN_WAIT_SECONDS = 5.0
_COMPANION_SKILL_NAME = "memos-lite-hint-policy"
_COMPANION_SKILL_DESCRIPTION = (
    "Interpret memos_lite skill-update hint metadata and decide whether native Hermes "
    "skill_manage should be used."
)
from .memos_tokenize import cjk_aware_tokens


_PREFETCH_STOPWORDS = {
    "a", "an", "and", "as", "continue", "for", "help", "i",
    "last", "me", "my", "of", "on", "please", "the", "to", "write",
}


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _query_tokens(text: str) -> set[str]:
    return cjk_aware_tokens(text) - _PREFETCH_STOPWORDS


def _cache_matches_query(cached_query: str, current_query: str) -> bool:
    cached_norm = _normalize_query(cached_query)
    current_norm = _normalize_query(current_query)
    if not cached_norm or not current_norm:
        return False
    if cached_norm == current_norm:
        return True
    cached_tokens = _query_tokens(cached_norm)
    current_tokens = _query_tokens(current_norm)
    if not cached_tokens or not current_tokens:
        return False
    overlap = cached_tokens & current_tokens
    return (len(overlap) / min(len(cached_tokens), len(current_tokens))) >= 0.6


class MemosLiteMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._session_id = ""
        self._config: Dict[str, Any] = {}
        self._data_dir = Path()
        self._store: LocalStore | None = None
        self._retrieval: RetrievalEngine | None = None
        self._embedding_provider = None
        self._summarizer = None
        self._skill_hint = None
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._closer_threads: List[threading.Thread] = []
        self._closer_lock = threading.Lock()
        self._prefetch_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._shutdown = False
        self._write_enabled = True

    @property
    def name(self) -> str:
        return "memos_lite"

    def is_available(self) -> bool:
        try:
            conn = sqlite3.connect(":memory:")
            conn.close()
            return True
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "embedding.enabled",
                "description": "Enable vector embeddings for semantic search (requires API key below).",
                "required": False,
                "default": "false",
            },
            {
                "key": "embedding.model",
                "description": "Embedding model (default: BAAI/bge-m3, supports BAAI/bge-m3, BAAI/bge-large-zh-v1.5).",
                "required": False,
                "default": "BAAI/bge-m3",
            },
            {
                "key": "embedding.api_key_env",
                "description": "SiliconFlow API key for embeddings.",
                "secret": True,
                "required": False,
                "env_var": "SILICONFLOW_API_KEY",
                "url": "https://www.siliconflow.cn",
            },
            {
                "key": "summarizer.enabled",
                "description": "Enable LLM summarization for long memories (>1200 chars).",
                "required": False,
                "default": "false",
            },
            {
                "key": "summarizer.model",
                "description": "LLM model for summarization (e.g. glm-4-flash or openai/gpt-4o-mini).",
                "required": False,
                "default": "",
            },
            {
                "key": "summarizer.base_url",
                "description": "Base URL for the summarization API.",
                "required": False,
                "default": "",
            },
            {
                "key": "summarizer.api_key_env",
                "description": "API key for the summarization endpoint.",
                "secret": True,
                "required": False,
                "env_var": "MEMOS_SUMMARIZER_API_KEY",
            },
            {
                "key": "retrieval.mode",
                "description": "Search mode: hybrid (FTS + vector), fts (text-only), or vector (embedding-only).",
                "required": False,
                "choices": ["hybrid", "fts", "vector"],
                "default": "hybrid",
            },
            {
                "key": "skill_hint.enabled",
                "description": "Emit skill-update hints when workflow evidence is detected in memory.",
                "required": False,
                "default": "true",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        data_dir = Path(hermes_home) / "memos_lite"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_path = data_dir / "config.json"
        try:
            config_path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error("memos_lite save_config failed: %s", exc)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cleanup_closer_threads(wait=True)
        self._stop_runtime(wait=True, timeout=None)
        hermes_home = Path(kwargs.get("hermes_home") or ".")
        self._session_id = session_id
        self._data_dir = hermes_home / "memos_lite"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._config = load_config(self._data_dir / "config.json")
        self._write_enabled = kwargs.get("agent_context", "primary") == "primary"
        self._embedding_provider = None
        self._summarizer = None
        self._retrieval = None
        self._skill_hint = None
        with self._cache_lock:
            self._prefetch_cache.clear()
        self._queue = queue.Queue()
        self._shutdown = False
        self._store = LocalStore(self._data_dir)
        embedding_cfg = self._config.get("embedding", {})
        if embedding_cfg.get("enabled") and os.getenv(embedding_cfg.get("api_key_env", "SILICONFLOW_API_KEY")):
            try:
                self._embedding_provider = SiliconFlowEmbeddingProvider(embedding_cfg)
            except Exception as exc:
                logger.debug("memos_lite embedding provider init skipped: %s", exc)
        summarizer_cfg = self._config.get("summarizer", {})
        if summarizer_cfg.get("enabled"):
            try:
                self._summarizer = OpenAICompatibleSummarizer(summarizer_cfg)
            except Exception as exc:
                logger.debug("memos_lite summarizer init skipped: %s", exc)
        self._retrieval = RetrievalEngine(
            self._store,
            config=self._config.get("retrieval", {}),
            embedding_provider=self._embedding_provider,
        )
        self._skill_hint = SkillHintEmitter(self._config.get("skill_hint", {}))
        worker_queue = self._queue
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(worker_queue,),
            daemon=True,
            name="memos-lite-worker",
        )
        self._worker.start()

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retrieval:
            return ""
        session_key = session_id or self._session_id
        with self._cache_lock:
            cached = self._prefetch_cache.pop(session_key, None)
        if cached and _cache_matches_query(str(cached.get("query") or ""), query):
            return str(cached.get("output") or "")
        return self._build_prefetch_output(query, session_key, allow_vector=False)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._enqueue_task({"type": "prefetch", "query": query, "session_id": session_id or self._session_id})

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._write_enabled or not self._config.get("capture", {}).get("enabled", True):
            return
        self._enqueue_task(
            {
                "type": "turn",
                "session_id": session_id or self._session_id,
                "user": user_content,
                "assistant": assistant_content,
            }
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._write_enabled:
            return
        self._enqueue_task({"type": "session_end", "messages": messages, "session_id": self._session_id})

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._write_enabled:
            self._enqueue_task({"type": "pre_compress", "messages": messages, "session_id": self._session_id})
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._write_enabled or action not in {"add", "replace"}:
            return
        self._enqueue_task(
            {
                "type": "memory_write",
                "action": action,
                "target": target,
                "content": content,
                "session_id": self._session_id,
            }
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._write_enabled:
            return list(TOOL_SCHEMAS)
        return [schema for schema in TOOL_SCHEMAS if schema["name"] in _READ_ONLY_TOOLS]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._store or not self._retrieval:
            return tool_error("memos_lite is not initialized")
        if tool_name in _WRITE_TOOLS and not self._write_enabled:
            return tool_error("memos_lite write tools are disabled outside the primary agent context")
        if tool_name == "memos_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return tool_error("memos_search requires a non-empty query")
            results = self._retrieval.retrieve(
                query,
                scope=args.get("scope"),
                tags=args.get("tags"),
                manual=True,
                top_k=args.get("top_k"),
            )
            return json.dumps(
                {
                    "count": len(results),
                    "results": [
                        {
                            "id": item["id"],
                            "scope": item.get("scope"),
                            "domain": item.get("domain"),
                            "created_at": item.get("created_at"),
                            "l0": item.get("l0"),
                            "summary": item.get("summary"),
                            "score": item.get("score"),
                        }
                        for item in results
                    ],
                },
                ensure_ascii=False,
            )
        if tool_name == "memos_get":
            memory_id = str(args.get("id", "")).strip()
            if not memory_id:
                return tool_error("memos_get requires a non-empty id")
            memory = self._store.get(memory_id)
            return json.dumps({"memory": memory}, ensure_ascii=False)
        if tool_name == "memos_remember":
            content = str(args.get("content", "")).strip()
            if not content:
                return tool_error("memos_remember requires a non-empty content field")
            if contains_secret(content):
                return tool_error("Refusing to store content that contains secrets")
            saved = self._remember_memory(
                content,
                session_id=self._session_id,
                source=args.get("source") or "manual",
                scope=args.get("scope"),
                domain=args.get("domain"),
                tags=args.get("tags"),
                metadata=args.get("metadata"),
                force=True,
            )
            return json.dumps({"saved": bool(saved), "memory": saved}, ensure_ascii=False)
        if tool_name == "memos_forget":
            memory_id = str(args.get("id", "")).strip()
            if not memory_id:
                return tool_error("memos_forget requires a non-empty id")
            forgotten = self._store.forget(memory_id)
            return json.dumps({"forgotten": forgotten, "id": memory_id}, ensure_ascii=False)
        if tool_name == "memos_timeline":
            try:
                raw_limit = int(args.get("limit", 20))
            except (TypeError, ValueError):
                return tool_error("memos_timeline: limit must be an integer")
            timeline = self._store.timeline(
                limit=max(1, min(raw_limit, 200)),
                scope=args.get("scope"),
                tags=args.get("tags"),
            )
            return json.dumps({"count": len(timeline), "timeline": timeline}, ensure_ascii=False)
        if tool_name == "memos_status":
            status = self._store.counts()
            status.update(
                {
                    "provider": self.name,
                    "data_dir": str(self._data_dir),
                    "queue_size": self._queue.qsize(),
                    "worker_alive": bool(self._worker and self._worker.is_alive()),
                    "embedding_enabled": bool(self._embedding_provider),
                    "summarizer_enabled": bool(self._summarizer),
                    "disable_fallback_to_agent_model": DISABLE_FALLBACK_TO_AGENT_MODEL,
                    "last_diagnostics": self._retrieval.last_diagnostics,
                }
            )
            return json.dumps(status, ensure_ascii=False)
        return tool_error(f"Unknown memos_lite tool: {tool_name}")

    def shutdown(self) -> None:
        if self._shutdown and self._worker is None and self._store is None:
            self._cleanup_closer_threads(wait=False)
            return
        self._stop_runtime(wait=False, timeout=_SHUTDOWN_WAIT_SECONDS)
        self._session_id = ""
        self._cleanup_closer_threads(wait=False)

    def _enqueue_task(self, task: Dict[str, Any]) -> bool:
        if self._shutdown:
            return False
        self._queue.put(task)
        return True

    def _build_prefetch_output(self, query: str, session_id: str, *, allow_vector: bool) -> str:
        if not self._retrieval:
            return ""
        results = self._retrieval.retrieve(query, manual=False, allow_vector=allow_vector)
        hint_line = self._skill_hint.maybe_emit(
            query,
            results,
            session_id=session_id,
        ) if self._skill_hint else ""
        formatted = format_recall(
            results,
            max_chars=int(self._config.get("retrieval", {}).get("max_chars", 6000)),
            hint_line=hint_line,
        )
        return formatted

    def _run_worker(self, runtime_queue: queue.Queue) -> None:
        while True:
            task = runtime_queue.get()
            try:
                if task is _SENTINEL:
                    return
                task_type = task.get("type")
                if task_type == "prefetch":
                    self._run_prefetch(task["query"], task["session_id"])
                elif task_type == "turn":
                    self._handle_turn(task)
                elif task_type in {"session_end", "pre_compress"}:
                    digest = extractive_digest(task.get("messages") or [])
                    if digest:
                        self._remember_memory(
                            digest,
                            session_id=task.get("session_id") or self._session_id,
                            source=task_type,
                            force=True,
                        )
                elif task_type == "memory_write":
                    self._remember_memory(
                        task.get("content", ""),
                        session_id=task.get("session_id") or self._session_id,
                        source=f"memory_write:{task.get('action')}:{task.get('target')}",
                        force=True,
                    )
            except Exception as exc:
                logger.debug("memos_lite background task failed: %s", exc, exc_info=True)
            finally:
                runtime_queue.task_done()

    def _run_prefetch(self, query: str, session_id: str) -> None:
        if not self._retrieval:
            return
        formatted = self._build_prefetch_output(query, session_id, allow_vector=True)
        with self._cache_lock:
            self._prefetch_cache[session_id] = {
                "query": query,
                "output": formatted,
                "diagnostics": dict(self._retrieval.last_diagnostics),
            }

    def _cleanup_closer_threads(self, *, wait: bool) -> None:
        with self._closer_lock:
            active: List[threading.Thread] = []
            for thread in self._closer_threads:
                if wait and thread.is_alive():
                    thread.join()
                if thread.is_alive():
                    active.append(thread)
            self._closer_threads = active

    def _spawn_deferred_close(
        self,
        worker: threading.Thread,
        runtime_queue: queue.Queue,
        store: LocalStore | None,
        retrieval: RetrievalEngine | None,
        embedding_provider,
        summarizer,
        skill_hint: SkillHintEmitter | None,
    ) -> None:
        if store is None and worker is None:
            return

        def _finalize() -> None:
            try:
                worker.join()
            finally:
                try:
                    if store:
                        store.close()
                    if summarizer:
                        summarizer.close()
                except Exception:
                    logger.debug("memos_lite deferred store/summarizer close failed", exc_info=True)
                self._release_runtime_refs(
                    worker=worker,
                    runtime_queue=runtime_queue,
                    store=store,
                    retrieval=retrieval,
                    embedding_provider=embedding_provider,
                    summarizer=summarizer,
                    skill_hint=skill_hint,
                )

        closer = threading.Thread(target=_finalize, daemon=True, name="memos-lite-closer")
        closer.start()
        with self._closer_lock:
            self._closer_threads.append(closer)

    def _release_runtime_refs(
        self,
        *,
        worker: threading.Thread | None,
        runtime_queue: queue.Queue,
        store: LocalStore | None,
        retrieval: RetrievalEngine | None,
        embedding_provider,
        summarizer,
        skill_hint: SkillHintEmitter | None,
    ) -> None:
        if self._worker is worker:
            self._worker = None
        if self._store is store:
            self._store = None
        if self._retrieval is retrieval:
            self._retrieval = None
        if self._embedding_provider is embedding_provider:
            self._embedding_provider = None
        if self._summarizer is summarizer:
            self._summarizer = None
        if self._skill_hint is skill_hint:
            self._skill_hint = None
        if self._queue is runtime_queue:
            self._queue = queue.Queue()

    def _stop_runtime(self, *, wait: bool, timeout: float | None) -> None:
        runtime_queue = self._queue
        worker = self._worker
        store = self._store
        retrieval = self._retrieval
        embedding_provider = self._embedding_provider
        summarizer = self._summarizer
        skill_hint = self._skill_hint
        session_id = self._session_id

        self._shutdown = True
        with self._cache_lock:
            self._prefetch_cache.clear()

        if skill_hint and session_id:
            skill_hint.clear_session(session_id)

        if worker and worker.is_alive():
            runtime_queue.put(_SENTINEL)
            worker.join(timeout=None if wait else timeout)
            if worker.is_alive():
                if wait:
                    worker.join()
                else:
                    logger.warning("memos_lite worker exceeded shutdown grace period; deferring store close")
                    self._spawn_deferred_close(
                        worker,
                        runtime_queue,
                        store,
                        retrieval,
                        embedding_provider,
                        summarizer,
                        skill_hint,
                    )
                    return

        if store:
            store.close()
        if summarizer:
            summarizer.close()
        self._release_runtime_refs(
            worker=worker,
            runtime_queue=runtime_queue,
            store=store,
            retrieval=retrieval,
            embedding_provider=embedding_provider,
            summarizer=summarizer,
            skill_hint=skill_hint,
        )

    def _handle_turn(self, task: Dict[str, Any]) -> None:
        capture_cfg = self._config.get("capture", {})
        parts = []
        user_text = str(task.get('user') or '').strip()
        assistant_text = str(task.get('assistant') or '').strip()
        if capture_cfg.get("include_user", True) and user_text:
            parts.append(f"User: {user_text}")
        if capture_cfg.get("include_assistant", True) and assistant_text:
            parts.append(f"Assistant: {assistant_text}")
        content = "\n".join(parts).strip()
        if not content:
            return
        self._remember_memory(
            content,
            session_id=task.get("session_id") or self._session_id,
            source="conversation",
            force=False,
        )

    def _remember_memory(
        self,
        content: str,
        *,
        session_id: str,
        source: str,
        scope: str | None = None,
        domain: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        force: bool,
    ) -> dict | None:
        if not self._store:
            return None
        raw_content = str(content or "").strip()
        if not raw_content or contains_secret(raw_content):
            return None
        min_chars = int(self._config.get("capture", {}).get("min_chars", 40))
        if not force and not should_store_text(raw_content, source=source, min_chars=min_chars):
            return None
        redacted = redact_text(raw_content)
        final_scope = infer_scope(
            content=redacted,
            scope=scope,
            domain=domain,
            metadata=metadata,
            config=self._config.get("scopes", {}),
        )
        summary = None
        l1_json = None
        if self._summarizer and len(redacted) >= int(self._config.get("summarizer", {}).get("threshold_chars", 1200)):
            try:
                summary_payload = self._summarizer.summarize(redacted)
                summary = summary_payload.get("summary")
                l1_json = summary_payload.get("l1_json")
            except Exception as exc:
                logger.debug("memos_lite summarizer failed: %s", exc)
        l0 = extract_l0(summary or redacted)
        embedding = None
        embedding_model = None
        embedding_dim = None
        if self._embedding_provider:
            try:
                embedding = self._embedding_provider.embed_texts([redacted])[0]
                embedding_model = getattr(self._embedding_provider, "model", None)
                embedding_dim = getattr(self._embedding_provider, "embedding_dim", None) or len(embedding)
            except Exception as exc:
                logger.debug("memos_lite embedding failed: %s", exc)
        workflow_key = infer_workflow_key(redacted, tags=tags, metadata=metadata)
        return self._store.remember(
            content=redacted,
            summary=summary,
            l0=l0,
            l1_json=l1_json,
            session_id=session_id,
            source=source,
            scope=final_scope,
            domain=domain or (final_scope.split(":", 1)[-1] if final_scope.startswith("domain:") else domain),
            project_id=(metadata or {}).get("project_id") if metadata else None,
            tags=tags or [],
            metadata=metadata or {},
            workflow_key=workflow_key or None,
            embedding=embedding,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )


# Plugin loader entry point (called by plugins/memory/__init__.py)
def get_provider() -> "MemosLiteMemoryProvider":
    """Return a MemosLiteMemoryProvider instance for the plugin loader."""
    return MemosLiteMemoryProvider()


def register(ctx) -> None:
    """Legacy plugin-style registration (still supported)."""
    provider = MemosLiteMemoryProvider()
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    skill_path = Path(__file__).parent / "skills" / _COMPANION_SKILL_NAME / "SKILL.md"
    if hasattr(ctx, "register_skill") and skill_path.exists():
        ctx.register_skill(_COMPANION_SKILL_NAME, skill_path, _COMPANION_SKILL_DESCRIPTION)
