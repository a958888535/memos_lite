from __future__ import annotations

import difflib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _as_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class LocalStore:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memos_lite.sqlite3"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self.fts_available = False
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._ensure_schema_version()
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories(
                  id TEXT PRIMARY KEY,
                  content TEXT NOT NULL,
                  summary TEXT,
                  l0 TEXT,
                  l1_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  session_id TEXT,
                  source TEXT,
                  scope TEXT NOT NULL DEFAULT 'global',
                  domain TEXT,
                  project_id TEXT,
                  tags TEXT,
                  metadata_json TEXT,
                  importance REAL DEFAULT 0.5,
                  confidence REAL DEFAULT 0.7,
                  access_count INTEGER DEFAULT 0,
                  last_accessed_at TEXT,
                  tier TEXT DEFAULT 'working',
                  workflow_key TEXT,
                  embedding_model TEXT,
                  embedding_dim INTEGER,
                  embedding_json TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events(
                  id TEXT PRIMARY KEY,
                  memory_id TEXT,
                  event_type TEXT,
                  created_at TEXT,
                  metadata_json TEXT
                )
                """
            )
            # --- Performance indexes ---
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_scope_updated ON memories(scope, updated_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_workflow_key ON memories(workflow_key) "
                "WHERE workflow_key IS NOT NULL"
            )
            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                      content,
                      summary,
                      l0,
                      content='memories',
                      content_rowid='rowid'
                    )
                    """
                )
                self._conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                      INSERT INTO memories_fts(rowid, content, summary, l0)
                      VALUES (new.rowid, new.content, COALESCE(new.summary, ''), COALESCE(new.l0, ''));
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                      INSERT INTO memories_fts(memories_fts, rowid, content, summary, l0)
                      VALUES('delete', old.rowid, old.content, COALESCE(old.summary, ''), COALESCE(old.l0, ''));
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                      INSERT INTO memories_fts(memories_fts, rowid, content, summary, l0)
                      VALUES('delete', old.rowid, old.content, COALESCE(old.summary, ''), COALESCE(old.l0, ''));
                      INSERT INTO memories_fts(rowid, content, summary, l0)
                      VALUES (new.rowid, new.content, COALESCE(new.summary, ''), COALESCE(new.l0, ''));
                    END;
                    """
                )
                self.fts_available = True
            except sqlite3.OperationalError:
                self.fts_available = False
            self._conn.commit()

    # ---- Schema migration -------------------------------------------------

    _SCHEMA_VERSION = 1

    def _ensure_schema_version(self) -> None:
        """Simple version-based migration for future schema changes."""
        meta_table = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'"
        ).fetchone()
        if not meta_table:
            self._conn.execute(
                "CREATE TABLE _meta(key TEXT PRIMARY KEY, value TEXT)"
            )
            self._conn.execute(
                "INSERT INTO _meta(key, value) VALUES('schema_version', ?)",
                (str(self._SCHEMA_VERSION),),
            )
            self._conn.commit()
            return
        row = self._conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()
        version = int(row[0]) if row else 0
        if version < self._SCHEMA_VERSION:
            # Future ALTER TABLE migrations go here, version by version.
            # Example:
            #   if version < 2:
            #       self._conn.execute("ALTER TABLE memories ADD COLUMN new_col TEXT")
            #       version = 2
            self._conn.execute(
                "UPDATE _meta SET value=? WHERE key='schema_version'",
                (str(self._SCHEMA_VERSION),),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        item["tags"] = _parse_json(item.get("tags"), [])
        item["metadata"] = _parse_json(item.get("metadata_json"), {})
        item["embedding"] = _parse_json(item.get("embedding_json"), None)
        return item

    def _tag_match(self, row_tags: List[str], tags: Iterable[str] | None) -> bool:
        if not tags:
            return True
        return set(tags).issubset(set(row_tags))

    def _find_duplicate(self, content: str, scope: str, l0: str) -> dict | None:
        """Find a near-duplicate memory within the same scope.

        Strategy:
          1. Exact match on normalised content (fast path).
          2. If FTS5 is available, use it to narrow candidates to rows that
             share at least one token with ``l0``, then run SequenceMatcher.
          3. Fall back to full scope scan if FTS5 is absent.
        """
        normalized = _normalize_text(content)
        normalized_l0 = _normalize_text(l0)

        # --- Fast path: exact content match via indexed query ---------------
        exact = self._conn.execute(
            "SELECT * FROM memories WHERE scope = ? AND content = ? LIMIT 1",
            (scope, content),
        ).fetchone()
        if exact:
            return self._row_to_dict(exact)

        # --- Candidate narrowing via FTS5 ----------------------------------
        if self.fts_available and normalized_l0.strip():
            # Use first few meaningful words as FTS query to avoid
            # over-constraining the match.
            fts_tokens = normalized_l0.split()[:6]
            fts_query = " OR ".join(fts_tokens)
            try:
                cursor = self._conn.execute(
                    """
                    SELECT m.* FROM memories m
                    JOIN memories_fts fts ON fts.rowid = m.rowid
                    WHERE m.scope = ?
                      AND memories_fts MATCH ?
                    ORDER BY m.updated_at DESC
                    LIMIT 50
                    """,
                    (scope, fts_query),
                )
            except sqlite3.OperationalError:
                # FTS query syntax error — fall through to full scan.
                cursor = None
            if cursor is not None:
                for row in cursor.fetchall():
                    item = self._row_to_dict(row)
                    # Re-check normalised exact match (FTS is token-based).
                    if _normalize_text(item["content"]) == normalized:
                        return item
                    ratio = difflib.SequenceMatcher(
                        None,
                        normalized_l0,
                        _normalize_text(item.get("l0") or item["content"]),
                    ).ratio()
                    if ratio >= 0.96:
                        return item
                return None

        # --- Fallback: full scope scan --------------------------------------
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE scope = ? ORDER BY updated_at DESC",
            (scope,),
        )
        for row in cursor.fetchall():
            item = self._row_to_dict(row)
            if _normalize_text(item["content"]) == normalized:
                return item
            ratio = difflib.SequenceMatcher(
                None,
                normalized_l0,
                _normalize_text(item.get("l0") or item["content"]),
            ).ratio()
            if ratio >= 0.96:
                return item
        return None

    def remember(
        self,
        *,
        content: str,
        summary: str | None = None,
        l0: str | None = None,
        l1_json: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        scope: str = "global",
        domain: str | None = None,
        project_id: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float = 0.7,
        tier: str = "working",
        workflow_key: str | None = None,
        embedding: List[float] | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> dict:
        with self._lock:
            timestamp = _utc_now()
            dedup = self._find_duplicate(content, scope, l0 or content)
            if dedup:
                merged_tags = sorted(set(dedup.get("tags", [])) | set(tags or []))
                metadata_out = dedup.get("metadata", {})
                if metadata:
                    metadata_out.update(metadata)
                self._conn.execute(
                    """
                    UPDATE memories
                    SET updated_at = ?, tags = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        timestamp,
                        _as_json(merged_tags),
                        _as_json(metadata_out),
                        dedup["id"],
                    ),
                )
                self._conn.commit()
                return self.get(dedup["id"], increment_access=False)

            memory_id = f"mem_{uuid.uuid4().hex[:12]}"
            self._conn.execute(
                """
                INSERT INTO memories(
                  id, content, summary, l0, l1_json, created_at, updated_at, session_id,
                  source, scope, domain, project_id, tags, metadata_json, importance,
                  confidence, tier, workflow_key, embedding_model, embedding_dim, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    content,
                    summary,
                    l0,
                    l1_json,
                    timestamp,
                    timestamp,
                    session_id,
                    source,
                    scope,
                    domain,
                    project_id,
                    _as_json(sorted(set(tags or []))),
                    _as_json(metadata or {}),
                    float(importance),
                    float(confidence),
                    tier,
                    workflow_key,
                    embedding_model,
                    embedding_dim,
                    _as_json(embedding),
                ),
            )
            self.record_event(memory_id, "remember", {"source": source or "unknown"})
            self._conn.commit()
            return self.get(memory_id, increment_access=False)

    def record_event(self, memory_id: str, event_type: str, metadata: Dict[str, Any] | None = None) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_events(id, memory_id, event_type, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"evt_{uuid.uuid4().hex[:12]}",
                memory_id,
                event_type,
                _utc_now(),
                _as_json(metadata or {}),
            ),
        )

    def get(self, memory_id: str, *, increment_access: bool = True) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            item = self._row_to_dict(row)
            if item and increment_access:
                last_accessed = _utc_now()
                self._conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (last_accessed, memory_id),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                item = self._row_to_dict(row)
            return item

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            deleted = self._conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            ).rowcount
            self._conn.execute(
                "DELETE FROM memory_events WHERE memory_id = ?",
                (memory_id,),
            )
            self._conn.commit()
            return deleted > 0

    def timeline(
        self,
        *,
        limit: int = 20,
        scope: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> List[dict]:
        with self._lock:
            base_query = "SELECT * FROM memories"
            params: list[Any] = []
            if scope:
                base_query += " WHERE scope = ?"
                params.append(scope)
            base_query += " ORDER BY updated_at DESC"
            if not tags:
                rows = self._conn.execute(f"{base_query} LIMIT ?", [*params, limit]).fetchall()
                return [self._row_to_dict(row) for row in rows]

            collected: List[dict] = []
            offset = 0
            batch_size = max(limit * 5, 50)
            while len(collected) < limit:
                rows = self._conn.execute(
                    f"{base_query} LIMIT ? OFFSET ?",
                    [*params, batch_size, offset],
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    item = self._row_to_dict(row)
                    if self._tag_match(item.get("tags", []), tags):
                        collected.append(item)
                        if len(collected) >= limit:
                            break
                offset += len(rows)
                if len(rows) < batch_size:
                    break
            return collected[:limit]

    def iter_memories(
        self,
        *,
        scope: str | None = None,
        tags: List[str] | None = None,
        has_embedding: bool = False,
    ) -> List[dict]:
        """Iterate memories with optional server-side filtering.

        When ``has_embedding`` is True, only rows with a non-null
        ``embedding_json`` are returned — useful for vector search.
        """
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if scope is not None:
                clauses.append("(scope = ? OR scope = 'global')")
                params.append(scope)
            if has_embedding:
                clauses.append("embedding_json IS NOT NULL")
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM memories{where} ORDER BY updated_at DESC",
                params,
            ).fetchall()
            result = [self._row_to_dict(row) for row in rows]
            if tags:
                tag_set = set(tags)
                result = [
                    r for r in result if tag_set.issubset(set(r.get("tags", [])))
                ]
            return result

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
        tags: Iterable[str] | None = None,
        include_global: bool = True,
    ) -> List[dict]:
        with self._lock:
            scope_values: list[str] = []
            allowed_scopes: set[str] | None = None
            if scope:
                scope_values.append(scope)
                if include_global and scope != "global":
                    scope_values.append("global")
            if scope_values:
                allowed_scopes = set(scope_values)

            def _fetch_fts(batch_limit: int, offset: int) -> list[sqlite3.Row]:
                sql = """
                    SELECT memories.*, bm25(memories_fts) AS fts_rank
                    FROM memories_fts
                    JOIN memories ON memories_fts.rowid = memories.rowid
                    WHERE memories_fts MATCH ?
                """
                params: list[Any] = [query]
                if scope_values:
                    placeholders = ", ".join("?" for _ in scope_values)
                    sql += f" AND memories.scope IN ({placeholders})"
                    params.extend(scope_values)
                sql += """
                    ORDER BY bm25(memories_fts)
                    LIMIT ? OFFSET ?
                """
                params.extend([batch_limit, offset])
                return self._conn.execute(sql, params).fetchall()

            def _fetch_like(batch_limit: int, offset: int) -> list[sqlite3.Row]:
                like = f"%{query}%"
                if scope_values:
                    placeholders = ", ".join("?" for _ in scope_values)
                    return self._conn.execute(
                        f"""
                        SELECT * FROM memories
                        WHERE scope IN ({placeholders}) AND (content LIKE ? OR summary LIKE ? OR l0 LIKE ?)
                        ORDER BY updated_at DESC
                        LIMIT ? OFFSET ?
                        """,
                        (*scope_values, like, like, like, batch_limit, offset),
                    ).fetchall()
                return self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE content LIKE ? OR summary LIKE ? OR l0 LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (like, like, like, batch_limit, offset),
                ).fetchall()

            batch_size = limit * 3 if not tags else max(limit * 3, 30)
            offset = 0
            items: List[dict] = []
            use_fts = self.fts_available

            while len(items) < limit:
                if use_fts:
                    try:
                        rows = _fetch_fts(batch_size, offset)
                    except sqlite3.OperationalError:
                        use_fts = False
                        offset = 0
                        items = []
                        continue
                else:
                    rows = _fetch_like(batch_size, offset)

                if not rows:
                    if use_fts and offset == 0 and not items:
                        use_fts = False
                        continue
                    break

                for row in rows:
                    item = self._row_to_dict(row)
                    if allowed_scopes is not None and item["scope"] not in allowed_scopes:
                        continue
                    if not self._tag_match(item.get("tags", []), tags):
                        continue
                    items.append(item)
                    if len(items) >= limit:
                        break

                if not tags:
                    break
                offset += len(rows)
                if len(rows) < batch_size:
                    break

            return items[:limit]

    def update_timestamps(self, memory_id: str, *, created_at: str, updated_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?
                """,
                (created_at, updated_at, memory_id),
            )
            self._conn.commit()

    def counts(self) -> dict:
        with self._lock:
            memory_count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            event_count = self._conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
            return {
                "memory_count": memory_count,
                "event_count": event_count,
                "fts_available": self.fts_available,
                "db_path": str(self.db_path),
            }
