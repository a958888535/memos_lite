"""CLI commands for memos_lite memory plugin.

Handles: hermes memos setup | status | stats | list | forget | search
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    return Path(get_hermes_home()) / "memos_lite"


def _config_path() -> Path:
    return _config_dir() / "config.json"


def _db_path() -> Path:
    return _config_dir() / "memos_lite.sqlite3"


def _read_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_config(cfg: Dict[str, Any]) -> None:
    _config_dir().mkdir(parents=True, exist_ok=True)
    _config_path().write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Provider context manager — eliminates repeated try/finally boilerplate
# ---------------------------------------------------------------------------

@contextmanager
def _provider(session_label: str = "cli"):
    """Yield an initialized MemosLiteMemoryProvider; auto-shutdown on exit."""
    from plugins.memory.memos_lite import MemosLiteMemoryProvider

    provider = MemosLiteMemoryProvider()
    initialized = False
    try:
        provider.initialize(
            session_id=session_label,
            hermes_home=str(get_hermes_home()),
            platform="cli",
            agent_context="primary",
        )
        initialized = True
        yield provider
    except Exception as e:
        print(f"  ✗ memos_lite error: {e}\n")
        if not initialized:
            yield None
    finally:
        try:
            provider.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------

def cmd_setup(args) -> None:
    """Interactive memos_lite setup wizard."""
    cfg = _read_config()
    print("\nmemos_lite memory setup\n" + "─" * 40)
    print("  Local-only memory with FTS5 search, optional embeddings.")
    print(f"  Config: {_config_path()}")
    print()

    # --- Memory provider selection ---
    print("  Select memory provider:")
    print("    memos_lite -- local FTS5 + optional semantic search (this)")
    current = cfg.get("backend", "local")
    print(f"    (current: {current})")
    print()

    # --- Embedding toggle ---
    print("  Vector embeddings (semantic search):")
    emb_enabled = cfg.get("embedding", {}).get("enabled", False)
    emb_choice = _prompt_bool("Enable embeddings?", default=emb_enabled)
    if emb_choice:
        cfg.setdefault("embedding", {})
        cfg["embedding"]["enabled"] = True
        cfg["embedding"]["model"] = _prompt(
            "Model",
            default=cfg.get("embedding", {}).get("model", "BAAI/bge-m3"),
        )
        api_key = cfg.get("embedding", {}).get("api_key_env", "SILICONFLOW_API_KEY")
        print(f"\n  SiliconFlow API key (env var: {api_key}):")
        print("  (Set via: hermes config set MEMOS_EMBEDDING_KEY <key>)")
        print("  Or edit ~/.hermes/.env directly.")
    else:
        cfg.setdefault("embedding", {})["enabled"] = False

    # --- Summarizer toggle ---
    print("\n  LLM summarization (for long memories):")
    sum_enabled = cfg.get("summarizer", {}).get("enabled", False)
    sum_choice = _prompt_bool("Enable summarization?", default=sum_enabled)
    if sum_choice:
        cfg.setdefault("summarizer", {})
        cfg["summarizer"]["enabled"] = True
        cfg["summarizer"]["model"] = _prompt(
            "Model (e.g. glm-4-flash)",
            default=cfg.get("summarizer", {}).get("model", ""),
        )
        cfg["summarizer"]["base_url"] = _prompt(
            "Base URL (e.g. https://open.bigmodel.cn/api/paas/v4)",
            default=cfg.get("summarizer", {}).get("base_url", ""),
        )
        print("\n  Set the API key via: hermes config set MEMOS_SUMMARIZER_KEY <key>")
    else:
        cfg.setdefault("summarizer", {})["enabled"] = False

    # --- Skill hints ---
    print("\n  Skill-update hints:")
    hint_enabled = cfg.get("skill_hint", {}).get("enabled", True)
    cfg.setdefault("skill_hint", {})["enabled"] = _prompt_bool(
        "Enable skill hints?", default=hint_enabled,
    )

    # --- Save ---
    _write_config(cfg)

    # --- Update memory.provider in hermes config ---
    print()
    _set_memory_provider("memos_lite")
    print()


def _set_memory_provider(name: str) -> None:
    """Set memory.provider in the global hermes config."""
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        cfg.setdefault("memory", {})["provider"] = name
        save_config(cfg)
        print(f"  ✓ memory.provider set to '{name}' in config")
    except Exception as e:
        print(f"  ⚠ Could not update config.yaml: {e}")
        print("    Run: hermes config set memory.provider memos_lite")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args) -> None:
    """Show memos_lite status."""
    with _provider("cli-status") as provider:
        if provider is None:
            return
        result = provider.handle_tool_call("memos_status", {}, session_id="cli-status")
        status = json.loads(result)

        print("\nmemos_lite status\n" + "─" * 40)
        print(f"  Provider:    {status.get('provider')}")
        print(f"  Data dir:    {status.get('data_dir')}")
        print(f"  DB:          {status.get('db_path')}")
        print(f"  FTS5:        {'✓ available' if status.get('fts_available') else '✗ unavailable (will use LIKE fallback)'}")
        print(f"  Embeddings:  {'✓ enabled' if status.get('embedding_enabled') else '✗ disabled'}")
        print(f"  Summarizer:  {'✓ enabled' if status.get('summarizer_enabled') else '✗ disabled'}")
        print(f"  Memories:    {status.get('memory_count', 0)}")
        print(f"  Events:     {status.get('event_count', 0)}")
        print(f"  Queue:      {status.get('queue_size', 0)}")
        print(f"  Worker:     {'✓ alive' if status.get('worker_alive') else '✗ dead'}")
        print(f"  Skill hints: {'✓ enabled' if not status.get('disable_fallback_to_agent_model') else '✗ disabled'}")

        diag = status.get("last_diagnostics", {})
        if diag:
            print("\n  Retrieval diagnostics:")
            print(f"    FTS candidates:     {diag.get('fts_candidate_count', 0)}")
            print(f"    Vector candidates:  {diag.get('vector_candidate_count', 0)}")
            print(f"    Fused:              {diag.get('fused_candidate_count', 0)}")
            print(f"    Final results:      {diag.get('final_result_count', 0)}")
            print(f"    Dropped (score):    {diag.get('dropped_by_min_score', 0)}")
            print(f"    Dropped (scope):    {diag.get('dropped_by_scope', 0)}")
            print(f"    Dropped (noise):    {diag.get('dropped_by_noise', 0)}")
            if diag.get("embedding_error"):
                print(f"    Embedding error:    {diag.get('embedding_error')}")
        print()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cmd_stats(args) -> None:
    """Show memory statistics by scope and domain."""
    with _provider("cli-stats") as provider:
        if provider is None:
            return
        timeline = provider.handle_tool_call(
            "memos_timeline", {"limit": 200}, session_id="cli-stats",
        )
        data = json.loads(timeline)

        items = data.get("timeline", [])
        print(f"\nmemos_lite — {len(items)} recent memories\n" + "─" * 40)

        by_scope: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        total_len = 0

        for item in items:
            scope = item.get("scope", "global")
            by_scope[scope] = by_scope.get(scope, 0) + 1
            domain = item.get("domain") or "-"
            by_domain[domain] = by_domain.get(domain, 0) + 1
            source = item.get("source") or "-"
            by_source[source] = by_source.get(source, 0) + 1
            total_len += len(item.get("content") or "")

        print("\n  By scope:")
        for scope, count in sorted(by_scope.items(), key=lambda x: -x[1]):
            print(f"    {scope:30s} {count:4d}")

        print("\n  By domain:")
        for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
            print(f"    {domain:30s} {count:4d}")

        print("\n  By source:")
        for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"    {source:30s} {count:4d}")

        if items:
            avg_len = total_len // len(items)
            print(f"\n  Avg content length: {avg_len} chars")
        print()


# ---------------------------------------------------------------------------
# List / search / forget
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    """List recent memories."""
    limit = min(getattr(args, "limit", None) or 20, 200)

    with _provider("cli-list") as provider:
        if provider is None:
            return
        raw = provider.handle_tool_call(
            "memos_timeline", {"limit": limit}, session_id="cli-list",
        )
        data = json.loads(raw)
        items = data.get("timeline", [])
        print(f"\nmemos_lite — {len(items)} memories\n" + "─" * 60)
        for item in items:
            date = ""
            try:
                from datetime import datetime
                date = datetime.fromisoformat(item.get("created_at", "")).date().isoformat()
            except Exception:
                pass
            l0 = item.get("l0") or item.get("content", "")[:80]
            print(f"  [{item.get('id', ''):14s} {date} {item.get('scope', 'global'):15s}] {l0}")
        print()


def cmd_search(args) -> None:
    """Search memories."""
    raw_query = getattr(args, "query", None)
    if not raw_query:
        print("  Error: provide a query, e.g. 'hermes memos search how to deploy'")
        return
    # query may be a list (nargs="+") or string (nargs="?")
    query = " ".join(raw_query) if isinstance(raw_query, list) else raw_query
    top_k = getattr(args, "top_k", None) or 8

    with _provider("cli-search") as provider:
        if provider is None:
            return
        raw = provider.handle_tool_call(
            "memos_search", {"query": query, "top_k": top_k}, session_id="cli-search",
        )
        data = json.loads(raw)
        results = data.get("results", [])
        print(f"\nmemos_lite search '{query}' — {len(results)} results\n" + "─" * 60)
        for item in results:
            l0 = item.get("l0") or ""
            score = item.get("score", 0)
            print(f"  [score={score:.3f} scope={item.get('scope', 'global'):15s} id={item.get('id', '')}]")
            print(f"    {l0[:120]}")
            print()


def cmd_forget(args) -> None:
    """Delete a memory by id."""
    memory_id = getattr(args, "memory_id", None)
    if not memory_id:
        print("  Error: provide a memory id, e.g. 'hermes memos forget mem_abc123def'")
        return

    with _provider("cli-forget") as provider:
        if provider is None:
            return
        raw = provider.handle_tool_call("memos_forget", {"id": memory_id}, session_id="cli-forget")
        data = json.loads(raw)
        if data.get("forgotten"):
            print(f"  ✓ Deleted {memory_id}")
        else:
            print(f"  ✗ Memory {memory_id} not found")


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def register_cli(parser) -> None:
    """Register 'hermes memos' subcommands onto the given parser."""
    sub = parser.add_subparsers(dest="memos_command", metavar="command")

    # setup
    p_setup = sub.add_parser("setup", help="Interactive setup wizard")
    p_setup.set_defaults(func=cmd_setup)

    # status
    p_status = sub.add_parser("status", help="Show provider status and diagnostics")
    p_status.set_defaults(func=cmd_status)

    # stats
    p_stats = sub.add_parser("stats", help="Show memory statistics")
    p_stats.set_defaults(func=cmd_stats)

    # list
    p_list = sub.add_parser("list", help="List recent memories")
    p_list.add_argument("-n", "--limit", type=int, default=20, help="Max results (default 20, max 200)")
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, dest="top_k", help="Max results")
    p_search.set_defaults(func=cmd_search)

    # forget
    p_forget = sub.add_parser("forget", help="Delete a memory by id")
    p_forget.add_argument("memory_id", help="Memory id (e.g. mem_abc123def)")
    p_forget.set_defaults(func=cmd_forget)


def memos_command(args) -> None:
    """Entry point called by the CLI dispatcher."""
    func = getattr(args, "func", None)
    if func:
        func(args)
    else:
        # No subcommand: show status
        cmd_status(args)


# Alias — discover_plugin_cli_commands() looks for "<provider>_command"
memos_lite_command = memos_command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    val = sys.stdin.readline().strip()
    return val or (default or "")


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    val = sys.stdin.readline().strip().lower()
    if not val:
        return default
    return val in ("y", "yes")
