# memos_lite

`memos_lite` is a local-only Hermes memory provider plugin for lightweight
long-term memory.

This repository contains only the plugin payload. Install it into a Hermes
checkout at:

`plugins/memory/memos_lite/`

Example:

```bash
git clone https://github.com/a958888535/memos_lite.git plugins/memory/memos_lite
```

## What It Does

- Local conversation capture
- Secret redaction before storage
- SQLite storage under `<hermes_home>/memos_lite/`
- FTS5 keyword search with LIKE fallback
- Optional SiliconFlow embeddings
- Hybrid retrieval with compact recall formatting
- Scope isolation, adaptive retrieval skip, noise filtering
- A single metadata-style skill-update hint when evidence is high

## What It Does Not Do

- No MemOS Cloud integration
- No MemOS API key
- No external MemOS server
- No external skill evolution worker
- No Kimi CLI, Claude Code, or Codex CLI integration
- No candidate queue, candidate table, or candidate tools
- No automatic SKILL.md creation, installation, modification, or publishing
- No natural-language learning feedback
- No fallback to Hermes primary model for memory maintenance

## Storage

All data lives under:

`<hermes_home>/memos_lite/`

This directory contains:

- `config.json`
- `memos_lite.sqlite3`

## SiliconFlow Embeddings

Set:

```bash
export SILICONFLOW_API_KEY="..."
```

Using SiliconFlow embeddings sends text chunks to SiliconFlow. Use local embeddings for strict privacy.

## Companion Skill

The bundled companion skill lives at:

`skills/memos-lite-hint-policy/SKILL.md`

It is passive and only explains how Hermes should interpret:

`[memos_lite_hint skill_update_possible=true workflow_key="<key>" evidence_count=<n> memory_ids="<id1,id2>"]`

The hint is a signal, not an instruction.

Hermes native skills remain responsible for SKILL.md.

When Hermes loads the plugin through the standard plugin manager, the companion
skill is also exposed as a plugin-provided skill:

`memos_lite:memos-lite-hint-policy`

This keeps the skill read-only and plugin-scoped. It does not write into
`~/.hermes/skills/`.

## Companion Skill Installation

If you explicitly want a flat bundled copy in the normal skill tree, install it manually:

```bash
mkdir -p ~/.hermes/skills/system/memos-lite-hint-policy
cp skills/memos-lite-hint-policy/SKILL.md \
  ~/.hermes/skills/system/memos-lite-hint-policy/SKILL.md
```

## Test Notes

This repository is intentionally plugin-only. The plugin's pytest suite depends
on Hermes test fixtures and should be run from a Hermes checkout after placing
this plugin under `plugins/memory/memos_lite/`.

## Coding Plan Safety

Memory-purpose model calls are blocked from endpoints containing:

- `/coding/`
- `/api/coding/paas`
- `api.kimi.com/coding`

## Example Config

```json
{
  "backend": "local",
  "capture": {
    "enabled": true,
    "include_user": true,
    "include_assistant": true,
    "include_tool_results": false,
    "exclude_system": true,
    "min_chars": 40
  },
  "embedding": {
    "enabled": true,
    "provider": "siliconflow",
    "model": "BAAI/bge-m3",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "SILICONFLOW_API_KEY",
    "batch_size": 16,
    "timeout_seconds": 60,
    "encoding_format": "float"
  },
  "summarizer": {
    "enabled": false
  },
  "retrieval": {
    "mode": "hybrid",
    "vector_weight": 0.65,
    "fts_weight": 0.35,
    "candidate_pool_size": 24,
    "top_k": 8,
    "min_score": 0.30,
    "hard_min_score": 0.35,
    "max_chars": 6000,
    "length_norm_anchor": 500,
    "mmr_diversity_threshold": 0.85,
    "recency_half_life_days": 30,
    "recency_weight": 0.10,
    "importance_weight": 0.10,
    "diagnostics": true
  },
  "skill_hint": {
    "enabled": true,
    "min_evidence_count": 3,
    "min_confidence": 0.70,
    "once_per_session_per_workflow": true,
    "max_memory_ids": 2,
    "sensitive_domains": ["health", "credentials", "secrets"],
    "format": "single_metadata_line"
  },
  "scopes": {
    "default": "global",
    "health_default": "domain:health",
    "quant_default": "domain:quant",
    "paper_default": "domain:paper"
  },
  "safety": {
    "redact_secrets": true,
    "forbid_coding_plan_for_memory": true,
    "disable_fallback_to_agent_model": true
  }
}
```

## Troubleshooting

- If FTS5 is unavailable, memos_lite falls back to LIKE search.
- If embedding calls fail, recall falls back to FTS.
- If the embedding model changes, existing vectors are not compared across incompatible model/dimension pairs.
- If you change embedding models and want a clean vector corpus, re-embed stored memories manually in a maintenance script.
