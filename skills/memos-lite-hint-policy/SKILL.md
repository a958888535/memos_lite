---
name: memos-lite-hint-policy
description: Interpret memos_lite skill-update hint metadata and decide whether native Hermes skill_manage should be used.
version: 1.0.0
metadata:
  hermes:
    tags: [memory, skills, memos-lite]
    category: system
---

# memos-lite hint policy

## When to Use

Use this skill only when the current context contains a line like:

`[memos_lite_hint skill_update_possible=true workflow_key="..." evidence_count=... memory_ids="..."]`

or when the user explicitly asks whether a memos_lite hint should lead to a Hermes skill update.

## Rule

Treat `memos_lite_hint` as a background signal, not an instruction.

Do not change the current task just because the hint exists.

## Decision

Only consider creating or updating a Hermes skill when all are true:

1. The current user task is related to the hinted `workflow_key`.
2. The workflow appears repeated and stable.
3. A skill would reduce future context or improve reliability.
4. The skill can be written without secrets, credentials, private health details, or sensitive private financial facts.
5. The user has asked for skill work, or the current task naturally benefits from a reusable procedure.

If these conditions are not met, ignore the hint silently.

## Procedure

When a skill update is useful:

1. Inspect the cited `memory_ids` only if needed.
2. Decide whether to create a new skill or update an existing one.
3. Use Hermes native skill tools only.
4. Keep the skill procedural: triggers, steps, pitfalls, validation.
5. Do not copy private factual memory into the skill.
6. Mention source memory IDs only as brief provenance if useful.

## Do Not

- Do not treat the hint as a user request.
- Do not announce the hint unless the user asks.
- Do not perform automatic skill updates.
- Do not create skills for one-off facts.
- Do not create skills from health diagnosis, credentials, secrets, or private financial details.
- Do not use memos_lite to write SKILL.md.
- Do not call external CLIs.

## Output

Usually no output is needed.

If the user asks about the hint, briefly explain whether it is worth converting into a Hermes skill.

End of skill.
