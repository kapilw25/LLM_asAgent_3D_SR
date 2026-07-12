# FactorJEPA — Claude Code session-onboarding memory

> Fresh Claude Code sessions on a new GPU instance: read this file FIRST before any code action. Index is one-line-per-entry; click into the linked .md for full detail.

## Project pulse

- [project_pulse.md](project_pulse.md) — current iter (iter14), v12/0.808 anchor, what's done on HF, the 3 approval gates blocking iter14 execution
- [hardware_split.md](hardware_split.md) — what runs on 24 GB vs 96 GB (ironclad — confirmed empirically 2026-05-03)
- [next_actions.md](next_actions.md) — concrete commands to resume iter14 on a fresh 96 GB instance

## Architecture

- [pipeline_layout.md](pipeline_layout.md) — module map (m04 → m11 + m04d motion features, scripts/, utils/), what consumes what
- [codebase_inventory.md](codebase_inventory.md) — file-level reference: 4 canonical scripts + every m*.py + every utils/ + every configs/ + every data/ entry, one line each
- [iter14_motion_aux_pivot.md](iter14_motion_aux_pivot.md) — current pivot: motion_aux (CE+MSE) replaced multi_task_probe in v12; pointer to iter14 plans
- [config_schema.md](config_schema.md) — per-mode YAML flatten convention, opt-in pattern, ckpt-schema dispatch, motion_aux block
- [legacy/iter13_multi_task.md](legacy/iter13_multi_task.md) — RETIRED iter13 multi_task_probe pivot (replaced by motion_aux in v12; kept for historical context)

## Graph view (derived)

- [memory_graph.md](memory_graph.md) — mermaid map of every memory + contract + enforcer and their relations; machine twin [memory_graph.jsonl](memory_graph.jsonl) (MCP memory-server schema). DERIVED: this MEMORY.md stays the source of truth — update memories first, mirror the graph second.

## Operating notes (copy-paste safe)

- [bug_log.md](bug_log.md) — known bug classes with their fixes (A/B/R8/OOM-frag/eval-ckpt-schema/Stage-8/plot-NaN — all merged but watch for regressions)
- [visual_mistakes.md](visual_mistakes.md) — append-only figure/demo layout mistake KB (VM1-VM6); the visual-audit agent (../agents/visual-audit.md) re-reads it on EVERY audit — run that agent on every figure before showing the user
- [conventions.md](conventions.md) — CLAUDE.md's load-bearing rules condensed (no hardcoded defaults, fail-hard, cache-policy contract, semicolon-not-&&, never-rm)

## Active feedback memories

- [feedback_no_hardcoded_defaults.md](feedback_no_hardcoded_defaults.md) — every numeric default lives in YAML, not as Python literal or argparse default
