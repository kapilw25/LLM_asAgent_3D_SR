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
- [visual_mistakes.md](visual_mistakes.md) — append-only figure/demo layout mistake KB (VM1-VM29); the visual-audit agent (../agents/visual-audit.md) re-reads it on EVERY audit — run that agent on every figure/demo before showing the user
- [iter20_demo_layman_win.md](iter20_demo_layman_win.md) — ⚠️ SUPERSEDED: the scene-W "layman win" was a FALSE APPROVE (VM27) the real user could not see; VM29 proved OURS>FROZEN is sub-perceptual on video. Kept as a cautionary record, NOT a shippable design
- [project_iter20_ood_edge_indomain_only.md](project_iter20_ood_edge_indomain_only.md) — 🔒 **CLOSED 2026-07-19**: OURS's surgery edge is IN-DOMAIN ONLY (loses OOD by up to 10.8pp, ghat-POV probe n=167 + Diving48 ×2); the visible demo_cosmos-style demo is NOT achievable — 4 independent closures. READ THIS BEFORE proposing another demo framing
- [feedback_metric_artifact_fake_win.md](feedback_metric_artifact_fake_win.md) — score BOTH arms with the IDENTICAL ruler (a mid-run parser fix fabricated a "+29.8pp win" that was really −0.2pp); save the model's RAW output, not just the parsed answer; check degeneracy before believing any accuracy
- [feedback_scratch_scripts_in_repo_tmp.md](feedback_scratch_scripts_in_repo_tmp.md) — scratch/probe scripts that produce a CITED result go in `src/utils/tmp/` (repo-tracked, survives box jumps), NEVER the /tmp session scratchpad (ephemeral, dies on box teardown — nearly lost ood_turn_probe.py). We jump boxes continuously
- [project_iter20_vlm_built.md](project_iter20_vlm_built.md) — iter20 VLM-head (App A) BUILT + 3060-verified (751 LoC: vlm.yaml + vlm_model + m18_vlm_{data,train,eval}); OURS-VLM vs FROZEN-VLM demo_cosmos; EARLY GATE on TempCompass = the OOD-transfer make-or-break; runs on 96GB box (runbook §E1)
- [project_iter20_demo_cosmos_impossible.md](project_iter20_demo_cosmos_impossible.md) — why the cheap layman-verifiable demo can't exist (motion invisible on WalkIndia radial flow); the exhaustive probe search; led to choosing the VLM
- [conventions.md](conventions.md) — CLAUDE.md's load-bearing rules condensed (no hardcoded defaults, fail-hard, cache-policy contract, semicolon-not-&&, never-rm)

## Active feedback memories

- [feedback_no_hardcoded_defaults.md](feedback_no_hardcoded_defaults.md) — every numeric default lives in YAML, not as Python literal or argparse default
- [feedback_tee_logs_on_terminal.md](feedback_tee_logs_on_terminal.md) — every operator command ends `2>&1 | tee logs/<process>_$(date +%Y%m%d_%H%M%S).log` (+ `set -o pipefail`) so logs stream on the terminal AND land in a timestamped file
- [feedback_no_hallucinated_victory.md](feedback_no_hallucinated_victory.md) — never fabricate an OURS>FROZEN demo win; gate every "OURS wins" claim behind a passing fail-loud measurement; a truthful negative is a valid stop, not giving up
- [project_iter20_demo_cosmos_impossible.md](project_iter20_demo_cosmos_impossible.md) — DEFINITIVE: OURS loses 0/15 taxonomy SCENE questions (demo_cosmos's exact format); no layman-verifiable question has OURS winning; present the forest plots, not a VQA
