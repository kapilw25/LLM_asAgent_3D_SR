---
name: visual-mistakes-kb
description: Pointer — the append-only figure/plot/demo mistake KB is now ONE FILE PER MISTAKE under visual_mistakes/ to stop this prompt from inflating; a PreToolUse hook surfaces the relevant ones on every figure edit
type: project
---

# Visual mistakes KB — split into one file per mistake

This file used to hold every VM inline and grew to ~500 lines, inflating every prompt. Verified lossless split (31,901 chars preserved exactly). It is now:

- **`.claude/memory/visual_mistakes/INDEX.md`** — one-line row per mistake (id · category · summary). Load this first.
- **`.claude/memory/visual_mistakes/VM<n>.md`** — the full rule for each mistake (VM1–VM38). Open only the ones that match the task.

**Automatic retrieval (no manual lookup):** the PreToolUse hook `.claude/hooks/surface-visual-mistakes.sh` fires on every `Edit`/`Write`. When the edit touches a figure / plot / `.tex` / demo, it injects the matching VM ids (by category) plus the universal **VM38 render-and-read gate** into context, before the edit. Non-figure edits are silent.

**The gate (VM38):** after ANY figure/layout edit, before presenting — rebuild, render every changed page ≥110 dpi, READ the labels for legibility, ensure connective text between consecutive figures and no figure before the abstract, and use any `_bold`/`_text_enlarge` figure variant. A single audit pass is insufficient.
