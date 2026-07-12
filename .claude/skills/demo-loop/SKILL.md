---
name: demo-loop
description: Continuous engineering loop for the visual metric demo (m14) — build → render → visual-audit → fix → re-render, never stopping until the audit passes AND the demo is humanly understandable. Use when asked to build/iterate/fix the metric visual demo or any presentation video/figure that must survive an audit.
---

# demo-loop — do not stop until the goal is achieved

**Goal**: `outputs/demo/metric_visual/demo.mp4` (+ contact_sheet.png, demo_metrics.json) that a
stranger can understand: what the model does, how each of the 4 headline metrics detects it, and
what the numbers mean. The loop ends ONLY on `AUDIT: PASS` from the visual-audit agent.

## The loop (repeat until PASS — no iteration limit)

0. **KB auto-audit (BEFORE any work, every iteration)** — Read
   `.claude/memory/visual_mistakes.md` + `.claude/plotting.md`. Every VM entry is a live checklist
   item; a fix that re-introduces a logged mistake is a double failure.
1. **Build/fix** — edit `src/m14_metric_demo.py` / `configs/demo.yaml` (Edit tool only, 3-check
   gate auto-fires via post-edit-lint).
2. **Render** — run m14 (USAGE block in its docstring; `source venv_walkindia/bin/activate`,
   background + tee log). FAIL LOUD errors → fix at the originating layer, never `|| true`.
3. **Self-eyeball** — Read `contact_sheet.png` yourself at full res FIRST (cheap pre-filter;
   render-log success is NOT visual verification).
4. **Visual-audit agent** — spawn `.claude/agents/visual-audit.md` on `contact_sheet.png` (and any
   standalone scene PNGs). It re-reads the KB, verdicts every checklist item + the
   "humanly-understandable" C8 check, and returns `AUDIT: PASS` or numbered FAILs.
5. **On FAIL** — for each finding: fix it; if it is a NEW mistake class, append a `VM<n>` entry to
   `.claude/memory/visual_mistakes.md` (Symptom → Root cause → Fix → Prevention) so it can never
   ship again. GOTO 0.
6. **On PASS** — present to the user: the mp4 path, the contact sheet, the demo_metrics.json
   numbers (spell out FULL metric names), and the audit verdict table.

## Hard rules

- NEVER present a visual artifact that has not passed step 4 this iteration (the
  `demo-audit-gate.sh` hook reminds after every render — it is not optional).
- Every mistake found at ANY step gets a KB entry BEFORE the fix lands — KB first, fix second.
- Metric parity is sacred: the demo must call the `utils.pt_*` / `utils.predictor_eval` cores; the
  built-in `np.allclose` parity guards in m14 must stay FATAL.
- GPU work runs on this box (user-assigned); never touch another session's GPU procs.
