---
name: feedback-scratch-scripts-in-repo-tmp
description: scratch/probe scripts that produce a CITED result go in src/utils/tmp/ (repo-tracked → git add . picks it up → survives box jumps), NEVER the /tmp session scratchpad which is ephemeral and dies on box teardown
metadata:
  type: feedback
---
**2026-07-19 incident.** `ood_turn_probe.py` — the script that produced the paper's **decisive OOD finding**
(FROZEN beats OURS by −6…−10.8pp on visible ego-yaw, [[project_iter20_ood_edge_indomain_only]]) — was written to
`/tmp/claude-0/<session-uuid>/scratchpad/`. The commit message CLAIMED "new ood_turn_probe", but `git add .`
never saw it (it lives OUTSIDE the repo tree), so the load-bearing script was about to be **lost with the 96GB box**
when the user jumped to a cheaper instance. The findings survived (in git); the reproducibility script nearly didn't.

**Why:** this project **continuously jumps between GPU boxes** (3060 demo ↔ 96GB Blackwell ↔ cheaper GPU). The
Claude Code session scratchpad `/tmp/claude-0/<uuid>/scratchpad/` is **per-box + per-session ephemeral** — it does
NOT get git-committed and is wiped on box teardown / session end. Anything with reproducibility value written there
is silently lost the moment we switch boxes. The harness *suggests* the session scratchpad for temp files, but that
guidance assumes one box — it does not fit a multi-box workflow.

**How to apply:**
- **Scratch/probe/one-off analysis `.py` that yields a RESULT we cite** (encoder probes, ablations, figure
  generators, eval harnesses) → write to **`src/utils/tmp/`** (repo-tracked → `git add .` picks it up → on GitHub →
  survives every box jump). Give it a real docstring; it IS a reproducibility artifact.
- **Truly-throwaway intermediates** (decoded frames, `.npy` feature caches, per-run logs, contact-sheet PNGs) →
  the session scratchpad is fine; they're regenerable and have no reproducibility value.
- **Before writing "new <script>" in a commit message, VERIFY it's in the repo tree** (`git status` / `ls`), not
  the session scratchpad. Same VERIFY-FIRST discipline as [[feedback_metric_artifact_fake_win]].
- Prior instance already in the repo: `src/utils/scratchpad/scan_turns.py`. Consolidate scratch scripts under one
  repo-tracked dir — `src/utils/tmp/` is the canonical one going forward.
