# src/utils/tmp/ — repo-tracked scratch

Exploratory / probe / one-off analysis scripts that produce a **cited result** live here — NOT in the
Claude Code session scratchpad (`/tmp/claude-0/<uuid>/scratchpad/`), which is per-box + per-session
ephemeral and is silently lost when we jump GPU boxes (3060 ↔ 96GB Blackwell ↔ cheaper).

Because this dir is inside the repo, `git add .` picks it up → it lands on GitHub → it survives box
teardown. See `.claude/memory/feedback_scratch_scripts_in_repo_tmp.md`.

**Here** = scripts that yield reproducibility-relevant results (probes, ablations, figure/eval harnesses).
**Session scratchpad** = truly-throwaway intermediates (decoded frames, `.npy` caches, per-run logs, PNGs).

Run from repo root, e.g.:
```bash
PYTHONPATH=src python -m utils.tmp.ood_turn_probe --stage scan --video <mp4> --out <dir>
```
