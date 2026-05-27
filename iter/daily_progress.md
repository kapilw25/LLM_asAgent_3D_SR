# Instruction: Date & daily udpate in 2 line for daily progress && 1 line for next step. Each line covers max 10 words
# AUDIENCE: Research leads focussed on > ## 🎯 Paper goal:  `vjepa_surgery` [X_epochs(surgery) +X_epochs(pretrain)] ≫ `vjepa_pretrain` [2X epochs] ≫ `vjepa_frozen` on motion / temporal features
# AUDIENCE has knowledge about ONLY plots from @iter/iter15_trainHead_freezeEncoder/result_outputs/v15a/poc/probe_plot/eval/
## Note: do NOT write pipeline jargon (SANITY/POC/eval_10k/Δ5/cells),

## Mon, May 25, 2026
- iter15 headline INVALID: test-leakage + 8× data-asymmetry found, fixed.
- Re-running leakage-safe paired-Δ; SANITY validates pipeline before FULL.
- Next: pass SANITY → FULL, test surgery≫pretrain≫frozen on motion.

## Tue, May 26, 2026
- First clean surgery-vs-pretrain numbers in (leakage-safe re-run).
- Surgery wins temporal (future-L1); pretrain wins action top1.
- Next: confirm on held-out test with CIs; add frozen.

## Wed, May 27, 2026
- Full-scale 116k-clip surgery data prepared: ~125 GPU-hr, ~$150.
- Held-out test evaluation running; frozen baseline now included.
- Next: read held-out motion-cos / future-L1 / top1 when done.
