---
name: feedback-metric-artifact-fake-win
description: A/B arms must be scored with the IDENTICAL ruler — changing the metric between arms fabricates a win; also always save the model's RAW output, not just the parsed answer
metadata:
  type: feedback
---
**2026-07-19 incident.** The iter20 VLM early gate printed **"✅ PASS — OURS beats FROZEN by +29.8pp"**.
It was **100% artefact**: FROZEN had been scored with a letter-only extractor that auto-failed all 340 yes/no
rows (60% of the eval set), while OURS ran after the parser was fixed. **Two arms, two different rulers.**
Re-scored on the matched metric: FROZEN 0.446 vs OURS 0.444 = **−0.2pp**. The "+29.8pp win" never existed.

**Why:** predicted-vs-actual arithmetic caught it BEFORE it was reported — frozen re-scored should be
≈ 0.36×(230/570) + 0.50×(340/570) ≈ 0.443, essentially OURS's 0.444 → gap ≈ 0. Measured: 0.446. Confirmed.

**How to apply:**
- 🚫 **NEVER compare arms scored at different times across a metric/parser/data change.** If ANY of
  (extractor, prompt, benchmark rows, label set) changed, RE-SCORE BOTH ARMS before computing a delta.
  Training is usually reusable — re-eval is cheap, retraining is not.
- 🔢 **Sanity-check a surprising win with arithmetic first.** A +29.8pp jump on an OOD task where 3 prior
  probes said −1pp is a red flag, not a triumph. Predict what the number SHOULD be; if the prediction
  explains the gap, it's an artefact.
- 💾 **Save the model's RAW output, not just the parsed answer** (`raw` field + `src/utils/vlm_eyeball.py`).
  Without it you cannot distinguish "answered wrongly" from "answered fine, our parser missed it" — exactly
  the bug that zeroed 60% of the set. Dump PROMPT / GT / RAW / PARSED so a human can verify by eye.
- 🗄️ **Quarantine invalid artefacts by renaming** (`*.BROKEN_METRIC.json`, `*.INVALID.json`) so a later
  session cannot mistake them for results.
- 📉 **Check for degeneracy before believing ANY accuracy**: prediction distribution (collapsed to one
  class?), unparsed rate, and per-subset accuracy vs the majority-class baseline. Both arms at chance means
  the experiment tested your training budget, NOT your hypothesis — report it as inconclusive, never as a
  negative result about the model. See [[feedback_no_hallucinated_victory]], [[project_iter20_ood_edge_indomain_only]].
