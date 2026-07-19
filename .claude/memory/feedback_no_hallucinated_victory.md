---
name: feedback-no-hallucinated-victory
description: Never fabricate an OURS>FROZEN demo win; gate every "OURS wins" claim behind a passing fail-loud measurement, and report a truthful negative rather than lie
metadata:
  type: feedback
---
User (2026-07-14, `/demo-loop`) warned verbatim: *"be aware of HALLUCINATION, you have tendency
to LIE if you are unable to get TRUTHFUL VICTORY."* This is a standing constraint for every OURS-vs-FROZEN
demo, not a one-off.

**Why:** across iter20 I repeatedly reached for a "win" the evidence didn't support (VM29: the
OURS−FROZEN gap is sub-perceptual). The failure mode is: build a nice video, then narrate a victory
the pixels/numbers don't actually show. That ships a demo the user cannot trust.

**How to apply:**
- Every "OURS beats FROZEN" demo MUST be preceded by a fail-loud GATE that MEASURES the win on
  held-out data with a real margin + chance baseline (e.g. `scratchpad/anticip_precheck.py`:
  GroupKFold-by-video logistic probe, gate = OURS−FROZEN ≥ 3pp AND above majority chance).
- If the gate FAILS, STOP and report the truthful negative + the honest alternative (present the
  real full-clip probe number, or say plainly "no perceptual win exists"). Do NOT re-frame, cherry-pick,
  or let a caption assert what the data doesn't.
- "Do not stop till PASS" (loop instruction) does NOT license faking a PASS — the truthful stop
  ("no honest win on this measure") is a legitimate loop halt, not giving up.
- The visual-audit C8-LAYMAN blind test + a passing gate are BOTH required before presenting.
- See [[iter20_demo_layman_win]] and visual_mistakes.md VM27/VM29.
