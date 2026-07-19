---
name: project-iter20-demo-cosmos-impossible
description: OURS loses ALL 15 taxonomy scene questions (demo_cosmos's exact format); no layman-verifiable question has OURS beating FROZEN — a demo_cosmos-style "OURS wins" VQA is not honestly achievable
metadata:
  type: project
---
**iter20 demo conclusion (2026-07-15) — DEFINITIVE.** A `demo_cosmos`-style VQA where OURS beats FROZEN
is NOT honestly achievable.

**Killer evidence:** `demo_cosmos` asks SCENE questions ("what type of road?"). Our taxonomy probe has
exactly that — 15 scene attributes — and OURS (`surgical_3stage_DI_diheavy`) LOSES **0/15** vs FROZEN
(`outputs/full/vjepa_2_1_vitg_1B/eval/full/probe_taxonomy/per_dim_acc.json`): scene_type 0.661 vs 0.705,
crowd_density 0.827 vs 0.845, pedestrian_vehicle_separation 0.856 vs 0.889, … every one FROZEN.

**Why (this is what surgery DOES, not a bug):** surgery specialized OURS for motion/temporal prediction,
trading a little static-scene accuracy → a genuine specialization trade-off.

**The OURS-wins ∩ layman-verifiable set is EMPTY:**
- scene attributes (verifiable) → FROZEN wins 0/15
- motion magnitude (OURS +8.2pp) → anti-correlates with vision (VM30, looks BACKWARDS)
- motion direction → degenerate (both below chance)
- visible-motion (OURS +3pp) → verifiable but weak/marginal
- motion-cosine +1432% / future-MSE / causal / action-top1 / pace (OURS wins BIG) → latent/technical, sub-perceptual

**How to apply:** OURS's real, large, defensible win is the eval metrics — present the **forest plots**
(`src/m13_eval_plot.py` outputs) as the paper figure. Do NOT build a VQA that shows OURS winning the
scene/semantic questions it actually LOSES. If a video is required, the only honest one is a SPECIALIZATION
trade-off ("OURS wins motion/prediction; FROZEN wins static scene"). See [[feedback_no_hallucinated_victory]], VM30.
