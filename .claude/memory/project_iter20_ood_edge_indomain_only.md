---
name: project-iter20-ood-edge-indomain-only
description: CLOSED 2026-07-19 — OURS's surgery edge is IN-DOMAIN ONLY (loses OOD by up to 10.8pp); the visible demo_cosmos-style OURS>FROZEN video is NOT achievable, 4 independent closures
metadata:
  type: project
---
**The iter20 visual-demo question is CLOSED.** The intersection of "eye-verifiable" and "OURS wins" is **EMPTY**,
proven at the **encoder** level (not just labels/probes). Stop hunting for a fifth framing.

## The four closures
| # | test | domain | n | result |
|---|---|---|---|---|
| 1 | scene / taxonomy | in-domain | — | OURS loses 0/15 |
| 2 | magnitude / direction | in-domain | 1878 | OURS wins +8.7…+15.7pp but **INVISIBLE** (VM30 radial flow) |
| 3 | WalkIndia straight-vs-turn | in-domain | 440 | no bimodal boundary; ~1-3/12 read as turns → dead |
| 4 | **ghat POV ego-yaw** | **OOD** | **167** | **OURS LOSES −6.0 (MLP) / −10.8pp (LINEAR)** |

## The decisive OOD probe (the load-bearing evidence)
`scratchpad/ood_turn_probe.py` — encoder-level (NO LLM, so immune to projector starvation), 18-min ghat-road POV
driving, 8 s windows, flow-derived {STRAIGHT, TURN_LEFT, TURN_RIGHT}, **leave-one-of-5-TEMPORAL-blocks-out**
(adjacent windows of one drive are autocorrelated — a random split leaks and fabricates a win).
FROZEN 0.737/0.731 vs OURS 0.629/0.671; majority baseline 0.503 → **both arms ≫ baseline, so the test had power.**
Replicates Diving48 (−1.1pp ×2). Cab-POV confirmation (n=55) was underpowered/mixed (LINEAR +1.8, MLP −3.6,
SE ±5.8pp) → neither confirms nor contradicts; do NOT quote its LINEAR number as a win.

**Mechanism:** surgery SPECIALISES the encoder to WalkIndia motion statistics (radial walking flow) and that
specialisation COSTS it on a different motion regime (large ego-yaw). This is a **publishable quantified
limitation** across 3 OOD experiments — report it, don't hide it. Reviewers trust authors who surface this.

## ⛔ The VLM early gate did NOT test the encoder — never cite it as OOD evidence
FROZEN 0.446 vs OURS 0.444 (−0.2pp) **but both at chance**: yes/no 0.503/0.474 (coin flip, balanced set),
MC 0.361/0.400 vs 0.374 majority. A from-scratch projector on 2 885 samples/1 epoch cannot align video→language
(LLaVA stage-1 uses 558K, ~200×). ✅ conclusive about the TRAINING BUDGET · ❌ inconclusive about the ENCODER.

**Two eval bugs found there (both would have produced fake verdicts) — see [[feedback_metric_artifact_fake_win]]:**
a letter-only extractor auto-failed all yes/no rows (60% of the set), and `_bench_jsonl` indexed the wrong cfg
level. An early chain printed "✅ PASS +29.8pp" that was PURE metric artefact; re-scored matched it is −0.2pp.
Artefacts quarantined as `*.BROKEN_METRIC.json` / `*.INVALID.json` in outputs/demo/vlm/.

## Still shippable
forest plots (in-domain, non-overlapping CIs) · `outputs/demo/mcq/demo_mcq.mp4` (+13.7pp retrieval, overlay-free,
captioned **metric-verified not eye-verified**) · the NEW OOD limitation section. See [[project_iter20_vlm_built]],
[[project_iter20_demo_cosmos_impossible]], [[feedback_no_hallucinated_victory]].
