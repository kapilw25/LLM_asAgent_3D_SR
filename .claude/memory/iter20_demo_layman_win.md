---
name: iter20 layman-visible demo — the design that finally passed
description: What made OURS-vs-FROZEN a layman-spottable win in scene W (after 5 rejected takes)
type: project
---

# iter20 scene-W demo — ⚠️ SUPERSEDED (the "win" was a FALSE APPROVE)

> ⚠️ **CORRECTION 2026-07-14 (VM27/VM29):** the "audit APPROVED" below was a FALSE POSITIVE — the
> visual-audit agent approved a squint/hue gestalt that the REAL user could NOT see ("I don't see any
> difference", demo1-5.png). VM29 then proved across 6 modalities that OURS>FROZEN is **sub-perceptual
> on video**. This whole scene-W design does NOT ship as an honest layman win. Kept as a cautionary
> record of how a foggy-decode gestalt fooled the audit. See [[feedback_no_hallucinated_victory]].

**Original (now-retracted) claim**: a video where a stranger (no ML, no numbers, text removed) spots
OURS diheavy > FROZEN 2.1. Was believed to pass on take v5, `outputs/demo/metric_visual/demo_W.mp4`.

## What FAILED (all logged as VM entries — never repeat)
- report-card heatmaps (green/red on real frames): twin panels, layman sees no winner (VM v1-v2).
- side-by-side decoded futures: fog-vs-fog, only a HUE difference (blue vs warm) — dismissible (VM23).
- ROI zoom insets, two ROI strategies (max-gap VM24, structure-weighted): at crop scale every
  predictor-latent decode is featureless fog — crops are fog-blind (VM24, VM25).

## What WORKED (scene W, `src/m14_metric_demo.py`)
1. **m15 tubelet-inversion decoders** (per model, own feature space) turn each model's PREDICTED
   hidden-half latents into real (blurry) pixels. Decoders: `outputs/demo/m15_{frozen,ours}/decoder.pt`
   (10.2M-param MLP 1408→2048→1536, L1, decode-sanity gate passed).
2. **Continuity cut**: each middle panel PLAYS the real first half, then CUTS to its own imagination —
   "which video breaks at the cut" is pre-attentive.
3. **Pixel-decisive hero + clip selection** (`select_decisive`): rank clips by mean|imag_FROZEN−real| −
   |imag_OURS−real|; FAIL LOUD if no clip favours OURS. Both Varanasi heroes favour OURS (+15.7,+12.4).
4. **SQUINT TEST row** (VM25): identical 8×8-block downsample of FROZEN-imag | OURS-imag | REAL —
   gestalt (hue+layout) survives squinting, fog cannot fake it; OURS-squint pairs with REAL-squint,
   FROZEN-squint is the odd cold-blue one out. Blue↔warm axis is colour-blind-safe.
5. **Real-future edge outline** (sparse, MIT Video-Diff style) on the imagined panels.
6. honest caption on every frame; verdict card renders real values + winner marks (VM21).

Data: `data/demo_clips_humans/` (4 Varanasi walking-crowd clips, shard 112). Runbook:
`iter/iter20_visual_DEMO/runbook.md`. Config knobs: `configs/demo.yaml` demo/m15 blocks.

**Reusable lesson (VM25)**: to make a BLURRY generative comparison layman-visible, compare at the SCALE
where the signal lives — gestalt/squint + continuity cut for foggy outputs, never crops or hue alone.
