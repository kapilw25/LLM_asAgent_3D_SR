# iter16 ablations — JEPA-variant (model) ablation · INTEGRATION CODE PLAN + HERO TABLE

> What this file answers: **which JEPA world-models we ablate, and exactly how each is wired
> into `scripts/run_train.sh` (train + val) and `scripts/run_eval.sh` (test/eval) + the m09
> trainers + the eval registry.** FT *techniques* (SAFE/SEEKR/SSIAT/SAPT) → `plan_FTtechniues.md`.
> Temporal metrics → `plan_metrics_temporal.md`. Broader code/exec ops → `plan_code.md`
> (+ retired `legacy/plan_model_FTtechniues.md`).

Status: DRAFT · scope = FULL 115k for paper numbers, 10k for dev/SANITY (see §0.5 in legacy doc).
Grounded in a 2026-05-28 re-read of run_train.sh / run_eval.sh / m09a1/a2/c1/c2 / probe_encoders.yaml.

═══════════════════════════════════════════════════════════════════════════════
§0 · iter17 EXECUTION STATUS (updated 2026-05-29)
═══════════════════════════════════════════════════════════════════════════════
INFRA ✅ : WS-A per-backbone namespace (run_train BACKBONE selector + run_eval name→(bb,arm)
  parser; vitG outputs migrated → outputs/<mode>/vjepa_2_1_vitG/) · WS-B cross-arch seams
  (get_vit_by_arch +vit_large/+vit_huge/+vit_giant_xformers_2_1 ; frozen_features.load_vjepa_frozen
  by-name + load_encoder_by_kind single dispatch + ijepa branch ; m12a/b/c rewired). 3-check green.
CORRECTION: 2.1 ViT-g AND ViT-L checkpoints DO exist (vjepa2_1_vitg_384.pt /
  vjepa2_1_vitl_dist_vitG_384.pt) — the scale axis is alive (earlier "ViT-G only" was wrong).

```text
┌──────────────────────┬───────────┬──────────────┬──────────────────────────────────────┐
│ model                │ integrated│ frozen-SANITY│ sanity-TRAIN (run_train --SANITY)      │
├──────────────────────┼───────────┼──────────────┼──────────────────────────────────────┤
│ vjepa_2_1_vitG       │ ✅        │ ✅ (iter16)  │ ✅ pretrain_encoder (namespace valid.) │
│ vjepa_2_1_vitg       │ ✅        │ ✅ 482/484   │ ✅ deep-sup train (vit_giant_xf_2_1)   │
│ vjepa_2_1_vitL       │ ✅        │ ✅ 290/292   │ N/A — distilled→frozen-only (no pred)  │
│ vjepa_2_0_vitg       │ ✅        │ ✅           │ ⛔ m09a teacher(…,training=True) is 2.1-│
│                      │           │              │    only; 2.0 base ViT.forward lacks it  │
│ vjepa_2_vitL_256     │ ✅        │ ✅ 292/292   │ N/A frozen-only                        │
│ vjepa_1_vitL         │ ✅        │ ✅ 292/292   │ N/A 1.x, no predictor                  │
│ vjepa_1_vitH         │ ✅        │ ✅           │ N/A 1.x, no predictor                  │
│ ijepa_vitH14         │ ✅        │ ✅           │ N/A image-JEPA                         │
│ ijepa_vitG16         │ ✅        │ ✅           │ N/A image-JEPA                         │
│ dinov2               │ ✅        │ ✅           │ N/A non-JEPA                           │
│ vjepa_2_0_vitg_ssv2  │ ⛔        │ ⛔           │ ⛔ needs kind=hf_vjepa2 forward (HF)   │
│ lejepa_vitH14        │ ⛔        │ ⛔           │ ⛔ raw artifact → custom loader        │
│ mc_jepa / d_jepa     │ ⛔        │ ⛔           │ ⛔ weights gated/unreleased            │
└──────────────────────┴───────────┴──────────────┴──────────────────────────────────────┘
```
✅=done · ⏳=running/next · N/A=arch can't (frozen-only) · ⛔=blocked (new loader / trainer / gated).
Net: ALL 10 weight-available models PASS frozen inference. Sanity-TRAIN ✅ for the 2.1 backbones
(vitG + vitg, deep-sup) — the scale axis trains end-to-end. Blocked tail (each = a CODE change,
not a run): (1) 2.0_vitg TRAIN — m09a is 2.1-coupled (`teacher(...,training=True)` + deep-sup loss);
needs version-aware forward to train the VERSION axis. (2) ssv2 — needs kind=hf_vjepa2 forward.
(3) lejepa — raw-artifact custom loader. (4) mc/d-jepa — weights gated. FROZEN 2.0_vitg works, so
the 2.0 baseline is covered for eval; only 2.0 *continual-pretrain/surgery* needs the trainer fix.


═══════════════════════════════════════════════════════════════════════════════
§A · The JEPA-variant roster (the "models" axis)
═══════════════════════════════════════════════════════════════════════════════

ONE canonical name per model, used identically in every table below.

```text
┌──────────────────────────┬───────────────────────────────┬──────────────────┬─────┬─────┬─────┬───────┬───────────────────────┐
│ EXACT NAME               │ full model / source           │ arch (vjepa2_*)  │ blk │ dim │ crop│ train?│ role / axis           │
├──────────────────────────┼───────────────────────────────┼──────────────────┼─────┼─────┼─────┼───────┼───────────────────────┤
│ vjepa_2_1_vitG (PRIMARY) │ V-JEPA 2.1 ViT-G/2B           │ vit_gigantic_xf  │ 48  │1664 │ 384 │ ✅    │ primary full pipeline │
│ vjepa_2_1_vitg           │ V-JEPA 2.1 ViT-g/1B           │ vit_giant_xf     │ 40  │1408 │ 384 │ ✅    │ scale axis (½ of G)   │
│ vjepa_2_1_vitL           │ V-JEPA 2.1 ViT-L/300M        │ vit_large_xf     │ 24  │1024 │ 384 │ ✅    │ scale axis (small)    │
│ vjepa_2_0_vitg           │ V-JEPA 2.0 ViT-g/1B          │ vit_giant_xf     │ 40  │1408 │ 384 │ ✅    │ version axis (2.0)    │
│ vjepa_2_0_vitg_ssv2      │ V-JEPA 2.0 ViT-g SSv2-FT     │ vit_giant_xf     │ 40  │1408 │ 384 │frozen │ supervised action bl. │
│ vjepa_1_vitL             │ V-JEPA 1 ViT-L/16 (2024)     │ vit_large (1.x)  │ 24  │1024 │ 224 │frozen │ version axis (v1)     │
│ vjepa_1_vitH             │ V-JEPA 1 ViT-H/16 (2024)     │ vit_huge (1.x)   │ 32  │1280 │ 224 │frozen │ version axis (v1)     │
│ vjepa_2_vitL_256         │ V-JEPA 2 ViT-L (fpc64-256)   │ vit_large (HF)   │ 24  │1024 │ 256 │frozen │ version/res axis (2.0)│
│ ijepa_vitH14             │ I-JEPA ViT-H/14 IN-1k        │ vit_huge (image) │ 32  │1280 │ 224 │frozen │ image baseline        │
│ ijepa_vitG16             │ I-JEPA ViT-G/16 IN-22k       │ vit_giant (image)│ —   │1408 │ 224 │frozen │ image baseline        │
│ lejepa_vitH14            │ LeJEPA-L ViT-H/14            │ vit_huge (image) │ 32  │1280 │ 224 │frozen │ image baseline        │
│ mc_jepa          🟡gated │ MC-JEPA motion+content       │ ViT (vid/img)    │ —   │ —   │ —   │frozen │ ★ motion contrast     │
│ d_jepa           🟡gated │ D-JEPA denoising (image)     │ ViT (image)      │ —   │ —   │ —   │frozen │ image-gen baseline    │
│ dinov2 (non-JEPA)        │ DINOv2 ViT-g + registers     │ HF AutoModel     │ 40  │1536 │ 224 │frozen │ non-JEPA contrast     │
└──────────────────────────┴───────────────────────────────┴──────────────────┴─────┴─────┴─────┴───────┴───────────────────────┘
```
train? : ✅ = full 5-arm train+eval · frozen = eval-only · 🟡 = weights GATED (add iff released).
ckpt sources: vjepa_2_1_* = checkpoints/vjepa2_1_vit{G,g,L}_384.pt (or torch.hub vjepa2_1_vit_*);
vjepa_2_0_* = facebook/vjepa2-vitg-fpc64-384(-ssv2); vjepa_1_* = facebookresearch/jepa (VideoMix2M);
vjepa_2_vitL_256 = HF facebook/vjepa2-vitl-fpc64-256(-fpc16-256-ssv2); ijepa_* = facebook/ijepa_*;
lejepa_vitH14 = HF asset lejepa-l.pt; mc_jepa = arXiv 2307.12698 (weights pending); dinov2 =
facebook/dinov2-with-registers-giant.
SKIP (not a per-clip video/image encoder): VL-JEPA (text query), A-JEPA (audio),
Point-JEPA/3D-JEPA, Graph-JEPA, T-JEPA/TS-JEPA, Brain-JEPA/ECG-JEPA/S-JEPA, Stem-JEPA (music).
Dropped post-M0: jepa-wms, LeWorldModel, H-JEPA (robotics/wrong-modality, no usable video weights).

═══════════════════════════════════════════════════════════════════════════════
§B · Per-variant FT-arm support (what each architecture can run)
═══════════════════════════════════════════════════════════════════════════════

The 5 FT arms = {frozen, pretrain_encoder, pretrain_head, surgery_encoder, surgery_head}.
Surgery/pretrain need a VIDEO JEPA with a predictor + the hierarchical (n_output_distillation=4)
output; image JEPAs have neither.

```text
┌──────────────────────┬────────┬─────────────┬───────────┬──────────────┬────────────┐
│ model                │ frozen │ pretrain_enc│ pretrain_hd│ surgery_enc  │ surgery_hd │
├──────────────────────┼────────┼─────────────┼───────────┼──────────────┼────────────┤
│ vjepa_2_1_vitG       │ ✅     │ ✅          │ ✅        │ ✅ (48-blk)  │ ✅         │
│ vjepa_2_1_vitg       │ ✅     │ ✅          │ ✅        │ ✅ (40-blk)  │ ✅         │
│ vjepa_2_1_vitL       │ ✅     │ ✅          │ ✅        │ ✅ (24-blk)  │ ✅         │
│ vjepa_2_0_vitg       │ ✅     │ ✅          │ ✅        │ ✅ (40-blk)  │ ✅         │
│ vjepa_2_0_vitg_ssv2  │ ✅     │ ❌ (FT ckpt)│ ❌        │ ❌           │ ❌         │
│ vjepa_1_vitL         │ ✅     │ ❌ 1.x arch │ ❌        │ ❌ 1.x arch  │ ❌         │
│ vjepa_1_vitH         │ ✅     │ ❌ 1.x arch │ ❌        │ ❌ 1.x arch  │ ❌         │
│ vjepa_2_vitL_256     │ ✅     │ ❌ HF-only  │ ❌        │ ❌           │ ❌         │
│ ijepa_vitH14         │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ ijepa_vitG16         │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ lejepa_vitH14        │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ mc_jepa      🟡gated │ ✅*    │ ❌          │ ❌        │ ❌           │ ❌         │
│ d_jepa       🟡gated │ ✅*    │ ❌          │ ❌        │ ❌           │ ❌         │
│ dinov2 (non-JEPA)    │ ✅     │ ❌          │ ❌        │ ❌           │ ❌         │
└──────────────────────┴────────┴─────────────┴───────────┴──────────────┴────────────┘
```
* = frozen support gated on weights surfacing (mc_jepa / d_jepa).
Frozen-only models → eval Stages 2/3/3.5/5/6 (+ m12 action/motion_cos/taxonomy) but SKIP Stage 8
future_mse + the new m12e predictor-temporal suite (no trained predictor — CLAUDE.md
TRUE-IMPOSSIBILITY carve-out). Surgery layer_freeze indices are depth-specific (§E).

═══════════════════════════════════════════════════════════════════════════════
§C · Integration architecture — the FOUR code seams (traced 2026-05-28)
═══════════════════════════════════════════════════════════════════════════════

```text
seam 1  configs/model/<variant>.yaml         backbone spec: arch / embed_dim / depth / heads /
        (clone vjepa2_0.yaml)                pred_* / crop / patch / tubelet / checkpoint_url /
                                             n_output_distillation / predict_all. CONSUMED by
                                             m09 build_student_predictor(model_cfg, data_cfg).
seam 2  scripts/run_train.sh                 L185 MODEL_CFG="configs/model/vjepa2_1.yaml" is
        (TRAIN — parameterize backbone)      HARDCODED → make it ${MODEL_CFG:-...} + a per-
                                             backbone output namespace so each variant trains
                                             into its own outputs/<mode>/<variant>/<arm>/ dir.
                                             Technique recipe stays in --train-config (unchanged).
seam 3  scripts/run_eval.sh                  (a) ENCODERS list (L137) ; (b) encoder_ckpt_for()
        (EVAL — register + resolve ckpt)     case (L182, →student_encoder.pt) ; (c) encoder_
                                             predictor_ckpt_for() case (L217, →*_ckpt_best.pt,
                                             Stage 8) ; (d) the FROZEN special-case (external .pt).
seam 4  configs/eval/probe_encoders.yaml     registry row per variant: kind / arch / crop /
        + src/utils/frozen_features.py       embed_dim. kind=vjepa already handles ALL V-JEPA
                                             ckpts (target_encoder→encoder→raw fallback) — NO new
                                             loader for V-JEPA variants. embed_dim propagates to
                                             the probe head dim automatically (probe_action
                                             _make_probe(d_in=embed_dim)). kind=ijepa = NEW (§F).
```

KEY WIN: because the m09 trainers load arch from `model_cfg["arch"]` via `get_vit_by_arch` and
probe heads size off the registry `embed_dim`, adding a V-JEPA variant is **config + shell-resolver
only — zero new Python** (surgery freeze auto-scales by depth §E; the one new file is the
image-JEPA loader §F).

═══════════════════════════════════════════════════════════════════════════════
§D · DETAILED CODE PLAN — adding one V-JEPA variant end-to-end (vitL worked example)
═══════════════════════════════════════════════════════════════════════════════

```text
D1  configs/model/vjepa2_1_vitL.yaml   (NEW — clone vjepa2_1.yaml, change:)
      arch: vit_large_xformers   embed_dim: 1024   depth: 24   num_heads: 16
      pred_embed_dim/-depth/-num_heads per V-JEPA-2.1-L predictor   crop_size: 384
      checkpoint_url/path: vjepa2_1 vit_large_384 hub weights   n_output_distillation: 4
D2  scripts/run_train.sh                (parameterize the backbone, ~6 lines)
      L185: MODEL_CFG="${MODEL_CFG:-configs/model/vjepa2_1.yaml}"
      derive a BACKBONE tag from MODEL_CFG basename → output ns:
        outputs/${mode_dir}/${BACKBONE}/m09a_pretrain_encoder/...  (keep arms under the backbone)
      surgery init (SURGERY_INIT) already derives from PRETRAIN_NS → make PRETRAIN_NS backbone-aware.
D3  scripts/run_eval.sh                 (register + resolve, ~8 lines)
      ENCODERS += vjepa_2_1_vitL_{frozen,pretrain_encoder,...,surgical_*}   (or a BACKBONES loop)
      encoder_ckpt_for():           new cases → outputs/${PFX}/${BACKBONE}/<arm>/student_encoder.pt
      encoder_predictor_ckpt_for(): new cases → .../<arm>/m09{a,c}_ckpt_best.pt
      frozen case for the backbone → its downloaded .pt
D4  configs/eval/probe_encoders.yaml    (NEW rows — one per variant×arm that gets evaluated)
      vjepa_2_1_vitL_frozen: {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
      ...(pretrain/surgery arms = same arch/dim, different ckpt resolved in run_eval)
D5  surgery freeze: NONE — already depth-agnostic via int(depth*unfreeze_below) (§E)
D6  SANITY smoke per new backbone       run_train --SANITY (1 arm) + run_eval --sanity ENCODERS=<one>
      → confirms ckpt loads at the new arch/dim, probe head sizes to embed_dim, no shape crash.
```

Scaling to ALL trainable variants: wrap D2/D3 in
`BACKBONES="vjepa_2_1_vitG vjepa_2_1_vitg vjepa_2_1_vitL vjepa_2_0_vitg"` so run_train trains every
(backbone × arm) and run_eval enumerates every (backbone × arm) encoder — the per-variant work
collapses to D1 (one yaml) + D4 (registry rows). Surgery freeze auto-scales (§E). Frozen-only models
(vjepa_2_0_vitg_ssv2, vjepa_1_vitL, vjepa_1_vitH, vjepa_2_vitL_256, ijepa_vitH14, ijepa_vitG16,
lejepa_vitH14, dinov2) skip D1/D2/D5 — registry + ckpt-resolver rows only.

═══════════════════════════════════════════════════════════════════════════════
§E · Surgery freeze is ALREADY depth-agnostic — NO per-backbone edit
═══════════════════════════════════════════════════════════════════════════════

CORRECTION (2026-05-28): an earlier draft of this section was WRONG. The surgery configs do NOT
hardcode 48-block indices. Each `configs/train/surgery_*.yaml` declares freeze as a DEPTH
FRACTION via `surgery.stages[*].unfreeze_below` (recipe-v3 = 0.083, 0.167 ≈ 4/8 of 48), and
`src/m09c1_surgery_encoder.py:1150` applies it as `n_trainable = int(depth * unfreeze_below)`,
where `depth` is the LOADED model's block count. So surgery auto-scales to ANY backbone with
ZERO recipe edit:

```text
model            depth  unfreeze_below=0.083 →  =0.167 →   (n_trainable = int(depth * frac))
vjepa_2_1_vitG   48     3–4                     8           recipe-v3 baseline
vjepa_2_1_vitg   40     3                       6           auto
vjepa_2_1_vitL   24     1–2                      4          auto
```
The ONLY per-backbone check: confirm `depth` is read from the loaded model (it is). NO new code.
(A draft `src/utils/surgery_freeze.py` was created then DELETED — redundant with m09c1:1150, and
its `(2/3,1/3)` was the retired legacy 12/24 reading, not recipe-v3. Per "hardcoded values live in
configs/ only", the fractions belong in the surgery yaml — where they already are.)

═══════════════════════════════════════════════════════════════════════════════
§F · Image-JEPA adapter (I-JEPA / LeJEPA) — frozen-only, NEW loader
═══════════════════════════════════════════════════════════════════════════════

```text
F1  configs/eval/probe_encoders.yaml   ijepa_vitH14_frozen: {kind: ijepa, arch: vit_huge,
                                       model_id: facebook/ijepa_vith14_1k, crop: 224, embed_dim: 1280}
F2  src/utils/frozen_features.py       NEW load_ijepa_frozen() + forward_ijepa(): encode each of
                                       the T frames as an IMAGE → (T, N_patch, D) → mean-pool over
                                       frames+patches → (D,). Mirrors forward_dinov2's per-frame path.
F3  dispatch                           probe_action/_motion_cos/_taxonomy `kind` dispatch already
                                       branches vjepa/dinov2 → add `ijepa`. Stage 8 + m12e: GUARD
                                       `if kind != "ijepa"` (no predictor → skip, log the N/A).
```
LeJEPA loads the same way (ViT-H/14 image encoder), just a different checkpoint asset.

═══════════════════════════════════════════════════════════════════════════════
§G · 🦸 HERO TABLE — JEPA-variant × FT-arm × metric (cells to fill at FULL)
═══════════════════════════════════════════════════════════════════════════════

Each cell = mean ± 95% BCa CI on FULL test split. ↑=higher-better, ↓=lower-better.
Headline claim lives in the bold cells: surgery_enc vs pretrain_enc per backbone, per metric.

```text
model · arm                       │ action↑ │ motion_cos↑│ future_mse↓│ rollout↓ │ teacher_free↓│ taxon↑
                                  │ (m12)   │ (m12b)     │ (m12d)     │ (m12e #1)│ (m12e #4)    │ (m12c)
──────────────────────────────────┼─────────┼────────────┼────────────┼──────────┼──────────────┼───────
vjepa_2_1_vitG · frozen           │   ·     │     ·      │     ·      │    ·     │      ·       │   ·
vjepa_2_1_vitG · pretrain_encoder │   ·     │     ·      │     ·      │    ·     │      ·       │   ·
vjepa_2_1_vitG · pretrain_head    │   ·     │     ·      │     ·      │    ·     │      ·       │   ·
vjepa_2_1_vitG · surgery_encoder ★│   ·     │     ·      │   **·**    │  **·**   │    **·**     │   ·
vjepa_2_1_vitG · surgery_head     │   ·     │     ·      │     ·      │    ·     │      ·       │   ·
──────────────────────────────────┼─────────┼────────────┼────────────┼──────────┼──────────────┼───────
vjepa_2_1_vitg · ×5 arms          │   ·     │     ·      │     ·      │    ·     │      ·       │   ·  scale axis
vjepa_2_1_vitL · ×5 arms          │   ·     │     ·      │     ·      │    ·     │      ·       │   ·  scale axis
vjepa_2_0_vitg · ×5 arms          │   ·     │     ·      │     ·      │    ·     │      ·       │   ·  version axis
──── FROZEN-only (frozen arm only) ┼─────────┼────────────┼────────────┼──────────┼──────────────┼───────
vjepa_2_0_vitg_ssv2               │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  supervised action bl.
vjepa_1_vitL                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v1 version axis
vjepa_1_vitH                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v1 version axis
vjepa_2_vitL_256                  │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v2/res axis
ijepa_vitH14                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
ijepa_vitG16                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
lejepa_vitH14                     │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
mc_jepa            🟡gated        │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  ★ motion contrast
d_jepa             🟡gated        │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image-gen baseline
dinov2 (non-JEPA)                 │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  non-JEPA contrast
```
N/A on frozen rows = future_mse/rollout/teacher_free need a trained predictor (none here).
Reads the table answers: (1) surgery>pretrain ACROSS backbones (vjepa_2_1_vitG/vitg/vitL) →
generality; (2) does the surgery edge GROW or SHRINK with scale (vitL→vitg→vitG) → scaling law;
(3) hold across VERSION (vjepa_2_0_vitg / vjepa_1_* / vjepa_2_vitL_256 vs vjepa_2_1_*); (4) how
the frozen SSv2 / I-JEPA / LeJEPA / DINOv2 baselines bound it.

**§G.2 · Full metric catalog — HERO (6) vs APPENDIX (8).** The hero stays the 6 headline columns
above; the remaining metrics go in a SAME-SHAPE appendix grid (models × metric), NOT the hero
(14 metrics × 14 models = unreadable). Same eval harness fills both.
```text
metric          module        dir  tier        surgery-win prob     status
action top1     m12  (probe)   ↑    HERO        pretrain ≳ surgery   ✅ live
motion_cos      m12b           ↑    HERO        ≈ (trained > frozen) ✅ live
taxonomy        m12c           ↑    HERO        capability check     ✅ live
future_mse      m12d           ↓    HERO        surgery wins ★       ✅ live
rollout         m12e #1        ↓    HERO ★      HIGHEST (predictor)  ✅ built · GPU-val pending
teacher_free    m12e #4        ↓    HERO ★      HIGHEST (predictor)  ✅ built · GPU-val pending
causal          m12e #2        ↓    APPENDIX    high (predictor)     ✅ built · GPU-val pending
tdist           m12e #3        ↓    APPENDIX    med  (predictor)     ✅ built · GPU-val pending
maskratio       m12e #5        ↓    APPENDIX    med  (predictor)     ✅ built · GPU-val pending
order           m12e #6        ↕    APPENDIX    med  (predictor)     ✅ built · GPU-val pending
TOV / VCOP      phase-2 enc    ↑    APPENDIX    lower (encoder)      ⏳ metrics§6 (Task #6)
Arrow-of-Time   phase-2 enc    ↑    APPENDIX    lower (encoder)      ⏳ metrics§6 (Task #6)
Pace            phase-2 enc    ↑    APPENDIX    lower (encoder)      ⏳ metrics§6 (Task #6)
TCC             phase-2 enc    ↑    APPENDIX    lower (no-train)     ⏳ metrics§6 (Task #7)
```
↕ = order's sign is the signal (reliance on temporal order), not strictly better/worse.
PROMOTION RULE: if an APPENDIX predictor metric shows a CI-separated surgery>pretrain that
future_mse/rollout do NOT, promote it into the hero. Encoder metrics stay appendix (lower prob,
phase-2) unless one surprises. Frozen models get only the encoder-feature metrics (action/
motion_cos/taxon/TOV/AoT/Pace/TCC) — all predictor columns are N/A.

═══════════════════════════════════════════════════════════════════════════════
§H · Sequencing + cost
═══════════════════════════════════════════════════════════════════════════════

```text
• POC/dev (10k): add vitL first (cheapest, 6.7× smaller) → validate the BACKBONES loop end-to-end
  before paying FULL. vitL full 5-arm train ≈ ¼ of vitG.
• Frozen-only variants (ssv2, ijepa, lejepa, dinov2) are eval-only → no training cost, just
  registry + (ijepa) the §F loader.
• Per-variant marginal work after the loop lands: 1 model-config + registry rows (surgery
  freeze auto-scales — §E).
• GATE: same as the rest of iter16 — no run_train.sh / run_eval.sh edits while the current eval
  runs (live-script). This plan is authored now; edits land in the post-eval pass.
```
