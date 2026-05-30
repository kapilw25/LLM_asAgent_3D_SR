# iter17 ablations — JEPA-variant (model) ablation · roster + integration + HERO TABLE

> WHAT/HOW JEPA world-models are wired into run_train.sh / run_eval.sh / m09 trainers / eval
> registry. FT techniques → legacy/plan_model_FTtechniues.md · temporal metrics → plan_metrics_temporal.md
> · detailed code diffs → legacy/plan_model_CODE.md (infra BUILT, git e8535c7). Scope: FULL 115k paper / 10k dev.

═══════════════════════════════════════════════════════════════════════════════
§0 · iter17 EXECUTION STATUS (updated 2026-05-29)
═══════════════════════════════════════════════════════════════════════════════

```text
┌──────────────────────┬───────────┬──────────────┬──────────────────────────────────────┐
│ model                │ integrated│ frozen-SANITY│ sanity-TRAIN (run_train --SANITY)      │
├──────────────────────┼───────────┼──────────────┼──────────────────────────────────────┤
│ vjepa_2_1_vitG       │ ✅        │ ✅ (iter16)  │ ✅ pretrain_encoder (namespace valid.) │
│ vjepa_2_1_vitg       │ ✅        │ ✅ 482/484   │ ✅ deep-sup train (vit_giant_xf_2_1)   │
│ vjepa_2_1_vitL       │ ✅        │ ✅ 290/292   │ N/A — distilled→frozen-only (no pred)  │
│ vjepa_2_0_vitg       │ ✅        │ ✅           │ ✅ sanity-train (deep-sup gated off;   │
│                      │           │              │    2.1 regress PASS — gates safe)      │
│ vjepa_2_vitL_256     │ ✅        │ ✅ 292/292   │ N/A frozen-only                        │
│ vjepa_1_vitL         │ ✅        │ ✅ 292/292   │ N/A 1.x, no predictor                  │
│ vjepa_1_vitH         │ ✅        │ ✅           │ N/A 1.x, no predictor                  │
│ ijepa_vitH14         │ ✅        │ ✅           │ N/A image-JEPA                         │
│ ijepa_vitG16         │ ✅        │ ✅           │ N/A image-JEPA                         │
│ dinov2               │ ✅        │ ✅           │ N/A non-JEPA                           │
│ vjepa_2_0_vitg_ssv2  │ ✅        │ ✅ (20,16,   │ N/A frozen-only (hf_vjepa2 fwd GPU-val │
│                      │           │    1408)     │    ✓; predictor N/A — skip_predictor) │
│ lejepa_vitL          │ ✅        │ ✅ (20,16,   │ N/A frozen-only (image JEPA; timm      │
│                      │           │    1024)     │    DINOv3-L + eva filter, 0miss/0unex) │
│ mc_jepa / d_jepa     │ ⛔        │ ⛔           │ ⛔ weights gated/unreleased            │
└──────────────────────┴───────────┴──────────────┴──────────────────────────────────────┘
```
✅=done · N/A=arch can't (frozen-only) · ⛔=blocked. 12 weight-available models PASS frozen; trainable
= vitG/vitg (2.1) + 2.0_vitg. Skipped (modality/no weights): V-JEPA 2-AC (encoder=2.0_vitg + robot-action
head), WavJEPA (audio), VL-JEPA (no public weights), mc/d-jepa (gated).

═══════════════════════════════════════════════════════════════════════════════
§A · The JEPA-variant roster (the "models" axis)
═══════════════════════════════════════════════════════════════════════════════

```text
{BACKBONE= , vjepa_2_1_vitg, vjepa_2_0_vitg, vjepa_2_1_vitG}
┌──────────────────────────┬───────────────────────────────┬──────────────────┬─────┬─────┬─────┬───────┬───────────────────────┐
│ EXACT NAME               │ full model / source           │ arch (vjepa2_*)  │ blk │ dim │ crop│ train?│ role / axis           │
├──────────────────────────┼───────────────────────────────┼──────────────────┼─────┼─────┼─────┼───────┼───────────────────────┤
│ vjepa_2_1_vitG (PRIMARY) │ V-JEPA 2.1 ViT-G/2B           │ vit_gigantic_xf  │ 48  │1664 │ 384 │ ✅    │ primary full pipeline │
│ vjepa_2_1_vitg           │ V-JEPA 2.1 ViT-g/1B           │ vit_giant_xf     │ 40  │1408 │ 384 │ ✅    │ scale axis (½ of G)   │
│ vjepa_2_1_vitL           │ V-JEPA 2.1 ViT-L/300M        │ vit_large_xf     │ 24  │1024 │ 384 │frozen │ scale axis (frozen)   │
│ vjepa_2_0_vitg           │ V-JEPA 2.0 ViT-g/1B          │ vit_giant_xf     │ 40  │1408 │ 384 │ ✅    │ version axis (2.0)    │
│ vjepa_2_0_vitg_ssv2      │ V-JEPA 2.0 ViT-g SSv2-FT     │ vit_giant_xf     │ 40  │1408 │ 384 │frozen │ supervised action bl. │
│ vjepa_1_vitL             │ V-JEPA 1 ViT-L/16 (2024)     │ vit_large (1.x)  │ 24  │1024 │ 224 │frozen │ version axis (v1)     │
│ vjepa_1_vitH             │ V-JEPA 1 ViT-H/16 (2024)     │ vit_huge (1.x)   │ 32  │1280 │ 224 │frozen │ version axis (v1)     │
│ vjepa_2_vitL_256         │ V-JEPA 2 ViT-L (fpc64-256)   │ vit_large (HF)   │ 24  │1024 │ 256 │frozen │ version/res axis (2.0)│
│ ijepa_vitH14             │ I-JEPA ViT-H/14 IN-1k        │ vit_huge (image) │ 32  │1280 │ 224 │frozen │ image baseline        │
│ ijepa_vitG16             │ I-JEPA ViT-G/16 IN-22k       │ vit_giant (image)│ —   │1408 │ 224 │frozen │ image baseline        │
│ lejepa_vitL              │ LeJEPA-L ViT-L/16 (DINOv3)   │ vit_large (timm) │ 24  │1024 │ 224 │frozen │ image baseline        │
│ dinov2 (non-JEPA)        │ DINOv2 ViT-g + registers     │ HF AutoModel     │ 40  │1536 │ 224 │frozen │ non-JEPA contrast     │
└──────────────────────────┴───────────────────────────────┴──────────────────┴─────┴─────┴─────┴───────┴───────────────────────┘
```
ckpt sources: native .pt in checkpoints/ (vjepa_2_1_*, vjepa_2_0_*, vjepa_1_*, lejepa_l.pt) · HF model_id
in $HF_HOME (ijepa_*, dinov2, *_ssv2). SKIP (wrong modality / no weights): VL-JEPA, WavJEPA/A-JEPA, V-JEPA 2-AC
(jepa-wms robotics), Point/3D/Graph/T/Brain/Stem-JEPA, mc_jepa & d_jepa (gated arXiv 2307.12698).

═══════════════════════════════════════════════════════════════════════════════
§B · Per-variant FT-arm support (what each architecture can run)
═══════════════════════════════════════════════════════════════════════════════

```text
┌──────────────────────┬────────┬─────────────┬───────────┬──────────────┬────────────┐
│ model                │ frozen │ pretrain_enc│ pretrain_hd│ surgery_enc  │ surgery_hd │
├──────────────────────┼────────┼─────────────┼───────────┼──────────────┼────────────┤
│ vjepa_2_1_vitG       │ ✅     │ ✅          │ ✅        │ ✅ (48-blk)  │ ✅         │
│ vjepa_2_1_vitg       │ ✅     │ ✅          │ ✅        │ ✅ (40-blk)  │ ✅         │
│ vjepa_2_0_vitg       │ ✅     │ ✅          │ ✅        │ ✅ (40-blk)  │ ✅         │
│ vjepa_2_1_vitL       │ ✅     │ ❌ distill  │ ❌        │ ❌           │ ❌         │
│ vjepa_2_0_vitg_ssv2  │ ✅     │ ❌ (FT ckpt)│ ❌        │ ❌           │ ❌         │
│ vjepa_1_vitL         │ ✅     │ ❌ 1.x arch │ ❌        │ ❌ 1.x arch  │ ❌         │
│ vjepa_1_vitH         │ ✅     │ ❌ 1.x arch │ ❌        │ ❌ 1.x arch  │ ❌         │
│ vjepa_2_vitL_256     │ ✅     │ ❌ HF-only  │ ❌        │ ❌           │ ❌         │
│ ijepa_vitH14         │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ ijepa_vitG16         │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ lejepa_vitL          │ ✅     │ ❌ no pred  │ ❌ no m_aux│ ❌ arch      │ ❌         │
│ dinov2 (non-JEPA)    │ ✅     │ ❌          │ ❌        │ ❌           │ ❌         │
└──────────────────────┴────────┴─────────────┴───────────┴──────────────┴────────────┘
```
Frozen-only → eval Stages 2/3/3.5/5/6 (action/motion_cos/taxonomy); SKIP Stage 8 future_mse + m12e
predictor-temporal (no trained predictor). Surgery freeze auto-scales by depth (§E).

═══════════════════════════════════════════════════════════════════════════════
§C · Integration architecture — the FOUR code seams (BUILT, git e8535c7)
═══════════════════════════════════════════════════════════════════════════════

```text
seam 1  configs/model/<variant>.yaml         backbone spec: arch / embed_dim / depth / heads /
        (clone vjepa2_0.yaml)                pred_* / crop / patch / tubelet / checkpoint_url /
                                             n_output_distillation / predict_all. CONSUMED by
                                             m09 build_student_predictor(model_cfg, data_cfg).
seam 2  scripts/run_train.sh                 BACKBONE selector (L187) → MODEL_CFG + per-backbone
        (TRAIN — parameterize backbone)      output ns outputs/<mode>/<variant>/<arm>/.
seam 3  scripts/run_eval.sh                  name→(backbone,arm) parser → encoder_ckpt_for /
        (EVAL — register + resolve ckpt)     encoder_predictor_ckpt_for / frozen_ckpt_for ;
                                             stage gates keyed on registry kind (not name prefix).
seam 4  configs/eval/probe_encoders.yaml     registry row per variant: kind / arch / crop /
        + src/utils/frozen_features.py       embed_dim (+ ckpt for kind=lejepa). load_encoder_by_kind
                                             dispatches vjepa / ijepa / dinov2 / hf_vjepa2 / lejepa.
```

═══════════════════════════════════════════════════════════════════════════════
§D · Adding one variant end-to-end (worked example)
═══════════════════════════════════════════════════════════════════════════════

```text
D1  configs/model/<bb>.yaml      arch / embed_dim / depth / num_heads / pred_* / crop / n_output_distillation
D2  scripts/run_train.sh         BACKBONE=<bb> → MODEL_CFG + outputs/<mode>/<bb>/<arm>/ (already parameterized)
D3  scripts/run_eval.sh          encoder name "<bb>_<arm>" auto-resolves via the parser (no per-variant case)
D4  configs/eval/probe_encoders  one row per <bb>_<arm>: {kind, arch, crop, embed_dim}
D5  surgery freeze               NONE — depth-agnostic int(depth*unfreeze_below) (§E)
D6  SANITY smoke                 run_train --SANITY (1 arm) + run_eval --SANITY ENCODERS=<one>
```

═══════════════════════════════════════════════════════════════════════════════
§E · Surgery freeze is depth-agnostic — NO per-backbone edit
═══════════════════════════════════════════════════════════════════════════════

`m09c1_surgery_encoder.py:1150` applies `n_trainable = int(depth * unfreeze_below)` (fractions in
configs/train/surgery_*.yaml). Auto-scales to any backbone:

```text
model            depth  unfreeze_below=0.083 →  =0.167 →   (n_trainable = int(depth * frac))
vjepa_2_1_vitG   48     3–4                     8           recipe-v3 baseline
vjepa_2_1_vitg   40     3                       6           auto
vjepa_2_1_vitL   24     1–2                      4          auto
```

═══════════════════════════════════════════════════════════════════════════════
§F · Image-JEPA / DINOv3 adapters — frozen-only loaders
═══════════════════════════════════════════════════════════════════════════════

```text
kind=ijepa   src/utils/ijepa_features.py     HF AutoModel (facebook/ijepa_*), per-frame image encode
kind=dinov2  frozen_features.load_dinov2      HF AutoModel (dinov2-with-registers-giant)
kind=lejepa  src/utils/lejepa_features.py     timm vit_large_patch16_dinov3_qkvb + eva.checkpoint_filter_fn
                                              on raw checkpoints/lejepa_l.pt (0 miss/0 unexpected)
all          forward → (B, T*n_tokens, D) → _pool_tokens; SKIP Stage 8/8b/8c (no predictor)
```

═══════════════════════════════════════════════════════════════════════════════
§G · 🦸 HERO TABLE — JEPA-variant × FT-arm × metric (cells to fill at FULL)
═══════════════════════════════════════════════════════════════════════════════

Each cell = mean ± 95% BCa CI on test split. ↑=higher-better ↓=lower-better. Headline = surgery_enc vs pretrain_enc.

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
vjepa_2_0_vitg · ×5 arms          │   ·     │     ·      │     ·      │    ·     │      ·       │   ·  version axis
─── FROZEN-only (frozen arm only) ┼─────────┼────────────┼────────────┼──────────┼──────────────┼───────
vjepa_2_1_vitL                    │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  scale axis (frozen)
vjepa_2_0_vitg_ssv2               │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  supervised action bl.
vjepa_1_vitL                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v1 version axis
vjepa_1_vitH                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v1 version axis
vjepa_2_vitL_256                  │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  v2/res axis
ijepa_vitH14                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
ijepa_vitG16                      │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
lejepa_vitL                       │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  image baseline
dinov2 (non-JEPA)                 │   ·     │     ·      │    N/A     │   N/A    │     N/A      │   ·  non-JEPA contrast
```
N/A on frozen rows = future_mse/rollout/teacher_free need a trained predictor. Reads: (1) surgery>pretrain
across backbones (vitG/vitg); (2) edge grows/shrinks with scale (vitg→vitG); (3) holds across version
(2.0_vitg / v1 / 2_vitL_256 vs 2.1); (4) frozen SSv2 / I-JEPA / LeJEPA / DINOv2 bound it.

```text
§G.2 metric catalog — HERO (6) vs APPENDIX (8)
metric          module        dir  tier        status
action top1     m12  (probe)   ↑    HERO        ✅ live
motion_cos      m12b           ↑    HERO        ✅ live
taxonomy        m12c           ↑    HERO        ✅ live
future_mse      m12d           ↓    HERO        ✅ live
rollout         m12e #1        ↓    HERO ★      ✅ built · GPU-val pending
teacher_free    m12e #4        ↓    HERO ★      ✅ built · GPU-val pending
causal          m12e #2        ↓    APPENDIX    ✅ built · GPU-val pending
tdist           m12e #3        ↓    APPENDIX    ✅ built · GPU-val pending
maskratio       m12e #5        ↓    APPENDIX    ✅ built · GPU-val pending
order           m12e #6        ↕    APPENDIX    ✅ built · GPU-val pending
TOV / VCOP      phase-2 enc    ↑    APPENDIX    ⏳ metrics§6
Arrow-of-Time   phase-2 enc    ↑    APPENDIX    ⏳ metrics§6
Pace            phase-2 enc    ↑    APPENDIX    ⏳ metrics§6
TCC             phase-2 enc    ↑    APPENDIX    ⏳ metrics§7
```
↕ = order's sign is the signal. WS-B3 DONE (predictor_eval arch-aware): vitg/2.0_vitg now get ALL predictor
cols — validated build+forward on all 3 (concat 6656/5632/1408; 2.1 deep-sup + 2.0 single-output).

═══════════════════════════════════════════════════════════════════════════════
§H · Sequencing + cost
═══════════════════════════════════════════════════════════════════════════════

```text
1. POC frozen-9 baseline eval (no train) → §G encoder rows                    runbook §1
2. WS-B3: predictor_eval arch-aware  ✅ DONE (validated build+forward, 3 backbones)
3. POC train+eval vjepa_2_1_vitg (scale) + vjepa_2_0_vitg (version)           runbook §2; pretrain_encoder first
   → now gets ALL 10 metrics (predictor stages no longer skipped for vitg/2.0_vitg)
4. Aggregate §G hero table + m13 plots across all backbones + 9 baselines
```
