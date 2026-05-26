
# iter17 — file-by-file implementation plan

## § A · One-paragraph framing

iter17 ships seven new candidate encoders against the iter15 v15a 8-stage paired-Δ matrix on `data/eval_10k_local/` (POC only). Three are FULL replicas of the iter15 recipe (V-JEPA 2.0 ViT-G 1B HF, V-JEPA 2.1 ViT-L 300M, V-JEPA 2.1 ViT-g 1B) and need the complete 8-encoder fan-out per backbone (1 model yaml + 6 train yamls + 8 registry rows). Four are FROZEN-/head-only (V-JEPA 2.0 SSv2-FT, I-JEPA ViT-H/14, I-JEPA ViT-G/16, LeJEPA-L ViT-H/14). The three I-/Le-JEPA image encoders need an extra adapter (`src/utils/encoder_loader.py`) plus a new image-encoder dispatch path in `m09a2_pretrain_head.py` + `m09c2_surgery_head.py`. A `MODEL_CFG_OVERRIDE` env var added to `scripts/run_train.sh` unlocks per-backbone trainer calls without forking the orchestrator. All work is read-only against `data/eval_10k_local/` — no FULL (115K) and no FactorPrep re-run (`m11_factor_datasets/` is reused as-is).

## § B · File-creation dependency DAG

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ DEPENDENCY DAG (top blocks bottom)                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L0  M0 audit (DONE per plan1 §10)                                           │
│        │                                                                     │
│        ▼                                                                     │
│  L1  Model yamls  (3 video + 4 image)                                        │
│      ├─ configs/model/vjepa2_0_HF.yaml          (FULL replica #1)            │
│      ├─ configs/model/vjepa2_1_vit_large.yaml   (FULL replica #2)            │
│      ├─ configs/model/vjepa2_1_vit_giant.yaml   (FULL replica #3)            │
│      ├─ configs/model/vjepa2_0_ssv2.yaml        (FROZEN only)                │
│      ├─ configs/model/ijepa_vith14.yaml         (FROZEN + head-only)         │
│      ├─ configs/model/ijepa_vitg16.yaml         (FROZEN + head-only)         │
│      └─ configs/model/lejepa_vith14.yaml        (FROZEN + head-only)         │
│        │                                                                     │
│        ▼                                                                     │
│  L2a Train yamls — V-JEPA video replicas (3 backbones × 6 yamls = 18)        │
│      For B ∈ {vjepa2_0_HF, vjepa2_1_vit_large, vjepa2_1_vit_giant}:          │
│      ├─ configs/train/pretrain_encoder_<B>.yaml                              │
│      ├─ configs/train/pretrain_head_<B>.yaml                                 │
│      ├─ configs/train/surgery_3stage_DI_encoder_<B>.yaml                     │
│      ├─ configs/train/surgery_2stage_noDI_encoder_<B>.yaml                   │
│      ├─ configs/train/surgery_3stage_DI_head_<B>.yaml                        │
│      └─ configs/train/surgery_2stage_noDI_head_<B>.yaml                      │
│                                                                              │
│  L2b Train yamls — image-encoder head-only (3 backbones × 3 yamls = 9)       │
│      For B ∈ {ijepa_vith14, ijepa_vitg16, lejepa_vith14}:                    │
│      ├─ configs/train/pretrain_head_<B>.yaml                                 │
│      ├─ configs/train/surgery_3stage_DI_head_<B>.yaml                        │
│      └─ configs/train/surgery_2stage_noDI_head_<B>.yaml                      │
│      ── Blocked by L1 AND L3 (image encoder dispatch)                        │
│                                                                              │
│  L3  Shared module + patches (image-encoder dispatch)                        │
│      ├─ src/utils/encoder_loader.py                 (NEW — adapter)          │
│      ├─ src/m09a2_pretrain_head.py                  (PATCH — dispatch)       │
│      ├─ src/m09c2_surgery_head.py                   (PATCH — dispatch)       │
│      └─ src/utils/frozen_features.py                (PATCH — eval-side)      │
│                                                                              │
│  L4  Orchestrator patches                                                    │
│      ├─ scripts/run_train.sh           (MODEL_CFG_OVERRIDE env var)          │
│      └─ scripts/run_eval.sh            (ENCODER_CKPT_OVERRIDE +              │
│                                         per-encoder ckpt-resolver rows)      │
│                                                                              │
│  L5  Registry update                                                         │
│      ├─ configs/eval/probe_encoders.yaml    (extend — 32+ new rows)          │
│      └─ configs/pipeline.yaml.encoders      (extend — match arch)            │
│                                                                              │
│  L6  M1 weight acquisition (HF Hub + torch.hub + wget pulls)                 │
│      checkpoints/iter17_ablations/{*.pt, ssv2/*.pt, …}                       │
│        │                                                                     │
│        ▼                                                                     │
│  L7  M5 SANITY smokes (Pro 4000, head-only paths only)                       │
│        │                                                                     │
│        ▼                                                                     │
│  L8  M6/M7 POC train (Pro 6000 #2 for video FULL replicas;                   │
│                       Pro 4000 for FROZEN + head-only)                       │
│        │                                                                     │
│        ▼                                                                     │
│  L9  run_eval.sh per-backbone (registry-driven encoder loop)                 │
│        │                                                                     │
│        ▼                                                                     │
│  L10 m07b_paired_delta aggregate → iter17_ablation_summary.{png,pdf}         │
└──────────────────────────────────────────────────────────────────────────────┘
```

Blocking edges of note:
- `encoder_loader.py` (L3) blocks every L2b yaml (each yaml needs `image_per_frame_adapter:` keys whose names come from the adapter).
- `MODEL_CFG_OVERRIDE` (L4) blocks all M6 invocations (without it, `MODEL_CFG="configs/model/vjepa2_1.yaml"` is hardcoded at `scripts/run_train.sh:156`).
- `m09a2/m09c2` image-encoder dispatch (L3) blocks any I-/Le-JEPA SANITY (their model yamls have `predictor: null` which would FATAL at `m09a2.py:222` `"FATAL: ckpt has no 'predictor' key"`).
- `configs/eval/probe_encoders.yaml` (L5) blocks every eval-side encoder (without a row, `frozen_features.ENCODERS[<name>]` KeyErrors → `frozen_features.py:81` FATAL).

### § B.1 — Post-Q5-Q9 DAG additions (2026-05-24)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ ADDITIONS (insert into the L-chain at the indicated levels)                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L0a  Q6 verification — torch.hub probe of ViT-L predictor depth             │
│       Blocks: vjepa2_1_vit_large.yaml authoring (L1).                        │
│       One-line Python:                                                       │
│         torch.hub.load('facebookresearch/jepa',                              │
│           'vjepa2_1_vit_large_384').predictor.depth                          │
│       Document actual value in vjepa2_1_vit_large.yaml.model.pred_depth.     │
│                                                                              │
│  L2c  Train yamls — image-encoder POOL SWEEP (Q5)                            │
│       For B ∈ {ijepa_vith14, ijepa_vitg16, lejepa_vith14}                    │
│       For P ∈ {mean, cls, max}:                                              │
│         configs/train/pretrain_head_<B>_pool<P>.yaml                         │
│         configs/train/surgery_3stage_DI_head_<B>_pool<P>.yaml                │
│         configs/train/surgery_2stage_noDI_head_<B>_pool<P>.yaml              │
│       = 3 B × 3 P × 3 positions = 27 SWEEP yamls                             │
│       (Supersedes L2b's 9 image yamls; L2b becomes the P=mean subset.)       │
│       Blocked by L3 (encoder_loader.image_temporal_pool field).              │
│                                                                              │
│  L2d  Train yamls — V-JEPA 2.0 SSv2-FT HEAD-ONLY extension (Q8)              │
│       configs/train/pretrain_head_vjepa2_0_ssv2.yaml                         │
│       configs/train/surgery_3stage_DI_head_vjepa2_0_ssv2.yaml                │
│       configs/train/surgery_2stage_noDI_head_vjepa2_0_ssv2.yaml              │
│       = 3 NEW yamls. SSv2-FT promotes FROZEN-only → FROZEN + head-only.      │
│       Blocked by L1 (vjepa2_0_ssv2.yaml model card already in § C.1).        │
│                                                                              │
│  L11  SOTA trainer modules (Q7 — IN SCOPE for iter17)                        │
│       src/m09s_safe.py     (NeurIPS 2024 slow+fast PET)                      │
│       src/m09s_seekr.py    (EMNLP   2024 replay + selective KD)              │
│       src/m09s_ssiat.py    (CVPR    2024 shared adapter)                     │
│       src/m09s_sapt.py     (ACL     2024 input-cond routing)                 │
│       Each ~500-1500 LoC; shared bookkeeping in src/utils/peft_modules.py    │
│       (NEW) for Adapter / SSF / LoRA / VPT building blocks.                  │
│                                                                              │
│  L12  SOTA train yamls (4 methods × 4 V-JEPA backbones = 16 yamls)           │
│       For M ∈ {safe, seekr, ssiat, sapt}:                                    │
│         configs/train/<M>_vjepa2_1_vit_giant_2b.yaml   (iter15 anchor)       │
│         configs/train/<M>_vjepa2_0_HF.yaml                                   │
│         configs/train/<M>_vjepa2_1_vit_large.yaml                            │
│         configs/train/<M>_vjepa2_1_vit_giant.yaml                            │
│       Blocked by L11 (each yaml's keys driven by trainer's argparse).        │
│                                                                              │
│  L13  hf_outputs.py PATCH (Q9 — --subdir iter17_ablations pass-through)      │
│       src/utils/hf_outputs.py: accept --subdir CLI arg, prefix uploaded      │
│       paths with it; default "" preserves existing iter15/16 behavior.       │
└──────────────────────────────────────────────────────────────────────────────┘
```

Blocking edges added by § B.1:
- L0a → L1 (vjepa2_1_vit_large.yaml authoring requires probed pred_depth).
- L2c supersedes L2b (9 yamls in L2b become P=mean subset of L2c's 27).
- L2d → L9 (run_eval.sh registry consumes 3 new SSv2-FT-head encoder names).
- L11 → L12 (yamls forced to match each trainer's required keys).
- L12 → L8 (M10-M13 POC train invocations blocked until trainer + yaml exist).
- L13 orthogonal — can land anytime, required for any iter17_ablations/ upload.

## § C · Per-file specs

### § C.1 — Model yamls (configs/model/*.yaml)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: configs/model/vjepa2_0_HF.yaml         (PRIMARY V-JEPA 2.0 loader, HF Apache 2.0)       │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/vjepa2_0.yaml                                                          │
│ DIFFS:                                                                                         │
│   model.version:       "2.0_HF"            ← new tag (legacy vjepa2_0.yaml stays as fbai dump) │
│   model.arch:          vit_giant_xformers  (UNCHANGED, 1408-dim 40-blk 22-heads)               │
│   model.embed_dim:     1408                                                                    │
│   model.depth:         40                                                                      │
│   model.num_heads:     22                                                                      │
│   model.mlp_ratio:     4.36                  (= 48/11)                                         │
│   model.pred_depth:    12                                                                      │
│   model.pred_embed_dim: 384                                                                    │
│   model.pred_num_heads: 12                                                                     │
│   model.num_mask_tokens: 2                                                                     │
│   model.zero_init_mask_tokens: true                                                            │
│   model.use_rope:      true                                                                    │
│   model.use_activation_checkpointing: true                                                     │
│   model.loss_exp:      1.0                                                                     │
│   model.predict_all:   false                 (2.0 = masked-only L1)                            │
│   model.weight_distance_loss: false          (2.0 = no deep supervision)                       │
│   model.n_output_distillation: 1             (2.0 = single output, required key, no .get())    │
│   model.crop_size:     384                                                                     │
│   model.patch_size:    16                                                                      │
│   model.tubelet_size:  2                                                                       │
│   model.hf_model_id:   facebook/vjepa2-vitg-fpc64-384                                          │
│   model.checkpoint_url: null               (NEW — signals HF Hub path, not wget)               │
│   model.checkpoint_path: checkpoints/iter17_ablations/vjepa2_0_HF_vitg_384.pt                  │
│   model.min_student_load_pct:  90                                                              │
│   model.min_predictor_load_pct: 50                                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/vjepa2_1_vit_large.yaml  (V-JEPA 2.1 ViT-L 300M, torch.hub)                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/vjepa2_1.yaml                                                          │
│ DIFFS:                                                                                         │
│   model.version:       "2.1_L"                                                                 │
│   model.arch:          vit_large            (deps/vjepa2/app/vjepa_2_1/.../vit_large)          │
│   model.embed_dim:     1024                                                                    │
│   model.depth:         24                                                                      │
│   model.num_heads:     16                                                                      │
│   model.mlp_ratio:     4                                                                       │
│   model.pred_depth:    12                    (Meta 2.1 default for ViT-L; verify against hub)  │
│   model.pred_embed_dim: 384                                                                    │
│   model.pred_num_heads: 12                                                                     │
│   model.num_mask_tokens: 2                                                                     │
│   model.zero_init_mask_tokens: true                                                            │
│   model.n_output_distillation: 4             (2.1: deep supervision at layers [5,11,17,23])    │
│   model.use_rope:      true                                                                    │
│   model.use_activation_checkpointing: true                                                     │
│   model.loss_exp:      1.0                                                                     │
│   model.predict_all:   true                                                                    │
│   model.weight_distance_loss: true                                                             │
│   model.crop_size:     384                                                                     │
│   model.patch_size:    16                                                                      │
│   model.tubelet_size:  2                                                                       │
│   model.hf_model_id:   null                                                                    │
│   model.torch_hub_id:  vjepa2_1_vit_large_384   (NEW key — see § C.4 loader patch)             │
│   model.checkpoint_url: null                                                                   │
│   model.checkpoint_path: checkpoints/iter17_ablations/vjepa2_1_vit_large_384.pt                │
│   model.min_student_load_pct:  90                                                              │
│   model.min_predictor_load_pct: 50                                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/vjepa2_1_vit_giant.yaml  (V-JEPA 2.1 ViT-g 1B, torch.hub)                  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/vjepa2_1.yaml                                                          │
│ DIFFS:                                                                                         │
│   model.version:       "2.1_g"                                                                 │
│   model.arch:          vit_giant_xformers   (deps/.../vjepa_2_1/.../vit_giant_xformers,        │
│                                              1408-dim 40-blk 22-heads — 2.1 variant w/         │
│                                              hierarchical output. NOT 2.0's vit_giant_xformers)│
│   model.embed_dim:     1408                                                                    │
│   model.depth:         40                                                                      │
│   model.num_heads:     22                                                                      │
│   model.mlp_ratio:     4.36                                                                    │
│   model.pred_depth:    24                                                                      │
│   model.pred_embed_dim: 384                                                                    │
│   model.pred_num_heads: 12                                                                     │
│   model.num_mask_tokens: 2                                                                     │
│   model.zero_init_mask_tokens: true                                                            │
│   model.n_output_distillation: 4                                                               │
│   model.use_rope:      true                                                                    │
│   model.use_activation_checkpointing: true                                                     │
│   model.loss_exp:      1.0                                                                     │
│   model.predict_all:   true                                                                    │
│   model.weight_distance_loss: true                                                             │
│   model.crop_size:     384                                                                     │
│   model.patch_size:    16                                                                      │
│   model.tubelet_size:  2                                                                       │
│   model.hf_model_id:   null                                                                    │
│   model.torch_hub_id:  vjepa2_1_vit_giant_384   (NEW)                                          │
│   model.checkpoint_path: checkpoints/iter17_ablations/vjepa2_1_vit_giant_384.pt                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/vjepa2_0_ssv2.yaml       (FROZEN only — supervised SSv2 FT)                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/vjepa2_0_HF.yaml (above)                                               │
│ DIFFS:                                                                                         │
│   model.version:       "2.0_HF_ssv2"                                                           │
│   model.hf_model_id:   facebook/vjepa2-vitg-fpc64-384-ssv2                                     │
│   model.checkpoint_path: checkpoints/iter17_ablations/vjepa2_0_HF_vitg_384_ssv2.pt             │
│   (everything else identical — same arch/dim/depth/predictor)                                  │
│   NOTE: NO train yamls — FROZEN-only is a registry-row + ckpt-resolver wire-up only.           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/ijepa_vith14.yaml        (IMAGE encoder, ViT-H/14 0.6B)                    │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/vjepa2_0.yaml — heavy diffs (image-only)                               │
│ DIFFS:                                                                                         │
│   model.version:       "ijepa_vith14"                                                          │
│   model.arch:          vit_huge_rope      (1280-dim, depth=32, 16 heads; deps/vjepa2/         │
│                                            src/models/vision_transformer.py:391 ALREADY EXISTS)│
│   model.embed_dim:     1280                                                                    │
│   model.depth:         32                                                                      │
│   model.num_heads:     16                                                                      │
│   model.mlp_ratio:     4                                                                       │
│   model.predictor:     null               (NEW key — image encoders ship encoder-only)         │
│   model.pred_depth:    null               (consumed by FAIL-LOUD guard in m09a2 — § C.4)       │
│   model.pred_embed_dim: null                                                                   │
│   model.pred_num_heads: null                                                                   │
│   model.num_mask_tokens: 0                                                                     │
│   model.zero_init_mask_tokens: false                                                           │
│   model.n_output_distillation: 1          (no deep supervision)                                │
│   model.use_rope:      true               (vit_huge_rope variant)                              │
│   model.use_activation_checkpointing: true                                                     │
│   model.loss_exp:      1.0                                                                     │
│   model.predict_all:   false                                                                   │
│   model.weight_distance_loss: false                                                            │
│   model.crop_size:     224                (I-JEPA pretrain is 224)                             │
│   model.patch_size:    14                                                                      │
│   model.tubelet_size:  1                  (image encoder — single-frame)                       │
│   model.modality:      image              (NEW key — dispatch hook for encoder_loader)         │
│   model.image_temporal_pool: mean         (NEW key — mean | cls | max; consumed by adapter)    │
│   model.hf_model_id:   facebook/ijepa_vith14_1k                                                │
│   model.checkpoint_path: checkpoints/iter17_ablations/ijepa_vith14_1k.pt                       │
│   model.min_student_load_pct:  90                                                              │
│   model.min_predictor_load_pct: 0         (no predictor; skip the % check)                     │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/ijepa_vitg16.yaml        (IMAGE encoder, ViT-G/16 1B)                      │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/ijepa_vith14.yaml (above)                                              │
│ DIFFS:                                                                                         │
│   model.version:       "ijepa_vitg16"                                                          │
│   model.arch:          vit_giant_xformers_rope (1408-dim, depth=40, 22-heads — same depth     │
│                                                 as V-JEPA 2.0 — deps/vjepa2/src/models/       │
│                                                 vision_transformer.py:435 ALREADY EXISTS).     │
│                                                 Verify embed_dim/heads against M0 audit; if    │
│                                                 I-JEPA-G uses 16 heads + 1408 dim, define a    │
│                                                 new constructor — see § F risk #2)             │
│   model.embed_dim:     1408                                                                    │
│   model.depth:         40                                                                      │
│   model.num_heads:     16                 (I-JEPA paper says 16; verify via M0 state-dict)     │
│   model.mlp_ratio:     4                                                                       │
│   model.crop_size:     224                                                                     │
│   model.patch_size:    16                                                                      │
│   model.hf_model_id:   facebook/ijepa_vitg16_22k                                               │
│   model.checkpoint_path: checkpoints/iter17_ablations/ijepa_vitg16_22k.pt                      │
│   (everything else inherits ijepa_vith14 image-encoder shape — predictor:null, modality:image) │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/model/lejepa_vith14.yaml       (IMAGE encoder, ViT-H/14 ~630M)                   │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/model/ijepa_vith14.yaml (above) — same arch, different ckpt + asset_loader   │
│ DIFFS:                                                                                         │
│   model.version:       "lejepa_vith14"                                                         │
│   model.arch:          vit_huge_rope (same as I-JEPA ViT-H/14)                                 │
│   model.embed_dim:     1280                                                                    │
│   model.depth:         32                                                                      │
│   model.num_heads:     16                                                                      │
│   model.hf_dataset_asset: gajeshladharai/artifacts/lejepa-l.pt   (NEW — see § C.4 loader)      │
│   model.hf_model_id:   null                                                                    │
│   model.checkpoint_path: checkpoints/iter17_ablations/lejepa_vith14.pt                         │
│   (modality, image_temporal_pool, predictor:null inherited from I-JEPA template)               │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Legend:
- `tag` = unique `model.version` string used in log lines (single-purpose; not consumed by code).
- `NEW key` = adds a yaml key that did not exist in the parent template.
- `image_temporal_pool` = sourced from § F risk #1 mitigation; pick "mean" as v17 default but log "cls"/"max" hooks in adapter.

### § C.2 — Train yamls (V-JEPA video replicas — 18 total)

All 18 V-JEPA video train yamls are mechanical clones of the iter15 trio (`pretrain_encoder.yaml`, `pretrain_head.yaml`, `surgery_*_{encoder,head}.yaml`). The ONLY changes are:

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ DIFF TABLE — per V-JEPA video backbone B                                             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ FILE                                          │ data.output_dir                       │
│ pretrain_encoder_<B>.yaml                     │ outputs/full/m09a_pretrain_encoder_<B>│
│ pretrain_head_<B>.yaml                        │ outputs/full/m09a_pretrain_head_<B>   │
│ surgery_3stage_DI_encoder_<B>.yaml            │ outputs/full/m09c_surgery_..._<B>     │
│ surgery_2stage_noDI_encoder_<B>.yaml          │ outputs/full/m09c_surgery_..._<B>     │
│ surgery_3stage_DI_head_<B>.yaml               │ outputs/full/m09c_surgery_..._<B>     │
│ surgery_2stage_noDI_head_<B>.yaml             │ outputs/full/m09c_surgery_..._<B>     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Other yaml-level changes (UNIFORM across the 6 yamls of each backbone):              │
│   data.adapted_encoder  (surgery only)  = vjepa_<B>_surgical_{3stage_DI,noDI}_{enc,h}│
│   layer_freeze.freeze_below (encoder)   = (see arithmetic table below)               │
│   surgery.stages[i].unfreeze_below      = (see arithmetic table below)               │
│   extends                              = base_optimization.yaml (pretrain_*)         │
│                                          OR surgery_base.yaml (surgery_*)            │
│                                          OR pretrain_encoder_<B>.yaml (pretrain_head)│
│                                          OR surgery_*_encoder_<B>.yaml (surgery_head)│
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Layer-freeze arithmetic (per § 4 M2 + user Q2):

```text
┌─────────────────────────┬───────┬──────────────────┬─────────────────────────────────────┐
│ Backbone                │ depth │ freeze_below     │ Surgery stage unfreeze_below        │
│                         │       │ (encoder-update) │ stage1 / stage2 / stage3            │
├─────────────────────────┼───────┼──────────────────┼─────────────────────────────────────┤
│ V-JEPA 2.1 ViT-G 2B     │ 48    │ 20 (iter15)      │ 0.083 / 0.167 / 0.167 (iter15)      │
│ (iter15 anchor)         │       │                  │ (= 4/8/8 of 48)                     │
│ V-JEPA 2.0 ViT-G 1B HF  │ 40    │ 17 (USER Q2)     │ 0.100 / 0.200 / 0.200               │
│                         │       │                  │ (= 4/8/8 of 40; matches iter15      │
│                         │       │                  │ "4/8/8 blocks" rule, not %)         │
│ V-JEPA 2.1 ViT-g 1B     │ 40    │ 17               │ 0.100 / 0.200 / 0.200               │
│ V-JEPA 2.1 ViT-L 300M   │ 24    │ 10               │ 0.167 / 0.333 / 0.333               │
│                         │       │                  │ (= 4/8/8 of 24)                     │
├─────────────────────────┼───────┼──────────────────┼─────────────────────────────────────┤
│ NOTE: fractions read by m09c1 are multiplied by len(student.blocks) at runtime.      │
│ Keep the 4/8/8-block constant (NOT the 0.083 fraction) when re-indexing — iter14    │
│ surgery_base.yaml:43-65 comment fixes block COUNT per Lee+ ICLR'23 ≤4 blocks/stage.  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Concrete example — `configs/train/pretrain_encoder_vjepa2_0_HF.yaml`:
```text
extends: pretrain_encoder.yaml
data:
  output_dir: outputs/full/m09a_pretrain_encoder_vjepa2_0_HF
layer_freeze:
  enabled: true
  freeze_below: 17    # 5/12 split (USER Q2 — 40-blk arch)
```

Concrete example — `configs/train/surgery_3stage_DI_encoder_vjepa2_1_vit_large.yaml`:
```text
extends: surgery_3stage_DI_encoder.yaml
data:
  adapted_encoder: vjepa_2_1_vit_large_surgical_3stage_DI_encoder
  output_dir: outputs/full/m09c_surgery_3stage_DI_encoder_vjepa2_1_vit_large
surgery:
  stages:
    - name: stage1_layout
      unfreeze_below: 0.167        # 4 of 24 layers
      mode_mixture: {L: 1.00, A: 0.00, I: 0.00}
      max_epochs_pct: 0.40
    - name: stage2_agent
      unfreeze_below: 0.333        # 8 of 24 layers
      mode_mixture: {L: 0.30, A: 0.70, I: 0.00}
      max_epochs_pct: 0.30
    - name: stage3_interaction
      unfreeze_below: 0.333
      mode_mixture: {L: 0.15, A: 0.15, I: 0.70}
      max_epochs_pct: 0.30
```

The 4 head yamls per backbone (`pretrain_head_<B>.yaml`, `surgery_3stage_DI_head_<B>.yaml`, `surgery_2stage_noDI_head_<B>.yaml`) extend their `_encoder_<B>.yaml` siblings (mirroring iter15) and ONLY override `data.module`, `data.adapted_encoder`, `data.output_dir`, `loss.weight_jepa=0`, `optimization.spd.enabled=false`, and `surgery.stages` (collapses to single `unfreeze_below=0.0` stage with the inherited mixture).

### § C.3 — Train yamls (image encoder head-only — 9 total)

Three yamls per image backbone (`ijepa_vith14`, `ijepa_vitg16`, `lejepa_vith14`). Image encoders have NO predictor and CANNOT run encoder-update or surgery-with-JEPA-loss → only the 3 head yamls are produced.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: configs/train/pretrain_head_<B>.yaml          (B ∈ image encoders)               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/train/pretrain_head.yaml                                             │
│ DIFFS:                                                                                 │
│   extends:                base_optimization.yaml    (NOT pretrain_encoder.yaml — that  │
│                                                      assumes the predictor exists)     │
│   data.module:            m09a2                                                        │
│   data.adapted_encoder:   <B>_pretrain_head                                            │
│   data.output_dir:        outputs/full/m09a_pretrain_head_<B>                          │
│   data.num_frames:        16                  (still 16 — adapter loops per-frame)     │
│   image_encoder:                              (NEW yaml block — read by m09a2 dispatch)│
│     enabled:              true                                                         │
│     temporal_pool:        mean                                                         │
│     per_frame_crop:       <model.crop_size>   (224 for I-JEPA / LeJEPA)                │
│     reuse_imagenet_norm:  true                                                         │
│   layer_freeze:                                                                        │
│     enabled:              true                                                         │
│     freeze_below:         <model.depth>       (full freeze; no encoder backward path)  │
│   drift_control.enabled:  false                                                        │
│   loss.weight_jepa:       0.0                 (no predictor; trivially zero anyway)    │
│   loss.weight_motion_aux: 1.0                                                          │
│   motion_aux.head.hidden_dim: 256                                                      │
│   motion_aux.head.dropout:    0.1                                                      │
│   motion_aux.head.embed_dim_in: <model.embed_dim>  (NEW key — head must size to image  │
│                                                     encoder D; iter15 head sized to    │
│                                                     V-JEPA 1664)                       │
│ Optimization keys inherited from base_optimization.yaml (lr 5e-4 head-only).           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/train/surgery_3stage_DI_head_<B>.yaml                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/train/surgery_3stage_DI_head.yaml                                    │
│ DIFFS:                                                                                 │
│   extends:                pretrain_head_<B>.yaml (NOT surgery_3stage_DI_encoder_<B> —  │
│                                                   that yaml DOES NOT EXIST for image   │
│                                                   encoders, by design)                 │
│   data.module:            m09c2                                                        │
│   data.adapted_encoder:   <B>_surgical_3stage_DI_head                                  │
│   data.output_dir:        outputs/full/m09c_surgery_3stage_DI_head_<B>                 │
│   interaction_mining.enabled: true                                                     │
│   surgery.warmup_mode:    single                                                       │
│   surgery.lp_ft_stage0.enabled: false                                                  │
│   surgery.stages:                                                                      │
│     - name: stage0_head_only_DI                                                        │
│       unfreeze_below: 0.0                                                              │
│       mode_mixture: {L: 0.15, A: 0.15, I: 0.70}                                        │
│       max_epochs_pct: 1.0                                                              │
│   optimization.spd.enabled: false                                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/train/surgery_2stage_noDI_head_<B>.yaml                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Same as above but mode_mixture: {L: 0.50, A: 0.50, I: 0.00}, interaction_mining: false.│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### § C.4 — NEW shared module + PATCHes (src/utils/encoder_loader.py + dispatch)

```text
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: src/utils/encoder_loader.py    (NEW — image-encoder adapter)                        │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ Module purpose: dispatch any new-iter17 ckpt onto a (encoder, embed_dim, forward_fn) tuple│
│ that m09a2 / m09c2 / probe_action / probe_motion_cos / probe_future_regress can consume   │
│ identically. Replaces ad-hoc loader code in m09a2/m09c2 build_model.                      │
│                                                                                           │
│ PUBLIC API:                                                                               │
│   def load_video_encoder(model_cfg: dict, device, num_frames: int) -> dict:               │
│       """V-JEPA 2.0 / 2.1 loader. Dispatches on model_cfg['arch'] +                       │
│       {checkpoint_url | hf_model_id | torch_hub_id}. Returns                              │
│       {'encoder': nn.Module, 'predictor': nn.Module|None, 'embed_dim': int,               │
│        'crop': int, 'init_ckpt_path': str, 'forward': Callable}."""                       │
│                                                                                           │
│   def load_image_encoder_per_frame(model_cfg: dict, train_cfg: dict, device,             │
│                                     num_frames: int) -> dict:                             │
│       """I-JEPA / LeJEPA loader. AUTHORITATIVE 4-arg signature (train_cfg supplies        │
│       the pool — see § C.6). Returns same dict shape; forward closure:                    │
│            x: (B, T, 3, H, W) → reshape (B*T, 3, H, W) → image_encoder(.)                 │
│            → spatial pool over tokens per train_cfg['image_encoder']['temporal_pool']     │
│            (mean | cls | max) → reshape (B, T, D) → mean over T → (B, D).                 │
│       FAIL LOUD if train_cfg lacks image_encoder.temporal_pool (no silent 'mean').       │
│       Logs per-frame FLOPs + warns if T*B exceeds 256 (memory hit)."""                    │
│                                                                                           │
│   def fetch_ckpt(model_cfg: dict) -> Path:                                                │
│       """Resolves ckpt onto disk via:                                                     │
│          1. model_cfg['checkpoint_path'] if file exists.                                  │
│          2. model_cfg['hf_model_id'] → huggingface_hub.snapshot_download (model repo)     │
│          3. model_cfg['hf_dataset_asset'] → huggingface_hub.hf_hub_download(repo_type=    │
│             'dataset')  ← LeJEPA-L path                                                   │
│          4. model_cfg['torch_hub_id'] → torch.hub.load(repo='facebookresearch/jepa', …)   │
│          5. model_cfg['checkpoint_url'] → torch.hub.load_state_dict_from_url             │
│       FAIL LOUD if none of 2-5 are set AND checkpoint_path doesn't exist."""              │
│                                                                                           │
│   def assert_embed_dim_match(ckpt_state_dict: dict, model_cfg: dict):                     │
│       """FAIL LOUD: inspect <first-norm>.weight shape; if shape[0] != embed_dim, FATAL    │
│       with both values. Catches the 'V-JEPA 2.0 (1408) loaded into V-JEPA 2.1 yaml (1664)'│
│       silent failure that would otherwise corrupt training."""                            │
│                                                                                           │
│   def assert_no_predictor_when_image(model_cfg: dict):                                    │
│       """FAIL LOUD: if model_cfg['modality'] == 'image' and model_cfg.get('predictor') !=│
│       None, FATAL. Forces image-encoder yamls to declare predictor: null explicitly."""   │
│                                                                                           │
│ INTERNALS:                                                                                │
│   _ENC_BUILDER_DISPATCH: dict mapping arch name → constructor closure                     │
│       'vit_giant_xformers'         → get_vit_by_arch (V-JEPA 2.0 shim)                    │
│       'vit_gigantic_xformers'      → get_vit_by_arch (V-JEPA 2.1 shim)                    │
│       'vit_large'                  → V-JEPA 2.1 vit_large (deps/.../app/.../vit_large)    │
│       'vit_huge_rope'              → deps/vjepa2/src/models/vision_transformer.py:391     │
│       'vit_giant_xformers_rope'    → deps/.../vision_transformer.py:435                   │
│                                                                                           │
│ CLI surface: NONE — pure-library. Imported by m09a2/m09c2/probe_* via utils.* path.       │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09a2_pretrain_head.py    (PATCH — image-encoder dispatch in build_model)       │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFF (against lines 126-250):                                                             │
│   1. At top of build_model (after model_cfg = cfg["model"]):                              │
│      from utils.encoder_loader import (load_video_encoder,                                │
│          load_image_encoder_per_frame, assert_no_predictor_when_image,                    │
│          assert_embed_dim_match)                                                          │
│      modality = model_cfg.get("modality", "video")  ← acceptable .get() ONLY because new │
│                                                       yaml key is opt-in; existing yamls │
│                                                       have no key → "video" default. All │
│                                                       new yamls MUST set explicitly.     │
│   2. Branch:                                                                              │
│        if modality == "image":                                                            │
│            assert_no_predictor_when_image(model_cfg)                                      │
│            built = load_image_encoder_per_frame(model_cfg, train_cfg, device,            │
│                                                 num_frames)  # 4-arg per § C.4/§ C.6     │
│            student   = built["encoder"]                                                   │
│            predictor = None             ← skip predictor build entirely                   │
│            init_ckpt = built["init_ckpt_path"]                                            │
│            embed_dim = built["embed_dim"]                                                 │
│        else:                                                                              │
│            (existing V-JEPA build path — unchanged)                                       │
│   3. Wrap the existing "if 'predictor' not in ckpt: FATAL" block in `if predictor is not │
│      None:` so the image branch doesn't trip it.                                          │
│   4. In the training loop, when computing JEPA L1: skip predictor.forward if predictor   │
│      is None (image branch never gets JEPA gradient — motion_aux head is the only loss). │
│      Already aligned with weight_jepa: 0.0 in the image yaml.                             │
│   5. motion_aux head is built with embed_dim_in = (image branch ? model_cfg['embed_dim'] │
│      : 1664 inherited from yaml). Currently motion_aux head reads embed_dim from cfg —    │
│      add a one-line override using cfg['motion_aux']['head'].get('embed_dim_in', None)    │
│      to swap to image dim WHEN set.                                                       │
│                                                                                           │
│ CLI surface: NO new args. The new behavior is yaml-driven via model_cfg['modality'].      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09c2_surgery_head.py    (PATCH — symmetric dispatch with m09a2)                │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ Same DIFF shape as m09a2 above. Additional guards:                                        │
│   - If modality == "image" AND cfg['init_from_ckpt'].startswith('hf://'):                 │
│       FATAL "image encoders init from the model.checkpoint_path or hf_model_id —          │
│       --init-from-ckpt is reserved for V-JEPA prior-run student+predictor schema."        │
│     (Surgery-on-image is initialised from the model's pretrained ckpt directly, NOT       │
│     from a prior m09a/c run that doesn't exist for the image branch.)                     │
│   - Skip the schema check on lines 232-241 that requires "student"+"predictor" keys.      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/utils/frozen_features.py    (PATCH — eval-side dispatch)                        │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFF (against lines 137-242):                                                             │
│   1. Add load_image_encoder_frozen(model_cfg, registry_row, num_frames):                  │
│        # eval side has no train yaml; synthesize the train_cfg the 4-arg loader needs    │
│        # from the registry row's image_temporal_pool field (Q5 — frozen runs once per    │
│        # pool, so registry_row carries the pool for THIS extraction).                     │
│        pool_cfg = {"image_encoder": {"temporal_pool": registry_row["image_temporal_pool"]}}│
│        encoder = encoder_loader.load_image_encoder_per_frame(model_cfg, pool_cfg, "cuda", │
│                                                              num_frames)["encoder"]       │
│        encoder = encoder.bfloat16().eval()                                                │
│        return encoder, model_cfg["crop_size"], model_cfg["embed_dim"]                     │
│   2. Add forward_image(model, batch, num_frames):                                         │
│        per-frame forward + token-mean + temporal_pool → (B, n_out=1, D).                  │
│   3. Extend ENCODERS dispatch (encoder_kind switch) in extract_features_for_keys to       │
│      route 'image' kind → forward_image.                                                  │
│   4. Acceptable .get('modality', 'video') ONLY at the dispatch site (single line).        │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: scripts/run_train.sh    (PATCH — MODEL_CFG_OVERRIDE)                                │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFF (against line 156):                                                                  │
│   - MODEL_CFG="configs/model/vjepa2_1.yaml"                                               │
│   + MODEL_CFG="${MODEL_CFG_OVERRIDE:-configs/model/vjepa2_1.yaml}"                        │
│   + [ -f "$MODEL_CFG" ] || { echo "❌ FATAL: MODEL_CFG=$MODEL_CFG missing"; exit 3; }     │
│   + echo "  model:     $MODEL_CFG  (override via MODEL_CFG_OVERRIDE env var)"             │
│                                                                                           │
│ Also patch SUBCMD validation (line 54) — accept the same subcmd names; the override is    │
│ orthogonal. Caller composes via:                                                          │
│   MODEL_CFG_OVERRIDE=configs/model/vjepa2_0_HF.yaml \                                     │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --SANITY                │
│                                                                                           │
│ ALSO patch TRAIN_CFG resolver (lines 232, 283, 287, 432-475): currently hardcodes one     │
│ train yaml per subcmd. Add a per-backbone TRAIN_CFG_OVERRIDE env var so the same          │
│ MODEL_CFG_OVERRIDE invocation can pick up the matching per-backbone train yaml:           │
│   TRAIN_CFG="${TRAIN_CFG_OVERRIDE:-configs/train/pretrain_encoder.yaml}"                  │
│ (Repeat for the 7 case branches that set TRAIN_CFG.)                                      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: scripts/run_eval.sh    (PATCH — extend the 3 ckpt resolvers + ENCODER_CKPT_OVERRIDE)│
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFFS:                                                                                    │
│   1. At line 130 (ENCODER_CKPT default), keep the V-JEPA 2.1 default but add a per-       │
│      encoder map keyed by encoder NAME prefix (not a single env var):                     │
│        case "$ENC" in                                                                     │
│          vjepa_2_0_HF*)             enc_ckpt="checkpoints/iter17_ablations/vjepa2_0_HF_…"│
│          vjepa_2_0_HF_ssv2*)        enc_ckpt="checkpoints/iter17_ablations/…ssv2.pt"     │
│          vjepa_2_1_vit_large*)      enc_ckpt="checkpoints/iter17_ablations/…vit_large.pt"│
│          vjepa_2_1_vit_giant*)      enc_ckpt="checkpoints/iter17_ablations/…vit_giant.pt"│
│          ijepa_vith14*)             enc_ckpt="checkpoints/iter17_ablations/ijepa_vith14"│
│          ijepa_vitg16*)             enc_ckpt="checkpoints/iter17_ablations/ijepa_vitg16"│
│          lejepa_vith14*)            enc_ckpt="checkpoints/iter17_ablations/lejepa_vith14"│
│          vjepa_2_1*|*)              enc_ckpt="$ENCODER_CKPT"                              │
│   2. Extend encoder_ckpt_for() / encoder_predictor_ckpt_for() / motion_aux_head_for()    │
│      with cases for every new registry name. ~25 new lines per resolver.                  │
│   3. Image-encoder predictor pre-flight: extend Stage 8 SKIP list to auto-drop encoder   │
│      names matching ijepa_*/lejepa_* (mirrors current `[[ "$ENC" == vjepa* ]]` guard).    │
│   4. Stage 1 frozen-only flag: add --frozen-only mode arg that drops Stage 3/3.5 head   │
│      training when the variant has no head ckpt to load.                                  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

### § C.5 — Registry yaml (configs/eval/probe_encoders.yaml) — extend by ~32 rows

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ NEW REGISTRY ROWS (extend top-level `encoders:` block — append after current line 76)   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Per V-JEPA video FULL backbone B ∈ {vjepa_2_0_HF, vjepa_2_1_vit_large, vjepa_2_1_vit_g}:│
│   <B>_frozen                          {kind: vjepa,  arch: <model.arch>,                │
│                                        crop: <model.crop_size>,                         │
│                                        embed_dim: <model.embed_dim>}                    │
│   <B>_pretrain_encoder                (same)                                            │
│   <B>_pretrain_2X_encoder             (same)                                            │
│   <B>_pretrain_head                   (same)                                            │
│   <B>_surgical_3stage_DI_encoder      (same)                                            │
│   <B>_surgical_noDI_encoder           (same)                                            │
│   <B>_surgical_3stage_DI_head         (same)                                            │
│   <B>_surgical_noDI_head              (same)                                            │
│   = 8 rows × 3 backbones = 24 rows                                                      │
│                                                                                         │
│ Per FROZEN-only V-JEPA video backbone:                                                  │
│   vjepa_2_0_HF_ssv2_frozen            (1 row)                                           │
│                                                                                         │
│ Per IMAGE encoder backbone B ∈ {ijepa_vith14, ijepa_vitg16, lejepa_vith14}:             │
│   <B>_frozen                          {kind: image, arch: <model.arch>,                 │
│                                        crop: <model.crop_size>,                         │
│                                        embed_dim: <model.embed_dim>,                    │
│                                        modality: image,                                 │
│                                        image_temporal_pool: mean,                       │
│                                        hf_model_id: <model.hf_model_id>}                │
│   <B>_pretrain_head                   (same)                                            │
│   <B>_surgical_3stage_DI_head         (same)                                            │
│   <B>_surgical_noDI_head              (same)                                            │
│   = 4 rows × 3 backbones = 12 rows                                                      │
│                                                                                         │
│ Total: 24 + 1 + 12 = 37 new rows.                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

`configs/pipeline.yaml.encoders` extension (mirrors the same 37 names, but only `model_id` + `dim` + `type` + `suffix` keys per the existing schema lines 234-294). The two registries are intentional redundancy — `pipeline.yaml.encoders` is the legacy m05/m05b path, `probe_encoders.yaml` is the iter13+ probe path. Both must list the new names or `frozen_features.py:81` / m05's dispatch will FATAL.

### § C.6 — Image-encoder POOL SWEEP (Q5) — 27 yamls + adapter contract

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ MECHANISM — three pools share one model.yaml; differ only in train.yaml image_encoder   │
│ block. encoder_loader.load_image_encoder_per_frame reads the pool from train_cfg, not   │
│ model_cfg, so each pool variant ships its OWN train yaml + own output_dir + own         │
│ registry row, but reuses one model yaml + one .pt ckpt.                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE PATTERN: configs/train/{pretrain_head, surgery_3stage_DI_head,                     │
│                 surgery_2stage_noDI_head}_<B>_pool<P>.yaml                              │
│   <B> ∈ {ijepa_vith14, ijepa_vitg16, lejepa_vith14}                                     │
│   <P> ∈ {mean, cls, max}                                                                │
│   = 3 positions × 3 backbones × 3 pools = 27 yamls                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/train/pretrain_head_<B>.yaml from § C.3 (P=mean variant becomes      │
│           the canonical parent; cls/max variants extend it).                            │
│ DIFFS per pool P:                                                                       │
│   image_encoder.temporal_pool: <P>          (mean | cls | max)                          │
│   data.adapted_encoder:        <B>_<position>_pool<P>                                   │
│   data.output_dir:             outputs/full/m09a_pretrain_head_<B>_pool<P>              │
│   (for surgery_*_head variants: same diff + parent surgery yaml + pool tag)             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ ADAPTER CONTRACT (extends § C.4 encoder_loader.py):                                     │
│   load_image_encoder_per_frame(model_cfg, train_cfg, device, num_frames) — adds        │
│   train_cfg arg so the closure can read train_cfg['image_encoder']['temporal_pool'].   │
│   Default (when key missing) FATALs — no silent fallback to 'mean'. Mirrors             │
│   m09c2:232-241 fail-loud pattern.                                                      │
│                                                                                          │
│   FORWARD CLOSURE:                                                                       │
│     per_frame_tokens = encoder(x.flatten(0,1))    # (B*T, N, D)                         │
│     pool ∈ {mean, max} → reduce over N axis → (B*T, D)                                  │
│     pool == 'cls'      → take pool_idx=0          → (B*T, D)  [if encoder has CLS;     │
│                                                                otherwise FATAL]         │
│     temporal_pool = mean(axis=1)  # always mean over T (the "spatial pool" is what we   │
│                                     vary; temporal stays mean — design choice; if user  │
│                                     wants temporal sweep too, add a `temporal_reduce`   │
│                                     yaml key in a future iter)                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ REGISTRY (extends § C.5): 27 new rows in BOTH probe_encoders.yaml + pipeline.yaml.      │
│ Naming: <B>_pretrain_head_pool<P>, <B>_surgical_3stage_DI_head_pool<P>, etc.            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

CLS-pool gotcha: I-JEPA does NOT use a CLS token by default; the encoder outputs N patch tokens. `image_temporal_pool: cls` requires the encoder to be configured with `use_cls_token=true` OR the adapter falls back to position 0 of the patch sequence (which is the top-left patch — meaningless globally). FAIL LOUD per § F if `cls` is requested for an encoder without a CLS token; document the workaround (mean-pool first row of attention maps).

### § C.7 — V-JEPA 2.0 SSv2-FT head-only extension (Q8) — 3 yamls

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: configs/train/pretrain_head_vjepa2_0_ssv2.yaml                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/train/pretrain_head_vjepa2_0_HF.yaml (from § C.2)                     │
│ DIFFS:                                                                                  │
│   extends:                  pretrain_head_vjepa2_0_HF.yaml                              │
│   model_cfg_path:           configs/model/vjepa2_0_ssv2.yaml  (NOT _HF.yaml)            │
│   data.adapted_encoder:     vjepa_2_0_HF_ssv2_pretrain_head                             │
│   data.output_dir:          outputs/full/m09a_pretrain_head_vjepa2_0_ssv2               │
│   init_from_ckpt:           <model.checkpoint_path> from vjepa2_0_ssv2.yaml             │
│                              (NOT a prior m09a run — supervised SSv2 init is the       │
│                              "Meta init" equivalent for this branch)                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/train/surgery_3stage_DI_head_vjepa2_0_ssv2.yaml                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: configs/train/surgery_3stage_DI_head_vjepa2_0_HF.yaml                         │
│ DIFFS: same model_cfg_path swap + adapted_encoder + output_dir as above.                │
│ SURGERY_INIT:               outputs/full/m09a_pretrain_head_vjepa2_0_ssv2/m09a_ckpt_    │
│                             best.pt    (the m09a SSv2 head-only pretrain ckpt)         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: configs/train/surgery_2stage_noDI_head_vjepa2_0_ssv2.yaml                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Same diffs as above; mixture {L:.5, A:.5, I:.0}, interaction_mining: false.             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ REGISTRY (extends § C.5): 3 NEW rows in both probe_encoders.yaml + pipeline.yaml:       │
│   vjepa_2_0_HF_ssv2_pretrain_head                                                       │
│   vjepa_2_0_HF_ssv2_surgical_3stage_DI_head                                             │
│   vjepa_2_0_HF_ssv2_surgical_noDI_head                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### § C.8 — SOTA continual-FT trainers (Q7) — 4 src modules + 16 yamls + 1 PEFT lib

✅ Q10 RESOLVED 2026-05-24 = Option A (sessions = surgery stages). User-verified
against iter/utils/teams_work/FactorJEPA-Alternatives_to_Vanilla_Continual_
Finetuning.md: the 4 methods are multi-session, backbone-frozen, PET-based →
they map onto the SURGERY family, NOT the vanilla pretrain family.

⚠️ BASE TRAINER (all 4): src/m09c1_surgery_encoder.py + src/m09c2_surgery_head.py
— NOT m09a1/m09a2. VERIFIED 2026-05-24:
  • multi-session/stage loop (`for stage_idx, stage_cfg in enumerate(stages)`,
    m09c1:1158) + per-stage build_optimizer (1219) + per-stage probe (962)
    + raw-replay buffer (SEEKR needs this) exist ONLY in m09c1.
  • frozen-backbone discipline (set_trainable_prefix(0) + assert_encoder_frozen,
    m09c2:271-276) is the PET "freeze the encoder, train a small module" pattern
    the doc mandates (§10.2.2 / §10.4.2 / §10.4.8).
  • m09a1 is explicitly single-stage ("no stage gate", m09a1:1223); m09a2 has
    zero stage machinery → neither can host a multi-session method.
CONSTRUCTION: clone m09c1's stage loop + replay; borrow m09c2's frozen-backbone
guards; REPLACE m09c1's progressive-unfreeze (set_trainable_prefix(n>0)) with
PET adapters (wrap_vit_with_peft + freeze_backbone_keep_peft). session_i ==
surgery stage_i; the SOTA anti-drift mechanism replaces progressive-unfreeze as
the per-stage-boundary forgetting control.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: src/utils/peft_modules.py    (NEW — shared PEFT building blocks)                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Module purpose: one place where Adapter / SSF / LoRA / VPT primitives live. All 4 SOTA  │
│ trainers import from here. Avoids 4 copies of the same Adapter code.                    │
│                                                                                         │
│ PUBLIC API:                                                                             │
│   class Adapter(nn.Module):          # bottleneck MLP — Pfeiffer 2020 style             │
│       __init__(self, dim, bottleneck=64, activation='gelu', init_scale=1e-3): ...       │
│       forward(x): return x + self.up(self.act(self.down(x))) * self.scale               │
│                                                                                         │
│   class SSF(nn.Module):              # scale + shift — Lian 2022                        │
│       __init__(self, dim, init_scale=1.0, init_shift=0.0): ...                          │
│                                                                                         │
│   class LoRA(nn.Module):             # rank-r decomp — Hu 2021                          │
│       __init__(self, in_dim, out_dim, r=8, alpha=16): ...                               │
│                                                                                         │
│   class VPT(nn.Module):              # prompt tokens — Jia 2022                         │
│       __init__(self, dim, num_prompts=10, deep=False): ...                              │
│                                                                                         │
│   def wrap_vit_with_peft(vit, mode, **kwargs):                                          │
│       """mode ∈ {adapter, ssf, lora, vpt}; injects into every ViT block.                │
│       Returns wrapped ViT + the PEFT param-group iterator for the optimizer."""         │
│                                                                                         │
│   def freeze_backbone_keep_peft(model):                                                 │
│       """Set requires_grad=False on all non-PEFT params. FAIL LOUD if no PEFT params    │
│       found (catches "forgot to wrap" bug)."""                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09s_safe.py    (NEW — SAFE NeurIPS 2024 slow+fast PET trainer)               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: src/m09c1_surgery_encoder.py (stage loop + replay) + m09c2 frozen-backbone    │
│   guards. NOT m09a1.                                                                    │
│ KEY DIFFS vs m09c1:                                                                     │
│   - Replace progressive-unfreeze (set_trainable_prefix(n>0)) with two PEFT branches:    │
│     slow (frozen after surgery stage 1) + fast (re-init each later stage). Base         │
│     backbone FROZEN throughout (freeze_backbone_keep_peft) — m09c2 discipline.          │
│   - Both branches wrap_vit_with_peft(student, mode=cfg['peft_mode'])                   │
│   - Loss: keep m09c1's JEPA L1 + motion_aux; ADD align λ*||slow_feat−fast_feat||^2      │
│   - Sessions = m09c1's existing cfg["surgery"]["stages"] (Q10 Option A). slow freezes   │
│     after stages[0]; fast re-inits at each later stages[i].                             │
│   - Output: student_encoder.pt (slow+fast PEFT params, base frozen) +                   │
│             m09s_ckpt_best.pt (carries predictor — reuse m09c1 save path)               │
│ CLI: m09c1's args (--model-config --train-config --subset --local-data --factor-dir     │
│      --init-from-ckpt ...) + --peft-mode {adapter|ssf|lora|vpt} --align-lambda          │
│      --sessions (defaults to surgery.stages per Option A) --no-wandb                     │
│ Acceptance: SANITY 10 clips × ≥2 stages exits 0 + non-zero student_encoder.pt +         │
│   FL-10 (≥2 sessions) + FL-9 (PET params > 0) pass.                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09s_seekr.py    (NEW — SEEKR EMNLP 2024 replay + selective KD)               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: src/m09c1_surgery_encoder.py — m09c1 ALREADY has the raw-replay 50% buffer    │
│   SEEKR needs (cfg["replay"]["raw_pretrain_pct"]); reuse it directly. NOT m09a1.        │
│ KEY DIFFS vs m09c1:                                                                     │
│   - Backbone frozen + PET (m09c2 discipline + wrap_vit_with_peft); keep m09c1's replay. │
│   - Selective KD on top-K retention-critical units. K from session-0 feature            │
│     attribution: per-unit Fisher info on probe_action loss.                              │
│   - Loss: m09c1 JEPA L1 + replay term + κ * KD(top-K-units || frozen prev-stage teacher)│
│   - Sessions = surgery stages (Q10 Option A); frozen teacher = end-of-prev-stage ckpt.  │
│ CLI: same as m09s_safe + --replay-frac --kd-kappa --topk-units                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09s_ssiat.py    (NEW — SSIAT CVPR 2024 shared adapter)                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: src/m09c1_surgery_encoder.py (stage loop) + m09c2 frozen backbone. NOT m09a1. │
│ KEY DIFFS vs m09c1:                                                                     │
│   - 1 adapter per ViT block, reused across all surgery stages (NOT re-init per stage).  │
│   - Backbone fully frozen (m09c2 discipline) — replaces m09c1's progressive-unfreeze.   │
│   - Gradient routing: adapter updates restricted to low-rank subspace defined by SVD   │
│     of stage-0 adapter weights (the "session-coordinated learn-and-select").            │
│   - Sessions = surgery stages (Q10 Option A).                                           │
│ CLI: same as m09s_safe + --adapter-bottleneck --subspace-rank                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: src/m09s_sapt.py    (NEW — SAPT ACL 2024 input-cond PET routing)                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ TEMPLATE: src/m09s_ssiat.py (extends the shared-adapter framework → inherits the m09c1  │
│   stage loop + m09c2 frozen backbone transitively). NOT m09a1.                          │
│ KEY DIFFS vs m09s_ssiat:                                                                │
│   - K adapters (instead of 1), each with router α_k(x) ∈ R^K.                           │
│   - Router is a tiny MLP attending over an input embedding (mean-pool of x's patches).  │
│   - Output: sum_k α_k(x) * Δ_k(x)                                                       │
│   - Sessions = surgery stages (Q10 Option A), inherited from m09s_ssiat.                │
│ CLI: same as m09s_ssiat + --num-adapters --router-hidden                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILES: configs/train/<M>_<B>.yaml × 16   (M × 4 V-JEPA backbones)                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ M ∈ {safe, seekr, ssiat, sapt}                                                          │
│ B ∈ {vjepa2_1_vit_giant_2b (iter15 anchor), vjepa2_0_HF, vjepa2_1_vit_large,            │
│      vjepa2_1_vit_giant}                                                                 │
│ TEMPLATE: configs/train/base_optimization.yaml + per-method peft block                  │
│ COMMON keys: peft_mode, peft_bottleneck, replay_frac (SEEKR only), align_lambda (SAFE   │
│   only), subspace_rank (SSIAT/SAPT), num_adapters (SAPT only).                          │
│ Per-backbone-specific keys: layer_freeze.freeze_below (matches § C.2 arithmetic).       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ REGISTRY (extends § C.5): 16 NEW rows in BOTH probe_encoders.yaml + pipeline.yaml:      │
│   <B>_<M>     for each (B, M) pair    (kind: vjepa, has_predictor: true, peft_mode:…)  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ FILE: scripts/run_train.sh    (PATCH — add 4 SOTA subcommands)                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFF: extend the subcmd case (line 54) + dispatch case (line 219) with:                 │
│   safe|seekr|ssiat|sapt) train via src/m09s_<SUBCMD>.py                                 │
│ Plus the TRAIN_CFG_OVERRIDE / MODEL_CFG_OVERRIDE plumbing already specified in § C.4.   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### § C.9 — hf_outputs.py --subdir patch (Q9)

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ FILE: src/utils/hf_outputs.py    (PATCH — accept --subdir for iter17 separation)        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ DIFF:                                                                                   │
│   1. Add argparse arg: --subdir TYPE=str DEFAULT="" HELP="HF repo sub-directory prefix" │
│   2. Before each hf_hub.upload_*(path_in_repo=X) call, replace X with                  │
│      (args.subdir + "/" + X) if args.subdir else X.                                     │
│   3. Document in --help: "iter17 callers must pass --subdir iter17_ablations to land   │
│      in the right shelf of anonymousML123/factorjepa-outputs."                          │
│ CLI example:                                                                            │
│   HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/poc \    │
│      --subdir iter17_ablations 2>&1 | tee logs/upload_iter17.log                        │
│ Acceptance: dry-run shows path_in_repo prefixed with "iter17_ablations/" for every file.│
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### § C.10 — SOTA session-mapping (BLOCKS M10-M13 — author + user-approve FIRST)

The four SOTA methods are MULTI-SESSION continual-learning algorithms. Their
defining mechanisms only have meaning across a SEQUENCE of training sessions:

```text
┌──────────┬───────────────────────────────────────────────────────────────────────────┐
│ Method    │ The mechanism that needs ≥2 sessions to mean anything                      │
├──────────┼───────────────────────────────────────────────────────────────────────────┤
│ SAFE      │ "slow branch frozen AFTER session 1" — needs a session 1 to freeze after  │
│ SEEKR     │ "replay buffer from prior sessions" + "KD against prior-session teacher"   │
│ SSIAT     │ "adapter REUSED across sessions" — one session = no reuse to measure       │
│ SAPT      │ "router learns per-session task identity" — one task = trivial router      │
└──────────┴───────────────────────────────────────────────────────────────────────────┘
```

iter17 is single-corpus (`data/eval_10k_local/`). Running these methods on ONE
session produces numbers, but they do NOT measure forgetting/transfer — the
exact axis the methods exist to measure. We MUST define a session sequence.

✅ Q10 RESOLVED 2026-05-24 = Option A (surgery stages = sessions). User verified
the 4 methods against the teams_work doc and confirmed they build on the SURGERY
family (m09c1/m09c2), whose multi-stage loop is the only session axis available —
which forces Option A. Options B/C are retained below for the record only.

```text
┌─────┬──────────────────────────────┬────────────────────────────────────────┬──────────────────┐
│ Opt │ Session sequence              │ What "forgetting" then means            │ Cost / fidelity  │
├─────┼──────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ A ✅│ Surgery STAGES as sessions:   │ Does unlocking deeper blocks (stage 3) │ FREE — reuses    │
│     │ session_1 = stage1_layout,    │ make the model forget the layout/agent │ existing surgery │
│     │ session_2 = stage2_agent,     │ skill learned in stage 1/2? This is    │ stage structure. │
│     │ session_3 = stage3_interaction│ the MOST defensible — our surgery IS a │ HIGH fidelity to │
│     │                              │ staged curriculum already.              │ method intent.    │
│     │                              │ SOTA replaces our progressive-unfreeze │                  │
│     │                              │ with SAFE/SEEKR/etc as the anti-drift  │                  │
│     │                              │ mechanism at each stage boundary.       │                  │
├─────┼──────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ B   │ FACTOR mixtures as sessions:  │ Does learning D_I (interaction) make   │ LOW cost — D_L/  │
│     │ session_1 = D_L only,         │ the model forget D_L (layout)? Maps to │ D_A/D_I already  │
│     │ session_2 = D_A only,         │ the factor-disentanglement story       │ on disk. MEDIUM  │
│     │ session_3 = D_I only          │ directly.                               │ fidelity.        │
├─────┼──────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ C   │ A held-out TASK split as      │ Classic CL: split the 13 taxonomy dims │ Cleanest CL      │
│     │ sessions: partition the eval  │ into 3-4 disjoint task groups, train   │ semantics but    │
│     │ corpus into N disjoint task   │ sequentially, measure BWT/FWT.         │ NEEDS new data   │
│     │ groups, train in sequence     │                                         │ splits — HIGHER  │
│     │                              │                                         │ cost, +1 day.    │
└─────┴──────────────────────────────┴────────────────────────────────────────┴──────────────────┘
```

Recommendation: **Option A** (surgery stages = sessions). It costs nothing extra
(our m09c surgery is already a 3-stage curriculum), and it answers the paper's
actual question: "is our progressive-unfreeze anti-drift better than SAFE/SEEKR/
SSIAT/SAPT's anti-drift, holding the staged curriculum fixed?" That is a clean,
publishable head-to-head — SOTA method swaps in as the per-stage-boundary
forgetting-control, our m09c1 is the baseline anti-drift.

Implementation contract for the SOTA trainers (§ C.8) under Option A:
- Each trainer accepts `--sessions <yaml-list>` where each entry is one surgery
  stage's {mode_mixture, unfreeze_below, max_epochs_pct}.
- Forgetting metric: re-run the held-out probe (probe_action top-1) at EACH
  session boundary; report BWT = mean_i (acc_after_all_sessions[i] −
  acc_right_after_session[i]). This is the cell that goes in a NEW § 12 sub-table
  (SOTA rows get a BWT column the V-JEPA-family rows don't need).
- FAIL LOUD if `--sessions` has < 2 entries (FL-10): a 1-session SOTA run is the
  silent-garbage case this whole section exists to prevent.

Q10 is now LOCKED (Option A) — M10-M13 are UNBLOCKED for implementation, built on
the m09c1/m09c2 surgery family per § C.8. The session axis is defined (surgery
stages), so the FL-10 (<2 sessions) guard is the only remaining runtime gate.

## § D · M-section ↔ file mapping

```text
┌──────┬─────────────────────────────────────────────────────┬──────────────────────────────────┐
│ M#   │ Files this stage PRODUCES or CONSUMES                │ Acceptance gate                 │
├──────┼─────────────────────────────────────────────────────┼──────────────────────────────────┤
│ M0   │ (read-only) iter/iter17_ablations/plan1.md §10        │ HF cards confirmed (DONE)      │
│ M1   │ PRODUCES: checkpoints/iter17_ablations/*.pt           │ 7 .pt files, sha256 logged     │
│      │ CONSUMES: configs/model/*.yaml (URI fields)           │ + total <180 GB                │
│ M2   │ PRODUCES:                                             │ yaml_extract.py walks the      │
│      │   configs/model/{vjepa2_0_HF, vjepa2_0_ssv2,          │ extends chain w/o KeyError.    │
│      │     vjepa2_1_vit_large, vjepa2_1_vit_giant,           │ Run:                           │
│      │     ijepa_vith14, ijepa_vitg16, lejepa_vith14}.yaml   │   for y in configs/model/*.yaml│
│      │   configs/train/{pretrain,surgery_*}_<B>.yaml × 27    │     ; do scripts/lib/          │
│      │ CONSUMES: configs/train/{pretrain_*, surgery_*}.yaml  │     yaml_extract.py "$y"       │
│      │           configs/model/vjepa2_{0,1}.yaml              │     model.embed_dim; done      │
│ M2a  │ PRODUCES: configs/train/*_vjepa2_0_HF.yaml (6 yamls)  │ Same.                          │
│ M2b  │ PRODUCES: configs/train/*_vjepa2_1_vit_large.yaml     │ Same.                          │
│ M2c  │ PRODUCES: configs/train/*_vjepa2_1_vit_giant.yaml     │ Same.                          │
│ M2d  │ PRODUCES: configs/train/*_<image_B>.yaml × 9 (head    │ Plus: each yaml's              │
│      │           variants only)                              │ image_encoder.enabled = true.  │
│ M3   │ PRODUCES: configs/eval/probe_encoders.yaml (extend)   │ python -c "from utils.         │
│      │           configs/pipeline.yaml (encoders block)      │   frozen_features import       │
│      │ CONSUMES: configs/model/*.yaml (embed_dim/arch/crop)  │   ENCODERS; print(len(         │
│      │                                                       │   ENCODERS))" returns ≥45.     │
│ M4   │ PRODUCES: src/utils/encoder_loader.py                  │ py_compile + ruff + unit       │
│      │ PATCHES:  src/m09a2_pretrain_head.py                   │ smoke (load each new model    │
│      │           src/m09c2_surgery_head.py                    │ yaml → ckpt → forward 1 clip   │
│      │           src/utils/frozen_features.py                 │ on CPU; embed_dim asserts).    │
│ M5   │ PRODUCES: outputs/sanity/m09a_pretrain_head_<B>/*.pt   │ exit 0 + non-empty             │
│      │           outputs/sanity/m09c_surgery_*_head_<B>/*.pt  │ motion_aux_head.pt for all 7   │
│      │ CONSUMES: configs/model/*.yaml, configs/train/*_<B>    │ candidates. layer_freeze logs  │
│      │           src/m09a2_*.py + image-encoder branch        │ show the expected             │
│      │ Gate before M6/M7.                                     │ [0,17)/[17,40) for vjepa2_0.   │
│ M6a  │ PRODUCES: outputs/poc/m09{a,c}_*_vjepa2_0_HF/*.pt      │ 7-cell paired-Δ matrix         │
│      │ PATCHES:  scripts/run_train.sh (MODEL_CFG_OVERRIDE)    │ written.                       │
│ M6b  │ Same for vjepa_2_1_vit_large.                          │ Same.                          │
│ M6c  │ Same for vjepa_2_1_vit_giant.                          │ Same.                          │
│ M7   │ PRODUCES: outputs/poc/m09{a,c}_*_<image_B>/*.pt        │ Stage 8 auto-skips per         │
│      │ CONSUMES: scripts/run_eval.sh (image-encoder           │ run_eval.sh dispatch.          │
│      │           Stage 8 skip path).                          │                                │
│ M7a  │ AMENDED (Q8): vjepa_2_0_ssv2 FROZEN + head-only.       │ Stages 1-7 + 11-13 for         │
│      │ PRODUCES: outputs/poc/m09{a,c}_*_vjepa2_0_ssv2/*.pt    │ frozen; full eval (incl Stage  │
│      │ CONSUMES: § C.7 3 yamls + vjepa2_0_ssv2.yaml ckpt.     │ 8) for head-only.              │
│ M8   │ DROPPED 2026-05-23 (per plan1 §10 — JEPA-WMS retired). │ —                              │
│ M9   │ PRODUCES: iter/iter17_ablations/iter17_ablation_       │ multi-encoder hero-table       │
│      │           summary.{png,pdf}  + high_level_outputs.md   │ matches § 12 schema.           │
│      │ CONSUMES: outputs/poc/probe_action/probe_paired_delta.json across all backbones (7×8).  │
│      │ PATCHES:  src/m07b_paired_delta.py (multi-model col)   │                                │
├──────┼─────────────────────────────────────────────────────┼──────────────────────────────────┤
│ § D.1 — Q5-Q9 ADDITIONS                                                                       │
├──────┼─────────────────────────────────────────────────────┼──────────────────────────────────┤
│ M0a  │ (Q6) PRODUCES: 1 docstring update in vjepa2_1_vit_   │ Python one-liner exits 0; the │
│      │ large.yaml with probed predictor.depth.                │ depth value lands in the yaml. │
│      │ CONSUMES: torch.hub network reachability.              │ Blocks M2b (ViT-L yaml write). │
│ M7b  │ (Q5) PRODUCES: 27 image-encoder pool-sweep yamls       │ All 27 trainer SANITY exits 0 │
│      │ (§ C.6); 27 trained heads (3 backbones × 3 pools ×    │ + 27 motion_aux_head.pt files. │
│      │ 3 positions); 27 registry rows.                        │ Hero table § 12 gets "pool"   │
│      │ CONSUMES: § C.6 yamls + encoder_loader.py pool patch.  │ sub-column.                    │
│ M10  │ (Q7) PRODUCES: src/m09s_safe.py + 4 SAFE yamls         │ SANITY: 1 backbone × SAFE     │
│      │ (one per V-JEPA backbone) + 4 trained encoders +       │ exits 0 + non-zero            │
│      │ 4 registry rows.                                       │ student_encoder.pt with PEFT   │
│      │ CONSUMES: src/utils/peft_modules.py + § C.8 specs.     │ params only (base frozen).     │
│ M11  │ (Q7) PRODUCES: src/m09s_seekr.py + 4 SEEKR yamls +     │ SANITY: replay buffer logs     │
│      │ 4 trained encoders + 4 registry rows.                  │ + KD term visible in losses.   │
│      │ CONSUMES: § C.8 + raw-replay buffer pattern from m09c1.│                                │
│ M12  │ (Q7) PRODUCES: src/m09s_ssiat.py + 4 SSIAT yamls +     │ SANITY: 1 adapter per block + │
│      │ 4 trained encoders + 4 registry rows.                  │ subspace-rank assertion.       │
│      │ CONSUMES: § C.8 + SVD subspace utilities.               │                                │
│ M13  │ (Q7) PRODUCES: src/m09s_sapt.py + 4 SAPT yamls +       │ SANITY: K adapters built +    │
│      │ 4 trained encoders + 4 registry rows.                  │ router α_k(x) sums to 1.       │
│      │ CONSUMES: § C.8 + m09s_ssiat shared framework.          │                                │
│ M14  │ (Q9) PATCHES: src/utils/hf_outputs.py --subdir flag.   │ Dry-run shows                  │
│      │ CONSUMES: existing hf_outputs.py at iter15+.            │ "iter17_ablations/" prefix on │
│      │                                                         │ every path_in_repo line.       │
└──────┴─────────────────────────────────────────────────────┴──────────────────────────────────┘
```

## § E · Smallest-SANITY smoke gates per file

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ SMOKE GATES (per CLAUDE.md smallest-SANITY-per-code-mod). Each must exit 0 BEFORE the next.  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-A · model-yaml resolvability                                                            │
│   for y in configs/model/{vjepa2_0_HF,vjepa2_0_ssv2,vjepa2_1_vit_large,vjepa2_1_vit_giant,   │
│              ijepa_vith14,ijepa_vitg16,lejepa_vith14}.yaml; do                               │
│     scripts/lib/yaml_extract.py "$y" model.embed_dim                                         │
│     scripts/lib/yaml_extract.py "$y" model.depth                                             │
│     scripts/lib/yaml_extract.py "$y" model.arch                                              │
│   done                                                                                       │
│   PASS: all 7 × 3 = 21 lookups print a value with no KeyError.                              │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-B · train-yaml inheritance walk                                                         │
│   for y in configs/train/*_{vjepa2_0_HF,vjepa2_1_vit_large,vjepa2_1_vit_giant,               │
│              ijepa_vith14,ijepa_vitg16,lejepa_vith14}.yaml; do                               │
│     scripts/lib/yaml_extract.py "$y" optimization.max_epochs.sanity                          │
│     scripts/lib/yaml_extract.py "$y" data.module                                             │
│   done                                                                                       │
│   PASS: extends-chain resolves without missing keys for all 27 train yamls.                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-C · encoder_loader.py CPU import + dispatch                                             │
│   python -c "                                                                                │
│     from utils.encoder_loader import (load_video_encoder, load_image_encoder_per_frame,      │
│         assert_no_predictor_when_image, assert_embed_dim_match, fetch_ckpt)                  │
│     import yaml                                                                              │
│     for m in ['ijepa_vith14','ijepa_vitg16','lejepa_vith14',                                 │
│               'vjepa2_0_HF','vjepa2_1_vit_large','vjepa2_1_vit_giant']:                      │
│       cfg = yaml.safe_load(open(f'configs/model/{m}.yaml'))['model']                         │
│       assert_no_predictor_when_image(cfg)                                                    │
│       print(m, cfg['embed_dim'], cfg['depth'])                                               │
│   "                                                                                          │
│   PASS: 6 lines printed; FATAL on any AssertionError or missing key.                         │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-D · registry parity (configs/eval/probe_encoders.yaml + pipeline.yaml.encoders)         │
│   python -c "                                                                                │
│     import yaml                                                                              │
│     a = set(yaml.safe_load(open('configs/eval/probe_encoders.yaml'))['encoders'].keys())     │
│     b = set(yaml.safe_load(open('configs/pipeline.yaml'))['encoders'].keys())                │
│     missing = a - b                                                                          │
│     assert not missing, f'pipeline.yaml.encoders missing: {missing}'                         │
│     print(len(a), 'rows in both')                                                            │
│   "                                                                                          │
│   PASS: count ≥ 45 (8 iter15 + 1 dinov2 + 37 iter17).                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-E · run_train.sh MODEL_CFG_OVERRIDE preflight (no GPU, --SANITY exits at ckpt-load)     │
│   for M in vjepa2_0_HF vjepa2_1_vit_large vjepa2_1_vit_giant; do                             │
│     MODEL_CFG_OVERRIDE=configs/model/${M}.yaml \                                             │
│     TRAIN_CFG_OVERRIDE=configs/train/pretrain_head_${M}.yaml \                               │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --SANITY 2>&1 |               │
│     tee logs/iter17_smoke_${M}.log                                                            │
│   done                                                                                       │
│   PASS: each exits 0; logs/iter17_smoke_<M>.log contains:                                    │
│     - "model: configs/model/<M>.yaml (override via MODEL_CFG_OVERRIDE env var)"              │
│     - "Student loaded: <X> params (<Y>/<Z> keys = ≥90%)"                                     │
│     - "[m09a2 STRICT HEAD-ONLY] encoder FROZEN: 0 trainable block params"                    │
│     - non-zero motion_aux_head.pt at outputs/sanity/m09a_pretrain_head_<M>/                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-F · image-encoder head SANITY (Pro 4000, ≤10 min each)                                  │
│   for M in ijepa_vith14 ijepa_vitg16 lejepa_vith14; do                                       │
│     MODEL_CFG_OVERRIDE=configs/model/${M}.yaml \                                             │
│     TRAIN_CFG_OVERRIDE=configs/train/pretrain_head_${M}.yaml \                               │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --SANITY                      │
│   done                                                                                       │
│   PASS criteria additional to GATE-E:                                                        │
│     - log contains "[image-encoder] modality=image; per-frame ViT + temporal_pool=mean"      │
│     - log does NOT contain "FATAL: ckpt has no 'predictor' key" (the guard at m09a2.py:222) │
│     - motion_aux head input dim = <embed_dim> in the head ckpt's `head_state_dict['fc1.       │
│        weight'].shape[1]` matches model.embed_dim (1280 for I-JEPA H/14; 1408 for G/16)      │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-G · surgery_head SANITY for each backbone (verifies no D_I tube schema drift)           │
│   for M in vjepa2_0_HF vjepa2_1_vit_large vjepa2_1_vit_giant; do                             │
│     SURGERY_INIT=outputs/sanity/m09a_pretrain_head_${M}/m09a_ckpt_best.pt \                  │
│     MODEL_CFG_OVERRIDE=configs/model/${M}.yaml \                                             │
│     TRAIN_CFG_OVERRIDE=configs/train/surgery_3stage_DI_head_${M}.yaml \                      │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_head --SANITY             │
│   done                                                                                       │
│   PASS: exit 0 + log contains "Stage: stage0_head_only_DI" + non-zero motion_aux_head.pt    │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-H · encoder-update SANITY (96 GB Pro 6000 #2 — gates M6a/b/c)                           │
│   For each of {vjepa2_0_HF, vjepa2_1_vit_large, vjepa2_1_vit_giant}:                         │
│     MODEL_CFG_OVERRIDE=... TRAIN_CFG_OVERRIDE=...pretrain_encoder_<M>.yaml                    │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --SANITY                   │
│   PASS: log shows "layer_freeze: blocks [0, <freeze_below>) frozen + [<freeze_below>, <N>)   │
│   trainable", and student_encoder.pt drift ‖Δ‖/‖init‖ > 1e-4 (matches iter15 v15a §3).       │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-I · SOTA trainer SANITY (Q7 — one gate per method, gates M10/M11/M12/M13)              │
│   PREREQ: § C.10 session-mapping authored + user-approved (else the run is meaningless).    │
│   For each M in {safe, seekr, ssiat, sapt}, smallest backbone (vjepa2_1_vit_large):         │
│     MODEL_CFG_OVERRIDE=configs/model/vjepa2_1_vit_large.yaml \                               │
│     TRAIN_CFG_OVERRIDE=configs/train/${M}_vjepa2_1_vit_large.yaml \                          │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh ${M} --SANITY                              │
│   PASS (per method):                                                                         │
│     - exit 0 + non-zero student_encoder.pt                                                   │
│     - log: "[peft] base backbone FROZEN: 0 trainable block params; PET params = <N> > 0"    │
│       (catches the freeze_backbone_keep_peft "forgot to wrap" bug, FL-9)                     │
│     - SAFE: log shows both "stage_S" + "stage_F"; SEEKR: "replay buffer size = <N>" +       │
│       "KD term" in loss; SSIAT: "adapter reused (not re-init)"; SAPT: "router α_k sums      │
│       to 1.0 ± 1e-4 per sample".                                                             │
│     - session-axis log line present: "[session-map] session i/N = <mapped entity>"          │
│       (proves § C.10 wiring is live, not a no-op).                                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ GATE-J · image-encoder POOL dispatch SANITY (Q5 — gates M7b)                                 │
│   For B in {ijepa_vith14, lejepa_vith14} (skip vitg16 — same code path) × P in {mean,cls,   │
│   max}:                                                                                       │
│     MODEL_CFG_OVERRIDE=configs/model/${B}.yaml \                                             │
│     TRAIN_CFG_OVERRIDE=configs/train/pretrain_head_${B}_pool${P}.yaml \                      │
│     CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --SANITY                      │
│   PASS:                                                                                       │
│     - P=mean,max: exit 0 + log "[image-encoder] temporal_pool=<P>"                          │
│     - P=cls on an encoder WITHOUT a CLS token: must FATAL with FL-8 message (this is a       │
│       PASS for the gate — the guard fired). If it silently runs, GATE-J FAILS.              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## § F · FAIL-LOUD guards to install

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Required FAIL-LOUD points (mirrors iter15 m09a1/m09c1 patterns where cited)              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-1 · encoder_loader.assert_embed_dim_match()                                           │
│   Mirrors m09a1_pretrain_encoder.py:268-273 (load_pct < min_student_load_pct FATAL).     │
│   NEW check: even if all keys load, the FIRST encoder block's norm.weight.shape[0] must  │
│   equal model_cfg['embed_dim']. Catches "loaded a 1408-dim ckpt into a 1664-dim yaml"    │
│   silent failure (would otherwise produce shape errors deep in attention layers, miles   │
│   from the actual ckpt-yaml mismatch).                                                   │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-2 · encoder_loader.assert_no_predictor_when_image()                                   │
│   FATAL if model_cfg['modality'] == 'image' AND any of                                   │
│      model_cfg.get('predictor') OR model_cfg.get('pred_depth') OR                        │
│      model_cfg.get('pred_embed_dim') OR model_cfg.get('pred_num_heads')                  │
│   is not None. Prevents an operator from copy-pasting a V-JEPA yaml and leaving stale    │
│   predictor fields → m09a2 would build a never-loaded predictor and corrupt cfg state.   │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-3 · m09a2.build_model — image-encoder must declare model_cfg['modality']              │
│   At top of build_model, before any branching:                                           │
│       if 'modality' not in model_cfg:                                                    │
│           print(f"FATAL: model.modality missing in cfg — every iter17 model yaml MUST    │
│              declare modality: video|image. legacy V-JEPA yamls add modality: video      │
│              explicitly (M2 patch).")                                                    │
│           sys.exit(1)                                                                    │
│   Mirrors m09a1:286 ("if model_cfg['predict_all'] or n_output_distillation > 1:" guard). │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-4 · m09c2.build_model — image branch cannot accept --init-from-ckpt that's a prior   │
│   V-JEPA run's m09{a,c}_ckpt_best.pt (which would carry "student"+"predictor" of a       │
│   different arch). Already covered by m09c2:232-241 schema check — extend the FATAL      │
│   message to add "OR you're trying to init an image encoder from a V-JEPA prior run;     │
│   image encoders init from model.checkpoint_path directly."                              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-5 · run_train.sh MODEL_CFG_OVERRIDE existence guard                                   │
│   After line 156 (the new MODEL_CFG resolver), add:                                      │
│     [ -f "$MODEL_CFG" ] || { echo "❌ FATAL: MODEL_CFG=$MODEL_CFG missing"; exit 3; }    │
│   Mirrors lines 75-76 (LOCAL_DATA/MASTER_MANIFEST guards). Without this, a typo in       │
│   MODEL_CFG_OVERRIDE would propagate to a Python ImportError 90 sec into the run.        │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-6 · frozen_features.load_image_encoder_frozen — embed_dim parity check                │
│   After building the encoder, assert encoder.embed_dim == registry['embed_dim']          │
│   (read from probe_encoders.yaml). Mirrors frozen_features.py:160-161 load_pct guard.    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-7 · pipeline.yaml + probe_encoders.yaml parity (GATE-D in § E above)                  │
│   Add as a post-edit hook OR as the first check inside frozen_features._load_encoders_   │
│   registry: validate that every key in probe_encoders.yaml.encoders is also present in   │
│   pipeline.yaml.encoders. Without this, m05's downstream extraction will silently fall   │
│   through to a None dispatcher.                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-8 · encoder_loader — CLS pool requested on a CLS-less encoder (Q5)                    │
│   In load_image_encoder_per_frame, when train_cfg['image_encoder']['temporal_pool']      │
│   == 'cls': FATAL unless the built encoder exposes a real CLS token (e.g.,               │
│   hasattr(encoder, 'cls_token') and encoder.cls_token is not None). Message:             │
│   "FATAL: temporal_pool=cls requested but <arch> ships no CLS token — I-JEPA emits N      │
│   patch tokens only. Use mean|max, or rebuild the encoder with use_cls_token=true."       │
│   Without this, the loader silently takes patch[0] (top-left patch) as a fake CLS →      │
│   garbage features that pass shape checks. Catches the § C.6 gotcha. Gated by GATE-J.    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-9 · peft_modules.freeze_backbone_keep_peft — no PET params found (Q7)                  │
│   After freezing, count params with requires_grad=True. If 0, FATAL:                     │
│   "FATAL: freeze_backbone_keep_peft left 0 trainable params — wrap_vit_with_peft was      │
│   never called or matched no layers. Check peft_mode=<mode> against the ViT block        │
│   module names." Without this, the SOTA trainer runs a no-op optimizer (loss flat, ckpt  │
│   == init) and burns GPU producing a checkpoint identical to the frozen baseline.        │
│   Mirrors m09a2's "0 trainable block params" assert. Gated by GATE-I.                    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ FL-10 · SOTA trainer — < 2 sessions in --sessions (Q7, § C.10)                            │
│   At trainer startup, if len(parse(--sessions)) < 2: FATAL                               │
│   "FATAL: SAFE/SEEKR/SSIAT/SAPT require ≥2 sessions; got <N>. A 1-session run cannot      │
│   measure forgetting/transfer — see § C.10 session-mapping. Pick Option A/B/C."           │
│   This is THE guard that prevents the headline failure (running multi-session methods    │
│   on a single corpus and reporting meaningless cells). Gated by GATE-I.                   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## § G · Risks not in plan1.md § 6

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Risk                                                  │ Mitigation                      │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-1 · Temporal mean-pool may underperform CLS-token   │ Adapter exposes image_temporal_ │
│      pool for I-JEPA / LeJEPA. Both papers ablate     │ pool: {mean,cls,max} per yaml.  │
│      attentive-probe-style pooling; mean is the safe  │ Iter17 default = mean. If frozen│
│      default but max often wins on motion-laden       │ baseline numbers underperform   │
│      datasets. Reviewers will ask.                    │ V-JEPA frozen by >10% on motion_│
│                                                       │ cos, A/B vs cls before M9       │
│                                                       │ aggregation. ~$1-2 extra GPU.    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-2 · I-JEPA ViT-G/16 head/dim values are not          │ M0 audit confirmed ckpt key set │
│      authoritative. Paper says 1408-dim ViT-G/16; HF  │ but did NOT confirm num_heads.  │
│      card may use 16 heads (image ViT convention)     │ Add a one-line probe in M5      │
│      vs V-JEPA 2.0's 22-head ViT-g/16. The two have   │ SANITY: print(encoder.blocks[0].│
│      DIFFERENT block weight shapes → state_dict load  │ attn.num_heads) before training │
│      succeeds with 0 unexpected keys but RoPE freq    │ kicks off; FATAL if not 16.     │
│      table is wrong → silent garbage features.        │                                 │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-3 · HF mirror's predictor weights may have different│ FL-1 (assert_embed_dim_match)   │
│      keys than fbai dump. facebook/vjepa2-vitg-fpc64- │ catches the most common         │
│      384 stores both encoder+predictor but the key    │ corruption. Add explicit        │
│      naming (target_encoder vs encoder, predictor.    │ key-set printout to            │
│      module.X vs predictor.X) varies. Loader's        │ resolve_encoder_state_dict()    │
│      resolve_encoder_state_dict already handles 5     │ side-effect logging when load_  │
│      schemas — extend test matrix to include the new  │ pct < 95% even if above the     │
│      HF mirror's exact key naming.                    │ 90% min_student_load_pct floor. │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-4 · LeJEPA-L lives in an HF DATASET repo            │ encoder_loader.fetch_ckpt has   │
│      (gajeshladharai/artifacts/lejepa-l.pt) not a    │ a dedicated branch for          │
│      MODEL repo. hf_hub_download(repo_type='dataset') │ hf_dataset_asset (new yaml key).│
│      is needed; using AutoModel.from_pretrained or    │ No AutoModel path attempted.    │
│      default model-repo path will FATAL.              │                                 │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-5 · iter15 motion_aux head was built for 1664-dim   │ motion_aux.head.embed_dim_in    │
│      V-JEPA G. iter17 backbones range 1024-1664. The  │ NEW yaml key consumed by        │
│      head's first Linear layer must size to backbone  │ utils.motion_aux_loss.build_    │
│      dim or the loss is uncomputable. Will silently   │ motion_aux_head_from_cfg.       │
│      load wrong-shape ckpt under strict=False.        │ FATAL if cfg key missing.       │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-6 · 27 train yamls + 7 model yamls + 37 registry    │ Add a single helper yaml        │
│      rows = 71 yaml files. Hand-authored typos in    │ generator script (CPU only) at  │
│      arch / embed_dim / freeze_below are the most     │ scripts/lib/gen_iter17_configs. │
│      likely silent failure mode. iter15 had two such  │ py that reads a single table of │
│      typos caught only by SANITY.                     │ (B, arch, dim, depth) and emits │
│                                                       │ all 71 files from templates.    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-7 · § 12 SOTA stretch (SAFE/SEEKR/SSIAT/SAPT × 4    │ Keep iter17 scope locked to the │
│      backbones) is estimated ~2000-6000 LoC of new    │ 7 candidates in § 11. Defer    │
│      trainer code. plan1 §12 hints "2-4 weeks". This  │ § 12 to a new iter18 plan; do   │
│      is optimistic — SSIAT alone has shared-adapter   │ NOT add SOTA cells to iter17    │
│      bookkeeping (gradient routing across sessions)   │ M-table. Surface this risk      │
│      that interacts non-trivially with our            │ explicitly in the iter17 commit │
│      surgery_base.yaml stage logic.                   │ message so the user can decide. │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ R-8 · Pro 6000 #2 provisioning lead time. plan1 §8    │ Add a "Pro 6000 #2 ready?"      │
│      §9 assumes the second instance is ready when M6  │ pre-flight as part of M5: the   │
│      starts. If it's not, M6 stalls behind M5 done    │ first invocation tries          │
│      → no parallelism gain (cost stays ~$30 but wall  │ nvidia-smi against the box and  │
│      time doubles vs the §5 estimate).                │ exits early with a clear msg.   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## § H · Open questions back to the user — Q5/Q6/Q7/Q8

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Q5 · Image-encoder temporal pooling default (§ G R-1)                                    │
│      [ ] mean       (current iter17 lock — safe but may underperform)                   │
│      [ ] cls        (better on global semantics)                                         │
│      [ ] max        (better on motion-laden datasets like ours — RAFT-binned classes)   │
│      [ ] sweep all 3 in M7  (+$2-3 GPU, ~1 day, gives a defensible ablation row)        │
│      Recommendation: pick mean for the headline numbers, add max as a footnote row.      │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q6 · V-JEPA 2.1 ViT-L pred_depth (§ C.1 vjepa2_1_vit_large.yaml)                         │
│      Meta's ViT-L pretrain config from `app/vjepa_2_1/configs/pretrain/vit_large.yaml`  │
│      uses pred_depth=12 (NOT the ViT-G's 24). Confirm M0 + torch.hub ckpt actually       │
│      ships a 12-depth predictor before M2b yaml lands. Without this confirmation, M5     │
│      SANITY's predictor-load gate will FATAL at <50% load_pct.                          │
│      Recommendation: read the torch.hub config (one Python line) BEFORE writing the     │
│      yaml.                                                                               │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q7 · iter17 SOTA stretch (§ G R-7) — defer or include?                                   │
│      [ ] DEFER to iter18  (iter17 stays a 7-candidate ablation; clean scope)            │
│      [ ] INCLUDE in iter17  (+$80-160 GPU, +2-4 weeks LoC; risk of slipping iter17)     │
│      Recommendation: DEFER. iter17's win condition is already 7-way paired-Δ tables.     │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q8 · Frozen-only vs head-only for SSv2-FT (§ 11 row #4)                                  │
│      Plan1 § 11 locks vjepa_2_0_HF_ssv2 as FROZEN only. But user has paired-Δ table for │
│      EVERY other backbone covering 4 paired axes (frozen / pretrain_head / surgical_*_  │
│      head). If a reviewer asks "does head-only training on SSv2-FT also win?" we won't  │
│      have the cell.                                                                      │
│      [ ] FROZEN only (status quo — saves $1-2 + 1 hr)                                   │
│      [ ] FROZEN + head-only (+$2 + 2 hr; symmetric with image encoders)                 │
│      Recommendation: add head-only (3 cells extra) for cross-table symmetry.            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q9 · Where do iter17 ckpts live? Plan1 § 11 manifest says `checkpoints/iter17_          │
│      ablations/` but doesn't specify whether they're inside the iter15 HF outputs repo  │
│      (anonymousML123/factorjepa-outputs) or a new repo. Affects M1 download recipe +    │
│      future repro.                                                                       │
│      [ ] Same repo, new subdir                                                          │
│      [ ] New repo `anonymousML123/factorjepa-iter17-ckpts`                              │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## § I · Implementer checklist (one-screen summary)

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ TODO ORDER (do in this sequence; check box when SANITY for that line is green)           │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  [ ] 0a. (Q6) torch.hub probe — print vjepa2_1_vit_large_384.predictor.depth; record    │
│         in vjepa2_1_vit_large.yaml.model.pred_depth BEFORE step 7.                       │
│  [ ] 1. Apply patches to src/m09a2_pretrain_head.py + src/m09c2_surgery_head.py per § C.4│
│  [ ] 2. Add src/utils/encoder_loader.py per § C.4 (CPU-only, no GPU calls)              │
│  [ ] 2b. (Q5) Extend encoder_loader.load_image_encoder_per_frame to read train_cfg      │
│         ['image_encoder']['temporal_pool']; FAIL LOUD if missing. Add CLS-availability  │
│         check per § C.6 gotcha.                                                          │
│  [ ] 3. Patch src/utils/frozen_features.py per § C.4 (image-encoder dispatch)           │
│  [ ] 4. Patch scripts/run_train.sh per § C.4 (MODEL_CFG_OVERRIDE + TRAIN_CFG_OVERRIDE   │
│         + 4 SOTA subcommands per § C.8)                                                  │
│  [ ] 5. Patch scripts/run_eval.sh per § C.4 (ckpt resolvers + image-Stage-8 skip)       │
│  [ ] 6. GATE-C smoke (CPU-side encoder_loader import)                                   │
│  [ ] 7. Write 7 model yamls per § C.1 → GATE-A                                          │
│  [ ] 8. Write 30 of 48 train yamls (18 video + 9 image P=mean + 3 SSv2-FT head per      │
│         § C.7) → GATE-B  [remaining 18 pool variants in 8b]                              │
│  [ ] 8b. (Q5) Write 18 ADDITIONAL pool-sweep yamls per § C.6 (P=cls + P=max variants)   │
│         → 27 total image-encoder head yamls. GATE-B re-run.                              │
│  [ ] 9. Extend configs/eval/probe_encoders.yaml + configs/pipeline.yaml per § C.5 +     │
│         § C.6 (27 pool rows) + § C.7 (3 SSv2-FT-head rows) + § C.8 (16 SOTA rows)       │
│         = 56 total NEW rows. GATE-D parity check.                                        │
│  [ ] 10. M1 downloads (network) → § C.1 ckpt paths populated                            │
│  [ ] 11. GATE-E / GATE-F / GATE-G SANITY smokes on Pro 4000                             │
│  [ ] 12. GATE-H (Pro 6000 #2) — verify encoder-update SANITY for 3 FULL replicas        │
│  [ ] 13. M6a/b/c POC (Pro 6000 #2)  +  M7/M7a/M7b POC (Pro 4000) in parallel            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q5-Q9 ADDITIONS (steps 14-22 — gated on the SOTA scope decision)                         │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  [ ] 14. (Q8) Run M7a SSv2-FT head-only POC (3 yamls per § C.7) on Pro 4000.            │
│  [ ] 15. (Q5) Run M7b image-encoder POOL SWEEP — 27 trainers across 3 backbones × 3     │
│         pools × 3 positions (3 backbones × 3 pools = 9 frozen + 9 head trainers; or     │
│         9 frozen + 27 head trainers if you do all 3 positions per pool). Pro 4000.       │
│  [ ] 16. (Q7) Add src/utils/peft_modules.py per § C.8 + the m09c1-clone base scaffold    │
│         (stage loop + replay + m09c2 frozen-backbone guards). HARD GATE: this base +     │
│         M10 must pass GATE-I before 18-20 (Q12 parallel build shares this base, so a     │
│         base bug blocks all 4 — prove it once).                                          │
│  [ ] 17. (Q7) M10 — src/m09s_safe.py (clone m09c1, PET slow/fast) + 4 SAFE yamls →      │
│         SANITY (GATE-I + FL-10) → POC (4 backbones).                                     │
│  [ ] 18-20. (Q7, PARALLEL per Q12) M11 m09s_seekr.py / M12 m09s_ssiat.py / M13          │
│         m09s_sapt.py — each + 4 yamls → SANITY → POC. All build on the step-16 base;     │
│         differ only in PET/replay/routing head. Start once step 16+17 base is green.    │
│  [ ] 21. (Q9) Patch src/utils/hf_outputs.py per § C.9 (--subdir iter17_ablations).      │
│  [ ] 22. run_eval.sh per backbone (POC, all 7 + SOTA × 4 backbones = 23 encoders).      │
│  [ ] 23. M9 aggregate: extend src/m07b_paired_delta.py for multi-model column           │
│         (now spans 8 V-JEPA positions × 4 backbones + image × 3 pools + SOTA × 4).      │
│  [ ] 24. Fill § 12 hero table cells in iter/iter17_ablations/plan1.md                   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Critical files for implementation
- /workspace/factorjepa/src/utils/encoder_loader.py (NEW — image-encoder adapter + pool sweep)
- /workspace/factorjepa/src/utils/peft_modules.py (NEW — Adapter/SSF/LoRA/VPT primitives for SOTA)
- /workspace/factorjepa/src/m09a2_pretrain_head.py (PATCH — image-encoder dispatch + FL-3 guard)
- /workspace/factorjepa/src/m09c2_surgery_head.py (PATCH — image-encoder dispatch + FL-4 guard)
- /workspace/factorjepa/src/m09s_safe.py  (NEW — SAFE trainer per § C.8)
- /workspace/factorjepa/src/m09s_seekr.py (NEW — SEEKR trainer per § C.8)
- /workspace/factorjepa/src/m09s_ssiat.py (NEW — SSIAT trainer per § C.8)
- /workspace/factorjepa/src/m09s_sapt.py  (NEW — SAPT trainer per § C.8)
- /workspace/factorjepa/src/utils/hf_outputs.py (PATCH — --subdir per § C.9, Q9)
- /workspace/factorjepa/scripts/run_train.sh (PATCH — MODEL_CFG_OVERRIDE + TRAIN_CFG_OVERRIDE env vars at lines 156, 232, 283-289, 432-475; + 4 SOTA subcommands)
- /workspace/factorjepa/configs/eval/probe_encoders.yaml (PATCH — append 56 registry rows: 37 base + 27 pool sweep + 3 SSv2-FT-head + 16 SOTA; line 76 is the current EOF)

### Post-Q5-Q9 scope summary

```text
┌──────────────────────────────────┬───────────┬────────────┬──────────────────────────────────┐
│ Bucket                            │ Pre-Q5-Q9 │ Post-Q5-Q9 │ Delta cause                       │
├──────────────────────────────────┼───────────┼────────────┼──────────────────────────────────┤
│ Model yamls                       │ 7         │ 7          │ unchanged                         │
│ Train yamls (V-JEPA + image)      │ 27        │ 48         │ +3 SSv2-FT head (§ C.7) +         │
│                                  │            │            │  18 pool-sweep cls/max (§ C.6)    │
│ Train yamls (SOTA)                │ 0         │ 16         │ +SAFE/SEEKR/SSIAT/SAPT × 4        │
│ NEW src modules                   │ 1         │ 6          │ +4 SOTA trainers + peft_modules.py│
│ PATCHES                           │ 4         │ 5          │ +hf_outputs.py --subdir           │
│ Registry rows                     │ 37        │ 83         │ +27 pool + 3 SSv2-FT + 16 SOTA    │
│ Estimated POC compute             │ ~$29-40   │ ~$111-205  │ +Q7 SOTA + Q5 pool sweep + Q8     │
│ Estimated calendar time           │ ~2 weeks  │ ~6-8 weeks │ Q7 SOTA LoC dominates             │
└──────────────────────────────────┴───────────┴────────────┴──────────────────────────────────┘
```

Reconciled with plan1 § 9b (both now show 48): "Train yamls (V-JEPA + image)" = 18 video
+ 27 image (pool sweep SUPERSEDES the 9 P=mean: 9 + 18 cls/max = 27) + 3 SSv2-FT head = 48.
SOTA (16) counted separately. NEW src modules = 6 = encoder_loader.py (pre-Q5-Q9) +
peft_modules.py + m09s_{safe,seekr,ssiat,sapt}.py (Q7). Registry rows 37 → 83 = +27 pool
+ 3 SSv2-FT + 16 SOTA. No remaining cross-doc disagreement.
