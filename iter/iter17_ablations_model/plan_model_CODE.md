# iter17 — JEPA-variant model-ablation · DETAILED CODE (executable spec)

> Sibling doc: `plan_model.md` = the WHAT (roster §A + arm-support §B + hero §G).
> **This file = the HOW**: code-complete diffs + new-file bodies for every ablation cell.
> Decisions (2026-05-28): (1) uniform `outputs/<mode>/<backbone>/<arm>/` namespace, migrate
> the existing ViT-G artifacts; (2) build infra for ALL trainable cross-arch
> (vitL/vitg/2.0_vitg) + ALL frozen-only baselines; (3) **do NOT execute** — user runs each
> (backbone × arm) train/eval sequentially. Every diff below is anchored to a verified
> 2026-05-28 read (line refs cited). `VERIFY-FIRST` = an upstream model fact to confirm
> before the run (NOT a code unknown).

═══════════════════════════════════════════════════════════════════════════════
§0 · Verified reality — corrects plan_model.md §C/§D "zero new Python"
═══════════════════════════════════════════════════════════════════════════════

```text
┌────────────────────────────────────┬───────────────────────────────────────────────┐
│ seam (verified file:line)          │ why it is NOT config-only for cross-arch        │
├────────────────────────────────────┼───────────────────────────────────────────────┤
│ vjepa2_imports.get_vit_by_arch     │ dispatch has ONLY vit_giant_xformers (2.0 g) +  │
│   :146-155                         │ vit_gigantic_xformers (2.1 G). No L/H/large.    │
│ frozen_features.load_vjepa_2_1_    │ reads arch/crop/embed_dim from HARDCODED        │
│   frozen :137-172 (enc=ENCODERS    │ ENCODERS["vjepa_2_1_frozen"] → any non-G ckpt   │
│   ["vjepa_2_1_frozen"] :143)       │ builds a G ViT → shape crash.                   │
│ predictor_eval :45-54 module consts│ PRED_*/ENCODER_EMBED_DIM from get_model_config  │
│   from get_model_config(None)=G    │ (None)=vjepa2_1.yaml → m12d/m12e predictor G-only│
│ m12a enc_kind dispatch :277-283    │ calls load_vjepa_2_1_frozen(ckpt,nf) w/o the    │
│   (repeats :373/420/434/480)       │ encoder name → loader can't pick the arch.      │
├────────────────────────────────────┼───────────────────────────────────────────────┤
│ surgery freeze m09c1 :763,:1150    │ ALREADY depth-agnostic int(depth*unfreeze_below)│
│                                    │ → ZERO edit (confirmed). §E.                     │
└────────────────────────────────────┴───────────────────────────────────────────────┘
```
Same-arch (ViT-G) variants ARE config+shell-only (the live 8 prove it). Cross-arch needs the
**3 contained Python edits** (B1/B2/B3) + new model configs. Predictor-TEMPORAL (m12e) on
NON-2.1 trainable backbones additionally needs the hierarchical-layer count from each config
(2.1 has n_output_distillation=4; 2.0 has none) — flagged inline.

═══════════════════════════════════════════════════════════════════════════════
§1 · Arm→on-disk-dir map + canonical encoder names (the naming contract)
═══════════════════════════════════════════════════════════════════════════════

Encoder name = `<backbone>_<arm>`. The existing 8 ViT-G encoders are renamed with an explicit
`vitG` token so the parser is unambiguous (`vjepa_2_1_pretrain_encoder` →
`vjepa_2_1_vitG_pretrain_encoder`).

```text
arm (suffix)                  on-disk arm dir              trainer   predictor-bearing ckpt
frozen                        — (external .pt / HF id)      —         (Meta predictor in ckpt)
pretrain_encoder              m09a_pretrain_encoder         m09a1     m09a_ckpt_best.pt
pretrain_2X_encoder           m09a_pretrain_2X_encoder      m09a1     m09a_ckpt_best.pt
pretrain_head                 m09a_pretrain_head            m09a2     m09a_ckpt_best.pt
surgical_3stage_DI_encoder    m09c_surgery_3stage_DI_encoder m09c1    m09c_ckpt_best.pt
surgical_noDI_encoder         m09c_surgery_noDI_encoder     m09c1     m09c_ckpt_best.pt
surgical_3stage_DI_head       m09c_surgery_3stage_DI_head   m09c2     m09c_ckpt_best.pt
surgical_noDI_head            m09c_surgery_noDI_head        m09c2     m09c_ckpt_best.pt
```
Trainable backbones (8 arms each): vjepa_2_1_vitG, vjepa_2_1_vitg, vjepa_2_1_vitL,
vjepa_2_0_vitg. Frozen-only (frozen arm only): vjepa_2_0_vitg_ssv2, vjepa_1_vitL,
vjepa_1_vitH, vjepa_2_vitL_256, ijepa_vitH14, ijepa_vitG16, lejepa_vitH14, dinov2.

═══════════════════════════════════════════════════════════════════════════════
WORKSTREAM A — uniform per-backbone namespace + migration
═══════════════════════════════════════════════════════════════════════════════

### A1 · `scripts/run_train.sh` (BACKBONE selector + namespaced OUT_DIR/SURGERY_INIT)

```diff
@@ L184 (was: MODEL_CFG="configs/model/vjepa2_1.yaml")
-MODEL_CFG="configs/model/vjepa2_1.yaml"
+# iter17: backbone selector. ViT-G keeps its existing canonical config file
+# (configs/model/vjepa2_1.yaml — also the get_model_config(None) default); other
+# backbones use configs/model/<backbone>.yaml. MODEL_CFG override still wins.
+BACKBONE="${BACKBONE:-vjepa_2_1_vitG}"
+case "$BACKBONE" in
+    vjepa_2_1_vitG) _MCFG="configs/model/vjepa2_1.yaml" ;;
+    *)              _MCFG="configs/model/${BACKBONE}.yaml" ;;
+esac
+MODEL_CFG="${MODEL_CFG:-$_MCFG}"
+[ -f "$MODEL_CFG" ] || { echo "FATAL: model config $MODEL_CFG missing for BACKBONE=$BACKBONE"; exit 3; }

@@ L207 SURGERY_INIT — surgery inits from THIS backbone's pretrain (per-backbone paired-Δ)
-SURGERY_INIT="${SURGERY_INIT:-outputs/${mode_dir}/${PRETRAIN_NS}/${PRETRAIN_CKPT}}"
+SURGERY_INIT="${SURGERY_INIT:-outputs/${mode_dir}/${BACKBONE}/${PRETRAIN_NS}/${PRETRAIN_CKPT}}"

@@ L266 / L274 / L326 / L401 / L441 — prefix every model OUT_DIR with ${BACKBONE}/
-        OUT_DIR="outputs/${mode_dir}/m09a_pretrain_2X_encoder"          # L266
+        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09a_pretrain_2X_encoder"
-            OUT_DIR="outputs/${mode_dir}/${PRETRAIN_NS}"                 # L274
+            OUT_DIR="outputs/${mode_dir}/${BACKBONE}/${PRETRAIN_NS}"
-        OUT_DIR="outputs/${mode_dir}/m09c_surgery_${VARIANT_TAG}"        # L326 (encoder surgery)
+        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09c_surgery_${VARIANT_TAG}"
-        OUT_DIR="outputs/${mode_dir}/m09a_pretrain_head"                 # L401
+        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09a_pretrain_head"
-        OUT_DIR="outputs/${mode_dir}/m09c_surgery_${VARIANT_TAG}"        # L441 (head surgery)
+        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09c_surgery_${VARIANT_TAG}"
```

**UNCHANGED (verified dataset-level / shared across backbones):** `TAXONOMY_LABELS` (L219),
the `m04f` output-root (L233), `action_labels.json` (L303/306/391/418/467 etc.),
`probe_action` — all stay at `outputs/${mode_dir}/…` root. `--model-config "$MODEL_CFG"`
is already passed to all four m09 trainers (L295/381/412/460); they build the ViT from
`model_cfg["arch"]` via `get_vit_by_arch`, so no `.py` change in the trainers.

### A2 · `scripts/run_eval.sh` (name→(backbone,arm) parser replaces the 4 case blocks)

Insert these helpers above `encoder_ckpt_for()` (replaces L197-247) and add the frozen
resolver:

```bash
# iter17: encoder name = "<backbone>_<arm>". Parse once; map arm→on-disk dir.
_arm_dir() {                                   # arm suffix → m09 output dir name
    case "$1" in
        pretrain_encoder)            echo m09a_pretrain_encoder ;;
        pretrain_2X_encoder)         echo m09a_pretrain_2X_encoder ;;
        pretrain_head)               echo m09a_pretrain_head ;;
        surgical_3stage_DI_encoder)  echo m09c_surgery_3stage_DI_encoder ;;
        surgical_noDI_encoder)       echo m09c_surgery_noDI_encoder ;;
        surgical_3stage_DI_head)     echo m09c_surgery_3stage_DI_head ;;
        surgical_noDI_head)          echo m09c_surgery_noDI_head ;;
        *) echo "" ;;
    esac
}
_split_enc() {                                  # "<bb>_<arm>" → echoes "BACKBONE ARM"
    local n="$1" arm
    for arm in pretrain_2X_encoder surgical_3stage_DI_encoder surgical_noDI_encoder \
               surgical_3stage_DI_head surgical_noDI_head pretrain_encoder pretrain_head frozen; do
        [[ "$n" == *"_$arm" ]] && { echo "${n%_$arm} $arm"; return; }
    done
    echo "$n "                                  # no arm suffix (unexpected) → empty arm
}
frozen_ckpt_for() {                             # external frozen ckpt; "" for HF-loaded kinds
    case "$1" in
        vjepa_2_1_vitG)       echo "$ENCODER_CKPT" ;;                       # checkpoints/vjepa2_1_vitG_384.pt
        vjepa_2_1_vitg)       echo "checkpoints/vjepa2_1_vitg_384.pt" ;;    # VERIFY-FIRST ckpt name
        vjepa_2_1_vitL)       echo "checkpoints/vjepa2_1_vitL_384.pt" ;;    # VERIFY-FIRST
        vjepa_2_0_vitg)       echo "checkpoints/vjepa2_0_vitg_384.pt" ;;    # VERIFY-FIRST
        vjepa_2_0_vitg_ssv2)  echo "checkpoints/vjepa2_0_vitg_384_ssv2.pt" ;; # VERIFY-FIRST
        vjepa_1_vitL)         echo "checkpoints/vjepa1_vitL_16.pt" ;;       # VERIFY-FIRST
        vjepa_1_vitH)         echo "checkpoints/vjepa1_vitH_16.pt" ;;       # VERIFY-FIRST
        *) echo "" ;;          # vjepa_2_vitL_256 / ijepa_* / lejepa_* / dinov2 → HF model_id in registry
    esac
}
encoder_ckpt_for() {                            # encoder-only — Stages 2/3
    local bb arm; read -r bb arm <<<"$(_split_enc "$1")"
    if [ "$arm" = frozen ]; then frozen_ckpt_for "$bb"; return; fi
    echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/$(_arm_dir "$arm")/student_encoder.pt"
}
motion_aux_head_for() {                         # head-vs-encoder paired-Δ; frozen → ""
    local bb arm; read -r bb arm <<<"$(_split_enc "$1")"
    [ "$arm" = frozen ] && { echo ""; return; }
    echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/$(_arm_dir "$arm")/motion_aux_head.pt"
}
encoder_predictor_ckpt_for() {                  # encoder+predictor — Stage 8 future_mse
    local bb arm d; read -r bb arm <<<"$(_split_enc "$1")"
    if [ "$arm" = frozen ]; then frozen_ckpt_for "$bb"; return; fi   # Meta predictor rides in frozen .pt
    d="$(_arm_dir "$arm")"
    case "$d" in
        m09a_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09a_ckpt_best.pt" ;;
        m09c_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09c_ckpt_best.pt" ;;
        *)      echo "" ;;
    esac
}
pretrain_cleanup_get_latest() {                 # _latest.pt to drop before Stage 3 (trainable arms)
    local bb arm d; read -r bb arm <<<"$(_split_enc "$1")"; d="$(_arm_dir "$arm")"
    case "$d" in
        m09a_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09a_ckpt_latest.pt" ;;
        m09c_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09c_ckpt_latest.pt" ;;
        *)      echo "" ;;
    esac
}
```

ENCODERS default (L152) — keep ViT-G as the back-compat default, document the full roster as
an override:
```bash
ENCODERS="${ENCODERS:-vjepa_2_1_vitG_frozen vjepa_2_1_vitG_pretrain_encoder vjepa_2_1_vitG_pretrain_2X_encoder vjepa_2_1_vitG_surgical_3stage_DI_encoder vjepa_2_1_vitG_surgical_noDI_encoder vjepa_2_1_vitG_pretrain_head vjepa_2_1_vitG_surgical_3stage_DI_head vjepa_2_1_vitG_surgical_noDI_head}"
# full iter17 sweep: prefix the 8 arms with each trainable backbone + append frozen-only baselines
# (see §1). Run subsets via ENCODERS="vjepa_2_1_vitL_frozen vjepa_2_1_vitL_pretrain_encoder …".
```

NOTE: the per-encoder predictor-stage guards already gate on `[[ "$ENC" == vjepa* ]]` (Stage
8b/8c) — extend to also honour `kind` (§C3). The shared probe roots (`OUTPUT_ACTION`,
`OUTPUT_TAXONOMY`, `OUTPUT_COS`, `OUTPUT_MSE`) at L131-149 are UNCHANGED (dataset-level).

### A3 · Migration (mv — never rm; user runs/confirms; expensive ckpts)

```bash
# Move the existing flat ViT-G train outputs under the uniform backbone dir.
for M in sanity poc; do
  for ARM in m09a_pretrain_encoder m09a_pretrain_2X_encoder m09a_pretrain_head \
             m09c_surgery_3stage_DI_encoder m09c_surgery_noDI_encoder \
             m09c_surgery_3stage_DI_head m09c_surgery_noDI_head; do
    SRC="outputs/$M/$ARM"
    [ -d "$SRC" ] && mkdir -p "outputs/$M/vjepa_2_1_vitG" && mv "$SRC" "outputs/$M/vjepa_2_1_vitG/$ARM"
  done
done
# outputs/full/ has no m09 train outputs yet (verified). Shared probe_*/ dirs stay at root.
```
Apply the A1+A2 edits **and** this migration together, then SANITY-smoke (no real run between).

═══════════════════════════════════════════════════════════════════════════════
WORKSTREAM B — cross-arch enablement (the 3 Python edits + model configs)
═══════════════════════════════════════════════════════════════════════════════

### B1 · `src/utils/vjepa2_imports.py` — add arch constructors (after L155)

```python
# ── V-JEPA 1.x / 2.0 large+huge (frozen baselines; base module, no deep supervision) ──
def get_vit_large():
    _ensure_loaded_base()
    return sys.modules["src.models.vision_transformer"].vit_large
def get_vit_huge():
    _ensure_loaded_base()
    return sys.modules["src.models.vision_transformer"].vit_huge
```
Extend the `get_vit_by_arch` dispatch dict (L148-151):
```python
    dispatch = {
        "vit_giant_xformers": get_vit_giant_xformers,
        "vit_gigantic_xformers": get_vit_gigantic_xformers,
        "vit_large": get_vit_large,        # V-JEPA 1 ViT-L, V-JEPA 2.0 ViT-L (frozen)
        "vit_huge": get_vit_huge,          # V-JEPA 1 ViT-H (frozen)
    }
```
**VERIFY-FIRST (gates the 2.1 SCALE axis):** confirm whether the 2.1 app module
(`app/vjepa_2_1/models/vision_transformer`) exposes `vit_giant_xformers` / `vit_large_xformers`
AND whether `vjepa2_1_vit{g,L}_384.pt` ckpts exist:
```bash
python - <<'PY'
import importlib, glob
m = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
print("2.1 constructors:", [n for n in dir(m) if n.startswith("vit_")])
print("ckpts:", glob.glob("checkpoints/vjepa2_1_vit*_384.pt"))
PY
```
If 2.1 g/L are NOT released → the scale axis must use **2.0 g/L** (which DO exist, `vit_large`/
`vit_giant_xformers`) and the trainable-cross-arch set becomes {2.1 G, 2.0 g, 2.0 L} (scale axis
then carries a 2.0-vs-2.1 confound — note it in the hero). Do not fabricate a 2.1 g/L config.

### B2 · `src/utils/frozen_features.py` — generalize the loader + add ijepa dispatch

```diff
@@ L137 generalize: read arch/crop/embed_dim from the encoder's OWN registry row
-def load_vjepa_2_1_frozen(ckpt_path: Path, num_frames: int):
-    ...
-    enc = ENCODERS["vjepa_2_1_frozen"]
+def load_vjepa_frozen(ckpt_path: Path, num_frames: int, encoder_name: str):
+    """Load ANY native V-JEPA frozen encoder by registry name (arch from ENCODERS[name])."""
+    if not ckpt_path.exists():
+        sys.exit(f"FATAL: encoder ckpt not found: {ckpt_path}")
+    enc = ENCODERS[encoder_name]
     crop = enc["crop"]
-    print(f"Loading V-JEPA 2.1 ViT-G ({enc['arch']}, crop={crop}, T={num_frames}) ...")
+    print(f"Loading V-JEPA frozen {encoder_name} ({enc['arch']}, crop={crop}, T={num_frames}) ...")
     ...
     vit_constructor = get_vit_by_arch(enc["arch"])
     model = vit_constructor(img_size=(crop, crop), patch_size=PATCH_SIZE, num_frames=num_frames,
                             tubelet_size=TUBELET_SIZE, use_sdpa=True, use_silu=False,
                             wide_silu=True, uniform_power=False, use_rope=True)
     ...
     return model, crop, enc["embed_dim"]
+
+# back-compat alias (existing call sites that pass no name resolve to the G frozen)
+def load_vjepa_2_1_frozen(ckpt_path, num_frames):
+    return load_vjepa_frozen(ckpt_path, num_frames, "vjepa_2_1_frozen")
```
**VERIFY-FIRST:** the constructor kwargs above (`use_silu=False, wide_silu=True, use_rope=True`)
are the 2.0/2.1 xformers family signature. For V-JEPA-1 `vit_large`/`vit_huge` (base module),
confirm the constructor accepts them; if not, branch the kwargs by arch family (read a
`ctor_kwargs` block from the registry/model-config rather than hardcoding). Also confirm v1
ckpt key — extend `resolve_encoder_state_dict` (L111) fallback list with `"module.encoder"` if
the facebookresearch/jepa ckpt uses it.

Add the ijepa forward branch in `_flush_batch` (L330-333) — lazy import avoids the
ijepa_features↔frozen_features circular import:
```diff
             if encoder_kind == "vjepa":
                 feats = forward_vjepa(model, sub)
+            elif encoder_kind == "ijepa":
+                from utils.ijepa_features import forward_ijepa
+                feats = forward_ijepa(model, sub, num_frames)
             else:
                 feats = forward_dinov2(model, sub, num_frames)
```
(`extract_features_for_keys` L350 already threads `encoder_kind` through to `_flush_batch` and
sizes the batch on `kind == "vjepa"` at L441 — add `or kind == "ijepa"` there so image JEPAs use
the larger per-frame batch path; verify VRAM in the smoke.)

### B3 · `src/utils/predictor_eval.py` — model-config as a parameter (not a module const)

The module-level constants (L45-54) are ViT-G. Make the loaders read a per-call config so
m12d/m12e/m12f work on any trainable backbone:
```diff
@@ L71 load_encoder_only — accept model_cfg, build the right ViT + hierarchical count
-def load_encoder_only(ckpt_path, num_frames):
+def load_encoder_only(ckpt_path, num_frames, model_cfg=None):
+    mc = get_model_config(model_cfg)["model"]
+    crop, patch, tub = mc["crop_size"], mc["patch_size"], mc["tubelet_size"]
+    embed_dim, arch = mc["embed_dim"], mc["arch"]
     ...
-    encoder = get_vit_gigantic_xformers()(img_size=(CROP, CROP), patch_size=PATCH_SIZE, ...)
+    encoder = get_vit_by_arch(arch)(img_size=(crop, crop), patch_size=patch, num_frames=num_frames,
+                                    tubelet_size=tub, use_sdpa=True, use_silu=False,
+                                    wide_silu=True, uniform_power=False, use_rope=True)
     ...
-    if embed_dim_concat != ENCODER_EMBED_DIM * 4:
+    n_distill = mc["n_output_distillation"]          # 2.1=4 ; 2.0=1 (no deep supervision)
+    if embed_dim_concat != embed_dim * n_distill:
         sys.exit(...)
```
`load_encoder_predictor` likewise reads `pred_embed_dim/pred_depth/pred_num_heads/
num_mask_tokens` from `mc`. Thread a `--model-config` arg through `m12d_future_mse.py`,
`m12e_predictor_temporal.py`, `m12f_encoder_temporal.py` (run_eval passes the backbone's config
— see C below). Keep `model_cfg=None` default = `vjepa2_1.yaml` (back-compat for ViT-G).
**VERIFY-FIRST:** 2.0 ViT-g has `n_output_distillation` absent / =1 and a shallower predictor;
confirm `m12e`'s hierarchical-concat + rollout logic is valid for 2.0 (it may be 2.1-only — if
so, gate predictor-temporal to 2.1 backbones and mark 2.0 cells "N/A (no deep-supervision pred)").

### B4 · NEW model configs (clones of `configs/model/vjepa2_1.yaml`)

`configs/model/vjepa_2_1_vitL.yaml` (illustrative — **VERIFY-FIRST** the predictor dims +
checkpoint URL against the actual Meta 2.1 ViT-L release before training):
```yaml
model:
  version: "2.1"
  arch: vit_large_xformers      # VERIFY constructor exists in 2.1 app module (B1 gate)
  embed_dim: 1024
  depth: 24
  num_heads: 16
  mlp_ratio: 4.0                # VERIFY
  pred_depth: 12                # VERIFY (2.1-L predictor)
  pred_embed_dim: 384           # VERIFY
  pred_num_heads: 12            # VERIFY
  num_mask_tokens: 2
  zero_init_mask_tokens: true
  n_output_distillation: 4      # 2.1 deep supervision
  use_rope: true
  use_activation_checkpointing: true
  loss_exp: 1.0
  predict_all: true
  weight_distance_loss: true
  crop_size: 384
  patch_size: 16
  tubelet_size: 2
  hf_model_id: null
  checkpoint_url: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitL_384.pt   # VERIFY
  checkpoint_path: checkpoints/vjepa2_1_vitL_384.pt
  min_student_load_pct: 90
  min_predictor_load_pct: 50
```
`vjepa_2_1_vitg.yaml` = same template with `arch: vit_giant_xformers, embed_dim 1408, depth 40,
num_heads 22`. `vjepa_2_0_vitg.yaml` = `version "2.0", arch vit_giant_xformers, 1408/40/22,
pred_depth 12, n_output_distillation: 1, predict_all: false, weight_distance_loss: false,
checkpoint facebook/vjepa2-vitg-fpc64-384` (hf_model_id set, native build via
`get_vit_giant_xformers`). Surgery freeze auto-scales by depth (§E) — no recipe edit.

### B5 · `configs/eval/probe_encoders.yaml` — trainable-backbone rows

Per trainable backbone × 8 arms; all arms of a backbone share kind/arch/crop/embed_dim (only
the ckpt — resolved in run_eval — differs). Pattern (ViT-L shown; repeat for vitg/2.0_vitg):
```yaml
  vjepa_2_1_vitL_frozen:                  {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_pretrain_encoder:        {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_pretrain_2X_encoder:     {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_pretrain_head:           {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_surgical_3stage_DI_encoder: {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_surgical_noDI_encoder:   {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_surgical_3stage_DI_head: {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
  vjepa_2_1_vitL_surgical_noDI_head:      {kind: vjepa, arch: vit_large_xformers, crop: 384, embed_dim: 1024}
```
(Rename the live 8 `vjepa_2_1_*` rows → `vjepa_2_1_vitG_*` to match §1; keep one
`vjepa_2_1_frozen` alias only if any legacy caller still needs it.)

═══════════════════════════════════════════════════════════════════════════════
WORKSTREAM C — frozen-only baselines (eval-only)
═══════════════════════════════════════════════════════════════════════════════

### C1 · `configs/eval/probe_encoders.yaml` — baseline rows

```yaml
  # native V-JEPA family (kind=vjepa, loaded by load_vjepa_frozen via get_vit_by_arch)
  vjepa_2_0_vitg_frozen:       {kind: vjepa, arch: vit_giant_xformers, crop: 384, embed_dim: 1408}
  vjepa_2_0_vitg_ssv2_frozen:  {kind: vjepa, arch: vit_giant_xformers, crop: 384, embed_dim: 1408}
  vjepa_1_vitL_frozen:         {kind: vjepa, arch: vit_large, crop: 224, embed_dim: 1024}   # VERIFY kwargs/keys
  vjepa_1_vitH_frozen:         {kind: vjepa, arch: vit_huge,  crop: 224, embed_dim: 1280}   # VERIFY
  vjepa_2_vitL_256_frozen:     {kind: vjepa, arch: vit_large, crop: 256, embed_dim: 1024}   # VERIFY (or kind=hf)
  # image JEPAs (kind=ijepa, loaded by the EXISTING src/utils/ijepa_features.py)
  ijepa_vitH14:  {kind: ijepa, model_id: facebook/ijepa_vith14_1k,  crop: 224, embed_dim: 1280}
  ijepa_vitG16:  {kind: ijepa, model_id: facebook/ijepa_vitg16_22k, crop: 224, embed_dim: 1408}
  lejepa_vitH14: {kind: ijepa, model_id: <HF asset>,                crop: 224, embed_dim: 1280}  # VERIFY id
```

### C2 · ijepa dispatch wiring (mirror B2 at every `enc_kind` site)

`src/utils/ijepa_features.py` ALREADY provides `load_ijepa_frozen(enc_name)` + `forward_ijepa`
(built, GPU-untested). In `m12a_action_top1.py` extend the dispatch (L277-283; **repeat the
identical block at L373/420/434/480 and in `m12b_motion_cos.py` / `m12c_taxonomy_f1.py`**):
```diff
     if enc_kind == "vjepa":
-        model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
+        model, crop, embed_dim = load_vjepa_frozen(args.encoder_ckpt, args.num_frames, args.encoder)
+    elif enc_kind == "ijepa":
+        from utils.ijepa_features import load_ijepa_frozen
+        model, _proc, crop, embed_dim = load_ijepa_frozen(args.encoder)
     elif enc_kind == "dinov2":
         model, _processor, crop, embed_dim = load_dinov2_frozen()
     else:
         sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")
```
The `kind == "vjepa" and args.encoder_ckpt is None` guard (L263/373) stays — ijepa/dinov2 need
no `--encoder-ckpt` (HF model_id from the registry).

### C3 · predictor-stage N/A guard (run_eval.sh)

Stages 8 (future_mse) / 8b (m12e) / 9b need a predictor. Image JEPAs have none. Replace the
existing `[[ "$ENC" == vjepa* ]]` gate with a kind-aware helper:
```bash
_has_predictor() {     # 1 if the encoder can run predictor metrics
    local k; k="$(scripts/lib/yaml_extract.py configs/eval/probe_encoders.yaml "encoders.$1.kind")"
    [ "$k" = vjepa ]   # ijepa/dinov2 → no predictor
}
# Stage 8 / 8b / 9b:  if _has_predictor "$ENC"; then … ; else log "  [N/A] $ENC has no predictor — skip"; fi
```
**VERIFY-FIRST:** confirm V-JEPA-1/2.0 frozen ckpts actually carry a predictor with compatible
dims before enabling Stage 8/8b for them; if not, treat them as predictor-N/A too (hero N/A
cells, matching plan_model.md §G).

### C4 · checkpoint acquisition

Add download notes per config/registry (URLs in plan_model.md §A footnotes). HF kinds
(`ijepa_*`, `dinov2`, `vjepa_2_vitL_256`) need no local .pt — `from_pretrained(model_id)` pulls
to HF_HOME. Native V-JEPA frozens need their `.pt` in `checkpoints/`. Verify each loads:
```bash
python - <<'PY'  # per native frozen baseline
from utils.frozen_features import load_vjepa_frozen
m,crop,d = load_vjepa_frozen("checkpoints/vjepa2_0_vitg_384.pt", 16, "vjepa_2_0_vitg_frozen")
print("OK", crop, d)
PY
```

═══════════════════════════════════════════════════════════════════════════════
§E · Surgery freeze — NO CODE (verified depth-agnostic)
═══════════════════════════════════════════════════════════════════════════════
`m09c1_surgery_encoder.py:763` reads `depth=cfg["model"]["depth"]`; `:1150`
`n_trainable=int(depth*stage_cfg["unfreeze_below"])`. Fractions live in
`configs/train/surgery_*.yaml`. vitL(24)/vitg(40)/vitG(48)/2.0g(40) auto-scale. Zero edit.

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION (build-only; SANITY; user executes full runs)
═══════════════════════════════════════════════════════════════════════════════
```bash
# 0. static — after every edit
ruff check --select F,E9 src/ ; bash -n scripts/run_train.sh scripts/run_eval.sh ; shellcheck scripts/*.sh
# A. namespace + migration (no real run)
ENCODERS=vjepa_2_1_vitG_frozen ./scripts/run_eval.sh --SANITY   # resolvers hit outputs/sanity/vjepa_2_1_vitG/…
# B1 gate. confirm 2.1 g/L constructors + ckpts exist (script in §B1) BEFORE writing those configs
# B. cross-arch trainable (smallest 1-arm SANITY)
BACKBONE=vjepa_2_1_vitL ./scripts/run_train.sh pretrain_encoder --SANITY   # ckpt loads @24-blk/1024, no shape crash
ENCODERS=vjepa_2_1_vitL_frozen ./scripts/run_eval.sh --SANITY              # loader builds vit_large_xformers, probe head=1024
# C. frozen baselines
ENCODERS="vjepa_2_0_vitg_frozen ijepa_vitH14" ./scripts/run_eval.sh --SANITY  # kind=vjepa + kind=ijepa + N/A guard
# DO NOT launch POC/FULL train or the multi-backbone sweep — hand back to user.
```

═══════════════════════════════════════════════════════════════════════════════
SEQUENCING
═══════════════════════════════════════════════════════════════════════════════
A (namespace+migration, unblocks all) → B1 gate (verify 2.1 g/L) → B2/B3 (loader + predictor
param) → B4/B5 (configs+registry, vitL first) → C (frozen baselines, parallel to B; no train
cost). Author all infra now; user runs each (backbone × arm) train/eval sequentially, debugging
per-backbone. Predictor-temporal (m12e) confined to 2.1 trainable backbones until the 2.0
hierarchical question (B3 VERIFY-FIRST) is resolved.
