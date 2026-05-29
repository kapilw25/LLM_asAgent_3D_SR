#!/usr/bin/env bash
# iter13 probe orchestrator — 3 modules × {V-JEPA, DINOv2} × stages.
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# and "DELETE PROTECTION — shells stay THIN, .py owns the cache-policy prompt".
#
# Pipeline (priority 1 — frozen V-JEPA vs frozen DINOv2 on Indian action probe
# applied to data/eval_10k_local):
#   STAGE 1   m04e_action_labels.py (labels)           (CPU, ~1 min — ANNOTATION band)
#   STAGE 2   m12a_action_top1.py --stage features     (GPU × 2 encoders, ~1 h)
#   STAGE 3   m12a_action_top1.py --stage train        (GPU × 2 encoders, ~30 min)
#   STAGE 11  m12c_taxonomy_f1.py --stage train        (GPU × 16 dims × N enc, ~30 min) [iter13]
#   STAGE 4   m12a_action.py     --stage paired_delta  (CPU, ~5 min)  ← P1 GATE
#   STAGE 12  m12c_taxonomy_f1.py --stage paired_delta (CPU, ~5 min, BCa across 16 dims) [iter13]
#   STAGE 13  m12c_taxonomy_f1.py --stage plot         (CPU, ~5s) [iter13]
#   STAGE 5   m12b_motion_cos.py  --stage features      (CPU mean-pool × 2 enc)
#   STAGE 6   m12b_motion_cos.py  --stage cosine        (CPU × 2 enc)
#   STAGE 7   m12b_motion_cos.py  --stage paired_delta  (CPU)
#   STAGE 8   m12d_future_mse.py  --stage forward       (GPU, V-JEPA only, ~30 min)
#   STAGE 9   m12d_future_mse.py  --stage paired_per_variant   (CPU)
#   STAGE 10  m13_eval_plot.py                                  (CPU, ~5s — plots)
#
# iter13 (2026-05-05): stages 11/12/13 added so the 16-dim taxonomy gets a
# proper paper-final eval pass (was previously train-time-only as multi-task aux).
#
# REFERENCES
#   plan_code_dev.md  — per-module specs + LoC budget
#   runbook.md        — full set of pre/post-flight one-liners
#
# USAGE
#   tmux new -s probe
#
#   # FULL run (default) — eval_10k (~9.9k clips), ~4h on 24GB / ~2.5h on 96GB
#   ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log
#
#   # SANITY smoke test — 150 stratified clips (50/class) from THE SAME eval_10k
#   # JSON, processed against the SAME eval_10k_local/ TARs. ~6-8 min on 24GB.
#   # Outputs sandboxed to outputs/sanity/. Pass --sanity OR set MODE=SANITY.
#   ./scripts/run_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log
#
#   # Bypass prompts (overnight / non-TTY)
#   CACHE_POLICY_ALL=1 ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # keep all caches
#   CACHE_POLICY_ALL=2 ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # recompute everything
#   CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log  # SANITY + recompute
#
#   # Skip a stage (resume after failure):
#   SKIP_STAGES="1,2" ./scripts/run_eval.sh
#   # Run only one variant (debug):
#   ENCODERS="vjepa_2_1_frozen" ./scripts/run_eval.sh
#   # Tune SANITY subset size (default 50 clips per class):
#   SANITY_N_PER_CLASS=20 ./scripts/run_eval.sh --sanity
#
#   # Probe-training knobs (Stage 3) — mode-aware defaults:
#   #   SANITY: EPOCHS=50  WARMUP_PCT=0.10  LR_SWEEP="5e-4"   (current behavior)
#   #   FULL:   EPOCHS=20  WARMUP_PCT=0.0   LR_SWEEP="5e-4"   (Meta-faithful single-LR)
#   # Override any of them via env vars:
#   EPOCHS=20 WARMUP_PCT=0.0 ./scripts/run_eval.sh --sanity   # apply Meta recipe to sanity
#
#   # Paper-faithful FULL with LR sweep (matches Meta's multihead 5-LR setup;
#   # ~4x stage-3 wall, but mirrors deps/vjepa2/configs/eval/vitg-384/ssv2.yaml):
#   LR_SWEEP="1e-4 3e-4 1e-3 3e-3" ./scripts/run_eval.sh
#   # Each LR trains a probe under outputs/full/probe_action/<encoder>/lr_<LR>/;
#   # the best-val-acc LR is symlinked to the canonical <encoder>/probe.pt path so
#   # Stage 4 paired_delta reads the winner (no code change needed).

# Fail-fast: probe stages have sequential dependencies (Stage N+1 needs N's outputs),
# so any stage failure → abort chain to avoid wasting GPU time on impossible
# downstream work. Differs from run_paired_eval_10k.sh which uses `set -uo` only
# because its variants are independent (errors_N_fixes #72).
set -euo pipefail
# Clear ERR trap surfaces line + exit code BEFORE the shell exits.
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_eval.sh aborted at line $LINENO (exit=$rc) — sequential dependency failed; downstream stages skipped" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

# ── Mode detection (CLI flag wins; falls back to MODE env-var; default FULL) ──
MODE="${MODE:-FULL}"
for arg in "$@"; do
    case "$arg" in
        --sanity|--SANITY) MODE="SANITY" ;;
        --poc|--POC)       MODE="POC" ;;
        --full|--FULL)     MODE="FULL" ;;
    esac
done
case "$MODE" in
    SANITY|POC|FULL) ;;
    *) echo "FATAL: MODE must be SANITY|POC|FULL (got: $MODE)" >&2; exit 2 ;;
esac

# ── Mode-gated defaults — read from configs/pipeline.yaml ──────────────
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# + "no hardcoded values" — probe-head training defaults (epochs, warmup_pct,
# lr_sweep) live in configs/pipeline.yaml under the probe_head_train block.
# Env vars EPOCHS / WARMUP_PCT / LR_SWEEP still override (overnight scripts +
# ad-hoc sweeps). Mode key is lower-case ('sanity'/'poc'/'full') matching the
# yaml structure. Single-element lr_sweep arrays render as "5.0e-4" → space-
# joined into a normal LR_SWEEP string for the for-loop below.
PIPELINE_YAML="configs/pipeline.yaml"
EX="scripts/lib/yaml_extract.py"
mode_key=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
DEFAULT_EPOCHS=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.epochs")
DEFAULT_WARMUP_PCT=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.warmup_pct")
DEFAULT_LR_SWEEP=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.lr_sweep")
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
WARMUP_PCT="${WARMUP_PCT:-$DEFAULT_WARMUP_PCT}"
LR_SWEEP="${LR_SWEEP:-$DEFAULT_LR_SWEEP}"

# iter16 M9 (2026-05-21): all 3 modes feed off the master manifest from
# pipeline.yaml > data.local_data_dir + data.master_manifest_name. M1
# Option X's subsample_manifest_for_mode helper (called inside probe_action /
# probe_labels) applies per-mode subsampling in-memory (sorted[:N] for SANITY,
# stratified_by_motion_class for POC, identity for FULL). No more per-mode
# pre-made eval_10k_{sanity,poc}.json — retired to data/eval_10k_local/legacy/.
LOCAL_DATA_M9=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)
MASTER_MANIFEST_NAME_M9=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.master_manifest_name)
DEFAULT_EVAL_SUBSET="${LOCAL_DATA_M9}/${MASTER_MANIFEST_NAME_M9}"
if [ "$MODE" = "SANITY" ]; then
    DEFAULT_OUTPUT_PREFIX="outputs/sanity"
elif [ "$MODE" = "POC" ]; then
    DEFAULT_OUTPUT_PREFIX="outputs/poc"
else                                   # FULL
    DEFAULT_OUTPUT_PREFIX="outputs/full"
fi

# ── Configurables (env-overridable; mode-gated defaults) ────────────────
EVAL_SUBSET="${EVAL_SUBSET:-$DEFAULT_EVAL_SUBSET}"
LOCAL_DATA="${LOCAL_DATA:-$LOCAL_DATA_M9}"
TAGS_JSON="${TAGS_JSON:-${LOCAL_DATA}/tags.json}"
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/vjepa2_1_vitG_384.pt}"
OUTPUT_ACTION="${OUTPUT_ACTION:-${DEFAULT_OUTPUT_PREFIX}/probe_action}"
OUTPUT_COS="${OUTPUT_COS:-${DEFAULT_OUTPUT_PREFIX}/probe_motion_cos}"
OUTPUT_MSE="${OUTPUT_MSE:-${DEFAULT_OUTPUT_PREFIX}/probe_future_mse}"
# iter16 §3.3: temporal-metric suites (m12e predictor 6 + m12f encoder 4) — Stages 8b/8c/9b/9c.
OUTPUT_PREDTEMP="${OUTPUT_PREDTEMP:-${DEFAULT_OUTPUT_PREFIX}/predictor_temporal}"
OUTPUT_ENCTEMP="${OUTPUT_ENCTEMP:-${DEFAULT_OUTPUT_PREFIX}/encoder_temporal}"
PT_BATCH="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.predictor_temporal.batch_size)"
ET_BATCH="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.batch_size)"
ET_TUBELET="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.tubelet_size)"
ET_TOV_NPERM="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.tov_n_permutations)"
ET_PACE_STRIDES="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.pace_strides)"
ET_PACE_SRCFRAMES="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.pace_source_frames)"
ET_TCC_TEMP="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.tcc_temperature)"
ET_HEAD_LR="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.head_lr)"
ET_HEAD_EPOCHS="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.head_epochs)"
ET_HEAD_WD="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.head_weight_decay)"
ET_HEAD_BS="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.head_batch_size)"
ET_HEAD_TRAIN_CAP="$(scripts/lib/yaml_extract.py configs/pipeline.yaml probe.encoder_temporal.head_train_cap)"
OUTPUT_TAXONOMY="${OUTPUT_TAXONOMY:-${DEFAULT_OUTPUT_PREFIX}/probe_taxonomy}"
OUTPUT_PLOTS="${OUTPUT_PLOTS:-${DEFAULT_OUTPUT_PREFIX}/probe_plot}"
TAG_TAXONOMY="${TAG_TAXONOMY:-configs/tag_taxonomy.json}"
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain_encoder vjepa_2_1_pretrain_2X_encoder vjepa_2_1_surgical_3stage_DI_encoder vjepa_2_1_surgical_noDI_encoder vjepa_2_1_pretrain_head vjepa_2_1_surgical_3stage_DI_head vjepa_2_1_surgical_noDI_head}"
SKIP_STAGES="${SKIP_STAGES:-}"
NUM_FRAMES="${NUM_FRAMES:-16}"

# iter13 v12 (2026-05-05) / iter15 (2026-05-15): MOTION-flow probe class
# derivation knobs. `motion_features.npy` is m04d's RAFT optical-flow output
# (23D × N_clips post-Phase-0), durably stored under
# <local_data>/m04d_motion_features/ and binned into magnitude×direction = 8
# motion classes by probe_action --stage labels. The entire subdir
# (.npy + .paths.npy + .meta.json + .m04d_checkpoint.npz) rides on
# hf_outputs.py upload-data → fresh GPU instances download via download-data.
MOTION_FEATURES="${MOTION_FEATURES:-${LOCAL_DATA}/m04d_motion_features/motion_features.npy}"
# Mode-aware filter floors: FULL/POC use paper-grade thresholds (34 clips/class
# → ≥5 per split at 70/15/15). SANITY relaxes them since 150-clip stratified
# subsets give ~9 clips per motion-flow class (would crash stratified_split).
if [ "$MODE" = "SANITY" ]; then
    # iter13 v13 (2026-05-07): floor=3 with greedy split (utils/action_labels.py
    # stratified_split). Keeps every motion-flow class with n≥3 → harder probe →
    # better signal for the paper goal `surgery > pretrain_encoder > frozen`. Mirror in
    # run_train.sh.
    DEFAULT_MIN_CLIPS_PER_CLASS=3
    DEFAULT_MIN_PER_SPLIT=1
else
    # iter17 (2026-05-27): POC + FULL read the SAME thresholds as the training
    # label bootstrap (pipeline.yaml probe_action_labels — single source). Was
    # hardcoded 10/2 (POC) / 34/5 (FULL), which diverged from training's yaml
    # 100/5 → eval rebuilt a 14-class label set vs the 11-class set the models
    # were trained on (motion_aux head mismatch) AND crashed the video-disjoint
    # split (val=5% needs ~5/0.05=100 clips/class). Single source = eval labels
    # match training labels by construction.
    DEFAULT_MIN_CLIPS_PER_CLASS=$(scripts/lib/yaml_extract.py configs/pipeline.yaml "probe_action_labels.min_clips_per_class.${MODE,,}")
    DEFAULT_MIN_PER_SPLIT=$(scripts/lib/yaml_extract.py configs/pipeline.yaml "probe_action_labels.min_per_split.${MODE,,}")
fi
MIN_CLIPS_PER_CLASS="${MIN_CLIPS_PER_CLASS:-$DEFAULT_MIN_CLIPS_PER_CLASS}"
MIN_PER_SPLIT="${MIN_PER_SPLIT:-$DEFAULT_MIN_PER_SPLIT}"

# Per-encoder checkpoint resolvers. Two functions because Stages 2/3 (probe-head
# training) need encoder-only ckpts, but Stage 8 (future_mse) also needs the
# predictor. m09a_pretrain_encoder / m09c_surgery_* write BOTH artifacts:
#   - student_encoder.pt    : encoder only      (export_student_for_eval)
#   - m09{a,c}_ckpt_best.pt : encoder+predictor (save_training_checkpoint full=True)
# Surgery has TWO variants — 3stage_DI (with interaction tubes) and noDI (without)
# — to test whether D_I helps; each writes its own dir.
# iter13 v13 T2-rename (2026-05-07): probe_pretrain → m09a_pretrain_encoder,
# probe_surgery_* → m09c_surgery_* (matches source-module naming + run_train.sh).
# iter17: encoder name = "<backbone>_<arm>". Parse once + map arm→on-disk m09 dir, so
# adding a backbone is config+registry only (no per-variant case growth). The legacy 8
# vitG encoders are renamed vjepa_2_1_vitG_<arm> (see ENCODERS default). frozen arm →
# external ckpt (frozen_ckpt_for); "" for HF-loaded kinds (ijepa/dinov2/2.0-HF).
_arm_dir() {                                    # arm suffix → m09 output dir name
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
_split_enc() {                                  # "<bb>_<arm>" → echoes "BACKBONE ARM" (longest arm match)
    local n="$1" arm bb
    for arm in pretrain_2X_encoder surgical_3stage_DI_encoder surgical_noDI_encoder \
               surgical_3stage_DI_head surgical_noDI_head pretrain_encoder pretrain_head frozen; do
        if [[ "$n" == *"_$arm" ]]; then
            bb="${n%_$arm}"
            [ "$bb" = vjepa_2_1 ] && bb=vjepa_2_1_vitG   # legacy vitG names (vjepa_2_1_<arm>) → uniform dir
            echo "$bb $arm"; return
        fi
    done
    echo "$n "
}
frozen_ckpt_for() {                             # external frozen ckpt; "" → HF model_id (registry)
    case "$1" in
        vjepa_2_1_vitG)       echo "$ENCODER_CKPT" ;;                        # checkpoints/vjepa2_1_vitG_384.pt
        vjepa_2_1_vitg)       echo "checkpoints/vjepa2_1_vitg_384.pt" ;;     # 2.1 ViT-g 1B (fbai vjepa2_1_vitg_384.pt)
        vjepa_2_1_vitL)       echo "checkpoints/vjepa2_1_vitl_dist_vitG_384.pt" ;;  # 2.1 ViT-L distilled-from-G
        vjepa_2_0_vitg)       echo "checkpoints/vjepa2_0_vitg_384.pt" ;;     # fbai vitg-384.pt (verified 40blk/1408)
        vjepa_2_vitL_256)     echo "checkpoints/vjepa2_0_vitl_256.pt" ;;     # fbai vitl.pt (2.0 ViT-L 256)
        vjepa_1_vitL)         echo "checkpoints/vjepa1_vitL_16.pt" ;;
        vjepa_1_vitH)         echo "checkpoints/vjepa1_vitH_16.pt" ;;
        *) echo "" ;;          # vjepa_2_vitL_256 / ijepa_* / lejepa_* / dinov2 → HF model_id in registry
    esac
}
encoder_ckpt_for() {                            # encoder-only — Stages 2/3
    local bb arm; read -r bb arm <<<"$(_split_enc "$1")"
    { [ "$arm" = frozen ] || [ -z "$arm" ]; } && { frozen_ckpt_for "$bb"; return; }   # frozen / bare HF name
    echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/$(_arm_dir "$arm")/student_encoder.pt"
}
# motion_aux head resolver (head-vs-encoder paired-Δ). Frozen / bare HF → "" → encoder-only path.
motion_aux_head_for() {
    local bb arm; read -r bb arm <<<"$(_split_enc "$1")"
    { [ "$arm" = frozen ] || [ -z "$arm" ]; } && { echo ""; return; }
    echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/$(_arm_dir "$arm")/motion_aux_head.pt"
}
encoder_predictor_ckpt_for() {                  # encoder+predictor — Stage 8 future_mse
    local bb arm d; read -r bb arm <<<"$(_split_enc "$1")"
    { [ "$arm" = frozen ] || [ -z "$arm" ]; } && { frozen_ckpt_for "$bb"; return; }
    d="$(_arm_dir "$arm")"
    case "$d" in
        m09a_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09a_ckpt_best.pt" ;;
        m09c_*) echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09c_ckpt_best.pt" ;;
        *)      echo "" ;;
    esac
}

T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }
should_skip() {
    local stage="$1"
    [[ ",${SKIP_STAGES}," == *",${stage},"* ]]
}

# ── Pre-flight ──────────────────────────────────────────────────────────
stamp "PRE-FLIGHT"

# V-JEPA 2.1 ViT-G ckpt sanity hint (~28 GB, comes from setup_env_uv.sh aria2c).
# Friendly ✓/✗ before the generic FATAL loop below — points user to the fix.
ls -lh "$ENCODER_CKPT" && echo "✓ ckpt present" || echo "✗ run setup_env_uv.sh first"

for f in "$EVAL_SUBSET" "$ENCODER_CKPT"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: required path missing: $f"
        exit 3
    fi
done
if [ ! -d "$LOCAL_DATA" ]; then
    echo "FATAL: --local-data dir missing: $LOCAL_DATA"
    exit 3
fi
if [ ! -f "$TAGS_JSON" ]; then
    echo "  WARN: $TAGS_JSON not found — probe_taxonomy stages (1/3.5/12/13) will be auto-skipped"
fi
# iter13 v12 (2026-05-05) / iter15 (2026-05-15): preflight m04d
# motion_features.npy. Required for Stage 1 (probe_action --stage labels)
# which derives motion-flow classes from RAFT optical-flow features. Run m04d
# once per local dataset (durable artifact stored under
# <local_data>/m04d_motion_features/, auto-uploaded by hf_outputs.py
# upload-data).
if [ ! -f "$MOTION_FEATURES" ]; then
    echo "❌ FATAL: motion_features.npy not found at: $MOTION_FEATURES" >&2
    echo "   Run m04d once for this dataset (durable artifact, ~30-60 min GPU):" >&2
    echo "     python -u src/m04d_motion_features.py --$MODE \\" >&2
    echo "         --subset $EVAL_SUBSET --local-data $LOCAL_DATA" >&2
    echo "     (writes to ${LOCAL_DATA}/m04d_motion_features/ by default)" >&2
    echo "   Or download via:  python -u src/utils/hf_outputs.py download-data" >&2
    exit 3
fi
MOTION_FEATURES_PATHS="${MOTION_FEATURES%.npy}.paths.npy"
if [ ! -f "$MOTION_FEATURES_PATHS" ]; then
    echo "❌ FATAL: motion_features.paths.npy not found at: $MOTION_FEATURES_PATHS (must be next to .npy)" >&2
    exit 3
fi
echo "  ✓ mode:                 $MODE"
echo "  ✓ eval_subset:          $EVAL_SUBSET"
echo "  ✓ local_data:           $LOCAL_DATA"
echo "  ✓ encoder_ckpt:         $ENCODER_CKPT  ($(du -h "$ENCODER_CKPT" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ motion_features:      $MOTION_FEATURES  ($(du -h "$MOTION_FEATURES" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ min_clips_per_class:  $MIN_CLIPS_PER_CLASS"
echo "  ✓ min_per_split:        $MIN_PER_SPLIT"
echo "  ✓ encoders:             $ENCODERS"
echo "  ✓ skip_stages:          ${SKIP_STAGES:-<none>}"
echo "  ✓ cache_policy:         ${CACHE_POLICY_ALL:-prompt-per-module}"
echo "  ✓ output_prefix:        $DEFAULT_OUTPUT_PREFIX"

# ── Cache-policy gather (UPFRONT — per scripts/delete_protection_for_each_variant.md) ──
# MUST: collect ALL decisions BEFORE compute starts. Mirrors run_train.sh:43-89.
# 3 prompts (one per output dir) instead of 9 .py-level input() prompts.
# Bypass: CACHE_POLICY_ALL=1|2 env. Non-TTY: silent default to 1.
declare -A POLICY
_check_and_prompt() {                  # $1=key  $2..=candidate cache paths/globs
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        # Use bash nullglob to expand $path; non-matching globs become empty
        # array, avoiding `compgen -G`'s non-zero exit which would trip set -e.
        shopt -s nullglob
        local matches=( $path )
        shopt -u nullglob
        if [ ${#matches[@]} -gt 0 ]; then
            found="${matches[0]}"
            break
        fi
    done
    if [ -z "$found" ]; then POLICY[$key]=1; return; fi
    if [ -n "${CACHE_POLICY_ALL:-}" ]; then
        POLICY[$key]=$CACHE_POLICY_ALL
        echo "  $key: cache at $found → policy=${POLICY[$key]} (CACHE_POLICY_ALL)"
        return
    fi
    if [ ! -t 0 ]; then
        POLICY[$key]=1
        echo "  $key: cache at $found → policy=1 (non-TTY default)"
        return
    fi
    local ans
    read -p "  $key cache at $found [1=keep / 2=recompute] (Enter=1): " ans
    case "${ans:-1}" in
        2|recompute) POLICY[$key]=2 ;;
        *)           POLICY[$key]=1 ;;
    esac
}

echo ""
echo "──────────────────────────────────────────────"
echo "run_probe cache-policy gather (UPFRONT — one prompt per module)"
echo "──────────────────────────────────────────────"
_check_and_prompt "probe_action" "${OUTPUT_ACTION}/*"
_check_and_prompt "probe_cos"    "${OUTPUT_COS}/*"
_check_and_prompt "probe_mse"    "${OUTPUT_MSE}/*"
P_ACTION="${POLICY[probe_action]:-1}"
P_COS="${POLICY[probe_cos]:-1}"
P_MSE="${POLICY[probe_mse]:-1}"
echo "  → ACTION=$P_ACTION  COS=$P_COS  MSE=$P_MSE"

# ── Pre-flight: drop P2/P3 encoders whose trainer outputs aren't on disk ─
# probe eval is the consumer; P2 (m09a continual SSL) and P3 (m09c surgery) are
# producers run separately via scripts/run_train.sh. If a producer hasn't
# run yet, silently dropping the encoder lets the rest of the pipeline complete
# (frozen + dinov2 always work — they're external ckpts).
echo ""
echo "──────────────────────────────────────────────"
echo "P2/P3 trainer-output pre-flight"
echo "──────────────────────────────────────────────"
NEW_ENCODERS=""
for ENC in $ENCODERS; do
    # iter17: parser-based. frozen/HF baselines (arm=frozen or bare HF name like dinov2/
    # ijepa_*) have NO trainer output → always pass. Trainable arms → require their
    # student_encoder.pt on disk (drop cleanly if a producer hasn't run for this backbone).
    read -r _bb _arm <<<"$(_split_enc "$ENC")"
    case "$_arm" in
        frozen|"")
            echo "  ✓ $ENC: external/HF frozen (no trainer needed)"
            ;;
        pretrain_encoder|pretrain_2X_encoder|surgical_3stage_DI_encoder|surgical_noDI_encoder|pretrain_head|surgical_3stage_DI_head|surgical_noDI_head)
            CKPT="$(encoder_ckpt_for "$ENC")"
            if [ ! -e "$CKPT" ]; then
                echo "  ⚠️  $ENC: $CKPT not found — train via:"
                echo "       BACKBONE=$_bb ./scripts/run_train.sh ${_arm/surgical_/surgery_} --$MODE"
                echo "  → dropping $ENC from this run; pipeline continues with remaining encoders"
                continue
            fi
            echo "  ✓ $ENC: $CKPT"
            ;;
        *)
            echo "  ⚠️  $ENC: unrecognized arm '$_arm' — dropping (check encoder name / registry)"
            continue
            ;;
    esac
    NEW_ENCODERS="$NEW_ENCODERS $ENC"
done
ENCODERS="$(echo "$NEW_ENCODERS" | xargs)"
if [ -z "$ENCODERS" ]; then
    echo "FATAL: ENCODERS list is empty after pre-flight — nothing to evaluate"
    exit 3
fi
echo "  → final ENCODERS: $ENCODERS"

# ── Pre-flight: Stage 8 needs predictor-bearing ckpt (NOT just encoder) ─
# Stages 2-7 use student_encoder.pt (encoder only); Stage 8 future_mse calls
# m12d_future_mse → predictor_eval.load_encoder_predictor which requires the "predictor" key —
# present only in m09{a,c}_ckpt_best.pt (save_training_checkpoint full=True).
# m09a's _best.pt was historically saved with full=False, so vjepa_2_1_pretrain_encoder
# may have student_encoder.pt but NOT m09a_ckpt_best.pt → Stage 8 in-loop
# FATAL'd in run_src_probe_sanity_v2.log line 778. Now: build a separate
# STAGE8_ENCODERS subset, drop variants missing the predictor ckpt with a
# clear warning. Stages 2-7 still include them (encoder-only is enough).
echo ""
echo "──────────────────────────────────────────────"
echo "Stage 8 predictor-ckpt pre-flight"
echo "──────────────────────────────────────────────"
STAGE8_NEW=""
for ENC in $ENCODERS; do
    [[ "$ENC" == vjepa* ]] || continue                         # DINOv2 has no predictor — skip
    PCKPT="$(encoder_predictor_ckpt_for "$ENC")"
    if [ -e "$PCKPT" ]; then
        echo "  ✓ $ENC: $PCKPT"
        STAGE8_NEW="$STAGE8_NEW $ENC"
    else
        echo "  ⚠️  $ENC: $PCKPT not found — Stage 8 will SKIP this encoder"
        case "$ENC" in
            vjepa_2_1_frozen)
                echo "       (frozen variant uses Meta's ckpt which always carries the predictor —"
                echo "        if this is missing, ENCODER_CKPT itself is wrong)" ;;
            vjepa_2_1_pretrain_encoder)
                echo "       Re-train (m09a_ckpt_best.pt is written via save_training_checkpoint full=True):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --$MODE" ;;
            vjepa_2_1_pretrain_2X_encoder)
                echo "       Re-train iter14 arm C (10 ep, ~20 GPU-h on FULL):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X_encoder --$MODE" ;;
            vjepa_2_1_surgical_*)
                echo "       Re-train (m09c writes m09c_ckpt_best.pt at end of surgery):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh ${ENC#vjepa_2_1_surgical_} --$MODE" ;;
        esac
    fi
done
STAGE8_ENCODERS="$(echo "$STAGE8_NEW" | xargs)"
if [ -z "$STAGE8_ENCODERS" ]; then
    echo "  → no V-JEPA variants have predictor ckpt; Stage 8/9 will be auto-skipped"
else
    echo "  → Stage 8 ENCODERS: $STAGE8_ENCODERS"
fi

# ── iter15 Phase 6 audit (2026-05-16): motion_aux head pre-flight ───────
# Mirrors the Stage 8 predictor-ckpt pre-flight. Reports which encoders have
# their motion_aux head present on disk (= will use augmented features for
# probe_action / probe_motion_cos paired-Δ). Frozen baseline correctly absent.
#
# iter15 Phase 7 (2026-05-17): split asymmetric handling for the two variant
# families:
#   - HEAD variants (*_head): motion_aux_head IS the sole trained artifact
#     (encoder is bit-identical Meta init). Missing head ⇒ variant is
#     meaningless to evaluate ⇒ DROP from ENCODERS (fail loud, no silent
#     fallback that would contaminate Δ5/Δ6/Δ7 paired tests with encoder-
#     only data masquerading as head-augmented).
#   - ENCODER variants (*_encoder): motion_aux_head is supplementary; the
#     encoder itself carries the trained signal. Missing head ⇒ WARN +
#     encoder-only fallback (preserves iter14 reproducibility).
echo ""
echo "──────────────────────────────────────────────"
echo "motion_aux_head pre-flight (head-vs-encoder paired-Δ readiness)"
echo "──────────────────────────────────────────────"
HEAD_AWARE_ENCODERS=""
NEW_ENCODERS=""
for ENC in $ENCODERS; do
    case "$ENC" in
        vjepa_2_1_frozen)
            echo "  ✓ $ENC: external frozen (no head — encoder-only path)"
            NEW_ENCODERS="$NEW_ENCODERS $ENC"
            continue
            ;;
    esac
    MA="$(motion_aux_head_for "$ENC")"
    if [ -e "$MA" ]; then
        echo "  ✓ $ENC: $MA"
        HEAD_AWARE_ENCODERS="$HEAD_AWARE_ENCODERS $ENC"
        NEW_ENCODERS="$NEW_ENCODERS $ENC"
    else
        case "$ENC" in
            vjepa_2_1_pretrain_head|vjepa_2_1_surgical_3stage_DI_head|vjepa_2_1_surgical_noDI_head)
                echo "  ❌ $ENC: $MA not found — head IS the trained artifact for *_head variants"
                echo "       → dropping $ENC (encoder-only fallback would contaminate Δ5/Δ6/Δ7 paired tests)"
                echo "       Re-train via: ./scripts/run_train.sh ${ENC#vjepa_2_1_} --$MODE"
                continue
                ;;
            *)
                echo "  ⚠️  $ENC: $MA not found — eval falls back to encoder-only for this variant"
                NEW_ENCODERS="$NEW_ENCODERS $ENC"
                ;;
        esac
    fi
done
ENCODERS="$(echo "$NEW_ENCODERS" | xargs)"
HEAD_AWARE_ENCODERS="$(echo "$HEAD_AWARE_ENCODERS" | xargs)"
if [ -z "$ENCODERS" ]; then
    echo "FATAL: ENCODERS list is empty after motion_aux_head pre-flight — nothing to evaluate"
    exit 3
fi
echo "  → head-augmented eval ENCODERS: ${HEAD_AWARE_ENCODERS:-<none>}"
echo "  → final ENCODERS after head preflight: $ENCODERS"

# ── Pre-eval pretrain_encoder-cleanup (iter13, 2026-05-05) ──────────────────────
# `m09a_ckpt_latest.pt` (~29 GB on ViT-G full=True) is a RESUME anchor for
# pretrain_encoder — it's only useful if pretrain_encoder crashes mid-run and needs to
# continue from the last step. Once pretrain_encoder has completed (proxy:
# student_encoder.pt is exported), latest.pt is dead weight on disk.
# Same for m09c_ckpt_latest.pt for surgery encoders.
# Deleting it here reclaims ~29 GB per pretrained encoder BEFORE eval starts —
# critical because Stage 3's transient train cache (~73 GB at fp16 / ~146 GB
# at fp32) would otherwise overflow our 199 GB disk budget.
# Opt-out: set EVAL_KEEP_LATEST=1 to skip this cleanup (e.g. during dev
# when you might want to resume a partial pretrain_encoder). Default: ON.
if [ "${EVAL_KEEP_LATEST:-0}" != "1" ]; then
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Pre-eval pretrain_encoder-cleanup (drop _latest.pt resume anchors)"
    echo "──────────────────────────────────────────────"
    pretrain_cleanup_get_latest() {
        # iter17: parser-based (per-backbone dir). Map encoder → its m09{a,c}_ckpt_latest.pt
        # (empty for frozen/head — only the trainable encoder arms write _latest.pt anchors).
        local bb arm d; read -r bb arm <<<"$(_split_enc "$1")"; d="$(_arm_dir "$arm")"
        case "$d" in
            m09a_pretrain_encoder|m09a_pretrain_2X_encoder)
                echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09a_ckpt_latest.pt" ;;
            m09c_surgery_3stage_DI_encoder|m09c_surgery_noDI_encoder)
                echo "${DEFAULT_OUTPUT_PREFIX}/${bb}/${d}/m09c_ckpt_latest.pt" ;;
            *) echo "" ;;
        esac
    }
    for ENC in $ENCODERS; do
        STUDENT_ENC="$(encoder_ckpt_for "$ENC")"
        # Only delete if pretrain_encoder/surgery has truly completed (student_encoder.pt
        # exists on the path encoder_ckpt_for resolves to). External (frozen)
        # encoders are skipped — their ckpt is shared and immutable.
        case "$ENC" in
            vjepa_2_1_frozen) continue ;;
        esac
        LATEST="$(pretrain_cleanup_get_latest "$ENC")"
        if [ -z "$LATEST" ] || [ ! -f "$LATEST" ]; then
            continue
        fi
        if [ ! -e "$STUDENT_ENC" ]; then
            echo "  [keep-latest] $ENC: student_encoder.pt missing → preserving $LATEST (pretrain_encoder not complete)"
            continue
        fi
        sz=$(du -h "$LATEST" 2>/dev/null | awk '{print $1}')
        rm -f "$LATEST"
        echo "  [pretrain_encoder-cleanup] removed $LATEST ($sz) — pretrain_encoder complete, latest.pt was resume-only"

        # iter13 (2026-05-05): also drop step ckpts (m09{a,c}_ckpt_step*.pt).
        # These are the keep_last_n=2 rotation buffers from training. Once
        # student_encoder.pt is exported and best.pt is preserved, the step
        # ckpts are dead weight — same student weights, no predictor (full=False).
        # Each step ckpt is ~7 GB at ViT-G; 2 of them = ~14 GB freed per encoder.
        # Opt-out via the same EVAL_KEEP_LATEST=1 env var as the latest.pt path.
        STEP_DIR="$(dirname "$LATEST")"
        for step_pt in "$STEP_DIR"/*ckpt_step*.pt; do
            if [ -f "$step_pt" ]; then
                step_sz=$(du -h "$step_pt" 2>/dev/null | awk '{print $1}')
                rm -f "$step_pt"
                echo "  [pretrain_encoder-cleanup] removed $(basename "$step_pt") ($step_sz) — step rotation buffer, training complete"
            fi
        done
    done
fi

# ── STAGE 1 — labels (action_probe + taxonomy, CPU, ~1-2 min) ──────────
# Two label artifacts emitted side-by-side from the same EVAL_SUBSET +
# TAGS_JSON so downstream stages and run_train.sh both find them.
#   - probe_action/action_labels.json   : 3-class action (P1 gate)
#   - probe_taxonomy/taxonomy_labels.json : 16 dims (action + 15 from
#                                           tag_taxonomy.json) — used by
#                                           m09a/m09c multi-task probe loss
#                                           when multi_task_probe.enabled=true
# Re-uses cache-policy P_ACTION (cheap CPU work; kept simple).
if ! should_skip 1; then
    stamp "STAGE 1 · action_probe + taxonomy labels (CPU, ~1-2 min)"
    # iter13 v12 (2026-05-05): probe_action labels derive MOTION-flow classes
    # from m04d's motion_features.npy (16 classes of magnitude×direction).
    # --tags-json is NOT passed any more — action labels no longer use VLM tags.
    python -u src/m04e_action_labels.py "--$MODE" \
        --eval-subset "$EVAL_SUBSET" \
        --motion-features "$MOTION_FEATURES" \
        --min-clips-per-class "$MIN_CLIPS_PER_CLASS" \
        --min-per-split "$MIN_PER_SPLIT" \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/m04e_action_labels.log
    if [ -f "$TAG_TAXONOMY" ]; then
        python -u src/m04f_taxonomy_labels.py "--$MODE" \
            --eval-subset "$EVAL_SUBSET" \
            --tags-json "$TAGS_JSON" \
            --tag-taxonomy "$TAG_TAXONOMY" \
            --output-root "$OUTPUT_TAXONOMY" \
            --cache-policy "$P_ACTION" \
            --no-wandb \
            2>&1 | tee logs/m04f_taxonomy_labels.log
    else
        echo "  WARN: $TAG_TAXONOMY missing — skipping probe_taxonomy labels."
        echo "    multi_task_probe in m09a/m09c will auto-disable for this run."
    fi
fi

# ── PER-ENCODER SEQUENTIAL PIPELINE (iter13, 2026-05-05) ───────────────
# Stages 2 + 3 + 5 + 6 + 8 fused into one per-encoder loop. The disk pressure
# during eval is dominated by `features_test.npy` (~31 GB at fp32, ~16 GB at
# fp16 per encoder) and Stage 3's transient train+val resume caches (~73 GB
# at fp16 per encoder). Running stages encoder-by-encoder lets Stage 5 free
# `features_test.npy` BEFORE the next encoder writes its own, so peak disk is
# bounded by ONE encoder's footprint instead of N_encoders × footprint.
# Stages 4/7/9/10 stay AFTER this loop — they aggregate small per-encoder
# result files (test_predictions.npy ~6 KB, per_clip_motion_cos.npy ~6 KB,
# per_clip_mse.npy ~6 KB) that comfortably fit on disk.
#
# Skip semantics: each stage's `should_skip N` is honoured INSIDE the per-
# encoder loop, so SKIP_STAGES="2,3" still works as expected.
#
# To restore the legacy 3-split layout (probe_taxonomy --stage train ad-hoc),
# pass `FEATURES_SPLITS="train val test"` env var.
FEATURES_SPLITS="${FEATURES_SPLITS:-test}"
# iter13 (2026-05-05): adaptive token pool. Default 16 (V-JEPA paper §4 attentive
# probe regime; 290× smaller .npy + 290× less probe attention compute). Set
# POOL_TOKENS=0 to disable (legacy 4608-token storage; will hit OOM at probe BS=64).
POOL_TOKENS="${POOL_TOKENS:-16}"
# Stream-and-discard mode (Stage 3 only). Skips ALL feature persistence; per-batch
# encoder→pool→head→loss→backward. Use STREAM_TRAIN=1 for >1M-clip datasets where
# even pooled features don't fit RAM. Single-LR only — overrides LR_SWEEP if set.
STREAM_TRAIN="${STREAM_TRAIN:-0}"
if [ "$STREAM_TRAIN" = "1" ]; then
    STREAM_FLAG="--stream-train"
else
    STREAM_FLAG=""
fi
n_lrs=$(echo "$LR_SWEEP" | wc -w)

PER_ENC_ANY=0
# iter16 §3.3: list ALL per-encoder stages so the loop isn't skipped when only a
# subset is requested. Was "2 3 5 6 8" — omitted 11 (taxonomy) AND the new 8b/8c
# (predictor/encoder temporal), so e.g. SKIP_STAGES=2,3,5,6,8 wrongly skipped them.
for s in 2 3 11 5 6 8 8b 8c; do should_skip "$s" || PER_ENC_ANY=1; done
if [ "$PER_ENC_ANY" -eq 1 ]; then
    stamp "PER-ENCODER pipeline (Stages 2/3/3.5/5/6/8/8b/8c) — ${ENCODERS//[^[:space:]]/x} encoders sequentially"
    # ── iter17: background aggregate-ETA heartbeat. Prints whole-pipeline progress
    # (tqdm-style bar + hh:mm ETA across ALL encoders) into THIS eval log every
    # ETA_HEARTBEAT_INTERVAL (default 120s), so no separate monitor script is needed —
    # the bulk/ETA shows up inline when you tail the eval log. Scrapes the freshest
    # eval log (= this run's outer-tee target). Self-terminates when run_eval exits
    # (the parent PID disappears). Best-effort: the `if …; then :; fi` guards keep it
    # alive (and set-e-safe) across transient parse failures on a partial log.
    _ETA_PARENT=$$
    _eta_heartbeat() {
        if python -u tests/eval_live_eta.py --reset >/dev/null 2>&1; then :; fi
        while kill -0 "$_ETA_PARENT" 2>/dev/null; do
            sleep "${ETA_HEARTBEAT_INTERVAL:-120}"
            local lg
            if lg=$(ls -t logs/iter15_v2_*eval*.log 2>/dev/null | head -1) \
                  && [ -n "$lg" ] && [ -n "$(find "$lg" -mmin -3 2>/dev/null)" ]; then
                if python -u tests/eval_live_eta.py --log "$lg" 2>/dev/null; then :; fi
            fi
        done
    }
    _eta_heartbeat &
    for ENC in $ENCODERS; do
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "ENCODER: $ENC  (Stages 2/3/5/6/8)"
        echo "════════════════════════════════════════════════════════════════"
        EXTRA_CKPT=""
        if [[ "$ENC" == vjepa* ]]; then
            CKPT="$(encoder_ckpt_for "$ENC")"
            [ -e "$CKPT" ] || { echo "FATAL: encoder ckpt missing for $ENC: $CKPT"; exit 3; }
            EXTRA_CKPT="--encoder-ckpt $CKPT"
        fi
        # iter15 Phase 6 audit (2026-05-16): motion_aux head flag. Empty for
        # frozen baseline / cells without a head → encoder-only path preserved.
        MA_CKPT="$(motion_aux_head_for "$ENC")"
        MA_FLAG=""
        if [ -n "$MA_CKPT" ] && [ -f "$MA_CKPT" ]; then
            MA_FLAG="--motion-aux-head $MA_CKPT"
            echo "  [motion_aux] head wired for $ENC: $MA_CKPT"
        fi

        # ─── Stage 2: features for this encoder ──────────────────────────
        if ! should_skip 2; then
            stamp "  STAGE 2 · features for $ENC (splits=${FEATURES_SPLITS}, dtype=fp16)"
            python -u src/m12a_action_top1.py "--$MODE" \
                --stage features \
                --encoder "$ENC" \
                $EXTRA_CKPT $MA_FLAG \
                --eval-subset "$EVAL_SUBSET" \
                --local-data "$LOCAL_DATA" \
                --output-root "$OUTPUT_ACTION" \
                --num-frames "$NUM_FRAMES" \
                --features-splits $FEATURES_SPLITS \
                --pool-tokens "$POOL_TOKENS" \
                --cache-policy "$P_ACTION" \
                --no-wandb \
                2>&1 | tee "logs/probe_action_features_${ENC}.log"
        fi

        # ─── Stage 3: probe head training (per-LR + select_best_lr) ──────
        # LR_SWEEP="5e-4" → single probe (canonical <enc>/probe.pt for Stage 4).
        # LR_SWEEP="1e-4 3e-4 1e-3 3e-3" → multihead sweep; best-val-acc LR
        #   is symlinked to canonical path. Matches Meta's eval/vitg-384/ssv2.yaml.
        # Lazy-extract: Stage 3 extracts train+val into transient .probe_features_
        # <split>_ckpt.npz cache; cache is reused across LR runs and cleaned at
        # end of run_select_best_lr_stage (multi-LR) or end of run_train_stage
        # (single-LR).
        if ! should_skip 3; then
            stamp "  STAGE 3 · train probe ($n_lrs LR(s); epochs=$EPOCHS warmup=$WARMUP_PCT) for $ENC"
            for LR in $LR_SWEEP; do
                if [ "$n_lrs" -gt 1 ]; then
                    SUBDIR_FLAG="--output-subdir lr_${LR}"
                    LOG_SUFFIX="_lr${LR}"
                else
                    SUBDIR_FLAG=""
                    LOG_SUFFIX=""
                fi
                # iter15 D7 fix (2026-05-16): thread $MA_FLAG so probe_action's
                # lazy-extract path (_load_or_extract) augments train+val to match
                # the already-augmented features_test.npy written by Stage 2. Without
                # this, AttentiveClassifier is built with embed_dim=1664 (lazy train)
                # then crashes at _eval_per_clip on the 1700-dim test split — see
                # logs/iter15_post_poc_eval_head_aware_20260516_012451.log:482.
                # iter15 D8 fix (2026-05-16): --keep-lazy-cache defers cleanup of
                # .probe_features_{train,val}_ckpt.npz so Stage 3.5 (probe_taxonomy)
                # can reuse them — saves ~7 min × N encoders of redundant re-extract.
                # Cleanup is invoked explicitly via `--stage cleanup_lazy` after Stage 3.5.
                python -u src/m12a_action_top1.py "--$MODE" \
                    --stage train \
                    --encoder "$ENC" \
                    $EXTRA_CKPT $MA_FLAG \
                    --eval-subset "$EVAL_SUBSET" \
                    --local-data "$LOCAL_DATA" \
                    --num-frames "$NUM_FRAMES" \
                    --output-root "$OUTPUT_ACTION" \
                    --epochs "$EPOCHS" \
                    --probe-lr "$LR" \
                    --warmup-pct "$WARMUP_PCT" \
                    --pool-tokens "$POOL_TOKENS" \
                    $STREAM_FLAG \
                    $SUBDIR_FLAG \
                    --keep-lazy-cache \
                    --cache-policy "$P_ACTION" \
                    --no-wandb \
                    2>&1 | tee "logs/probe_action_train_${ENC}${LOG_SUFFIX}.log"
            done
            if [ "$n_lrs" -gt 1 ]; then
                # iter15 D8 (2026-05-16): same --keep-lazy-cache contract — defer
                # cleanup so Stage 3.5 reuses the resume ckpt.
                python -u src/m12a_action_top1.py "--$MODE" \
                    --stage select_best_lr \
                    --encoder "$ENC" \
                    --output-root "$OUTPUT_ACTION" \
                    --keep-lazy-cache \
                    --cache-policy "$P_ACTION" \
                    --no-wandb \
                    2>&1 | tee "logs/probe_action_select_best_lr_${ENC}.log"
            fi
        fi

        # ─── Stage 3.5: probe_taxonomy --stage train (16-dim per-encoder) ─
        # iter13 (2026-05-05): the 16-dim taxonomy must be evaluated end-to-
        # end (train + val + test), not just used as multi-task aux loss
        # during m09a/m09c training. probe_taxonomy.run_train_stage trains
        # 16 attentive heads (1 single-label "action" + 15 multi-label tag
        # dims) on the SAME features that probe_action --stage train just
        # consumed. Lazy-extract logic in probe_taxonomy reuses the
        # .probe_features_<split>_ckpt.npz resume cache from probe_action's
        # extraction → near-zero re-extraction cost. Disabled by SKIP_STAGES
        # entry 11 (new stage number) or by --features-splits set excluding
        # train/val (taxonomy can't run without those).
        if ! should_skip 11 && [ -f "$TAG_TAXONOMY" ]; then
            stamp "  STAGE 3.5 · taxonomy probe (16 dims) for $ENC"
            # iter15 D7 fix (2026-05-16): same lazy-extract issue as Stage 3 —
            # probe_taxonomy._load_or_extract augments only when --motion-aux-head
            # is passed (D1 fix in src/probe_taxonomy.py). Without this flag the
            # taxonomy probe would crash with the same LN[1664] vs input[*,1700]
            # mismatch on the test split. Mirrors Stage 3 threading above.
            python -u src/m12c_taxonomy_f1.py "--$MODE" \
                --stage train \
                --encoder "$ENC" \
                $EXTRA_CKPT $MA_FLAG \
                --local-data "$LOCAL_DATA" \
                --num-frames "$NUM_FRAMES" \
                --features-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_TAXONOMY" \
                --epochs "$EPOCHS" \
                --probe-lr 5e-4 \
                --warmup-pct "$WARMUP_PCT" \
                --pool-tokens "$POOL_TOKENS" \
                --cache-policy "$P_ACTION" \
                --no-wandb \
                2>&1 | tee "logs/probe_taxonomy_train_${ENC}.log"
        fi

        # iter15 D8 fix (2026-05-16): clean up lazy-extract resume caches now
        # that Stage 3.5 (last consumer) is done. Stage 3 used --keep-lazy-cache
        # to defer this. UNCONDITIONAL (outside the Stage-3.5 conditional) so
        # SKIP_STAGES=11 (Stage 3.5 disabled) doesn't leak ~30 GB/encoder of
        # transient .probe_features_{train,val}_ckpt.npz at FULL scale.
        # Idempotent — no-op when files absent (e.g., Stage 3 also skipped).
        # CLAUDE.md: "All destructive deletes live in .py" — shell never rm's.
        python -u src/m12a_action_top1.py "--$MODE" \
            --stage cleanup_lazy \
            --encoder "$ENC" \
            --output-root "$OUTPUT_ACTION" \
            --cache-policy "$P_ACTION" \
            --no-wandb \
            2>&1 | tee "logs/probe_action_cleanup_lazy_${ENC}.log"

        # ─── Stage 5: motion_cos features (mean-pool features_test.npy) ──
        # Stage 5 is the LAST consumer of features_test.npy in normal flow,
        # but iter15 Phase 5 V6 (2026-05-15) MOVED the cleanup to after
        # Stage 8 so a Stage 8 crash doesn't force re-extraction on resume.
        # See end-of-per-encoder-loop cleanup block below.
        if ! should_skip 5; then
            stamp "  STAGE 5 · motion_cos features for $ENC"
            # iter15 Phase 6 audit (2026-05-16): $MA_FLAG threading — Path A
            # (--share-features, default) inherits augmentation from probe_action's
            # features_test.npy automatically. Flag is forwarded for Path B
            # (--no-share-features) parity + future-proofing.
            python -u src/m12b_motion_cos.py "--$MODE" \
                --stage features \
                --encoder "$ENC" \
                $MA_FLAG \
                --action-probe-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_COS" \
                --pool-tokens "$POOL_TOKENS" \
                --share-features \
                --cache-policy "$P_COS" \
                --no-wandb \
                2>&1 | tee "logs/probe_motion_cos_features_${ENC}.log"
        fi

        # ─── Stage 6: motion_cos cosine ───────────────────────────────────
        # Reads pooled_features_test.npy (~10 MB) emitted by Stage 5. Does NOT
        # read features_test.npy. Stage 6's cosine math runs on the mean-pooled
        # tensor only. (iter15 Phase 5 V6: features_test.npy now lives until
        # the post-Stage-8 cleanup below — irrelevant to Stage 6.)
        if ! should_skip 6; then
            stamp "  STAGE 6 · motion_cos cosine for $ENC"
            python -u src/m12b_motion_cos.py "--$MODE" \
                --stage cosine \
                --encoder "$ENC" \
                --action-probe-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_COS" \
                --cache-policy "$P_COS" \
                --no-wandb \
                2>&1 | tee "logs/probe_motion_cos_cosine_${ENC}.log"
        fi

        # ─── Stage 8: future_mse forward (V-JEPA + predictor only) ───────
        # Skipped for DINOv2 (no predictor) and for any V-JEPA variant whose
        # predictor-bearing ckpt (m09{a,c}_ckpt_best.pt) wasn't found in the
        # Stage 8 preflight (line ~298). Defense-in-depth: re-check inside the
        # loop and SKIP+continue (NOT FATAL) so SKIP_STAGES bypassing the
        # preflight can't crash the rest of the per-encoder pipeline.
        if ! should_skip 8 && [[ "$ENC" == vjepa* ]]; then
            if [[ " $STAGE8_ENCODERS " == *" $ENC "* ]]; then
                PCKPT="$(encoder_predictor_ckpt_for "$ENC")"
                if [ -e "$PCKPT" ]; then
                    stamp "  STAGE 8 · future_mse forward for $ENC"
                    # iter15 D7 fix (2026-05-16): pass $MA_FLAG so m12d_future_mse
                    # emits the parallel per_clip_motion_aux_l1.npy + aggregate_
                    # motion_aux_l1.json (D2 wiring in src/m12d_future_mse.py). The
                    # original JEPA per_clip_mse.npy is still emitted in parallel —
                    # this is an ADDITIONAL 4th paired-Δ axis for the paper claim,
                    # not a replacement.
                    python -u src/m12d_future_mse.py "--$MODE" \
                        --stage forward \
                        --variant "$ENC" \
                        --encoder-ckpt "$PCKPT" \
                        $MA_FLAG \
                        --action-probe-root "$OUTPUT_ACTION" \
                        --local-data "$LOCAL_DATA" \
                        --output-root "$OUTPUT_MSE" \
                        --num-frames "$NUM_FRAMES" \
                        --cache-policy "$P_MSE" \
                        --no-wandb \
                        2>&1 | tee "logs/probe_future_mse_forward_${ENC}.log"
                else
                    echo "  ⚠️  Stage 8 SKIP $ENC: predictor-bearing ckpt missing ($PCKPT)"
                fi
            else
                echo "  Stage 8 SKIP $ENC (not in STAGE8_ENCODERS preflight set)"
            fi
        fi

        # ── STAGE 8b — predictor_temporal (m12e, 6 metrics) ── (iter16 §3.3, default-on)
        if ! should_skip 8b && [[ "$ENC" == vjepa* ]]; then
            if [[ " $STAGE8_ENCODERS " == *" $ENC "* ]]; then
                PCKPT="$(encoder_predictor_ckpt_for "$ENC")"
                if [ -e "$PCKPT" ]; then
                    stamp "  STAGE 8b · predictor_temporal (6 metrics) for $ENC"
                    python -u src/m12e_predictor_temporal.py "--$MODE" \
                        --stage forward --metric all \
                        --variant "$ENC" --encoder-ckpt "$PCKPT" \
                        $MA_FLAG \
                        --action-probe-root "$OUTPUT_ACTION" \
                        --local-data "$LOCAL_DATA" \
                        --output-root "$OUTPUT_PREDTEMP" \
                        --num-frames "$NUM_FRAMES" \
                        --batch-size "$PT_BATCH" \
                        --cache-policy "$P_MSE" \
                        --no-wandb \
                        2>&1 | tee "logs/m12e_predictor_temporal_forward_${ENC}.log"
                else
                    echo "  ⚠️  Stage 8b SKIP $ENC: predictor-bearing ckpt missing ($PCKPT)"
                fi
            else
                echo "  Stage 8b SKIP $ENC (not in STAGE8_ENCODERS preflight set)"
            fi
        fi

        # ── STAGE 8c — encoder_temporal (m12f, 4 metrics: aot/tov/pace/tcc) ── (iter16 §6/§3.3)
        if ! should_skip 8c && [[ "$ENC" == vjepa* ]]; then
            if [[ " $STAGE8_ENCODERS " == *" $ENC "* ]]; then
                ECKPT="$(encoder_ckpt_for "$ENC")"   # encoder-only (lighter; m12f needs no predictor — §3.3 R2)
                if [ -e "$ECKPT" ]; then
                    stamp "  STAGE 8c · encoder_temporal (4 metrics) for $ENC"
                    python -u src/m12f_encoder_temporal.py "--$MODE" \
                        --stage forward --metric all \
                        --variant "$ENC" --encoder-ckpt "$ECKPT" \
                        --action-probe-root "$OUTPUT_ACTION" \
                        --local-data "$LOCAL_DATA" \
                        --output-root "$OUTPUT_ENCTEMP" \
                        --num-frames "$NUM_FRAMES" \
                        --tubelet-size "$ET_TUBELET" \
                        --batch-size "$ET_BATCH" \
                        --tov-n-permutations "$ET_TOV_NPERM" \
                        --pace-strides "$ET_PACE_STRIDES" \
                        --pace-source-frames "$ET_PACE_SRCFRAMES" \
                        --tcc-temperature "$ET_TCC_TEMP" \
                        --head-lr "$ET_HEAD_LR" \
                        --head-epochs "$ET_HEAD_EPOCHS" \
                        --head-weight-decay "$ET_HEAD_WD" \
                        --head-batch-size "$ET_HEAD_BS" \
                        --head-train-cap "$ET_HEAD_TRAIN_CAP" \
                        --cache-policy "$P_MSE" \
                        --no-wandb \
                        2>&1 | tee "logs/m12f_encoder_temporal_forward_${ENC}.log"
                else
                    echo "  ⚠️  Stage 8c SKIP $ENC: encoder ckpt missing ($ECKPT)"
                fi
            else
                echo "  Stage 8c SKIP $ENC (not in STAGE8_ENCODERS preflight set)"
            fi
        fi

        # ─── End-of-per-encoder cleanup (iter15 Phase 5 V6, 2026-05-15) ──
        # Delete features_test.npy + clip_keys_test.npy + .probe_features_test_
        # ckpt.npz NOW — this encoder has finished ALL stages (5/6/8) that read
        # those artifacts. Peak disk unchanged (still 1× encoder footprint since
        # NEXT encoder's Stage 2 writes its own features_test.npy in its own
        # subdir AFTER this delete). Moved from post-Stage-5 → post-Stage-8 so a
        # Stage 8 crash leaves features_test.npy on disk → resume skips Stage 2
        # re-extract. Critical for FULL mode (~16 GB features_test.npy/encoder,
        # ~30 sec re-extract waste per encoder × N completed encoders).
        #
        # iter15 Phase 8 (2026-05-17): mode-gated cleanup. The cleanup was
        # designed for FULL (16 GB × 8 enc = 128 GB would overflow the 199 GB
        # partition during Stage 3's transient 73 GB train+val cache). At POC
        # the file is 12 MB and at SANITY <1 MB — wiping it forces a useless
        # ~2 min Stage 2 re-extract per encoder on every rerun, making
        # cache_policy=1 nearly worthless. So:
        #   FULL                  → DELETE (disk-safety dominant)
        #   POC / SANITY          → KEEP   (negligible disk cost, ~12 min saved on rerun)
        #   cache_policy=2 (any)  → DELETE (user explicitly asked to recompute)
        if [ "$MODE" = "FULL" ] || [ "$P_ACTION" = "2" ]; then
            FEATS_TEST="${OUTPUT_ACTION}/${ENC}/features_test.npy"
            KEYS_TEST="${OUTPUT_ACTION}/${ENC}/clip_keys_test.npy"
            CKPT_TEST="${OUTPUT_ACTION}/${ENC}/.probe_features_test_ckpt.npz"
            for f in "$FEATS_TEST" "$KEYS_TEST" "$CKPT_TEST"; do
                if [ -f "$f" ]; then
                    sz=$(du -h "$f" 2>/dev/null | awk '{print $1}')
                    rm -f "$f"
                    echo "  [post-stage8-cleanup] removed $f ($sz)"
                fi
            done
        else
            echo "  [post-stage8-cleanup] SKIPPED for $ENC (mode=$MODE, P_ACTION=$P_ACTION) — features_test.npy preserved for cache_policy=1 reruns"
        fi
    done
fi

# ── STAGE 4 — paired Δ (action_probe) ←── 🔥 PRIORITY 1 GATE ──────────
# Reads test_predictions.npy + test_metrics.json from EACH encoder's
# canonical path. All encoders' Stage 3 outputs were written above.
if ! should_skip 4; then
    stamp "STAGE 4 · action_probe paired_delta (🔥 P1 GATE — CPU, ~5 min, BCa 10K)"
    python -u src/m12a_action_top1.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_action_paired_delta.log
fi

# ── STAGE 12 — paired Δ (probe_taxonomy, per-dim across 16 dims) ──────
# Aggregate stage — reads per-dim test predictions from each encoder's
# OUTPUT_TAXONOMY/<encoder>/ subdirs (written by Stage 3.5 above).
# Skipped when tag_taxonomy.json is absent (Stage 1 also skipped) or via
# SKIP_STAGES=12.
if ! should_skip 12 && [ -f "$TAG_TAXONOMY" ]; then
    stamp "STAGE 12 · probe_taxonomy paired_delta (CPU, 16 dims × encoders, BCa 10K each)"
    python -u src/m12c_taxonomy_f1.py "--$MODE" \
        --stage paired_delta \
        --features-root "$OUTPUT_ACTION" \
        --output-root "$OUTPUT_TAXONOMY" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_taxonomy_paired_delta.log
fi

# ── STAGE 13 — plot (probe_taxonomy per-dim metric across encoders) ───
if ! should_skip 13 && [ -f "$TAG_TAXONOMY" ]; then
    stamp "STAGE 13 · probe_taxonomy plot (CPU, ~5s)"
    python -u src/m12c_taxonomy_f1.py "--$MODE" \
        --stage plot \
        --features-root "$OUTPUT_ACTION" \
        --output-root "$OUTPUT_TAXONOMY" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_taxonomy_plot.log
fi

# ── STAGE 7 — paired Δ (motion_cos) ────────────────────────────────────
if ! should_skip 7; then
    stamp "STAGE 7 · motion_cos paired_delta (CPU)"
    python -u src/m12b_motion_cos.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_COS" \
        --cache-policy "$P_COS" \
        --no-wandb \
        2>&1 | tee logs/probe_motion_cos_paired_delta.log
fi

# ── STAGE 8 (forward) is run inside the per-encoder loop above ─────────
# (see "PER-ENCODER SEQUENTIAL PIPELINE" block). Kept here as a marker for
# anyone grepping for the stage; the actual `python m12d_future_mse --stage
# forward` invocation lives in the per-encoder loop so its V-JEPA forward
# pass runs while features_test.npy for that encoder is still small/freed.

# ── STAGE 9 — paired Δ (future_mse, per-variant) ───────────────────────
if ! should_skip 9; then
    stamp "STAGE 9 · future_mse paired_per_variant (CPU)"
    python -u src/m12d_future_mse.py "--$MODE" \
        --stage paired_per_variant \
        --output-root "$OUTPUT_MSE" \
        --cache-policy "$P_MSE" \
        --no-wandb \
        2>&1 | tee logs/probe_future_mse_paired.log
fi

# ── STAGE 9b — predictor_temporal paired (m12e, 6 metrics) ── (iter16 §3.3)
if ! should_skip 9b; then
    stamp "STAGE 9b · predictor_temporal paired_per_variant (CPU)"
    python -u src/m12e_predictor_temporal.py "--$MODE" \
        --stage paired_per_variant --metric all \
        --output-root "$OUTPUT_PREDTEMP" \
        --cache-policy "$P_MSE" \
        --no-wandb \
        2>&1 | tee logs/m12e_predictor_temporal_paired.log
fi

# ── STAGE 9c — encoder_temporal paired (m12f, 4 metrics) ── (iter16 §6/§3.3)
if ! should_skip 9c; then
    stamp "STAGE 9c · encoder_temporal paired_per_variant (CPU)"
    python -u src/m12f_encoder_temporal.py "--$MODE" \
        --stage paired_per_variant --metric all \
        --output-root "$OUTPUT_ENCTEMP" \
        --tubelet-size "$ET_TUBELET" \
        --cache-policy "$P_MSE" \
        --no-wandb \
        2>&1 | tee logs/m12f_encoder_temporal_paired.log
fi

# ── STAGE 10 — plots (m08d, CPU, always-recompute) ─────────────────────
# Pure visualization — no cache_policy. Wipes its own output_dir on entry
# (single-owner, m08b-style). Reads JSONs + train_log.jsonl from prior stages.
if ! should_skip 10; then
    stamp "STAGE 10 · m13 eval plot — 14 bars + HERO (CPU, ~5s)"
    python -u src/m13_eval_plot.py "--$MODE" \
        --action-probe-root "$OUTPUT_ACTION" \
        --motion-cos-root   "$OUTPUT_COS" \
        --future-mse-root   "$OUTPUT_MSE" \
        --taxonomy-root     "$OUTPUT_TAXONOMY" \
        --predictor-temporal-root "$OUTPUT_PREDTEMP" \
        --encoder-temporal-root   "$OUTPUT_ENCTEMP" \
        --output-dir        "$OUTPUT_PLOTS" \
        --no-wandb \
        2>&1 | tee logs/m13_eval_plot.log
fi

# ── Final summary ──────────────────────────────────────────────────────
stamp "DONE · total wall = $(( ($(date +%s) - T0) / 60 )) min"
echo "Artifacts:"
echo "  🔥 P1 GATE     $OUTPUT_ACTION/probe_paired_delta.json"
echo "  motion_cos     $OUTPUT_COS/probe_motion_cos_paired.json"
echo "  future_mse     $OUTPUT_MSE/probe_future_mse_per_variant.json"
echo "  taxonomy lbls  $OUTPUT_TAXONOMY/taxonomy_labels.json (consumed by run_train.sh multi-task path)"
echo "  plots          $OUTPUT_PLOTS/eval/{probe_action_loss,probe_action_acc,probe_action_acc_compare,probe_motion_cos_compare,probe_future_mse_compare}.{png,pdf}"
echo "Per-encoder probe ckpts:"
for ENC in $ENCODERS; do
    echo "  $OUTPUT_ACTION/$ENC/probe.pt"
done
