#!/usr/bin/env bash
# iter13 probe encoder trainer — produces vjepa_2_1_pretrain_encoder (P2) + vjepa_2_1_surgical (P3)
# checkpoints that scripts/run_eval.sh consumes for the 4-encoder paired-Δ
# action-probe gate. Mirrors the run_train.sh / run_eval.sh split (iter11 v2).
#
# Subcommands:
#   pretrain_encoder          — m09a continual SSL pretrain_encoder (LR 1e-5, 5 epochs, drift L2 anchor)
#   surgery_3stage_DI_encoder — m09c factor surgery WITH interaction tubes (D_L → D_A → D_I, 40/30/30 split)
#   surgery_noDI_encoder      — m09c factor surgery WITHOUT D_I  (D_L → D_A,    50/50 split)
#
# Why two surgery variants? iter9 v15c showed Stage 3 (D_I unfreeze 75%) caused
# downstream-metric BWT=-0.33 (legacy retrieval score dropped 20.50→19.83).
# Whether D_I helps under the iter13 motion-flow probe-accuracy metric is
# genuinely open — train both and let probe_eval's paired-Δ decide.
#
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python" — this
# wrapper only:
#   1. Pre-flights that probe Stage 1 ran (action_labels.json must exist)
#   2. Generates eval_10k_{train,val}_split.json via src/utils/probe_train_subset.py
#   3. Dispatches m09a or m09c with the right yaml + output dir
#
# USAGE:
#   tmux new -s probe_train
#   ./scripts/run_train.sh pretrain_encoder          --SANITY   # ~3 min
#   ./scripts/run_train.sh surgery_3stage_DI_encoder --SANITY   # ~10 min
#   ./scripts/run_train.sh surgery_noDI_encoder      --SANITY   # ~7 min  (no Stage 3)
#   ./scripts/run_train.sh pretrain_encoder          --FULL     # ~3 GPU-h
#   ./scripts/run_train.sh surgery_3stage_DI_encoder --FULL     # ~6-8 GPU-h
#   ./scripts/run_train.sh surgery_noDI_encoder      --FULL     # ~4-6 GPU-h
#
# Bypass the cache-policy prompt for overnight runs:
#   CACHE_POLICY_ALL=1 ./scripts/run_train.sh pretrain_encoder --FULL  # keep stale outputs
#   CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --FULL  # wipe + recompute

set -euo pipefail
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_train.sh aborted at line $LINENO (exit=$rc)" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

if [ $# -lt 2 ]; then
    echo "USAGE: $0 {pretrain_encoder|pretrain_2X_encoder|pretrain_head|surgery_3stage_DI_encoder|surgery_noDI_encoder|surgery_3stage_DI_head|surgery_noDI_head} {--SANITY|--POC|--FULL}" >&2
    exit 2
fi

SUBCMD="$1"; shift
MODE_FLAG="$1"; shift

# iter15 Phase 4 (2026-05-14): added 3 new head-only SUBCMDs (pretrain_head,
# surgery_3stage_DI_head, surgery_noDI_head) wrapping m09a2_pretrain_head.py
# + m09c2_surgery_head.py. The *_head variants run on 24 GB Pro 4000 (no
# encoder backward → no activation storage); _encoder variants need 96 GB.
case "$SUBCMD" in
    pretrain_encoder|pretrain_2X_encoder|pretrain_head| \
    surgery_3stage_DI_encoder|surgery_noDI_encoder| \
    surgery_3stage_DI_head|surgery_noDI_head) ;;
    *) echo "FATAL: subcommand must be {pretrain_encoder|pretrain_2X_encoder|pretrain_head|surgery_3stage_DI_encoder|surgery_noDI_encoder|surgery_3stage_DI_head|surgery_noDI_head} (got: $SUBCMD)" >&2; exit 2 ;;
esac

case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY"; mode_dir="sanity" ;;
    --POC|--poc)       MODE="POC";    mode_dir="poc" ;;
    --FULL|--full)     MODE="FULL";   mode_dir="full" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

# iter16 M9 (2026-05-21): active local data dir + master manifest from yaml.
# Single source of truth — flip configs/pipeline.yaml > data.local_data_dir +
# data.master_manifest_name to migrate the whole pipeline from eval_10k_local
# (iter15 / iter16 SANITY+POC) to full_local (iter16 FULL).
LOCAL_DATA=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)
MASTER_MANIFEST_NAME=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.master_manifest_name)
MASTER_MANIFEST="${LOCAL_DATA}/${MASTER_MANIFEST_NAME}"
[ -d "$LOCAL_DATA" ] || { echo "❌ FATAL: \$LOCAL_DATA=$LOCAL_DATA missing"; exit 3; }
[ -f "$MASTER_MANIFEST" ] || { echo "❌ FATAL: \$MASTER_MANIFEST=$MASTER_MANIFEST missing"; exit 3; }
echo "[M9] LOCAL_DATA=$LOCAL_DATA · MASTER_MANIFEST=$MASTER_MANIFEST"

# iter17 (2026-05-26): canonical factor dir from the SAME yaml key src/utils/data_paths.py
# reads (data.factor_subdir) — so the --factor-dir passed to m09c1 matches data_paths.
# factor_dir(local_data) and m09c1's consistency assert passes. Single source, no divergence.
FACTOR_SUBDIR=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.factor_subdir)
FACTOR_DIR_CANONICAL="${LOCAL_DATA}/${FACTOR_SUBDIR}"

# ── Probe Stage 1 (action_labels.json) — auto-bootstrap before split ─────
# iter14 recipe-v3 (2026-05-09): TECH-DEBT — this shell-level bootstrap is
# REDUNDANT with src/utils/probe_labels.ensure_probe_labels_for_mode(cfg=cfg),
# which m09a/m09c already call at startup (in-process, reads cfg, no shell
# orchestration). It's kept here ONLY because probe_train_subset.py at lines
# ~93-97 below runs BEFORE m09a/m09c and depends on ACTION_LABELS existing.
# Full refactor — option 🅱 in iter/iter14_surgery_on_pretrain/plan_no_discrepancy.md
# § "Phase E (NEW)" — moves probe_train_subset.py invocation into m09a/m09c
# (via in-process split_subset() calls) so this shell block can be deleted.
# Until then: this shell calls probe_labels-equivalent logic via the same
# CLI surface (subprocess to probe_action.py + eval_subset.py) for parity.
ACTION_LABELS="outputs/${mode_dir}/probe_action/action_labels.json"
if [ ! -f "$ACTION_LABELS" ]; then
    echo "  [run_probe_train] $ACTION_LABELS missing — auto-bootstrapping via m04e_action_labels.py (CPU, ~1 min)"
    MOTION_FEATURES_BOOTSTRAP="${LOCAL_DATA}/m04d_motion_features/motion_features.npy"
    if [ ! -f "$MOTION_FEATURES_BOOTSTRAP" ]; then
        echo "❌ FATAL: $MOTION_FEATURES_BOOTSTRAP not found — run m04d_motion_features.py first" >&2
        exit 3
    fi
    # iter16 M1 Option X (2026-05-21 PM): all 3 modes feed off the master
    # manifest. The Python helper subsample_manifest_for_mode — called from
    # probe_action.run_labels_stage — applies per-mode subsample (sorted[:N]
    # for SANITY, stratified_by_motion_class for POC, identity for FULL).
    # Shell stays thin (CLAUDE.md "no logic in shell"). The legacy mode-
    # specific subset selection + POC eval_subset.py call were retired in
    # this same pass; eval_10k_{sanity,poc}.json moved to legacy/ per M1.
    EVAL_SUBSET_BOOTSTRAP="$MASTER_MANIFEST"
    if [ "$MODE" = "SANITY" ]; then
        # SANITY: stratified_split floors (val=1/test=1/train=1) for tiny pool.
        MIN_CLIPS_BOOTSTRAP=3
        MIN_SPLIT_BOOTSTRAP=1
    else
        # POC + FULL: thresholds from pipeline.yaml (single source; calibrated to the
        # 75/5/20 probe split — see probe_action_labels.min_clips_per_class comment).
        MIN_CLIPS_BOOTSTRAP=$(scripts/lib/yaml_extract.py configs/pipeline.yaml "probe_action_labels.min_clips_per_class.${mode_dir}")
        MIN_SPLIT_BOOTSTRAP=$(scripts/lib/yaml_extract.py configs/pipeline.yaml "probe_action_labels.min_per_split.${mode_dir}")
    fi
    python -u src/m04e_action_labels.py "${MODE_FLAG}" \
        --eval-subset "$EVAL_SUBSET_BOOTSTRAP" \
        --motion-features "$MOTION_FEATURES_BOOTSTRAP" \
        --min-clips-per-class "$MIN_CLIPS_BOOTSTRAP" \
        --min-per-split "$MIN_SPLIT_BOOTSTRAP" \
        --output-root "outputs/${mode_dir}/probe_action" \
        --cache-policy "${CACHE_POLICY_ALL:-1}" \
        --no-wandb \
        2>&1 | tee "logs/m04e_action_labels_${mode_dir}.log"
fi

# ── Pre-flight: bitsandbytes for SANITY 8-bit optim path ────────────────
if [ "$MODE" = "SANITY" ]; then
    if ! python -c "import bitsandbytes" 2>/dev/null; then
        echo "❌ FATAL: bitsandbytes not installed (required for SANITY 24 GB 8-bit AdamW)." >&2
        echo "  Run: pip install bitsandbytes" >&2
        exit 3
    fi
fi

# ── Generate train/val/test split JSONs from action_labels.json ──────────
# action_labels.json carries 70/15/15 split (train=6964, val=1491, test=1496)
# — paper-final m06d trio uses TEST clips that m09a never sees during pretrain_encoder.
# iter13 Task #23 (2026-05-04): test split externalised here so eval stages
# downstream can reference ${LOCAL_DATA}/test_split.json directly.
# iter16 M9 (2026-05-21): paths derived from $LOCAL_DATA (yaml-keyed) AND
# filenames are corpus-agnostic ({train,val,test}_split.json — no "eval_10k_"
# prefix) so flipping LOCAL_DATA → data/full_local works without re-renaming.
TRAIN_SPLIT="${LOCAL_DATA}/train_split.json"
VAL_SPLIT="${LOCAL_DATA}/val_split.json"
TEST_SPLIT="${LOCAL_DATA}/test_split.json"
echo "═══ $(date '+%H:%M:%S') · Generating train/val/test split JSONs ═══"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split train --output "$TRAIN_SPLIT"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split val --output "$VAL_SPLIT"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split test --output "$TEST_SPLIT"

# ── iter17 (2026-05-26): leakage-safe TRAIN POOL — built ONCE, consumed by ALL ──
# Every trainer's --subset = this pool = universe − (val ∪ test). Fixes the iter15
# bugs where each trainer derived its own pool internally (m09c1 excluded val but
# NOT test → eval-test leaked into surgery; pretrain used the ~1k train_split while
# surgery used the ~9k broad manifest = 8× confound). universe from
# base_optimization.yaml training_pool.universe (single source). All logic in
# src/utils/clip_splits.py (CLAUDE.md "SHARED DERIVATION VIA CLI" — shell stays thin).
TRAIN_POOL="${LOCAL_DATA}/train_pool.json"
UNIVERSE=$(scripts/lib/yaml_extract.py configs/train/base_optimization.yaml training_pool.universe)
# iter17 (2026-05-26): scale the pool by clip_pool_ratio[mode] of the corpus — works on
# eval_10k_local (10k) AND full_local (115k). The per-mode fractions live ONLY in
# configs/pipeline.yaml clip_pool_ratio (single source; NOT hardcoded here) — we yaml_extract
# the mode's value. Replaces the fixed in-trainer sanity_train_clips cap (removed). Deterministic
# via data.seed so all 7 matrix cells draw the SAME ratio-scaled pool. mode_dir = sanity|poc|full.
POOL_RATIO=$(scripts/lib/yaml_extract.py configs/pipeline.yaml "clip_pool_ratio.${mode_dir}")
POOL_SEED=$(scripts/lib/yaml_extract.py configs/train/base_optimization.yaml data.seed)
echo "═══ $(date '+%H:%M:%S') · Building leakage-safe train pool (universe=${UNIVERSE} · ratio=${POOL_RATIO} [${MODE}]) ═══"
python -u src/utils/clip_splits.py \
    --manifest "$MASTER_MANIFEST" \
    --train-split "$TRAIN_SPLIT" --val-split "$VAL_SPLIT" --test-split "$TEST_SPLIT" \
    --universe "$UNIVERSE" --pool-ratio "$POOL_RATIO" --seed "$POOL_SEED" --out "$TRAIN_POOL"

# (LOCAL_DATA defined earlier via M9 yaml_extract — no re-declaration needed)
# iter17: backbone selector. ViT-G keeps its canonical config (also the
# get_model_config(None) default); other backbones use configs/model/<backbone>.yaml.
# Per-backbone output namespace below: outputs/<mode>/<backbone>/<arm>/.
BACKBONE="${BACKBONE:-vjepa_2_1_vitG}"
case "$BACKBONE" in
    vjepa_2_1_vitG) _MCFG="configs/model/vjepa2_1.yaml" ;;       # 2B ViT-G (canonical)
    vjepa_2_1_vitg) _MCFG="configs/model/vjepa2_1_vitg.yaml" ;;  # 1B ViT-g scale axis
    vjepa_2_1_vitL) _MCFG="configs/model/vjepa2_1_vitL.yaml" ;;  # 300M ViT-L scale axis
    vjepa_2_0_vitg) _MCFG="configs/model/vjepa2_0.yaml" ;;       # 2.0 ViT-g version axis
    *)              _MCFG="configs/model/${BACKBONE}.yaml" ;;
esac
MODEL_CFG="${MODEL_CFG:-$_MCFG}"
[ -f "$MODEL_CFG" ] || { echo "FATAL: model config $MODEL_CFG missing for BACKBONE=$BACKBONE"; exit 3; }
P_M09="${CACHE_POLICY_ALL:-1}"

# iter17 (2026-05-26): surgery init source — LOCAL per-mode m09a pretrain_encoder
# output. SINGLE SOURCE: pretrain_namespace + ckpt_filename from pipeline.yaml drive
# BOTH the m09a write-dir (OUT_DIR below) AND this surgery read-path, so they can
# never drift (SHARED DERIVATION, src/CLAUDE.md). Removed the prior hf:// default:
# a remote base of unknown provenance was a paired-Δ confound — surgery must build
# on the exact LOCAL pretrain it is compared against, per run-mode.
#
# DEPENDENCY: pretrain_encoder MUST have run+succeeded for this mode first (it writes
# outputs/<mode>/<ns>/<ckpt>). If absent, m09c FATALs loud on the missing init ckpt.
#
# Why m09a_ckpt_best.pt NOT student_encoder.pt: m09c needs BOTH student weights
# (key="student") AND predictor weights (key="predictor"). student_encoder.pt has
# only the encoder (schema="student_state_dict"); m09a_ckpt_best.pt bundles student
# + predictor + teacher (schema="student") — single artifact covers full init.
#
# Override (advanced; m09c1 dispatcher still accepts local paths AND hf:// URIs):
#   SURGERY_INIT=outputs/poc/m09a_pretrain_encoder/m09a_ckpt_best.pt \
#     bash scripts/run_train.sh surgery_3stage_DI_encoder --POC
PRETRAIN_NS=$(scripts/lib/yaml_extract.py configs/pipeline.yaml surgery_init.pretrain_namespace)
PRETRAIN_CKPT=$(scripts/lib/yaml_extract.py configs/pipeline.yaml surgery_init.ckpt_filename)
SURGERY_INIT="${SURGERY_INIT:-outputs/${mode_dir}/${BACKBONE}/${PRETRAIN_NS}/${PRETRAIN_CKPT}}"

# ── Multi-task probe-loss labels (iter13) ────────────────────────────────
# When base_optimization.yaml `multi_task_probe.enabled` is true for this
# mode, m09a/m09c add CrossEntropy/BCE losses on 16 taxonomy dims to JEPA L1
# — needs taxonomy_labels.json from probe_taxonomy --stage labels.
#
# Auto-generate if missing so the wrapper is self-sufficient (matches the
# pattern of probe-train-subset auto-gen above). probe_taxonomy --stage
# labels is CPU-only (~30 s on FULL); the inputs (eval_subset, tags.json,
# tag_taxonomy.json) are already on disk for any working repo. If sources
# are missing, we fall back to silent-disable + a fix-it hint.
TAXONOMY_LABELS="outputs/${mode_dir}/probe_taxonomy/taxonomy_labels.json"
TAG_TAXONOMY="configs/tag_taxonomy.json"
# iter16 M9 (2026-05-21): master manifest for all modes; M1 Option X
# subsamples in-process. No more per-mode pre-made JSONs.
EVAL_SUBSET_TX="$MASTER_MANIFEST"
TAGS_JSON_TX="${LOCAL_DATA}/tags.json"
TAXONOMY_ARGS=()
if [ ! -f "$TAXONOMY_LABELS" ]; then
    if [ -f "$TAG_TAXONOMY" ] && [ -f "$EVAL_SUBSET_TX" ] && [ -f "$TAGS_JSON_TX" ]; then
        echo "  [multi-task] $TAXONOMY_LABELS missing — auto-generating via m04f_taxonomy_labels.py"
        python -u src/m04f_taxonomy_labels.py "${MODE_FLAG}" \
            --eval-subset "$EVAL_SUBSET_TX" \
            --tags-json "$TAGS_JSON_TX" \
            --tag-taxonomy "$TAG_TAXONOMY" \
            --output-root "outputs/${mode_dir}/probe_taxonomy" \
            --cache-policy "$P_M09" \
            --no-wandb \
            2>&1 | tee "logs/m04f_taxonomy_labels_${mode_dir}.log"
    else
        echo "  [multi-task] cannot auto-generate $TAXONOMY_LABELS — sources missing:"
        [ -f "$TAG_TAXONOMY" ]   || echo "      ✗ $TAG_TAXONOMY"
        [ -f "$EVAL_SUBSET_TX" ] || echo "      ✗ $EVAL_SUBSET_TX"
        [ -f "$TAGS_JSON_TX" ]   || echo "      ✗ $TAGS_JSON_TX"
        echo "    → multi_task_probe will auto-disable for this run"
    fi
fi
if [ -f "$TAXONOMY_LABELS" ]; then
    TAXONOMY_ARGS=(--taxonomy-labels-json "$TAXONOMY_LABELS")
    echo "  [multi-task] Using taxonomy labels: $TAXONOMY_LABELS"
fi

# ── Dispatch ──────────────────────────────────────────────────────────────
case "$SUBCMD" in
    pretrain_encoder|pretrain_2X_encoder)
        # iter13 v12+ (2026-05-06): renamed probe_pretrain → m09a_pretrain_encoder to
        # match the source module's name (CLAUDE.md "m*.py = each module is
        # independent"). Mirror rename in run_eval.sh + yaml output_dir.
        # iter14 (2026-05-08): pretrain_2X_encoder shares pretrain_encoder.yaml (no new
        # yamls) but writes to m09a_pretrain_2X_encoder/ + passes --max-epochs 10 for
        # FULL only (G1=🅰️ "5+5 vs 10").
        # iter15 (2026-05-17): extend 2X to POC mode so Δ3 (factor-patching causal,
        # surg vs compute-matched pretrain) is POC-testable. POC was previously dropped
        # by run_eval.sh preflight ("pretrain_2X_encoder ... not found"). Both POC + FULL
        # now compute 2X from yaml's max_epochs (single source of truth — no hardcoded
        # 4/10). SANITY left at single-epoch since it's a code-path validator.
        TRAIN_CFG="configs/train/pretrain_encoder.yaml"
        if [ "$SUBCMD" = "pretrain_2X_encoder" ]; then
            OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09a_pretrain_2X_encoder"
            EPOCHS_OVERRIDE_FLAG=""
            if [ "$MODE" = "POC" ] || [ "$MODE" = "FULL" ]; then
                _BASE_EP=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" "optimization.max_epochs.${mode_dir}")
                EPOCHS_OVERRIDE_FLAG="--max-epochs $((_BASE_EP * 2))"
            fi
        else
            # iter17: single source with surgery read-path (pipeline.yaml surgery_init.pretrain_namespace)
            OUT_DIR="outputs/${mode_dir}/${BACKBONE}/${PRETRAIN_NS}"
            EPOCHS_OVERRIDE_FLAG=""
        fi
        # Read lambda_reg from YAML so it stays the single source of truth.
        # Passing it explicitly to m09a bypasses the legacy auto-ablation gate
        # (m09a1_pretrain_encoder.py:1187 enters EWC sweep when --lambda-reg is unset;
        # pretrain_encoder.yaml intentionally has ablation_lambdas=[] which would
        # then trip select_ablation_winner's non-empty assertion).
        LAMBDA_REG=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" drift_control.lambda_reg)
        echo "═══ $(date '+%H:%M:%S') · m09a continual SSL ${SUBCMD} (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_POOL  (leakage-safe: universe − val − test)"
        echo "  val:       $VAL_SPLIT"
        echo "  local:     $LOCAL_DATA"
        echo "  output:    $OUT_DIR"
        echo "  lambda:    $LAMBDA_REG (read from yaml; bypasses auto-ablation gate)"
        if [ -n "$EPOCHS_OVERRIDE_FLAG" ]; then
            echo "  epochs:    $EPOCHS_OVERRIDE_FLAG (iter14 arm C — overrides yaml's full=5)"
        fi
        mkdir -p "$OUT_DIR"
        python -u src/m09a1_pretrain_encoder.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_POOL" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --lambda-reg "$LAMBDA_REG" \
            $EPOCHS_OVERRIDE_FLAG \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "${LOCAL_DATA}/tags.json" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09a_${SUBCMD}_${mode_dir}.log"
        ;;
    surgery_3stage_DI_encoder|surgery_noDI_encoder)
        # Map subcommand → yaml + variant tag (used in output dir + log name).
        case "$SUBCMD" in
            surgery_3stage_DI_encoder)
                TRAIN_CFG="configs/train/surgery_3stage_DI_encoder.yaml"
                VARIANT_TAG="3stage_DI_encoder"
                ;;
            surgery_noDI_encoder)
                TRAIN_CFG="configs/train/surgery_2stage_noDI_encoder.yaml"
                VARIANT_TAG="noDI_encoder"
                ;;
        esac
        # iter13 v12+ (2026-05-06): renamed probe_surgery_* → m09c_surgery_* to
        # match m09c1_surgery_encoder.py module name. Mirror in run_eval.sh.
        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09c_surgery_${VARIANT_TAG}"
        # iter13 v12+ Task 3 (2026-05-06): m11 outputs co-located with input
        # under <--local-data>/m11_factor_datasets/. m11 derives this default
        # itself; this consumer just mirrors the same convention.
        FACTOR_DIR="$FACTOR_DIR_CANONICAL"   # iter17: yaml-derived (data.factor_subdir), single source
        VAL_TAGS="${LOCAL_DATA}/tags.json"
        # Surgery prereq: m10/m11 factor datasets (D_L/D_A/D_I) generated by run_factor_prep.sh.
        # noDI variant only needs D_L + D_A but the factor dir holds all three; same prereq.
        if [ ! -d "$FACTOR_DIR" ]; then
            echo "❌ FATAL: $FACTOR_DIR missing — run scripts/run_factor_prep.sh first" >&2
            exit 3
        fi
        if [ ! -f "$VAL_TAGS" ]; then
            echo "❌ FATAL: $VAL_TAGS missing — surgery's mid-training probe requires VLM tags" >&2
            exit 3
        fi
        # iter17 (2026-05-26): the recipe knobs are now MANDATORY CLI args on m09c1
        # (required=True — no silent None→yaml fallback inside the trainer, the hidden-
        # default bug class). Resolve each from the train yaml (single source) with an
        # optional env-var override winning (ad-hoc manual ablation), then ALWAYS pass the
        # resolved value → active recipe is visible in the command + log. argparse
        # choices=[...] validates each value (single validation point). NOTE: the iter14
        # run_recipe_v3_sweep.sh drop-one orchestrator was retired to scripts/legacy/ on
        # 2026-05-26 (recipe settled; the --subset-mode legacy arm was removed).
        _y() { scripts/lib/yaml_extract.py "$TRAIN_CFG" "$1"; }
        _onoff() { case "$1" in True|true|1) echo on ;; *) echo off ;; esac; }
        TEACHER="${TEACHER_MODE_OVERRIDE:-$(_y surgery.teacher_mode)}"
        LPFT="${LP_FT_OVERRIDE:-$(_onoff "$(_y surgery.lp_ft_stage0.enabled)")}"
        WARMUP="${WARMUP_OVERRIDE:-$(_y surgery.warmup_mode)}"
        SALIENCY="${SALIENCY_OVERRIDE:-$(_onoff "$(_y optimization.loss.saliency_weighting)")}"
        SPD="${SPD_OVERRIDE:-$(_onoff "$(_y optimization.spd.enabled)")}"
        _RP="$(_y replay.raw_pretrain_pct)"
        case "$_RP" in 0|0.0|0.00|"") _REPLAY=off ;; *) _REPLAY=on ;; esac
        REPLAY="${REPLAY_OVERRIDE:-$_REPLAY}"
        # subset-mode: the legacy 12/24/24 arm was retired 2026-05-26 → recipe_v3 (4/8/8,
        # Lee ICLR'23, the yaml-declared unfreeze_below) is the only valid value. No env
        # override (SUBSET_OVERRIDE removed with the legacy arm); m09c1 argparse rejects
        # anything but recipe_v3.
        SUBSETM=recipe_v3
        RECIPE_V2_ARGS=(--teacher-mode "$TEACHER" --lp-ft-stage0 "$LPFT" --warmup-mode "$WARMUP" \
                        --subset-mode "$SUBSETM" --saliency "$SALIENCY" --spd "$SPD" --replay "$REPLAY")

        echo "═══ $(date '+%H:%M:%S') · m09c factor surgery [variant=${VARIANT_TAG}] (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_POOL  (leakage-safe: universe − val − test)"
        echo "  factor:    $FACTOR_DIR"
        echo "  output:    $OUT_DIR"
        echo "  init:      $SURGERY_INIT (iter15 — accepts hf:// URI or local path; m09c1 dispatches by prefix)"
        echo "  recipe (MANDATORY, resolved from $TRAIN_CFG + optional env overrides): ${RECIPE_V2_ARGS[*]}"
        mkdir -p "$OUT_DIR"
        # iter14 (2026-05-08): --init-from-ckpt is REQUIRED in m09c (argparse
        # required=True). Always pass the HF URI — single source per CLAUDE.md
        # FAIL LOUD. m09c downloads via hf_hub_download (cached in HF_HOME after
        # first call; subsequent surgery runs hit cache instantly).
        python -u src/m09c1_surgery_encoder.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_POOL" --local-data "$LOCAL_DATA" \
            --factor-dir "$FACTOR_DIR" \
            --probe-subset "$VAL_SPLIT" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "$VAL_TAGS" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --init-from-ckpt "$SURGERY_INIT" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            "${RECIPE_V2_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09c_surgery_${VARIANT_TAG}_${mode_dir}.log"
        ;;
    pretrain_head)
        # iter15 Phase 4 (2026-05-14): head-only m09a2. Frozen encoder + frozen
        # predictor; only the ~432K motion_aux head trains. 24 GB sufficient.
        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09a_pretrain_head"
        TRAIN_CFG="configs/train/pretrain_head.yaml"
        echo "═══ $(date '+%H:%M:%S') · m09a2 HEAD-ONLY continual SSL (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_POOL  (leakage-safe: universe − val − test)"
        echo "  val:       $VAL_SPLIT"
        echo "  local:     $LOCAL_DATA"
        echo "  output:    $OUT_DIR"
        echo "  contract:  all 48 ViT-G blocks + predictor FROZEN; trainable = motion_aux head"
        mkdir -p "$OUT_DIR"
        python -u src/m09a2_pretrain_head.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_POOL" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "${LOCAL_DATA}/tags.json" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09a2_pretrain_head_${mode_dir}.log"
        ;;
    surgery_3stage_DI_head|surgery_noDI_head)
        # iter15 Phase 4 (2026-05-14): head-only m09c2. Same freeze contract as
        # pretrain_head + StreamingFactorDataset for factor-aug clips. Single
        # head-only stage (no progressive unfreeze).
        case "$SUBCMD" in
            surgery_3stage_DI_head)
                TRAIN_CFG="configs/train/surgery_3stage_DI_head.yaml"
                VARIANT_TAG="3stage_DI_head"
                ;;
            surgery_noDI_head)
                TRAIN_CFG="configs/train/surgery_2stage_noDI_head.yaml"
                VARIANT_TAG="noDI_head"
                ;;
        esac
        OUT_DIR="outputs/${mode_dir}/${BACKBONE}/m09c_surgery_${VARIANT_TAG}"
        FACTOR_DIR="$FACTOR_DIR_CANONICAL"   # iter17: yaml-derived (data.factor_subdir), single source
        VAL_TAGS="${LOCAL_DATA}/tags.json"
        if [ ! -d "$FACTOR_DIR" ]; then
            echo "❌ FATAL: $FACTOR_DIR missing — run scripts/run_factor_prep.sh first" >&2
            exit 3
        fi
        echo "═══ $(date '+%H:%M:%S') · m09c2 HEAD-ONLY surgery [variant=${VARIANT_TAG}] (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_POOL  (leakage-safe: universe − val − test)"
        echo "  factor:    $FACTOR_DIR"
        echo "  output:    $OUT_DIR"
        echo "  init:      $SURGERY_INIT (iter15 — paired-Δ parity with m09c1; accepts hf:// or local path)"
        echo "  contract:  all 48 ViT-G blocks + predictor FROZEN; trainable = motion_aux head"
        mkdir -p "$OUT_DIR"
        # m09c2 derives factor_manifest from --local-data internally (see m09c2.py:284
        # `manifest_path = Path(local_data) / "m11_factor_datasets" / "factor_manifest.json"`).
        # Unlike m09c1 it doesn't accept --factor-dir CLI — StreamingFactorDataset-only path.
        python -u src/m09c2_surgery_head.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_POOL" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --init-from-ckpt "$SURGERY_INIT" \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "$VAL_TAGS" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09c2_surgery_${VARIANT_TAG}_${mode_dir}.log"
        ;;
esac

# ── Verify expected outputs ──────────────────────────────────────────────
echo ""
echo "═══ $(date '+%H:%M:%S') · DONE ═══"
echo "Expected outputs (consumed by scripts/run_eval.sh):"
if [ -f "${OUT_DIR}/student_encoder.pt" ]; then
    ls -lh "${OUT_DIR}/student_encoder.pt"
else
    echo "  ⚠️  ${OUT_DIR}/student_encoder.pt NOT produced (check logs/probe_${SUBCMD}_${mode_dir}.log)"
fi
case "$SUBCMD" in
    pretrain_encoder|pretrain_2X_encoder|pretrain_head)
        FULL_CKPT="${OUT_DIR}/m09a_ckpt_best.pt"
        ;;
    surgery_3stage_DI_encoder|surgery_noDI_encoder|surgery_3stage_DI_head|surgery_noDI_head)
        FULL_CKPT="${OUT_DIR}/m09c_ckpt_best.pt"
        ;;
esac
if [ -f "$FULL_CKPT" ]; then
    ls -lh "$FULL_CKPT"
    echo "  → Stage 8 future_mse will use this (carries 'predictor' key)."
else
    echo "  ⚠️  $FULL_CKPT NOT produced — Stage 8 future_mse will FATAL for ${SUBCMD}"
fi
