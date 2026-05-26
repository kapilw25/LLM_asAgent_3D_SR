# iter15-v2 — Runbook: re-run the 8-encoder paired-Δ with the leakage/universe/path fixes

WHY v2: iter15 v15a was invalidated by (A) test-leakage, (B) 8× universe-asymmetry,
(C) split-drift, (D) path-divergence. All fixed in code — `run_train.sh` now builds ONE
leakage-safe `train_pool.json` (= corpus − val − test) via `src/utils/clip_splits.py` and
feeds it as `--subset` to every trainer; paths come from `src/utils/data_paths.py`. The
trainer commands are therefore ~identical to iter15; the fix is internal. Re-run on the NEW
96 GB instance. iter15 checkpoints are INVALID — do not reuse them.

Run-mode order is mandatory: SANITY → POC → FULL. All three subsample the SAME corpus
(`data/full_local`) via `pipeline.yaml clip_pool_ratio` (sanity 1% / poc 10% / full 100%).

## 0 · Pre-flight: fixed code + corpus download

```bash
# fixed-code artifacts present
ls src/utils/clip_splits.py src/utils/data_paths.py
grep -q "^training_pool:" configs/train/base_optimization.yaml && echo "training_pool key ✓"
grep -q "SHARED DERIVATION VIA CLI" src/CLAUDE.md && echo "rule ✓"

# corpus download — eval_10k_local was migrated/emptied; re-pull it to re-run iter15's
# EXACT experiment with the fixed code at the same scale. (corrected path: data/eval_10k_local,
# not data/data/eval_10k_local). For the 115k paper tier, also: download-data data/full_local.
python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 \
  | tee logs/iter15_v2_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log

# single source → corpus dir (default eval_10k_local; flip to full_local for the 115k paper
# tier: edit configs/pipeline.yaml data.local_data_dir + data.master_manifest_name)
LD=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)
echo "LOCAL_DATA=$LD"

# data prereqs present after download (probe labels + factor prep)
ls "$LD/m04d_motion_features/motion_features.npy"          # m04d done (probe labels need it)
test -f "$LD/m11_factor_datasets/factor_manifest.json" && echo "m11 factor manifest ✓"
```

## 0.5 · GPU-SANITY checkpoint — validate the iter17 code changes BEFORE the full re-run

```bash
# Gate (task #35): validate the 2026-05-26 edits (hardcoded-values→yaml single-source,
# getattr/.get removal→dtype whitelist, --subset-mode legacy retired, refactor #29 shared
# compute_val_motion_aux_loss) end-to-end on GPU. ~3–10 min each. Run BEFORE §1; if any FATAL,
# fix before trusting §1-4 numbers. 6 subcmds cover all touched code: m09a1/a2/c1/c2 ×
# (DI + noDI) yaml variants.
./scripts/run_train.sh pretrain_encoder          --SANITY 2>&1 | tee logs/sanity_a1.log          # m09a1
./scripts/run_train.sh pretrain_head             --SANITY 2>&1 | tee logs/sanity_a2.log          # m09a2
./scripts/run_train.sh surgery_3stage_DI_encoder --SANITY 2>&1 | tee logs/sanity_c1.log          # m09c1 DI yaml
./scripts/run_train.sh surgery_noDI_encoder      --SANITY 2>&1 | tee logs/sanity_c1_noDI.log     # m09c1 noDI yaml
./scripts/run_train.sh surgery_3stage_DI_head    --SANITY 2>&1 | tee logs/sanity_c2.log          # m09c2 DI head yaml
./scripts/run_train.sh surgery_noDI_head         --SANITY 2>&1 | tee logs/sanity_c2_noDI.log     # m09c2 noDI head yaml

# ── VALIDATION GATE (each grep confirms a specific change; last one MUST be empty) ──
grep -E "recipe-v3 receipts" logs/sanity_c1*.log                    # m09c1 (DI+noDI): subset-mode recipe_v3, no legacy
grep -E "leakage-guard.*train pool" logs/sanity_c2.log              # m09c2: streaming + leakage filter (sanity=true)
grep -E "variant_tag|3stage_DI_head|noDI_head" logs/sanity_c2*.log  # m09c2: data.variant_tag read (was .replace)
grep -E "val_loss=" logs/sanity_a2.log logs/sanity_c2.log           # a2/c2: shared compute_val_motion_aux_loss ran ≥1 val
grep -iE "FATAL|Traceback|KeyError|AttributeError|invalid choice" logs/sanity_*.log   # MUST be empty
# NOTE: m09c2 SANITY uses StreamingFactorDataset (factor_streaming.sanity now config-true) —
#   unchanged vs old force-True behavior; logs should match prior SANITY. Once green → run §1.
```

## 1 · Set the tier (run this block 3× — SANITY, then POC, then FULL)

```bash
MODE=--SANITY ; MD=sanity      # tier 1: validate fixed code paths
# MODE=--POC  ; MD=poc         # tier 2: quick paired-Δ
# MODE=--FULL ; MD=full        # tier 3: paper numbers (115k)
```

## 2 · 7-cell paired-Δ matrix (pretrain FIRST → provides SURGERY_INIT)

```bash
# (1) pretrain encoder — provides the shared init for all 4 surgery cells
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log

# (2) pretrain head (Δ6 pair with m09a1; both Meta init)
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log

# (3) compute-matched control for Δ3 (Meta init, 2× epochs)
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_2X_encoder $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09a1_pretrain_2X_encoder_$(date +%Y%m%d_%H%M%S).log

# (4-7) surgery cells — all share ONE init from the POC/FULL pretrain ckpt (paired-Δ validity)
SI=outputs/${MD}/m09a_pretrain_encoder/m09a_ckpt_best.pt
SURGERY_INIT=$SI CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_encoder $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09c1_3stage_DI_encoder_$(date +%Y%m%d_%H%M%S).log
SURGERY_INIT=$SI CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_encoder $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09c1_noDI_encoder_$(date +%Y%m%d_%H%M%S).log
SURGERY_INIT=$SI CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_head $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09c2_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log
SURGERY_INIT=$SI CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_head $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_m09c2_noDI_head_$(date +%Y%m%d_%H%M%S).log
```

## 3 · LEAKAGE-FIX VERIFICATION GATE (the reason v2 exists — must pass before trusting numbers)

```bash
source venv_walkindia/bin/activate
LD=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)

# (A) train_pool is leakage-free: ZERO test AND zero val clips in the training pool
python -c "
import json
P=lambda f: set(json.load(open(f))['clip_keys'])
pool=P('$LD/train_pool.json'); val=P('$LD/val_split.json'); test=P('$LD/test_split.json')
assert not (pool & test), f'TEST LEAK: {len(pool&test)} test clips in pool'
assert not (pool & val),  f'VAL LEAK:  {len(pool&val)} val clips in pool'
print(f'OK leakage-free: pool={len(pool)} test_in_pool=0 val_in_pool=0')
"

# (B) m09c1 factor-dir assert passed + m09c2 streaming-universe restricted + clip_splits ran
grep -hE "\[clip_splits\] universe=|leakage-guard.*restricted to train pool" logs/iter15_v2_${MD}_*.log

# (C) ZERO FATAL across the matrix
grep -c FATAL logs/iter15_v2_${MD}_*.log

# (D) pretrain (m09a) and surgery (m09c) now train on the SAME pool size (universe-symmetry)
grep -hE "Train clips:|train/val split:|universe=broad_manifest" logs/iter15_v2_${MD}_*.log
```

## 4 · Eval + Δ5 paper claim

```bash
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh $MODE 2>&1 \
  | tee logs/iter15_v2_${MD}_eval_$(date +%Y%m%d_%H%M%S).log

# Δ5 = surgery_3stage_DI_encoder − surgery_3stage_DI_head (headline). Want non-overlapping CI.
source venv_walkindia/bin/activate
python -c "
import json
d = json.load(open('outputs/${MD}/probe_action/probe_paired_delta.json'))['iter14_paper_deltas']
d5 = d.get('delta_5_surgical_vs_surgical_head')
if not d5 or d5.get('skipped'):
    print('Δ5 not available — cells incomplete')
else:
    print(f'Δ5 mean {d5[\"delta_mean\"]:+.4f}  95% CI [{d5[\"delta_ci_lo\"]:+.4f},{d5[\"delta_ci_hi\"]:+.4f}]  p={d5[\"p_value\"]:.4f}')
    print(d5['interpretation'])
"

# training-side trajectory plots (12 PNGs)
python -u src/probe_plot.py $MODE --training-side \
  --training-root outputs/${MD} --output-dir outputs/${MD}/probe_plot --no-wandb 2>&1 \
  | tee logs/iter15_v2_${MD}_probe_plot_$(date +%Y%m%d_%H%M%S).log
```

## 5 · Monitor / triage / cleanup

```bash
# live GPU + latest tqdm (2nd pane)
watch -n 5 'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv; \
  ls -t logs/iter15_v2_*.log 2>/dev/null | head -1 | xargs tail -1'

# health sweep
for log in logs/iter15_v2_*.log; do [ -f "$log" ] || continue; \
  echo "$log: FATAL=$(grep -c FATAL "$log") OOM=$(grep -c 'IMMINENT\|OutOfMemoryError' "$log")"; done

# common failures: OOM@BS1 → VRAM math · assert_encoder_frozen → freeze wiring ·
#   factor_manifest.json missing → run scripts/run_factor_prep.sh first ·
#   "--factor-dir != canonical" FATAL → pass the canonical m11 dir (yaml-derived) ·
#   "streaming universe EMPTY after train-pool filter" → --subset pool vs factor_manifest mismatch

# verify all 7 cells produced ckpts, then upload (POC/FULL only — SANITY is throw-away)
find outputs/${MD}/m09a_pretrain_encoder outputs/${MD}/m09a_pretrain_head \
     outputs/${MD}/m09c_surgery_3stage_DI_encoder outputs/${MD}/m09c_surgery_noDI_encoder \
     outputs/${MD}/m09c_surgery_3stage_DI_head outputs/${MD}/m09c_surgery_noDI_head \
     -name '*ckpt_best.pt' -o -name 'student_encoder.pt' | sort
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/${MD} 2>&1 \
  | tee logs/iter15_v2_${MD}_upload_$(date +%Y%m%d_%H%M%S).log
```
