# iter18 — Runbook · 🎯 cross-set 10k retest (`subset_10k`) — tighten the 17 trained 2B arms' CIs ~2×

## 🟢 Box 1 · 2060 8 GB — CPU only

```bash
# class_edges = eval_10k's motion-bin definition (CPU, ~3 min) — reused by the retest
python -u src/m04e_action_labels.py --POC --eval-subset data/eval_10k_local/eval_10k.json \
--motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
--output-root outputs/poc/_xset_edges --cache-policy 2

set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/**" --exclude "**/.*" \
2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
```

---

## 🟠 Box 2 · 1× RTX 6000 — single-GPU (m04d + taxonomy heads + smoke)

```bash
# tars + tags.json for the taxonomy regen
# encoders for the head regen

python -u src/utils/hf_outputs.py download-data data/subset_10k_local 2>&1 | tee logs/download_data_subset_10k_local_$(date +%Y%m%d_%H%M%S).log && \
python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 | tee logs/download_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log && \ 
python -u src/utils/hf_outputs.py download-data outputs/poc 2>&1 | tee logs/download_outputs_poc_$(date +%Y%m%d_%H%M%S).log    # FULL folder, NO --ext (local==HF; partial hides stale HF files)    



# m04d motion-features (compiled, ~1–1.5 h)
python -u src/m04d_motion_features.py --POC \
--subset data/subset_10k_local/subset_10k.json \
--local-data data/subset_10k_local \
--cache-policy 1 \
--no-wandb \
2>&1 | tee logs/m04d_subset_10k_local_$(date +%Y%m%d_%H%M%S).log

# taxonomy heads — 17 encoders sequential on 1 GPU (~15 h)
DROP="pretrain_2X_encoder pretrain_head surgical_3stage_DI_head surgical_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder"
ENCS="vjepa_2_1_frozen"; for t in $(python src/utils/arm_registry.py eval-tokens); do case " $DROP " in *" $t "*) :;; *) ENCS="$ENCS vjepa_2_1_$t";; esac; done

KEEP_PROBE_HEADS=1 CACHE_POLICY_ALL=2 SKIP_STAGES="1,2,3,4,5,6,7,8,8b,8c,9,9b,9c,10,12,13" \
bash scripts/run_eval.sh --POC --encoders "$ENCS" \
2>&1 | tee logs/regen_taxheads_$(date +%Y%m%d_%H%M%S).log

# (encoder-temporal heads built on Box-3 via --etheads-only, not here)

# m12f self-smoke (fresh node) — temporal metrics wire up; stage 1 builds the sanity labels 8c needs
SKIP_STAGES="2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen 2>&1 | tee logs/m12f_sanity_$(date +%Y%m%d_%H%M%S).log

# 200-clip smoke → subset_10k_smoke (scheduler self-bootstraps the corpus labels via its Stage-1 L: job)
E10=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k) && \
SKIP="surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder" && \
python -c "import json; d=json.load(open('data/subset_10k_local/subset_10k.json')); d['clip_keys']=d['clip_keys'][:200]; d['n']=200; json.dump(d, open('data/subset_10k_local/subset_10k_smoke.json','w'))"

export ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k_smoke PROBE_SPLIT=test-all && \
export LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k_smoke.json && \
export CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json EVAL_HEAD_REUSE_ROOT="$E10"

python -u scripts/iter18_poc_ngpu.py --mode POC \
--gpus 1 --cache 1 \
--skip-arms $SKIP \
2>&1 | tee logs/iter18_xset_smoke_$(date +%Y%m%d_%H%M%S).log

# HF UPLOAD >> motion_features.npy
python -u src/utils/hf_outputs.py upload-data data/subset_10k_local \
2>&1 | tee logs/upload_data_subset_10k_local_$(date +%Y%m%d_%H%M%S).log

# HF UPLOAD >> outputs/poc/
set -a; . .env; set +a
# additive — never deletes
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/**" --exclude "**/.*" \
2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
# VERIFY UPLOAD
python -u src/utils/hf_outputs.py verify outputs/poc 2>&1 | tee logs/verify_outputs_poc_$(date +%Y%m%d_%H%M%S).log

```

---

## 🔵 Box 3 · 4× RTX 6000 — 4-GPU (full retest)

```bash
python -u src/utils/hf_outputs.py download-data data/subset_10k_local
python -u src/utils/hf_outputs.py download-data data/eval_10k_local                 # the et-heads gen corpus (Stage 8c trains on eval_10k)
python -u src/utils/hf_outputs.py download-data outputs/poc      # FULL folder, NO --ext (local==HF; partial hides stale HF files)

E10=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k)
SKIP="surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder"

# m12f self-smoke (fresh node) — temporal metrics wire up; stage 1 builds the sanity labels 8c needs
SKIP_STAGES="2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen 2>&1 | tee logs/m12f_sanity_$(date +%Y%m%d_%H%M%S).log

# encoder-temporal aot/tov/pace heads on eval_10k = cross-set reuse SOURCE, 4-wide (~5 h). MUST precede the retest exports below.
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --etheads-only --cache 2 --skip-arms $SKIP \
2>&1 | tee logs/regen_etheads_ngpu_$(date +%Y%m%d_%H%M%S).log

# full retest → subset_10k
export ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k PROBE_SPLIT=test-all
export LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k.json
export CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json EVAL_HEAD_REUSE_ROOT="$E10"
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_xset_subset10k_$(date +%Y%m%d_%H%M%S).log

# ship the retest results + et-heads back to HF (whole-folder ADDITIVE — never partial)
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/**" --exclude "**/.*" \
2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log

# pick 2 OURS from the per-backbone scorecard
ls $(python src/utils/output_paths.py plot-dir poc vjepa_2_1_vitG subset_10k)/eval/
```

---


## ⏱️ Durations (estimates · test-all = ~5× the test clips of the old per-split eval)

| op | box | wall (h:mm) |
|---|---|---:|
| class_edges regen (m04e, CPU) | Box-1 · 2060 | ~0:03 |
| m04d motion-features (compiled) | Box-2 · 1× RTX 6000 | ~1:00–1:30 |
| taxonomy-head regen (17 enc, sequential) | Box-2 · 1× RTX 6000 | ~1:30–3:00 |
| cross-set retest (16 arms + frozen) | Box-3 · 4× RTX 6000 | ~8:00–12:00 |
