# iter18 — Runbook · 🎯 cross-set 10k retest (`subset_10k`) — tighten the 17 trained 2B arms' CIs ~2×

## 🟢 Box 1 · 2060 8 GB — CPU only

```bash
# class_edges = eval_10k's motion-bin definition (CPU, ~3 min) — reused by the retest
python -u src/m04e_action_labels.py --POC --eval-subset data/eval_10k_local/eval_10k.json \
--motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
--output-root outputs/poc/_xset_edges --cache-policy 2

set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset --include "outputs/poc/_xset_edges/**"
```

---

## 🟠 Box 2 · 1× RTX 6000 — single-GPU (m04d motion-features only)

```bash
# subset_10k tars for m04d
python -u src/utils/hf_outputs.py download-data data/subset_10k_local 2>&1 | tee logs/download_data_subset_10k_local_$(date +%Y%m%d_%H%M%S).log

# m04d motion-features (compiled, ~1 h)
python -u src/m04d_motion_features.py --POC \
--subset data/subset_10k_local/subset_10k.json \
--local-data data/subset_10k_local \
--cache-policy 1 \
--no-wandb \
2>&1 | tee logs/m04d_subset_10k_local_$(date +%Y%m%d_%H%M%S).log

# ships motion_features.npy (box3 downloads it). taxonomy heads are built 4-wide on box3 — NOT here.
python -u src/utils/hf_outputs.py upload-data data/subset_10k_local
```

---

## 🔵 Box 3 · 4× RTX 6000 — 4-GPU (taxheads → m12f → smoke → full retest, all on this node)

```bash
# downloads: subset_10k (retest corpus) · eval_10k_local (taxheads reuse-source corpus) · outputs/poc (encoders + class_edges)
python -u src/utils/hf_outputs.py download-data data/subset_10k_local
python -u src/utils/hf_outputs.py download-data data/eval_10k_local
python -u src/utils/hf_outputs.py download-data outputs/poc --ext json,csv,pt

E10=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k)
SKIP="surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder"
# unattended (sleep through it): join the next 4 commands [m12f · taxheads · smoke · retest] with && so the first failure stops the rest

# m12f self-smoke (fresh node, once) — confirms encoder-temporal metrics wire up
SKIP_STAGES="1,2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen 2>&1 | tee logs/m12f_sanity_$(date +%Y%m%d_%H%M%S).log

# taxonomy-head reuse source — 4-wide (~2.5 h); --cache 1 resumes a partial. (--taxheads-only = Stage-11-only, one encoder per GPU)
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --taxheads-only --cache 1 --skip-arms $SKIP \
2>&1 | tee logs/regen_taxheads_ngpu_$(date +%Y%m%d_%H%M%S).log

# 200-clip smoke → subset_10k_smoke (cheap pre-flight; validates the 4-GPU retest path before the full run)
python -c "import json; d=json.load(open('data/subset_10k_local/subset_10k.json')); d['clip_keys']=d['clip_keys'][:200]; d['n']=200; json.dump(d, open('data/subset_10k_local/subset_10k_smoke.json','w'))"
ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k_smoke PROBE_SPLIT=test-all \
LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k_smoke.json \
CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json EVAL_HEAD_REUSE_ROOT="$E10" \
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_xset_smoke_$(date +%Y%m%d_%H%M%S).log

# full retest → subset_10k
export ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k PROBE_SPLIT=test-all
export LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k.json
export CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json EVAL_HEAD_REUSE_ROOT="$E10"
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_xset_subset10k_$(date +%Y%m%d_%H%M%S).log

# pick 2 OURS from the per-backbone scorecard
ls $(python src/utils/output_paths.py plot-dir poc vjepa_2_1_vitG subset_10k)/eval/
```

---


## ⏱️ Durations (estimates · test-all = ~5× the test clips of the old per-split eval)

| op | box | wall (h:mm) |
|---|---|---:|
| class_edges regen (m04e, CPU) | Box-1 · 2060 | ~0:03 |
| m04d motion-features (compiled) | Box-2 · 1× RTX 6000 | ~1:00 |
| taxonomy-head regen (17 enc · --taxheads-only 4-wide) | Box-3 · 4× RTX 6000 | ~2:30 |
| cross-set retest (16 arms + frozen) | Box-3 · 4× RTX 6000 | ~8:00–12:00 |
