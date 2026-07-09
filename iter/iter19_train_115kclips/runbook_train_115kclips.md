# 🚀 iter19 — FULL 115k runbook (1B `vjepa_2_1_vitg`)

> 🎯 Two-box, seed-then-parallel FULL run driven by the **verified** `scripts/ngpu_run.py` (`--mode FULL`).
> Trains the paper's headline trio on **116k clips (1 epoch)**, evals 6 encoders on the **~28k held-out test**.
> Box A trains the SSL seed serially; Box B trains Best-OUR ∥ Best-COMP, evals all 6, runs the §3 finale.

---

## 🗂️ Roster — 6 eval encoders (3 trained + frozen + 2 merges)

| # | encoder | arm (`run_train` name) | how it is produced | box |
|---|---|---|---|---|
| 1 | `vjepa_2_1_vitg_pretrain_encoder` | `pretrain_encoder` | SSL seed — every arm's init ckpt | **A** (serial) |
| 2 | `vjepa_2_1_vitg_surgical_3stage_DI_diheavy_encoder` | `surgery_3stage_DI_diheavy_encoder` | Best-OUR — inits from seed | **B** (GPU0) |
| 3 | `vjepa_2_1_vitg_peft_lora_encoder` | `peft_lora_encoder` | Best-COMP — inits from seed | **B** (GPU1) |
| 4 | `vjepa_2_1_vitg_frozen` | — | anchor, eval-only (no train) | **B** (finale) |
| 5 | `vjepa_2_1_vitg_surgical_diheavy_wiseft_f50_encoder` | `surgical_diheavy_wiseft_f50_encoder` | `kind: merge` — post-hoc WiSE-FT (α=0.5), built by `run_eval` | **B** (finale) |
| 6 | `vjepa_2_1_vitg_surgical_diheavy_wiseft_f70_encoder` | `surgical_diheavy_wiseft_f70_encoder` | `kind: merge` — post-hoc WiSE-FT (α=0.3), built by `run_eval` | **B** (finale) |

DAG: `pretrain` (Box A, serial) → `diheavy` (GPU0) ∥ `peft_lora` (GPU1) → eval all 6 + §3 finale (Box B).
The two merges have **no train job** — the scheduler builds them from `merge_recipe()` once `diheavy`'s ckpt exists.

---

## ✅ Pre-flight (verify BEFORE any GPU spend)

| ✅ | check | expected | source |
|---|---|---|---|
| ☐ | `max_epochs.full` | **1** (confirmed) | `configs/train/base_optimization.yaml:187` |
| ☐ | disk free — Box A (`--cache 2`) | **≥ 500 G** | `ngpu_run.py` gate `FULL {2:500}` |
| ☐ | disk free — Box B (`--cache 1`) | **≥ 350 G** | `ngpu_run.py` gate `FULL {1:350}` |
| ☐ | `EVAL_CORPUS` | **`full`** (export; also default-derived after the yaml flip) | scheduler L77–78 |
| ☐ | pipeline data dir flipped | `data.local_data_dir: data/full_local` + `master_manifest_name: full_local.json` | Stage-1 sed below |
| ☐ | `max_epochs` / `saves_per_epoch` / `cache_policy` carry `full:` | `base_optimization.yaml`: `full = 1 / 9 / 2` (verified). `batch_size` is a **scalar `32`** (all modes — not mode-keyed). No `full:` key ⇒ status ledger falls back to priors **loudly, no crash** | base_optimization.yaml |
| ☐ | `SKIP` string | the 18 non-roster arms (see bottom) | `configs/arm_registry.yaml` |
| ☐ | backbone | `ITER18_BACKBONE=vjepa_2_1_vitg` (1B) on **every** scheduler + status pane | scheduler L53 default is 2B — MUST export |

---

## 🌐 Shared env — set on BOTH boxes, every pane

```bash
export ITER18_BACKBONE=vjepa_2_1_vitg     # 1B (scheduler default is 2B vitG → MUST export)
export EVAL_CORPUS=full                    # score against the 'full' corpus

# Point the pipeline at the 116k data (single source); corpus derives to 'full'.
sed -i -e 's|local_data_dir:.*|local_data_dir: "data/full_local"|' -e 's|master_manifest_name:.*|master_manifest_name: "full_local.json"|' configs/pipeline.yaml

# Drop the 18 non-roster arms (keep the 5 roster arms).
export SKIP="surgery_3stage_DI_encoder surgery_noDI_encoder surgery_3stage_DI_head surgery_noDI_head \
surgical_autorgn_encoder surgery_raw_encoder full_ft_encoder lpft_encoder peft_dora_encoder \
cassle_encoder ewc_encoder surgery_3stage_DI_replay25_encoder surgery_3stage_DI_tccaux_encoder \
surgery_3stage_DI_intervene_encoder surgical_3stage_DI_wiseft_encoder surgical_intervene_wiseft_f30_encoder \
surgical_intervene_wiseft_f50_encoder surgical_intervene_wiseft_f70_encoder \
pretrain_encoder surgical_diheavy_wiseft_f50_encoder surgical_diheavy_wiseft_f70_encoder"

# Verify the yaml flip landed (must print data/full_local + full_local.json).
grep -E "^\s*local_data_dir|^\s*master_manifest_name" configs/pipeline.yaml
```

---

## 📦 BOX A — 1× 96 GB — train the SEED only (~19 h)

```bash
# A1 · pull the prepped 116k data (~214 GB).
python -u src/utils/hf_outputs.py download-data data/full_local 2>&1 | tee logs/iter19_dl_full_local_$(date +%F_%H%M%S).log

# A2 · shared env (ITER18_BACKBONE, EVAL_CORPUS, yaml flip) — from the block above.

# A3 · SANITY smoke (200 clips) — code-path green-light before the seed.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode SANITY --gpus 1 --cache 2 --only pretrain_encoder 2>&1 | tee logs/iter19_sanity_seed_$(date +%F_%H%M%S).log
# remove sanity artifacts to save disk space form explooding during full training
rm -rf outputs/sanity/

# A4 · train the SEED (~18 h, 1 GPU) → student_encoder.pt + m09a_ckpt_best.pt.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 1 --cache 1 --only pretrain_encoder 2>&1 | tee logs/iter19_full_seed_$(date +%F_%H%M%S).log

# A5 · push the seed for Box B (additive, no-delete).
python -u src/utils/hf_outputs.py upload-additive outputs/full \
2>&1 | tee logs/iter19_upload_pretrain_seed_$(date +%F_%H%M%S).log
```

---

## 🔀 BOX B — 2× 96 GB — Best-OUR ∥ Best-COMP + eval + §3 finale

> Spin up **after** Box A's seed lands on HF. `--cache 1` resume-skips the seed's train job, then trains
> `diheavy` (GPU0) ∥ `peft_lora` (GPU1), evals all kept encoders, and the §3 finale evals `frozen` + the
> two diheavy WiSE-FT merges on the ~28k held-out test.

```bash
# B1 · pull the seed + the 116k data. (NO outputs/poc — the eval trains its own heads inline; see B2b.)
python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/iter19_dl_seed_$(date +%F_%H%M%S).log
python -u src/utils/hf_outputs.py download-data data/full_local 2>&1 | tee logs/iter19_dl_full_local_$(date +%F_%H%M%S).log

# B2 · shared env (ITER18_BACKBONE, EVAL_CORPUS, yaml flip, SKIP) — from the block above.

# B2b · NO extra eval env, NO new command. Default eval (PROBE_SPLIT=stratified) trains action+taxonomy
#       heads INLINE (Stage 3/11, KEEP_PROBE_HEADS=1) → all 15 metrics for all 6 encoders, scored on the
#       held-out test (leakage-free: clip_splits excludes test from the SSL train_pool). Same as v5_1B POC.

# B3 · SANITY smoke (2 GPU, same SKIP) — green-light before the arms.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode SANITY --gpus 2 --cache 2 --skip-arms $SKIP 2>&1 | tee logs/iter19_sanity_rest_$(date +%F_%H%M%S).log
# remove sanity artifacts to save disk space form explooding during full training
rm -rf outputs/sanity/

# B4 · FULL run with the E0 SSL-head gate folded in (--eval-first, Prof Das): pretrain + frozen eval run
#      FIRST (~9h, 2-wide), THEN diheavy ∥ peft_lora train + eval all 6 + §3 finale. --cache 1 skips the seed.
#      Go/no-go lands BEFORE the 20h train spend → watch the E0 rows; Ctrl-C if pretrain ≪ frozen (SSL hurt).
#      The status pane shows a "🚦 E0 SSL-head gate" line (HELD → CLEARED). Drop --eval-first to run straight.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 2 --cache 1 --eval-first pretrain_encoder frozen --skip-arms $SKIP 2>&1 | tee logs/iter19_full_rest_$(date +%F_%H%M%S).log
# the gate's go/no-go rows: outputs/full/vjepa_2_1_vitg_1B/eval/full/probe_plot/*/eval_metrics.csv
```

### 🔧 B4b · Finalize-recovery — re-write a crashed arm's predictor ckpt (`m09c_ckpt_best.pt`)

```bash
# Crashed-at-finalize arm has student_encoder.pt but no m09c_ckpt_best.pt → eval silently drops its 7 predictor metrics.
# is_finalized gate re-runs only un-finalized arms; a clean arm is a no-op. resume→finalize ~10-15 min, no training.
# peft_lora — NEEDS this (missing _best):
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 1 --cache 1 --only peft_lora_encoder 2>&1 | tee logs/iter19_peftlora_finalize_$(date +%F_%H%M%S).log
# diheavy — already clean → safe no-op (gate skips it); run only to double-check:
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 1 --cache 1 --only surgery_3stage_DI_diheavy_encoder 2>&1 | tee logs/iter19_diheavy_finalize_$(date +%F_%H%M%S).log
# confirm both m09c_ckpt_best.pt exist, then resume B4 → peft_lora P:all + Stage-8 future_mse recompute:
ls -lh \
outputs/full/vjepa_2_1_vitg_1B/train/m09b_peft_lora_encoder/m09c_ckpt_best.pt \
outputs/full/vjepa_2_1_vitg_1B/train/m09b_peft_lora_encoder/student_encoder.pt \
outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/student_encoder.pt
```

### 📟 Live status pane (separate terminal on Box B)

```bash
# BACKBONE must match; auto-backs up outputs/full to HF every 45 min. Point --log at B4's tee.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run_status.py --mode FULL --log logs/iter19_full_rest_<ts>.log
# Unattended/overnight — detached so an SSH drop can't kill the pane or its 45-min HF backup:
nohup env ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run_status.py --mode FULL \
  --log logs/iter19_full_rest_<ts>.log > logs/status_pane_$(date +%F_%H%M%S).log 2>&1 &
# live refresh:
# watch -n60 'ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run_status.py --mode FULL --log logs/iter19_full_rest_<ts>.log'
```

---

## 🏁 FINALIZE — cross-plots + HF persist

```bash
# Cross-backbone forest + scale-replication plots from the 'full' tree.
# --skip-arms "$SKIP" hides the 21 non-roster arms from the forest + combined scorecard (same as the live
# pane); WITHOUT it the finale would overwrite the pane's skip-filtered figures with all-24-arm versions.
python -u src/m13_eval_plot.py --cross-plots --cross-mode full --skip-arms "$SKIP" \
  2>&1 | tee logs/iter19_cross_plots_$(date +%F_%H%M%S).log

# Persist the full outputs tree (additive, no-delete).
python -u src/utils/hf_outputs.py upload-additive outputs/ 2>&1 | tee logs/iter19_upload_outputs_FULL_$(date +%F_%H%M%S).log
```

---

## 📋 Reference — the 18 `SKIP` arms (dropped from the DAG)

```text
surgery_3stage_DI_encoder            surgery_noDI_encoder                surgery_3stage_DI_head
surgery_noDI_head                    surgical_autorgn_encoder            surgery_raw_encoder
full_ft_encoder                      lpft_encoder                        peft_dora_encoder
cassle_encoder                       ewc_encoder                         surgery_3stage_DI_replay25_encoder
surgery_3stage_DI_tccaux_encoder     surgery_3stage_DI_intervene_encoder surgical_3stage_DI_wiseft_encoder
surgical_intervene_wiseft_f30_encoder  surgical_intervene_wiseft_f50_encoder  surgical_intervene_wiseft_f70_encoder
```

KEPT (5): `pretrain_encoder` · `surgery_3stage_DI_diheavy_encoder` · `peft_lora_encoder` ·
`surgical_diheavy_wiseft_f50_encoder` · `surgical_diheavy_wiseft_f70_encoder` (+ `frozen`, always evaled).

> ⚠️ `--skip-arms` will FATAL on an unknown arm. `pretrain_encoder` can NEVER be skipped (it is every arm's
> init dependency — the scheduler refuses it). All 18 above are valid `scheduler: true` arm names.
