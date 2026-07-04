# 🚀 iter19 — FULL 115k runbook (1B `vjepa_2_1_vitg`)

> 🎯 Two-box, seed-then-parallel FULL run driven by the **verified** `scripts/ngpu_run.py` (`--mode FULL`).
> Trains the paper's headline trio on **116k clips (1 epoch)**, evals 6 encoders on the **23k test**.
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
| ☐ | `batch_size` / `max_epochs` / `saves_per_epoch` carry `full:` | inherited from `base_optimization.yaml` (`full: 2 / 1 / 9`) → status ledger prices it; no `full` key ⇒ ledger falls back to priors **loudly, no crash** | base_optimization.yaml |
| ☐ | `SKIP` string | the 18 non-roster arms (see bottom) | `configs/arm_registry.yaml` |
| ☐ | backbone | `ITER18_BACKBONE=vjepa_2_1_vitg` (1B) on **every** scheduler + status pane | scheduler L53 default is 2B — MUST export |

---

## 🌐 Shared env — set on BOTH boxes, every pane

```bash
export ITER18_BACKBONE=vjepa_2_1_vitg     # 1B (scheduler default is the 2B vitG → MUST export)
export EVAL_CORPUS=full                    # score against the 'full' corpus (23k test)

# Point the whole pipeline at the prepped 116k data (single source; run_train.sh:74-77 reads this).
# After this flip, TRAINED_CORPUS derives to 'full' automatically for every mode.
sed -i -e 's|local_data_dir:.*|local_data_dir: "data/full_local"|' \
       -e 's|master_manifest_name:.*|master_manifest_name: "full_local.json"|' configs/pipeline.yaml

# The 18 arms to DROP = every scheduler:true arm EXCEPT the 5 roster arms
# (pretrain_encoder, surgery_3stage_DI_diheavy_encoder, peft_lora_encoder,
#  surgical_diheavy_wiseft_f50_encoder, surgical_diheavy_wiseft_f70_encoder).
export SKIP="surgery_3stage_DI_encoder surgery_noDI_encoder surgery_3stage_DI_head surgery_noDI_head \
surgical_autorgn_encoder surgery_raw_encoder full_ft_encoder lpft_encoder peft_dora_encoder \
cassle_encoder ewc_encoder surgery_3stage_DI_replay25_encoder surgery_3stage_DI_tccaux_encoder \
surgery_3stage_DI_intervene_encoder surgical_3stage_DI_wiseft_encoder surgical_intervene_wiseft_f30_encoder \
surgical_intervene_wiseft_f50_encoder surgical_intervene_wiseft_f70_encoder"
```

---

## 📦 BOX A — 1× 96 GB — train the SEED only (~19 h)

```bash
# A1 · pull the prepped full data (~120 GB; m04d + m10 SAM + m11 factor are already in the tree).
python -u src/utils/hf_outputs.py download-data data/full_local \
  2>&1 | tee logs/iter19_dl_full_local_$(date +%F_%H%M%S).log

# A2 · shared env (ITER18_BACKBONE, EVAL_CORPUS, the yaml flip) — from the block above.

# A3 · SANITY smoke first (~200 clips, fresh) — code-path green-light before the 19 h seed.
#      --only trains ONLY pretrain_encoder (no eval, no §3 finale). rm the throwaway smoke tree after.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode SANITY --gpus 1 --cache 2 --only pretrain_encoder \
  2>&1 | tee logs/iter19_sanity_seed_$(date +%F_%H%M%S).log
rm -rf outputs/sanity/

# A4 · train the SEED (fresh — outputs/full/ is empty on HF). ~19 h, 1 GPU.
#      Writes outputs/full/vjepa_2_1_vitg/train/m09a_pretrain_encoder/{student_encoder.pt, m09a_ckpt_best.pt}.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 1 --cache 2 --only pretrain_encoder \
  2>&1 | tee logs/iter19_full_seed_$(date +%F_%H%M%S).log

# A5 · push the seed so Box B can pull it (additive, no-delete, token-safe).
python -u src/utils/hf_outputs.py upload-additive outputs/full \
  2>&1 | tee logs/iter19_up_seed_$(date +%F_%H%M%S).log
```

---

## 🔀 BOX B — 2× 96 GB — Best-OUR ∥ Best-COMP + eval + §3 finale

> Spin up **after** Box A's seed lands on HF. `--cache 1` resume-skips the seed's train job, then trains
> `diheavy` (GPU0) ∥ `peft_lora` (GPU1), evals all kept encoders, and the §3 finale evals `frozen` + the
> two diheavy WiSE-FT merges on the 23k test.

```bash
# B1 · pull the full data + the seed Box A trained.
python -u src/utils/hf_outputs.py download-data data/full_local \
  2>&1 | tee logs/iter19_dl_full_local_$(date +%F_%H%M%S).log
python -u src/utils/hf_outputs.py download outputs/full \
  2>&1 | tee logs/iter19_dl_seed_$(date +%F_%H%M%S).log

# B2 · shared env (ITER18_BACKBONE, EVAL_CORPUS, the yaml flip, SKIP) — from the block above.

# B3 · SANITY smoke first (2 GPU, fresh, same SKIP) — green-light before the 19 h arms.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode SANITY --gpus 2 --cache 2 --skip-arms $SKIP \
  2>&1 | tee logs/iter19_sanity_rest_$(date +%F_%H%M%S).log
rm -rf outputs/sanity/

# B4 · the real FULL run. --cache 1 resume-skips pretrain, trains diheavy ∥ peft_lora, evals all 6,
#      then the §3 finale builds frozen + the two diheavy WiSE-FT merges. ~19 h + eval.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 2 --cache 1 --skip-arms $SKIP \
  2>&1 | tee logs/iter19_full_rest_$(date +%F_%H%M%S).log
```

### 📟 Live status pane (separate terminal on Box B)

```bash
# BACKBONE must match the run or every cell reads pending. Auto-backs up outputs/full to HF every 45 min
# (POC+FULL are backed up; SANITY is throwaway). Point --log at the B4 main tee.
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run_status.py --mode FULL \
  --log logs/iter19_full_rest_<ts>.log
# live refresh:
# watch -n60 'ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run_status.py --mode FULL --log logs/iter19_full_rest_<ts>.log'
```

---

## 🏁 FINALIZE — cross-plots + HF persist

```bash
# Cross-backbone forest + scale-replication + combined scorecard, discovered from the 'full' tree.
python -u src/m13_eval_plot.py --cross-plots --cross-mode full \
  2>&1 | tee logs/iter19_cross_plots_$(date +%F_%H%M%S).log

# Persist the full outputs tree (additive, no-delete, token-safe).
python -u src/utils/hf_outputs.py upload-additive outputs/full \
  2>&1 | tee logs/iter19_up_full_$(date +%F_%H%M%S).log
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
