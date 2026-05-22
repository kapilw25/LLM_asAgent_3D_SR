# 🚀 iter16 — 115K FULL training runbook

> **Status legend**: ⏳ pending · 🟡 in-progress · ✅ done · ❌ blocked · ⏭️ skipped

🛠️ **All 9 code modifications (M1-M9) ✅ LANDED 2026-05-21**. Full design +
verification + recipe sources archived in
`iter/iter16_train_115kclips/legacy/plan_code_modifications.md` (post-T40).

```
┌────┬────────────────────────────────────────────────────────────────────────┐
│ M  │ ✅ Status                                                                │
├────┼────────────────────────────────────────────────────────────────────────┤
│ M1 │ clip_pool_ratio + subsample_manifest_for_mode + Option X stratified-   │
│    │ for-POC. CLI + in-process symmetric wiring. probe_split mode-keyed.    │
│ M2 │ max_epochs single source — base_optimization.yaml {1,2,1};             │
│    │ pretrain_encoder + surgery_base overrides DELETED                      │
│ M3 │ src/utils/gen_full_local_manifest.py — wrote full_local.json           │
│    │ (n=115,687 · num_videos=1,559 · ~74 clips/video)                       │
│ M4 │ checkpoint.saves_per_epoch 2 → 9 (9 trajectory points/cell)            │
│ M5 │ Video-disjoint stratified_split (StratifiedGroupKFold) — 2026-05-20.  │
│    │ Mode-keyed: SANITY clip-level, POC/FULL video-disjoint                 │
│ M6 │ DINO 4-anchor batched inference (R3) — ~18% per-clip speedup           │
│ M7 │ torch.compile DINO yaml gate (R5, Pro 6000) — default OFF on Pro 4000 │
│ M8 │ torch.compile m04d RAFT — iter13 disable REVERSED with WebSearch-     │
│    │ validated recipe (mode=default + dynamic=False)                        │
│ M9 │ yaml-keyed local_data_dir + master_manifest_name (Option III) —       │
│    │ 30+ hardcoded refs eliminated. ONE-LINE migration to full_local.      │
└────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Terminal commands (in execution order)

### 🚦 Pre-flight — flip yaml to data/full_local (after Stage 1 + M3 + Stage 2-3 outputs ready)

```bash
# Pre-flight gate: data/full_local/ must already contain:
#   1. tags.json (Stage 1 metadata)
#   2. full_local.json (M3 generated — already done)
#   3. m04d_motion_features/motion_features.{npy,paths.npy} (Stage 2 output)
#   4. m10_sam_segment/masks/ (Stage 3 output)
#   5. m11_factor_datasets/ (Stage 4 — streaming or pre-computed)
#   6. subset-*.tar shards (Stage 1 — RUNNING)

# When gate passes, one-line flip migrates the whole pipeline:
sed -i \
    -e 's|local_data_dir:        "data/eval_10k_local"|local_data_dir:        "data/full_local"|' \
    -e 's|master_manifest_name:  "eval_10k.json"|master_manifest_name:  "full_local.json"|' \
    configs/pipeline.yaml
```

### ⏳ Stage 1 — HF download walkindia-200k via m00d (~10-15 min, Pro 4000 OK)

```bash
# Pre-req: data/full_local/tags.json exists (115,687 entries; 156 MB) — already on disk
python -u src/m00d_download_subset.py --FULL \
--master-tags data/full_local/tags.json \
--no-wandb 2>&1 | tee logs/iter16_m00d_full_$(date +%Y%m%d_%H%M%S).log

# Pull auxiliary outputs (if any exist on HF)
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download-data 2>&1 \
  | tee logs/iter16_dl_factorjepa_data_$(date +%Y%m%d_%H%M%S).log

# Expected under data/full_local/:
#   subset-00000.tar … subset-00115.tar     ~115 shards × ~1 GB ≈ 120 GB total
#   manifest.json                           {n, shards, saved_keys, processed_hf_shards}
#   tags.json                               115,687 entries (unchanged from input)
```

### 🟡 Stage 2 — m04d motion features (Pro 6000 ONLY · ~30-60 min RESUME from 85%)

⚠️ **Pro 4000 (36 GB cgroup) STRUCTURALLY TOO SMALL** for this stage. The 2026-05-22 attempt
ran 23 hr to 85% (clip 98,400 / 115,687) before kernel SIGKILL'd it on cgroup OOM (memory
slowly crept from 80% → 100% as Inductor cache + producer queue + page cache + Python heap
accumulated). Restart on the same box → same crash within 12-24 hr. **Migrate to Pro 6000.**
The 14.42 MB `.m04d_checkpoint.npz` (98,400 clips processed) is on HF for cross-box resume.

#### 🚀 Pro 6000 migration recipe — paste ONE line at a time (avoid bracketed-paste corruption)

```bash
# Setup (5-10 min)
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
bash setup_env_uv.sh
```

```bash
# Re-fetch the 116 subset-*.tar shards from walkindia-200k (~10 min on Pro 6000's fat I/O)
python -u src/m00d_download_subset.py --FULL --master-tags data/full_local/tags.json --no-wandb
```

```bash
# Pull tags.json + manifest.json + full_local.json + m04d_motion_features/ (~30 sec, ~160 MB)
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download-data data/full_local
```

```bash
# Confirm checkpoint landed (must be ~14.42 MB; otherwise resume falls back to fresh start)
ls -la data/full_local/m04d_motion_features/.m04d_checkpoint.npz
```

```bash
# Resume m04d on Pro 6000. M10 auto-picks the 256 GB cgroup row (decode=16, queue=16).
# M8 torch.compile stays ON — Pro 6000's 96 GB VRAM lets AdaptiveBatchSizer converge
# instead of thrashing (the failure mode that hit Pro 4000).
CACHE_POLICY_ALL=1 python -u src/m04d_motion_features.py --FULL --local-data data/full_local --subset data/full_local/full_local.json --no-wandb 2>&1 | tee logs/iter16_m04d_resume_$(date +%Y%m%d_%H%M%S).log
```

Expected startup log lines (sanity-check before walking away):
```
[M10 motion_decode_scaling] cgroup CPU-RAM=241.0 GB → row[cpu_ram_gb_max=inf]: decode_workers=16, producer_queue=16
[M8 m04d_compile] mode=default dynamic=False fullgraph=False — compiling RAFT-Large
RAFT-Large loaded on cuda (weights: C_T_SKHT_V2, compiled, fp16)
Checkpoint loaded: 98,400 clips from .m04d_checkpoint.npz
Resuming: 98,400 clips already processed
m04d motion features:  85%|████████▌ | 98400/115687 [00:00<?, ?clip/s]
```

#### 🔬 Verification post-2b (run after m04d completes)

```bash
ls -la data/full_local/m04d_motion_features/
venv_walkindia/bin/python -c "
import numpy as np
f = np.load('data/full_local/m04d_motion_features/motion_features.npy')
p = np.load('data/full_local/m04d_motion_features/motion_features.paths.npy', allow_pickle=True)
print(f'features shape: {f.shape}   dtype: {f.dtype}')
print(f'paths shape   : {p.shape}   first: {p[0]}')
"
# Expected: features (115687, 23) float32 · paths (115687,) · first key starts with goa/walking/04YK...
```

### ⏳ Stage 3 — m10 + m11 factor prep [⚠️ migrate to Pro 6000 for FULL]

```bash
# 3a) SANITY parallel smoke (Pro 4000 OK, ~5 min, 20 clips × 2 workers)
#     CACHE_POLICY_ALL=2 = fresh smoke each time (no resume from prior smoke).
CACHE_POLICY_ALL=2 LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 2 --SANITY

# 3b) FULL parallel — Pro 6000 96 GB recommended (~6 hr with M6 + M7 enabled)
#     Enable M7 (torch.compile DINO) on Pro 6000:
sed -i 's|enabled:        false             # Pro 4000 default|enabled:        true              # Pro 6000|' \
    configs/pipeline.yaml

# CACHE_POLICY_ALL=1 = SAFE RESUME (default — keeps per-worker checkpoints
# so a kill+restart resumes from .m10_checkpoint_*_<fp>.npz; saves ~25-30
# min per worker). Use =2 only when you want a fresh restart.
# (Interactive prompts also available if env var unset on a TTY — see
# `scripts/run_factor_prep_parallel.sh` docstring "Resume semantics".)
CACHE_POLICY_ALL=1 LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 6 --FULL

# 3c) FALLBACK serial m10 (~180 hr Pro 6000)
CACHE_POLICY_ALL=1 LOCAL_DATA=data/full_local \
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI_encoder.yaml --FULL
```

### ⏳ Stage 4 — Train 3 HEAD cells on Pro 4000 24 GB (~9 hr total)

```bash
# 4a) pretrain_encoder FIRST (provides SURGERY_INIT for 4c-4d)
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --FULL 2>&1 \
  | tee logs/iter16_full_m09a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log

# 4b) pretrain_head
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --FULL 2>&1 \
  | tee logs/iter16_full_m09a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log

# 4c) surgery_3stage_DI_head
SURGERY_INIT=outputs/full/m09a_pretrain_encoder/m09a_ckpt_best.pt \
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_head --FULL 2>&1 \
  | tee logs/iter16_full_m09c2_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log

# 4d) surgery_noDI_head
SURGERY_INIT=outputs/full/m09a_pretrain_encoder/m09a_ckpt_best.pt \
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_head --FULL 2>&1 \
  | tee logs/iter16_full_m09c2_noDI_head_$(date +%Y%m%d_%H%M%S).log
```

### ⏳ Stage 5 — Train 3 ENCODER cells on Pro 6000 96 GB [⚠️ migrate instance]

```bash
# Pre-req on Pro 6000: pull pretrain_encoder ckpt from previous box
#   rsync -av outputs/full/m09a_pretrain_encoder/ <pro6000>:outputs/full/m09a_pretrain_encoder/
# OR: push to HF here, pull on Pro 6000
#   HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full

# 5a) pretrain_2X_encoder (2 ep total = 2 × max_epochs.full=1 via shell override)
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_2X_encoder --FULL 2>&1 \
  | tee logs/iter16_full_m09a1_pretrain_2X_encoder_$(date +%Y%m%d_%H%M%S).log

# 5b) surgery_3stage_DI_encoder
SURGERY_INIT=outputs/full/m09a_pretrain_encoder/m09a_ckpt_best.pt \
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_encoder --FULL 2>&1 \
  | tee logs/iter16_full_m09c1_3stage_DI_encoder_$(date +%Y%m%d_%H%M%S).log

# 5c) surgery_noDI_encoder
SURGERY_INIT=outputs/full/m09a_pretrain_encoder/m09a_ckpt_best.pt \
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_encoder --FULL 2>&1 \
  | tee logs/iter16_full_m09c1_noDI_encoder_$(date +%Y%m%d_%H%M%S).log
```

### ⏳ Stage 6 — Full 13-stage eval (~3 hr Pro 4000, ~1.5 hr Pro 6000)

```bash
# Pre-req: pull all 7 cell ckpts to whichever box runs eval
#   HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download outputs/full

CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --FULL 2>&1 \
  | tee logs/iter16_post_full_eval_$(date +%Y%m%d_%H%M%S).log
```

### ⏳ Stage 7 — Plot N-run comparisons (CPU, ~5 min)

```bash
python -u src/probe_plot.py --FULL --training-side \
    --training-root outputs/full \
    --output-dir    outputs/full/probe_plot \
    --no-wandb 2>&1 | tee logs/iter16_probe_plot_train_$(date +%Y%m%d_%H%M%S).log
```

### ⏳ Stage 8 — Persist to HF (after all 7 cells + eval complete)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full 2>&1 \
  | tee logs/upload_outputs_full_$(date +%Y%m%d_%H%M%S).log

HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload-data data/full_local 2>&1 \
  | tee logs/upload_data_full_$(date +%Y%m%d_%H%M%S).log
```
