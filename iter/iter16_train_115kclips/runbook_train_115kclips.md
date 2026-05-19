# 🚀 iter16 — 115K FULL training runbook

## 🛠️ Code modifications (one-time, before any run)

### M1. Wire 75:5:20 split via YAML (single source of truth — no shell, no CLI)

```yaml
# configs/pipeline.yaml — add new top-level block, mode-keyed (mirrors probe_head_train)
probe_split:
  sanity:
    train_pct: 0.70                  # legacy default — keep for SANITY
    val_pct:   0.15
  poc:
    train_pct: 0.70                  # legacy default — keep for POC parity
    val_pct:   0.15
  full:
    train_pct: 0.75                  # iter16 — was 0.70
    val_pct:   0.05                  # iter16 — was 0.15  (test_pct = 1 − train − val = 0.20)
```

```python
# src/utils/config.py — add loader (no defaults; FAIL LOUD if missing)
def get_probe_split(mode: str) -> dict:
    """Return {train_pct, val_pct} for the given mode (sanity|poc|full)."""
    return get_pipeline_config()["probe_split"][mode]
```

```python
# src/probe_action.py — read from yaml at the call site (no argparse, no CLI flag)
from utils.config import get_probe_split
split_cfg = get_probe_split(args.mode)   # args.mode is the existing --SANITY/--POC/--FULL flag
splits = stratified_split(
    records,
    seed=args.seed,
    train_pct=split_cfg["train_pct"],
    val_pct=split_cfg["val_pct"],
    min_per_split=args.min_per_split,
)
```

```python
# src/utils/probe_train_subset.py — same yaml read at the stratified_split() call
from utils.config import get_probe_split
split_cfg = get_probe_split(args.mode)
splits = stratified_split(records, seed=args.seed,
                          train_pct=split_cfg["train_pct"],
                          val_pct=split_cfg["val_pct"])
```

```
# scripts/run_train.sh + scripts/run_eval.sh — NO CHANGES.
# The .sh files keep forwarding only --SANITY / --POC / --FULL; the .py files
# resolve the split ratios from configs/pipeline.yaml via get_probe_split(mode).
```

### M2. max_epochs.full = 1

```yaml
# configs/train/pretrain_encoder.yaml — line 79
max_epochs:
  sanity: 1
  poc: 2
  full: 1                          # iter16 — was 5

# configs/train/surgery_base.yaml — line 89
max_epochs:
  sanity: 1
  poc: 2
  full: 1                          # iter16 — was 5

# configs/train/base_optimization.yaml — line 196
max_epochs:
  sanity: 1
  poc: 1
  full: 1                          # iter16 — was 15
```

### M3. full_local.json generator (one-time)

```bash
# Locate / write the master-manifest generator (mirrors eval_10k.json shape).
# Search first; only write if missing:
grep -rn "eval_10k.json\|full_local.json" src/m00*.py src/utils/*.py 2>/dev/null
# If no generator → ~30 LoC: load outputs/data_prep/clip_durations.json,
#   video-level uniform sampling, seed=99, write {n, seed, source, sampling,
#   clips_per_video, num_videos, clip_keys[]} JSON to data/full_local/full_local.json
```

---

## 🖥️ Terminal commands (in execution order)

### Stage 1 — HF download walkindia-200k (~10-15 min, Pro 4000 OK)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -u - <<'PY' 2>&1 | tee logs/dl_walkindia_full_$(date +%Y%m%d_%H%M%S).log
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(dotenv_path="/workspace/factorjepa/.env")
snapshot_download(
    repo_id="anonymousML123/walkindia-200k",
    repo_type="dataset",
    local_dir="data/full_local",
    allow_patterns=["data/*.tar"],
    token=os.getenv("HF_TOKEN"),
    max_workers=16,
)
PY

HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download-data 2>&1 \
  | tee logs/dl_factorjepa_data_$(date +%Y%m%d_%H%M%S).log
```

### Stage 2 — full_local.json + m04d motion features (~45 min, Pro 4000 OK)

```bash
# 2a) Generate full_local.json (after M3 generator is in place)
python -u src/utils/<full_local_generator>.py \
    --clip-durations outputs/data_prep/clip_durations.json \
    --n 115000 --seed 99 \
    --output data/full_local/full_local.json

# 2b) m04d motion features → data/full_local/m04d_motion_features/
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --local-data data/full_local \
    --no-wandb 2>&1 | tee logs/m04d_full_$(date +%Y%m%d_%H%M%S).log
```

### Stage 3 — m10 + m11 factor prep (Pro 4000 OK)

```bash
# 3a) SANITY parallel first (~5 min, catches m10_split_subset bugs)
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 2 --SANITY

# 3b) FULL parallel (N=4 workers, ~5-7 hr on Pro 4000)
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 4 --FULL

# 3c) FALLBACK if 3a fails — serial (~16 hr m10 on Pro 4000)
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI_encoder.yaml --FULL
```

### Stage 4 — Train 3 HEAD cells on Pro 4000 24 GB (~9 hr total)

```bash
# 4a) pretrain_encoder FIRST — provides SURGERY_INIT for cells 4c-4d
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --FULL 2>&1 \
  | tee logs/iter16_full_m09a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log

# 4b) pretrain_head — no SURGERY_INIT dependency
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

### Stage 5 — Train 3 ENCODER cells on Pro 6000 96 GB [⚠️ migrate instance]

```bash
# Pre-req on new Pro 6000 box: pull pretrain_encoder ckpt from this instance
#   rsync -av outputs/full/m09a_pretrain_encoder/ <pro6000>:outputs/full/m09a_pretrain_encoder/
# OR: push to HF here, pull on Pro 6000:
#   HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full

# 5a) pretrain_2X_encoder — 2 ep total (= 2 × max_epochs.full=1)
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

### Stage 6 — Full 13-stage eval (~3 hr Pro 4000, ~1.5 hr Pro 6000)

```bash
# Pre-req: pull all 7 cell ckpts back to whichever box runs eval
#   HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download outputs/full

CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --FULL 2>&1 \
  | tee logs/iter16_post_full_eval_$(date +%Y%m%d_%H%M%S).log
```

### Stage 7 — Plot N-run comparisons (CPU, ~5 min)

```bash
python -u src/probe_plot.py --FULL --training-side \
    --training-root outputs/full \
    --output-dir    outputs/full/probe_plot \
    --no-wandb 2>&1 | tee logs/iter16_probe_plot_train_$(date +%Y%m%d_%H%M%S).log
```

### Stage 8 — Persist to HF (after all 7 cells + eval complete)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full 2>&1 \
  | tee logs/upload_outputs_full_$(date +%Y%m%d_%H%M%S).log

HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload-data data/full_local 2>&1 \
  | tee logs/upload_data_full_$(date +%Y%m%d_%H%M%S).log
```
