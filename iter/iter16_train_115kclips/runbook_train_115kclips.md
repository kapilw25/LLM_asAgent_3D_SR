# 🚀 iter16 — 115K FULL training runbook

> **Status legend** (update as tasks complete):
> ⏳ pending · 🟡 in-progress · ✅ done · ❌ blocked · ⏭️ skipped
>
> Update the emoji next to each section header as that task moves through the lifecycle.

---

## 🛠️ Code modifications (one-time, before any run)

### ⏳ M1. Mode-invariant clip pool sizing + train:val:test split (POC↔FULL parity)

**1a — `clip_pool_ratio`** — derive POC/SANITY clip count from `FULL × ratio`. No hardcoded numbers anywhere; corpus size is the single source of truth.

```yaml
# configs/pipeline.yaml — NEW top-level block. Mode-keyed FRACTION of FULL corpus.
# FULL corpus size N_full is read at runtime from data/full_local/full_local.json.
clip_pool_ratio:
  full:   1.00      # 100% — every clip in the master manifest
  poc:    0.10      # 10%  → ~11,500 @ N_full = 115,000
  sanity: 0.01      # 1%   → ~ 1,150 @ N_full = 115,000
```

```python
# src/utils/config.py — derive n_clips at runtime; FAIL LOUD if mode key missing.
def get_clip_pool_size(mode: str, n_full: int) -> int:
    """Mode-keyed clip count: round(n_full × clip_pool_ratio[mode])."""
    ratio = get_pipeline_config()["clip_pool_ratio"][mode.lower()]
    return int(round(n_full * ratio))
```

**1b — `probe_split`** — flat 75:5:20 train:val:test, identical across sanity / poc / full (parity).

```yaml
# configs/pipeline.yaml — flat block, NO per-mode keys (POC↔FULL parity per CLAUDE.md)
probe_split:
  train_pct: 0.75
  val_pct:   0.05      # test_pct = 1 − train − val = 0.20
```

```python
# src/utils/config.py — single source of truth, no mode arg
def get_probe_split() -> dict:
    """Return {train_pct, val_pct} — identical for sanity/poc/full (parity rule)."""
    return get_pipeline_config()["probe_split"]
```

**1c — Caller updates** — `probe_action.py` + `probe_train_subset.py` derive BOTH pool size and split from yaml:

```python
# src/probe_action.py
from utils.config import get_clip_pool_size, get_probe_split

manifest = json.load(open(args.subset))            # data/full_local/full_local.json
n_full = len(manifest["clip_keys"])
n_clips = get_clip_pool_size(args.mode, n_full)    # 115000 / 11500 / 1150 per mode
clip_keys = sorted(manifest["clip_keys"])[:n_clips]   # deterministic head-slice
records = [build_record(k) for k in clip_keys]

split_cfg = get_probe_split()
splits = stratified_split(
    records, seed=args.seed,
    train_pct=split_cfg["train_pct"],
    val_pct=split_cfg["val_pct"],
    min_per_split=args.min_per_split,
)
```

```python
# src/utils/probe_train_subset.py — identical pattern; same two helpers
from utils.config import get_clip_pool_size, get_probe_split
n_full = len(json.load(open(args.subset))["clip_keys"])
n_clips = get_clip_pool_size(args.mode, n_full)
split_cfg = get_probe_split()
# ... build records[:n_clips] + stratified_split as above
```

```
# scripts/run_train.sh + scripts/run_eval.sh — NO CHANGES.
# .sh files forward only --SANITY / --POC / --FULL; .py files resolve BOTH
# clip_pool_ratio[mode] and probe_split (flat) from configs/pipeline.yaml.
```

> **Cross-mode clip counts at the new ratios** (parity-preserving — same schema, different N):
> ```
> ┌────────┬────────┬──────────┬──────────┬─────────┬──────────┐
> │ Mode   │ Ratio  │ N_clips  │ N_train  │ N_val   │ N_test   │
> ├────────┼────────┼──────────┼──────────┼─────────┼──────────┤
> │ FULL   │ 100 %  │ 115,000  │  86,250  │  5,750  │  23,000  │
> │ POC    │  10 %  │  11,500  │   8,625  │    575  │   2,300  │
> │ SANITY │   1 %  │   1,150  │     862  │     58  │     230  │
> └────────┴────────┴──────────┴──────────┴─────────┴──────────┘
> ```
> POC val = 575 clips → stable val metric at the parity ratio. SANITY val = 58 → fine for
> code-correctness. The hallucinated "poc ≈ 220" warning in the prior draft was wrong:
> POC N is `FULL × clip_pool_ratio.poc`, never a hardcoded literal.

### ⏳ M2. `max_epochs` — single source of truth in `base_optimization.yaml`

**Problem**: three yamls each redefined `max_epochs` via `load_merged_config()` inheritance overrides. Iter11 set `base.full=15`, iter14 cut `surgery.full=5`, user spec set `pretrain.full=5`. Iter16 collapses everything to `full=1` — the override layer no longer carries information; pure duplication waiting for silent skew.

**Fix**: define `max_epochs` ONLY in `base_optimization.yaml`. Delete the two redundant overrides. All techniques inherit.

```yaml
# configs/train/base_optimization.yaml — line 196 — SOLE definition site
max_epochs:
  sanity: 1                       # code-path validation only
  poc:    2                       # POC↔FULL parity exception (CLAUDE.md): "the ONLY
                                  # legitimate POC vs FULL deltas are poc_total_clips
                                  # and max_epochs.poc"
  full:   1                       # iter16 — was 15 (iter11 v2 SSL canon)
```

```yaml
# configs/train/pretrain_encoder.yaml — line 79 — DELETE the override block:
#   max_epochs:
#     sanity: 1
#     poc: 2
#     full: 5                       # iter15 carried this from user spec — now obsolete
# All techniques inherit from base_optimization.yaml.
```

```yaml
# configs/train/surgery_base.yaml — line 89 — DELETE the override block:
#   max_epochs:
#     sanity: 1
#     poc: 2
#     full: 5                       # iter14 override from base's 15 — now obsolete
# All techniques inherit from base_optimization.yaml.
```

> **Exception — `pretrain_2X_encoder.yaml`**: this variant intentionally doubles the
> pretrain budget (compute-matched Δ3 control). Keep one explicit override
> `max_epochs.full: 2` with a comment citing the paired-Δ rationale. No other
> technique config should override `max_epochs`.

### ⏳ M3. `full_local.json` generator (one-time, ~25 LoC)

**Status**: no existing generator references `full_local.json` (`grep -rn` clean across src/).
**Source of truth**: `data/full_local/tags.json` (already on disk, 115,687 entries from m04 VLM
tagging). `clip_durations.json` is NOT needed — tags.json carries `{section, video_id,
source_file}` for every clip, which is all the manifest needs.

```python
# src/utils/gen_full_local_manifest.py — NEW FILE
"""Generate data/full_local/full_local.json master manifest from tags.json.

USAGE:
    python -u src/utils/gen_full_local_manifest.py
"""
import json
from pathlib import Path

TAGS = Path("data/full_local/tags.json")
OUT  = Path("data/full_local/full_local.json")

tags = json.load(open(TAGS))
clip_keys = sorted(f"{t['section']}/{t['video_id']}/{t['source_file']}" for t in tags)
num_videos = len({t["video_id"] for t in tags})

OUT.write_text(json.dumps({
    "n":               len(clip_keys),
    "seed":            99,
    "source":          str(TAGS),
    "sampling":        "all clips (full corpus, from master tags.json)",
    "clips_per_video": f"~{len(clip_keys) // max(num_videos, 1)}",
    "num_videos":      num_videos,
    "clip_keys":       clip_keys,
}, indent=2))
print(f"Wrote {OUT} — n={len(clip_keys):,}, num_videos={num_videos:,}")
```

Output shape mirrors `data/eval_10k_local/eval_10k.json` so downstream loaders
(probe_action.py, probe_train_subset.py, m04d/m10/m11) need NO changes.

### ✅ M5. Video-disjoint `stratified_split()` (close train↔val↔test leakage gap)

**Status**: completed 2026-05-20. Edits land in `src/utils/action_labels.py`
(`_extract_video_id` helper + rewritten `stratified_split` using sklearn
`StratifiedGroupKFold`). Full rationale + algorithm + risks: see
`iter/iter16_train_115kclips/legacy/plan_video_disjoint_stratified_split.md`.

```bash
# Quick verification — should print "ZERO straddlers" + ratios near 70/15/15:
CACHE_POLICY_ALL=2 python -u src/probe_action.py --FULL \
    --stage labels \
    --eval-subset data/eval_10k_local/eval_10k.json \
    --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
    --min-clips-per-class 34 --min-per-split 5 \
    --output-root outputs/full/probe_action --no-wandb

python -u tests/test_action_labels.py        # 10/10 pass on Pro 4000 24 GB (CPU-side)
```

**Why M5 lands before Stage 1**: the FIRST iter16 FULL `probe_action --stage
labels` call writes `outputs/full/probe_action/action_labels.json`. With M5,
every `video_id` lands in exactly one of {train, val, test} → no visual-style
leakage. Downstream `probe_action`, `probe_motion_cos`, `probe_future_mse`,
`probe_future_regress`, `probe_taxonomy`, `probe_train_subset` consume the
per-clip split dict unchanged.

⚠️ **SANITY operator note**: `--SANITY` mode currently FAILS LOUD on
`eval_10k_sanity.json` (≈ 115 clips × 11 classes → 1-3 videos/class, too sparse
for video-disjoint splits). This is **by design** per `src/CLAUDE.md > FAIL
LOUD` + the iter16 plan. To validate the iter16 pipeline at SANITY scale,
either: (a) raise the sanity input pool size, OR (b) smoke-test stages with
`--FULL` against `eval_10k.json` (the M5 verification command above already
does this). Do NOT branch on mode — POC↔FULL parity rule.

**Expected research impact**: probe metrics (top-1 / mAP@K / future-MSE) will
DROP by ≈ 1-3 pp absolute vs iter15 because visual-style leakage between
train/val/test is now gone. This is research-correct, not regression.
Re-baseline iter15 with the new split BEFORE publishing iter16↔iter15 deltas.

---

## 🖥️ Terminal commands (in execution order)

### ⏳ Stage 1 — HF download walkindia-200k via m00d (~10-15 min, Pro 4000 OK)

Use the project's canonical downloader `src/m00d_download_subset.py`. It produces the
**flat** `subset-NNNNN.tar` layout the downstream `f"{local_data}/*.tar"` globs (m04d /
m10 / m11) require — `snapshot_download` would write to `data/full_local/data/*.tar` and
break that glob.

`m00d --FULL` ALSO writes `manifest.json` (saved_keys + shards) and idempotently
re-filters `tags.json` (no-op in FULL mode since saved_keys = all master entries).

```bash
# Pre-req: data/full_local/tags.json must exist (115,687 entries from m04 VLM tagging).
# It's already on disk (156 MB) — confirm with:
#   python3 -c "import json; print(len(json.load(open('data/full_local/tags.json'))))"

python -u src/m00d_download_subset.py --FULL \
    --master-tags data/full_local/tags.json \
    --no-wandb 2>&1 | tee logs/iter16_m00d_full_$(date +%Y%m%d_%H%M%S).log

# Pull any auxiliary outputs (e.g., outputs/data_prep/ if it exists on HF):
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download-data 2>&1 \
  | tee logs/iter16_dl_factorjepa_data_$(date +%Y%m%d_%H%M%S).log
```

Expected output under `data/full_local/`:
```
subset-00000.tar … subset-00115.tar     ~115 shards × ~1 GB each ≈ 120 GB total
manifest.json                           {n, shards, saved_keys, processed_hf_shards}
tags.json                               115,687 entries (unchanged from input)
```

### ⏳ Stage 2 — full_local.json + m04d motion features (~10-15 hr m04d on Pro 4000)

> **m04d time recalibration**: eval_10k (9,297 clips) took 6,974 s on the previous run
> (see `data/eval_10k_local/m04d_motion_features/motion_features.meta.json`).
> Scaling: 115,687 × 0.75 sec/clip ≈ **24 hours on the same hardware**.
> On Pro 4000 24 GB with the AdaptiveBatchSizer auto-shrinking, expect **10-15 hours**
> via fp16 RAFT autocast (~2× speedup) — but plan tmux + watchdog accordingly.

```bash
# 2a) Generate full_local.json from tags.json (M3 helper — no clip_durations.json needed)
python -u src/utils/gen_full_local_manifest.py 2>&1 \
  | tee logs/iter16_gen_full_local_manifest_$(date +%Y%m%d_%H%M%S).log

# 2b) m04d motion features → data/full_local/m04d_motion_features/
#     Auto-resumes from .m04d_checkpoint.npz if interrupted (per CLAUDE.md DELETE PROTECTION).
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --local-data data/full_local \
    --subset data/full_local/full_local.json \
    --no-wandb 2>&1 | tee logs/iter16_m04d_full_$(date +%Y%m%d_%H%M%S).log
```

### ⏳ Stage 3 — m10 + m11 factor prep [⚠️ migrate to Pro 6000 for FULL]

**Driver yaml**: `surgery_3stage_DI_encoder.yaml` — the MAXIMAL factor config
(`interaction_mining.enabled: true`). m10 mines interactions + m11 streams D_L / D_A / D_I.
**All 4 surgery variants** (3stage_DI / noDI × encoder / head) consume from this single
factor-data output dir, so we only ever run m10/m11 **once per corpus**.

> **GPU sizing — honest timings at 115K**:
> The script's docstring speedup table was measured on **FULL 10K** on **Pro 6000 96 GB Blackwell**.
> Extrapolated to **115K** (11.5×) and across GPUs (Pro 4000 ≈ ⅓ throughput of Pro 6000):
>
> ```
> ┌─────────────┬─────────────────┬──────────────────┐
> │ Config       │ Pro 6000 96 GB  │ Pro 4000 24 GB   │
> ├─────────────┼─────────────────┼──────────────────┤
> │ Serial m10   │ ~180 hr (~8 d)  │ ~540 hr (~22 d)  │
> │ Parallel × 4 │ ~ 92 hr (~4 d)  │ ~230 hr (~10 d)  │
> │ Parallel × 6 │ ~ 75 hr (~3 d)  │  N/A (VRAM tight)│
> └─────────────┴─────────────────┴──────────────────┘
> ```
> Each worker holds DINO (~500 MB) + SAM3 (~3.5 GB) ≈ **4 GB VRAM**. Pro 4000 24 GB
> fits at most 4 workers; Pro 6000 96 GB fits 6 cleanly (CPU saturates at 32 cores).
> **Stage 3 should run on Pro 6000.** Keep Pro 4000 for Stage 2 (m04d ~12 hr) and the
> head-cell training in Stage 4.

```bash
# 3a) SANITY parallel — Pro 4000 OK (~5 min · 20 clips · 2 workers)
#     Smoke-tests m10_split_subset.py + worker-merge + m11-streaming chain.
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 2 --SANITY

# 3b) FULL parallel — RECOMMENDED ON PRO 6000, 6 workers (~75 hr / ~3 days).
#     On Pro 4000 with 4 workers it's ~230 hr (~10 d). Use tmux + GPU watchdog.
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI_encoder.yaml 6 --FULL
# On Pro 4000 (if you must), drop workers to 4:
#   ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml 4 --FULL

# 3c) FALLBACK if 3b parallel infra fails (worker-merge bugs, OOM, etc.) —
#     serial m10 (~180 hr Pro 6000 / ~540 hr Pro 4000).
LOCAL_DATA=data/full_local \
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI_encoder.yaml --FULL
```

> **Stage 2 ↔ Stage 3 GPU sharing**: m04d and m10 both want the same Pro 4000.
> Run them **sequentially**, not in parallel: Stage 2 (m04d ~12 hr Pro 4000) → migrate
> instance to Pro 6000 → Stage 3 (m10+m11 ~75 hr Pro 6000). Total data-prep wall ≈ 90 hr.

### ⏳ Stage 4 — Train 3 HEAD cells on Pro 4000 24 GB (~9 hr total)

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

### ⏳ Stage 5 — Train 3 ENCODER cells on Pro 6000 96 GB [⚠️ migrate instance]

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

### ⏳ Stage 6 — Full 13-stage eval (~3 hr Pro 4000, ~1.5 hr Pro 6000)

```bash
# Pre-req: pull all 7 cell ckpts back to whichever box runs eval
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
