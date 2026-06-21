# iter18 — Output-tree restructure · 🎯 3-dial tree: scale → backbone → corpus (per-backbone plots, ZERO dup)

**Why:** today the tree is *metric-first* and interleaves both backbones by encoder name
(`outputs/poc/predictor_temporal/vjepa_2_1_<arm>/` next to `…/vjepa_2_1_vitg_<arm>/`) → miserable to
eyeball + forces one stacked plot. Re-key it by the **three orthogonal dials** below.

## 🎛️ the 3 dials (separating these is the whole point)

```text
dial 1 — training SCALE (mode)  :  poc (10k)   │  full (116k)        [sanity = throwaway, stays isolated]
dial 2 — MODEL (backbone)       :  vitG_2B │ vitg_1B │ 2.0_vitg_1B
dial 3 — SCORE corpus           :  eval_10k │ subset_10k
```

`full` is **dial 1** (how much we trained), NOT a third corpus. The trained encoder is **one artifact per
(scale,backbone)** that BOTH corpora read — so `train/` sits ABOVE the corpus split → **no symlink/proxy, no
duplication.** (Earlier drafts nested `train/` under each corpus and needed a proxy; hoisting it deletes that.)

## 🌳 Target tree

```text
outputs/poc/                                   ← dial 1: training scale (sanity → outputs/sanity/, full → outputs/full/)
└── vjepa_2_1_vitG_2B/                          ← dial 2: <backbone>_<size> (size from config, eyeball-clear)
    ├── train/   <arm>/student_encoder.pt       ← the encoders, stored ONCE (both corpora read these — no proxy)
    └── eval/
        ├── eval_10k/    <metric>/<arm>/ + plot/   ← dial 3a: scored on eval_10k   → its OWN per-backbone plot
        └── subset_10k/  <metric>/<arm>/ + plot/   ← dial 3b: scored on subset_10k → its OWN per-backbone plot
# vjepa_2_1_vitg_1B/ and vjepa_2_0_vitg_1B/ are siblings (same shape) · future 116k run → outputs/full/<bb>/…
```

- **enc leaf KEEPS its `vjepa_2_1_` prefix** (as-built decision) — the `<bb>_2B` dir already names the backbone
  so it's redundant, but keeping it avoids the deep `enc_name` / HF-loader / status ripple; `enc_prefix()` stays.
- `<metric>` = the 6 families (`probe_action`, `probe_motion_cos`, `probe_future_mse`, `predictor_temporal`,
  `encoder_temporal`, `probe_taxonomy`); `plot/` = m13's figures for THAT (backbone × corpus).
- Corpus naming standardized to **`subset_10k`** (matches `data/subset_10k_local/`); replaces the ad-hoc
  `ITER18_EVAL_TAG=subset10k` — the eval corpus is now a first-class path arg.

## 🧩 old → new path map

```text
TRAIN  outputs/poc/<bb>/<arm>/student_encoder.pt
   →   outputs/poc/<bb>_<size>/train/<arm>/student_encoder.pt          (corpus-independent — ONE copy)
EVAL   outputs/poc/<metric>/vjepa_2_1_<arm>/…           (eval_10k, untagged today)
   →   outputs/poc/<bb>_<size>/eval/eval_10k/<metric>/<arm>/…          (strip enc prefix)
XSET   outputs/poc/subset10k/<metric>/<enc>/…           (my ITER18_EVAL_TAG today)
   →   outputs/poc/<bb>_<size>/eval/subset_10k/<metric>/<arm>/…
PLOT   outputs/poc/probe_plot/eval/                     (stacked both backbones)
   →   outputs/poc/<bb>_<size>/eval/<corpus>/plot/      (ONE plot per backbone × corpus)
```

## 🛠️ code changes — route EVERY path through ONE helper (single source, never drifts)

| file | change |
|---|---|
| `configs/pipeline.yaml` | + `backbone_size_labels` (vitG→2B, vitg→1B, 2.0_vitg→1B, vitL→300M) · + `eval_corpora` list (FAIL-LOUD validation) |
| `src/utils/output_paths.py` **(NEW)** | the single source: `bb_dir / train_dir / eval_dir / plot_dir` (py fns + bash CLI); unknown backbone/corpus → FATAL |
| `scripts/run_train.sh` | train `OUT_DIR` → `output_paths train-dir <mode> <bb>`; label roots → `eval/<corpus>/probe_*` |
| `scripts/run_eval.sh` | `DEFAULT_OUTPUT_PREFIX`→`eval/<corpus>` · 7 `OUTPUT_*` → that root · `encoder_ckpt_for` → `<bb>/train` (NO symlink) |
| `scripts/iter18_poc_ngpu.py` | delete `enc_prefix` special-case · `_eval_out_env`+resume markers → `eval/<corpus>` · train-read → `<bb>/train` |
| `src/m13_eval_plot.py` | already takes `--*-root` args → caller passes the per-(backbone,corpus) `eval/<corpus>` roots; plot → `eval/<corpus>/plot` |
| `scripts/iter18_poc_status.py` | `outputs/{mtag}/…` reads → helper (per-backbone, per-corpus) |

No `train_source` / proxy machinery — `train/` is one dir per (scale,backbone), read by every corpus directly.

## 📦 migration (existing ~200 GB · idempotent · `mv` only, never `rm`)

```text
1. 2B encoders : outputs/poc/vjepa_2_1_vitG/<arm>/        → outputs/poc/vjepa_2_1_vitG_2B/train/<arm>/
2. 2B eval     : outputs/poc/<metric>/vjepa_2_1_<arm>/    → …/vjepa_2_1_vitG_2B/eval/eval_10k/<metric>/<arm>/  (strip prefix)
3. 1B encoders : outputs/poc/vjepa_2_1_vitg/<arm>/        → …/vjepa_2_1_vitg_1B/train/<arm>/
4. 1B eval     : outputs/poc/<metric>/vjepa_2_1_vitg_<arm>→ …/vjepa_2_1_vitg_1B/eval/eval_10k/<metric>/<arm>/
5. plots       : outputs/poc/probe_plot/                  → …/vjepa_2_1_vitG_2B/eval/eval_10k/plot/
6. HF          : re-upload new tree (xet dedups moved blobs → little real transfer) + super_squash old paths
```

A one-off `src/utils/migrate_output_tree.py` (dry-run first, `mv`+guard, FAIL-LOUD on collisions) does 1–5.

## ✅ verify before declaring done
- `output_paths` selftest (every verb resolves; unknown backbone/corpus FATALs).
- scheduler `--dry-run` shows new paths; resume markers land under `<bb>_<size>/eval/<corpus>/`.
- migrate `--dry-run` lists every `mv` with 0 collisions; the eval_10k scorecard re-plots from the moved tree.
- a 200-clip smoke (POC mode, `subset_10k`) writes the full new sub-tree on the real endpoints.

## ✅ resolved
`full` = training MODE → `outputs/full/<bb>_<size>/eval/{eval_10k,subset_10k}/…` by the same scheme (NOT a
corpus). `sanity` stays isolated at `outputs/sanity/`. Three dials, three levels, zero conflation.
