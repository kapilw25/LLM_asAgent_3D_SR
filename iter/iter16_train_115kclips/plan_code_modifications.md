# 🛠️ iter16 Code Modifications — M1 + M2 + M3 + M4 + M6 + M7 + M8 (NEW)

> **Status legend** (update each task as it moves through the lifecycle):
> ⏳ pending · 🟡 in-progress · ✅ done · ❌ blocked · ⏭️ skipped
>
> When ALL T35-T39 (+ T18 + T41 + T42) below are ✅, this file moves to
> `iter/iter16_train_115kclips/legacy/` per T40.

---

## 🔄 Resume-from-24h-pause checklist (READ FIRST when coming back)

📅 **Last touched**: 2026-05-20 (this plan + M5 already landed). Coming back
2026-05-21+ — do these 4 things BEFORE any tool call that mutates state:

```
┌────┬─────────────────────────────────────────────────────────────────────┐
│ #  │ Pre-flight on resume                                                 │
├────┼─────────────────────────────────────────────────────────────────────┤
│ 1  │ git status — confirm commit landed; no surprise files in WT          │
│ 2  │ source venv_walkindia/bin/activate — venv ready                      │
│ 3  │ nvidia-smi — Pro 4000 24 GB attached, no zombie GPU process         │
│ 4  │ Re-read this plan's "📋 Execution checklist" table — confirm         │
│    │ no T<N> drifted; M5 line in iter16 runbook still ✅                   │
└────┴─────────────────────────────────────────────────────────────────────┘
```

🚦 **Safe to start immediately** (no external prereqs):
   T35 (M1) · T36 (M2) · T38 (M4) · T18 (M6) — all CPU-side or Pro 4000

🛑 **Blocked on Stage 1 data download** (data/full_local/tags.json):
   T37 (M3 — gen_full_local_manifest.py reads tags.json). If Stage 1 not
   run yet → write the .py during T37 but skip the smoke-test; re-run
   smoke-test after `m00d_download_subset.py` completes.

🔬 **WebSearch first**:
   T42 (M8) — pull iter13's exact InductorError from
   `iter/iter13/errors_N_fixes.md`, confirm `dynamic=False` +
   `reduce-overhead` is the upstream-validated fix (≥2 sources) before
   flipping `m04d_compile.enabled: true`.

🔗 **Sequencing constraints**:
   - M6 (T18) MUST land before M7 (T41) — M7 wraps the batched DINO helper
     that M6 creates.
   - M1, M2, M3, M4, M6, M8 are mutually independent (parallelizable).
   - M7 is Pro 6000-only (yaml gate defaults to OFF on Pro 4000).
   - Runbook update (T39) + plan retirement (T40) are LAST.

🎯 **Non-negotiables that bit me this session — re-read before editing**:
   1. 🛑 NO DEFER, NO TECH DEBT (src/CLAUDE.md) — every R-item is an
      M-section in iter16; no "park it for iter17" / "marginal gain" framing.
   2. 🛑 NO RATIONALIZING (memory: feedback_no_rationalizing.md) — when an
      asymmetry is flagged, JUST FIX IT; don't write a "why this is by
      design" paragraph.
   3. 🛑 NO HARDCODED VALUES (memory: feedback_no_hardcoded_python_defaults
      .md) — every magic number / model ID / path / threshold lives in
      `configs/*.yaml`, including function-signature defaults.
   4. 🛑 FAIL LOUD — no eager fallback, no `.get(key, default)` on yaml,
      no `try: ... except: pass`. Silent failures = garbage metrics.

---

## 🎯 Why this change

Iter16 runbook (`iter/iter16_train_115kclips/runbook_train_115kclips.md`)
**promised** M1/M2/M3 as code-mod sections. Only M5 (video-disjoint
`stratified_split`) actually landed. The 2026-05-20 audit confirmed that the
yaml blocks + helper functions + caller updates described in the runbook text
were **never written into the codebase**.

```
┌────┬──────────────────────────────────────────────────────────────┐
│ 🧪 Audit verdict (2026-05-20)                                      │
├────┼──────────────────────────────────────────────────────────────┤
│ M1 │ ❌ probe_split block present BUT get_probe_split() helper      │
│    │ MISSING, clip_pool_ratio block MISSING,                        │
│    │ get_clip_pool_size() MISSING, probe_action caller NOT wired   │
├────┼──────────────────────────────────────────────────────────────┤
│ M2 │ ❌ base.full=15 (not iter16 target=1); pretrain_encoder.yaml +│
│    │ surgery_base.yaml still have their max_epochs blocks          │
├────┼──────────────────────────────────────────────────────────────┤
│ M3 │ ❌ src/utils/gen_full_local_manifest.py MISSING                │
├────┼──────────────────────────────────────────────────────────────┤
│ M5 │ ✅ Already done (2026-05-20). See legacy/                      │
│    │ plan_video_disjoint_stratified_split.md.                       │
└────┴──────────────────────────────────────────────────────────────┘
```

Plus **2 adjacent code-mods discussed but not yet in runbook**:

```
┌────┬──────────────────────────────────────────────────────────────┐
│ M4 │ ⏳ saves_per_epoch=2 → 9 for iter16 1-epoch FULL trajectory   │
│    │ (currently 2 → only 2 probe-trio data points per cell;         │
│    │ 9 → 1 validation every ~9.6K train clips = smooth curve)       │
├────┼──────────────────────────────────────────────────────────────┤
│ M6 │ ⏳ DINO 4-anchor batching in m10_sam_segment.py:326-336        │
│    │ (~18% per-clip speedup, compounds with SAM3.1 upgrade;         │
│    │ HF #32206 confirms safe for identical compound_prompt)         │
└────┴──────────────────────────────────────────────────────────────┘
```

🔒 **User decisions captured in plan-mode Phase 3**:

```
┌────┬──────────────────────────────────────────────────────────────┐
│ #  │ Decision                                                      │
├────┼──────────────────────────────────────────────────────────────┤
│ 1  │ ⭐ clip_pool_ratio + get_clip_pool_size() — INCLUDE           │
│    │   (non-negotiable, no shortcuts). POC/SANITY clip counts      │
│    │   derive from FULL × ratio at runtime — NOT from per-mode     │
│    │   pre-made eval_10k_{poc,sanity}.json files.                  │
├────┼──────────────────────────────────────────────────────────────┤
│ 2  │ 🚫 pretrain_2X_encoder.yaml — SKIP. Shell-level                │
│    │   --max-epochs override in run_train.sh:248-252 already        │
│    │   handles 2×. With M2 base.full=1 → shell passes 2 epochs.    │
├────┼──────────────────────────────────────────────────────────────┤
│ 3  │ 🔼 saves_per_epoch — BUMP to 9 (mode-invariant per parity).   │
├────┼──────────────────────────────────────────────────────────────┤
│ 4  │ 📒 DINO 4-anchor batching — ADD as ⏳ M6 in runbook.          │
└────┴──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Throughput recommendations status (R1–R5 cross-reference)

Five throughput recs identified across the session. Each one is now either
landed in the codebase, tracked in this plan, or explicitly deferred — none
are forgotten.

```
┌─────┬────────────────────────────────────────────────────────────┬──────┬───────────────────────────┐
│ #   │ Recommendation                                              │Status│ Where to find it          │
├─────┼────────────────────────────────────────────────────────────┼──────┼───────────────────────────┤
│ R1  │ ⭐ Upgrade m10 HF model: facebook/sam3 → facebook/sam3.1     │ ✅   │ surgery_base.yaml:156      │
│     │   (Object Multiplex: ~3-7× speedup on multi-agent clips,    │ DONE │ sam_hf_model edited        │
│     │   ½ VRAM → 8 workers Pro 4000 / 12-16 Pro 6000;             │ 2026-│ 2026-05-20.                │
│     │   Stage 3 wall: ~75 hr → ~11-18 hr)                          │ 05-20│                            │
├─────┼────────────────────────────────────────────────────────────┼──────┼───────────────────────────┤
│ R2  │ m04d sizer: decouple initial_size from max_size, add        │ ✅   │ pipeline.yaml gpu.        │
│     │   motion_initial_bs: 8 (AdaptiveBatchSizer can finally grow │ DONE │ motion_initial_bs: 8 +    │
│     │   toward 0.85 × VRAM; ~30-50% Pro 6000 gain)                │ 2026-│ m04d_motion_features.py:  │
│     │                                                              │ 05-20│ 608-612 wiring.            │
├─────┼────────────────────────────────────────────────────────────┼──────┼───────────────────────────┤
│ R3  │ Batch Grounding DINO across 4 anchor frames (m10:326-336):  │ ⏳   │ 🅼6 (T18 in execution      │
│     │   one DINO forward with pixel_values = stack(frames_np[     │ TODO │ checklist below)          │
│     │   [0,4,8,12]]) vs 4 sequential calls; ~18% per-clip,         │      │                            │
│     │   compounds atop R1                                          │      │                            │
├─────┼────────────────────────────────────────────────────────────┼──────┼───────────────────────────┤
│ R4  │ torch.compile m04d RAFT (~1.5-2× compute speedup)           │ ⏳   │ 🅼8 (T42 — NEW). Iter13   │
│     │                                                              │ M8   │ trap was dynamic=True;     │
│     │                                                              │      │ fix is dynamic=False +     │
│     │                                                              │      │ reduce-overhead + fixed    │
│     │                                                              │      │ input shape (same recipe   │
│     │                                                              │      │ as M7). FAIL LOUD on       │
│     │                                                              │      │ Inductor — no eager        │
│     │                                                              │      │ fallback per CLAUDE.md.    │
├─────┼────────────────────────────────────────────────────────────┼──────┼───────────────────────────┤
│ R5  │ torch.compile DINO with fixed-resolution input              │ ⏳   │ 🅼7 (T41 — new; executes  │
│     │   (Pro 6000 only, after M6); ~1.3-1.5× DINO speedup atop    │ POST-│ ONLY on Pro 6000 after M6 │
│     │   M6's batching; same compound_prompt sidesteps HF #32206   │ M6   │ lands).                    │
└─────┴────────────────────────────────────────────────────────────┴──────┴───────────────────────────┘
```

📊 **Compound Stage-2 + Stage-3 wall projection** (Pro 6000, FULL=115K corpus,
6 parallel workers):

```
Stage 2 (m04d RAFT)
  baseline (iter15)          ~8 hr   ████████
  + R2 (sizer fix)           ~5 hr   █████
  + R2 + R4 (M8)             ~3 hr   ███

Stage 3 (m10 DINO+SAM3)
  baseline (iter15)          ~75 hr  ████████████████████████████████████████
  + R1 (SAM3.1)              ~11 hr  ██████
  + R1 + R3 (M6)             ~ 9 hr  █████
  + R1 + R3 + R5 (M7)        ~ 6 hr  ███
```

---

## 📋 Execution checklist

```
┌──────┬───────────────────────────────────────────────────────────────────┬────────┐
│  #   │ Task                                                                │ Status │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T35  │ 🅼1 — clip_pool_ratio yaml + get_probe_split + get_clip_pool_size + │ ⏳     │
│      │      load_subset_with_labels kwarg + probe_action caller            │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T36  │ 🅼2 — max_epochs single source of truth in base_optimization.yaml   │ ⏳     │
│      │      (delete duplicates in pretrain/surgery yamls)                  │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T37  │ 🅼3 — gen_full_local_manifest.py (NEW, ~30 LoC)                     │ ⏳     │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T38  │ 🅼4 — checkpoint.saves_per_epoch: 2 → 9                             │ ⏳     │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T18  │ 🅼6 — DINO 4-anchor batched inference in m10_sam_segment.py        │ ⏳     │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T42  │ 🅼8 — torch.compile m04d RAFT (R4 unparked) — pin shapes + yaml    │ ⏳     │
│      │      gate; iter13 InductorError fix is dynamic=False + fixed shape │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T41  │ 🅼7 — torch.compile DINO on Pro 6000 (R5) — POST-M6 + post-migrate │ ⏳     │
│      │      (only relevant once Stage 3 runs on Pro 6000)                  │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T39  │ 📒 Runbook: flip ⏳ M1/M2/M3 → ✅ + add ⏳ M4 + ⏳ M6 + ⏳ M7 + ⏳ M8 │ ⏳     │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T40  │ 🪦 Move this plan → iter16/legacy/ after all above ✅              │ ⏳     │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T19  │ (existing) 3-check + GPU SANITY smoke test on Pro 4000 24 GB        │ ⏳     │
└──────┴───────────────────────────────────────────────────────────────────┴────────┘
```

---

## 🗂️ Critical files

```
┌──────────────────────────────────────────────────┬───────────────────────────────────┬──────────┐
│ 📄 Path                                           │ ✏️ Change                          │ 📐 ~LoC   │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🆕 configs/pipeline.yaml                          │ + clip_pool_ratio block (M1)       │ +10      │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 configs/train/base_optimization.yaml           │ M2: update max_epochs L196         │ +1/−2    │
│                                                   │ M4: saves_per_epoch L315 (2 → 9)   │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ ❌ configs/train/pretrain_encoder.yaml            │ M2: DELETE max_epochs L79-82       │ −5       │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ ❌ configs/train/surgery_base.yaml                │ M2: DELETE max_epochs L89-97       │ −9       │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 src/utils/config.py                            │ M1: + get_probe_split()            │ +25      │
│                                                   │ + get_clip_pool_size(mode, n_full) │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 src/utils/action_labels.py                     │ M1: load_subset_with_labels        │ +5       │
│                                                   │ accepts optional clip_keys kwarg    │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 src/probe_action.py                            │ M1: wire mode → subsample →        │ +15      │
│                                                   │ get_probe_split → stratified_split │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🆕 src/utils/gen_full_local_manifest.py           │ M3: ~30 LoC manifest generator     │ +30      │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 src/m10_sam_segment.py                         │ M6: refactor L326-336 to batch     │ +30/−20  │
│                                                   │ 4 DINO calls into one               │          │
│                                                   │ M7: torch.compile DINO wrapper      │ +8       │
│                                                   │ (Pro 6000 only, guarded by env)     │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 🔧 src/m04d_motion_features.py                    │ M8: torch.compile RAFT-Large wrap  │ +12/−5   │
│                                                   │ (dynamic=False + reduce-overhead);  │          │
│                                                   │ replaces L222-230 eager-fallback    │          │
│                                                   │ comment block with real compile     │          │
├──────────────────────────────────────────────────┼───────────────────────────────────┼──────────┤
│ 📒 iter/iter16_train_115kclips/                   │ Flip ⏳ M1/M2/M3 → ✅; ADD ⏳ M4 +  │ +90/−5   │
│   runbook_train_115kclips.md                      │ ⏳ M6 + ⏳ M7 + ⏳ M8 sections       │          │
└──────────────────────────────────────────────────┴───────────────────────────────────┴──────────┘
```

🛑 **NO** new `pretrain_2X_encoder.yaml` (shell-level --max-epochs handles 2×).
🛑 **NO** consumer changes downstream of `action_labels.json` (schema unchanged).

---

## 🧠 Implementation per section

### 🅼1 — clip_pool_ratio + probe_split helpers + caller

**(a) `configs/pipeline.yaml`** — new top-level block after `probe_split`:

```yaml
# iter16 M1 (2026-05-20): mode-keyed FRACTION of FULL corpus. N_full read at
# runtime from data/full_local/full_local.json. All POC/SANITY clip counts
# are derived; NO hardcoded literals anywhere in src/*.py.
clip_pool_ratio:
  full:   1.00      # 100% — every clip in the master manifest
  poc:    0.10      # 10%  → ~11,500 @ N_full = 115,000
  sanity: 0.01      # 1%   → ~ 1,150 @ N_full = 115,000
```

**(b) `src/utils/config.py`** — two helpers (after `get_pipeline_config`):

```python
def get_probe_split() -> dict:
    """Return {train_pct, val_pct} — identical for sanity/poc/full (parity)."""
    return get_pipeline_config()["probe_split"]


def get_clip_pool_size(mode: str, n_full: int) -> int:
    """Mode-keyed clip count: round(n_full × clip_pool_ratio[mode]).

    iter16 M1: derives POC/SANITY clip counts from the FULL corpus size at
    runtime. mode ∈ {sanity, poc, full} (case-insensitive). FAIL LOUD on
    unknown mode via dict KeyError per CLAUDE.md.
    """
    ratio = get_pipeline_config()["clip_pool_ratio"][mode.lower()]
    return int(round(n_full * ratio))
```

**(c) `src/utils/action_labels.py:144`** — add optional `clip_keys` kwarg:

```python
def load_subset_with_labels(subset_path, motion_features_path, *,
                            min_clips_per_class=MIN_CLIPS_PER_CLASS_DEFAULT,
                            clip_keys=None):                       # 🆕 NEW kwarg
    """...

    iter16 M1: when `clip_keys` is supplied, it OVERRIDES the clip_keys read
    from subset_path. Used by probe_action to inject a clip_pool_ratio-derived
    subsample of the master manifest WITHOUT writing a temp JSON file.
    """
    ...
    if clip_keys is None:                                          # 🆕 NEW gate
        with open(subset_path) as f:
            clip_keys = json.load(f)["clip_keys"]
    ...
```

**(d) `src/probe_action.py`** — `run_labels_stage` (~L280-286):

```python
from utils.config import (
    add_local_data_arg, check_gpu, get_paired_deltas,
    get_pipeline_config, get_probe_split, get_clip_pool_size,    # 🆕 NEW
)
...
def run_labels_stage(args, wb):
    # 🅼1: derive mode string for yaml lookup
    mode = "sanity" if args.SANITY else ("poc" if args.POC else "full")

    # 🅼1: subsample master manifest to mode-appropriate clip count
    with open(args.eval_subset) as f:
        manifest = json.load(f)
    n_full = len(manifest["clip_keys"])
    n_clips = get_clip_pool_size(mode, n_full)
    pool_keys = sorted(manifest["clip_keys"])[:n_clips]
    print(f"[M1 clip_pool_ratio] mode={mode}: subsampled "
          f"{n_clips:,}/{n_full:,} clips (ratio = {n_clips/n_full:.2%})")

    records, class_names = load_subset_with_labels(
        args.eval_subset, args.motion_features,
        min_clips_per_class=args.min_clips_per_class,
        clip_keys=pool_keys,                                        # 🆕 NEW
    )

    # 🅼1: probe_split also from yaml (single source of truth)
    split_cfg = get_probe_split()
    splits = stratified_split(
        records,
        train_pct=split_cfg["train_pct"],
        val_pct=split_cfg["val_pct"],
        seed=args.seed,
        min_per_split=args.min_per_split,
    )
    ...
```

### 🅼2 — max_epochs single source of truth

**(e) `configs/train/base_optimization.yaml:196-199`** — UPDATE values:

```yaml
max_epochs:
  sanity: 1        # code-path validation only
  poc:    2        # POC↔FULL parity exception per CLAUDE.md
  full:   1        # iter16 — was 15 (iter11 v2 SSL canon, now obsolete)
```

**(f) `configs/train/pretrain_encoder.yaml:79-82`** — DELETE the block:

```yaml
# DELETE these lines (current state: sanity:1, poc:2, full:5).
# pretrain_encoder.yaml now inherits max_epochs from base_optimization.yaml.
```

**(g) `configs/train/surgery_base.yaml:89-97`** — DELETE the block.

> 🚫 No new `pretrain_2X_encoder.yaml`. `run_train.sh:248-252` reads
> `optimization.max_epochs.${mode}` from `pretrain_encoder.yaml` (which now
> inherits base full=1) and passes `--max-epochs 2` to m09a1 → pretrain_2X
> gets 2 epochs via shell-level override.

### 🅼3 — gen_full_local_manifest.py

**(h)** New file `src/utils/gen_full_local_manifest.py` (~30 LoC):

```python
"""Generate data/full_local/full_local.json master manifest from tags.json.

USAGE:
    python -u src/utils/gen_full_local_manifest.py 2>&1 \\
        | tee logs/gen_full_local_manifest_$(date +%Y%m%d_%H%M%S).log
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

### 🅼4 (NEW) — saves_per_epoch bump for iter16 1-epoch FULL trajectory

**(i) `configs/train/base_optimization.yaml:315`** — single value bump:

```yaml
checkpoint:
  saves_per_epoch: 9    # iter16 (2026-05-20): bumped 2 → 9. 1 epoch × ~86K
                        # train clips ÷ 9 saves ≈ 9.6K clips between probe-trio
                        # validations → 9 trajectory points per cell.
                        # Mode-invariant per POC↔FULL parity.
  keep_last_n: 5
```

### 🅼6 (NEW) — DINO 4-anchor batched inference in m10_sam_segment.py

**(j)** Refactor `src/m10_sam_segment.py:326-336` — the per-anchor loop
becomes one batched DINO call:

```python
# ❌ OLD (4 sequential DINO forwards per clip):
dino_per_anchor = {}
H = W = 0
all_detected_categories = set()
for a in anchors:
    dino_out = detect_boxes_grounding_dino(
        dino_processor, dino_model, frames_np[a], compound_prompt,
        box_threshold, text_threshold,
    )
    H, W = dino_out["H"], dino_out["W"]
    dino_per_anchor[a] = dino_out["boxes_by_cat"]
    all_detected_categories.update(dino_out["boxes_by_cat"].keys())

# ✅ NEW (1 batched DINO forward per clip):
batch_outs = detect_boxes_grounding_dino_batched(
    dino_processor, dino_model,
    frames_batch=frames_np[anchors],     # shape (4, H, W, 3)
    compound_prompt=compound_prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
)
H, W = frames_np.shape[1], frames_np.shape[2]
dino_per_anchor = {a: batch_outs[i]["boxes_by_cat"] for i, a in enumerate(anchors)}
all_detected_categories = set()
for out in batch_outs:
    all_detected_categories.update(out["boxes_by_cat"].keys())
```

New helper `detect_boxes_grounding_dino_batched` lives next to the existing
`detect_boxes_grounding_dino` (kept for callers that want single-frame
inference). HF processor accepts a list of PIL images + identical
compound_prompt for all → sidesteps HF Transformers issue #32206 safely.

### 🅼7 (NEW) — torch.compile DINO on Pro 6000 (R5)

🚦 **Activation gate**: M7 lands ONLY after BOTH (a) M6 is merged AND
(b) Stage 3 has migrated to Pro 6000 96 GB (see runbook Stage 3 ⚠️ migrate
note). On Pro 4000 the compile overhead is not amortized — explicit gate
prevents premature activation.

**(k) `configs/pipeline.yaml`** — add `dino_compile` block:

```yaml
# iter16 M7 (R5): torch.compile gate for Grounding DINO. Pro 6000 only.
# Default = false on Pro 4000; flip to true ONLY when running Stage 3 on
# the 96 GB box.
dino_compile:
  enabled:        false     # Pro 4000 default — compile overhead not worth it
  mode:           "reduce-overhead"
  dynamic:        false     # fixed pixel_values shape (HF #34556)
  fullgraph:      false     # tolerate graph breaks at non-tensor ops
```

**(l) `src/m10_sam_segment.py`** — wrap DINO model after load (~L200, near
`detect_boxes_grounding_dino_batched` helper from M6):

```python
# 🅼7: torch.compile wrapper, gated on yaml + GPU class.
cfg = get_pipeline_config()
if cfg["dino_compile"]["enabled"]:
    print(f"[M7 dino_compile] mode={cfg['dino_compile']['mode']} "
          f"dynamic={cfg['dino_compile']['dynamic']} — compiling DINO")
    dino_model = torch.compile(
        dino_model,
        mode=cfg["dino_compile"]["mode"],
        dynamic=cfg["dino_compile"]["dynamic"],
        fullgraph=cfg["dino_compile"]["fullgraph"],
    )
    # Warmup: first 2 forward passes JIT-compile; FAIL LOUD on Inductor errors
    # (NO eager fallback — per CLAUDE.md "no silent fallback").
else:
    print(f"[M7 dino_compile] disabled — running DINO in eager mode")
```

> ⚠️ **FAIL LOUD policy**: if `torch.compile` raises an InductorError on
> warmup, the script MUST exit non-zero (NO `try/except` that falls back to
> eager). Per CLAUDE.md "No CPU/eager fallback in inference scripts" + "no
> silent failures = garbage metrics". The operator either fixes the
> upstream issue or sets `dino_compile.enabled: false` in pipeline.yaml.

🔬 **WebSearch citation**: HF Transformers issue #34556 confirms
`reduce-overhead` + `dynamic=False` works on Grounding DINO when caption is
fixed across batch — exactly the M6 case (compound_prompt is identical
across all 4 anchor frames).

### 🅼8 (NEW) — torch.compile m04d RAFT-Large

🎯 **Root cause**: m04d:222-230 currently runs RAFT-Large in eager mode
with a comment citing an InductorError. The actual fix is `dynamic=False`
+ `reduce-overhead` mode + the fixed (H,W) resize that m04d already
performs — same recipe M7 uses for DINO.

**(m) `configs/pipeline.yaml`** — add `m04d_compile` block:

```yaml
# iter16 M8: torch.compile gate for m04d RAFT-Large on PyTorch 2.12 nightly.
# dynamic=False is mandatory — RAFT's correlation pyramid breaks under
# Inductor dynamic-shape recompilation. m04d already pins input shape via
# fixed resize, so the constraint is satisfied.
m04d_compile:
  enabled:        true
  mode:           "reduce-overhead"
  dynamic:        false
  fullgraph:      false
```

**(n) `src/m04d_motion_features.py:222-230`** — replace the eager-only
block with the compile call:

```python
raft_model = raft_model.eval()
cfg = get_pipeline_config()
if cfg["m04d_compile"]["enabled"]:
    print(f"[M8 m04d_compile] mode={cfg['m04d_compile']['mode']} "
          f"dynamic={cfg['m04d_compile']['dynamic']} — compiling RAFT-Large")
    raft_model = torch.compile(
        raft_model,
        mode=cfg["m04d_compile"]["mode"],
        dynamic=cfg["m04d_compile"]["dynamic"],
        fullgraph=cfg["m04d_compile"]["fullgraph"],
    )
else:
    print(f"[M8 m04d_compile] disabled — running RAFT in eager mode")
```

🛑 **FAIL LOUD**: if compile raises an InductorError on warmup, the script
exits non-zero. No `try/except` eager-fallback — per CLAUDE.md "No CPU/eager
fallback in inference scripts".

🔬 **WebSearch action required during T42**:
1. Pull the exact iter13 InductorError from `iter/iter13/errors_N_fixes.md`.
2. Confirm on PyTorch GitHub (≥2 sources) that `dynamic=False` +
   `reduce-overhead` resolves it for RAFT-Large on 2.12 nightly.
3. If WebSearch reveals a different upstream-validated fix, revise the
   yaml + code above — no blind recipe-copy from M7.

📊 **Expected impact**: Stage 2 wall (m04d motion features) ~8 hr → ~3 hr
on Pro 6000 = 5 GPU-hr saved per training cell.

---

## 📒 Runbook updates

In `iter/iter16_train_115kclips/runbook_train_115kclips.md`:

```
┌────┬──────────────────────────────────────────────────────────────────┐
│ #  │ Edit                                                              │
├────┼──────────────────────────────────────────────────────────────────┤
│ 1  │ L12  — flip ⏳ M1 → ✅; body updates to reference landed code      │
│ 2  │ L99  — flip ⏳ M2 → ✅                                              │
│ 3  │ L138 — flip ⏳ M3 → ✅                                              │
│ 4  │ ADD ⏳ M4 between M3 and M5 — saves_per_epoch bump                 │
│ 5  │ ADD ⏳ M6 after M5 — DINO 4-anchor batched inference               │
│ 6  │ ADD ⏳ M7 after M6 — torch.compile DINO (Pro 6000 gate)            │
│ 7  │ ADD ⏳ M8 after M7 — torch.compile m04d RAFT (dynamic=False)        │
│ 8  │ ADD reference to "R1-R5 status" table in this plan (Stage 2 +     │
│    │ Stage 3 sections) — all 5 R-items addressed in iter16              │
└────┴──────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Verification (per-task, all CPU-side on Pro 4000 24 GB)

```bash
# === 🅼1 ===
venv_walkindia/bin/python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import get_probe_split, get_clip_pool_size
print('probe_split:', get_probe_split())
for m in ('full', 'poc', 'sanity'):
    print(f'clip_pool_size({m}, 115687) =', get_clip_pool_size(m, 115687))
"
# Expected: probe_split {train_pct:0.75, val_pct:0.05}; clip counts
#   full=115687, poc=11569, sanity=1157

# End-to-end probe_action.py --FULL with the new subsampling path:
CACHE_POLICY_ALL=2 venv_walkindia/bin/python -u src/probe_action.py --FULL \
    --stage labels --eval-subset data/eval_10k_local/eval_10k.json \
    --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
    --min-clips-per-class 34 --min-per-split 5 \
    --output-root outputs/full/probe_action --no-wandb 2>&1 | grep '\[M1'

# === 🅼2 ===
grep -A4 '^max_epochs:\|^  max_epochs:' \
    configs/train/base_optimization.yaml \
    configs/train/pretrain_encoder.yaml \
    configs/train/surgery_base.yaml
# Expected: only base_optimization.yaml shows max_epochs; others absent.

# === 🅼3 ===
venv_walkindia/bin/python src/utils/gen_full_local_manifest.py
venv_walkindia/bin/python -c "
import json
d = json.load(open('data/full_local/full_local.json'))
print(f\"n={d['n']:,} num_videos={d['num_videos']:,} first_key={d['clip_keys'][0]}\")
"
# Expected: n=115687 num_videos=714 first_key=<some clip path>

# === 🅼4 ===
grep saves_per_epoch configs/train/base_optimization.yaml
# Expected: saves_per_epoch: 9

# === 🅼6 ===
# Smoke-test signature (full GPU run is part of T19):
venv_walkindia/bin/python -c "
import sys; sys.path.insert(0, 'src')
from m10_sam_segment import detect_boxes_grounding_dino_batched
import inspect; print(inspect.signature(detect_boxes_grounding_dino_batched))
"

# === 🅼7 ===
# Pro 4000: verify gate defaults to OFF
grep -A4 '^dino_compile:' configs/pipeline.yaml
# Expected: dino_compile.enabled: false

# Pro 6000 only — after migration, flip flag + run SANITY:
# sed -i 's/enabled:        false/enabled:        true/' configs/pipeline.yaml
# (then) CACHE_POLICY_ALL=2 ./scripts/m10_pretrain_subset.sh SANITY 2>&1 | \
#     tee logs/m10_m7_smoke_$(date +%Y%m%d_%H%M%S).log
# Expected: [M7 dino_compile] mode=reduce-overhead dynamic=False — compiling DINO

# === 🅼8 ===
grep -A4 '^m04d_compile:' configs/pipeline.yaml
# Expected: m04d_compile.enabled: true, dynamic: false

# SANITY smoke (Pro 4000): verify RAFT compiles + produces non-NaN features
CACHE_POLICY_ALL=2 venv_walkindia/bin/python -u src/m04d_motion_features.py \
    --SANITY --no-wandb 2>&1 | tee logs/m04d_m8_smoke_$(date +%Y%m%d_%H%M%S).log | \
    grep -E '\[M8|raft|NaN'
# Expected: [M8 m04d_compile] mode=reduce-overhead dynamic=False — compiling RAFT-Large
#           NO "NaN" lines in stderr (iter13 trap was NaN-producing kernel fusion)

# === Final 3-check gate (per CLAUDE.md TESTING & VALIDATION) ===
for f in src/utils/config.py src/utils/action_labels.py src/probe_action.py \
         src/utils/gen_full_local_manifest.py src/m10_sam_segment.py \
         src/m04d_motion_features.py; do
  venv_walkindia/bin/python -m py_compile "$f"
  venv_walkindia/bin/ruff check --select F,E9 "$f" | tail -1
done

# === Regression: M5 tests still pass ===
venv_walkindia/bin/python tests/test_action_labels.py    # expect 10/10 pass
```

---

## ⚠️ Risks & explicit non-decisions

```
┌──────────────────────────────────────────────────────────┬────────────────────────────┐
│ ⚠️ Risk                                                   │ 🛡️ Mitigation               │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M1 caller change in probe_action.py touches the canonical│ tests/test_action_labels.py │
│ run_labels_stage path                                     │ regression-checked (10/10); │
│                                                           │ SANITY + FULL paths re-     │
│                                                           │ verified end-to-end.        │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M1: clip_pool_ratio.sanity = 0.01 × 115K = 1,150 clips    │ Already discussed in M5     │
│ may still FAIL min_per_split=5 at SANITY due to sparse    │ runbook. Operator action:   │
│ per-class video coverage                                  │ raise sanity ratio or lower │
│                                                           │ min_per_split.              │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M2: deleting max_epochs blocks may break a downstream     │ run_train.sh:250 uses       │
│ yaml_extract call in run_train.sh:248-252                 │ pretrain_encoder.yaml;      │
│                                                           │ inheritance returns base.   │
│                                                           │ full=1 → 1×2=2 epochs OK.   │
│                                                           │ Verify post-edit.           │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M3: data/full_local/full_local.json will be ~120 MB       │ Disk space verified earlier:│
│ (115K entries × ~1 KB each)                               │ 417 GB free.                │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M4: saves_per_epoch=9 adds ~27 min/cell wall (9 probe-    │ Acceptable trade for        │
│ trio invocations at ~3 min each)                          │ trajectory plot fidelity.   │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M6: HF Grounding DINO batched inference works only when   │ ✅ Our compound_prompt is    │
│ all images share the same caption (HF #32206)             │ identical across anchors.   │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M7: torch.compile compile-time warmup adds ~30 s before   │ One-shot cost per m10       │
│ first DINO forward (cached after that)                    │ worker startup; amortized   │
│                                                           │ over 100s of clips.         │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ M8: iter13 RAFT InductorError may not be fully resolved   │ WebSearch (per CLAUDE.md    │
│ by dynamic=False alone                                    │ "NO LAZY FIX") during T42   │
│                                                           │ to confirm upstream fix     │
│                                                           │ before flipping enabled:    │
│                                                           │ true. SANITY smoke catches  │
│                                                           │ NaN regressions early.      │
└──────────────────────────────────────────────────────────┴────────────────────────────┘
```

### 🚫 Explicit non-decisions

- 🚫 No `pretrain_2X_encoder.yaml` (shell-level --max-epochs handles 2×).
- 🚫 No yaml `mode_invariant: bool` flag for saves_per_epoch (just bump it).
- 🚫 No per-mode pre-made manifest JSON shortcuts (would bypass clip_pool_ratio's
       single-source-of-truth design).

---

## 🚀 Execution order

```
1. T35 — 🅼1 — clip_pool_ratio yaml + helpers + load_subset_with_labels kwarg
              + probe_action caller update + smoke-test
2. T36 — 🅼2 — base_optimization.yaml max_epochs update + DELETE blocks from
              pretrain_encoder.yaml + surgery_base.yaml + verify shell still works
3. T37 — 🅼3 — write gen_full_local_manifest.py + run + verify
              full_local.json output (~120 MB)
4. T38 — 🅼4 — base_optimization.yaml saves_per_epoch 2 → 9
5. T18 — 🅼6 — refactor m10_sam_segment.py:326-336 to batched DINO + smoke-test
6. T42 — 🅼8 — torch.compile m04d RAFT (yaml + m04d:222-230 + WebSearch
              + SANITY smoke confirming no NaN regression)
7. T41 — 🅼7 — torch.compile DINO yaml gate + m10 wrapper (Pro 4000 lands
              gate as OFF; Pro 6000 flips ON post-migration)
8. T39 — 📒 — flip ⏳ M1/M2/M3 → ✅ in runbook + add ⏳ M4 + ⏳ M6 + ⏳ M7 + ⏳ M8
9. T40 — 🪦 — move this plan → iter16/legacy/
10.T19 — (separate) 3-check gate + GPU SANITY smoke test on Pro 4000
```

🕒 **Estimated wall**: ~5-6 hours including verification.
   M1 ~1.5h · M2/M3/M4 ~minutes each · M6 ~1h · M7 ~30 min (yaml + wrapper) ·
   M8 ~1.5h (WebSearch + SANITY smoke) · runbook/cleanup ~10 min.

---

## 📦 Commit message (when all M1-M8 + runbook caught up)

```
iter16 M1+M2+M3+M4+M6+M7+M8 — close runbook code-mod gaps + throughput R1-R5

M1: clip_pool_ratio + probe_split helpers + load_subset_with_labels kwarg +
    probe_action wires both for runtime mode-keyed subsampling
M2: max_epochs consolidated to base_optimization.yaml (single source);
    duplicates DELETED from pretrain_encoder + surgery_base yamls
M3: src/utils/gen_full_local_manifest.py — manifest writer for 115K corpus
M4: base.checkpoint.saves_per_epoch 2 → 9 (iter16 1-epoch FULL trajectory)
M6: m10_sam_segment.py DINO batched 4-anchor inference (~18% per-clip; R3)
M7: m10 torch.compile DINO yaml gate (Pro 6000 only; ~1.3-1.5× DINO; R5)
M8: m04d torch.compile RAFT-Large (dynamic=False + reduce-overhead; R4
    unparked from iter13; Stage 2 wall ~8 hr → ~3 hr)

All 5 throughput recommendations addressed in iter16:
  R1 ✅ SAM3 → SAM3.1 (surgery_base.yaml:156, landed earlier)
  R2 ✅ m04d sizer decoupling (motion_initial_bs: 8, landed earlier)
  R3 ✅ M6 (this PR)
  R4 ✅ M8 (this PR)
  R5 ✅ M7 (this PR; gate OFF on Pro 4000, flip ON for Pro 6000)

Plan archived at iter/iter16_train_115kclips/legacy/plan_code_modifications.md
Tests: tests/test_action_labels.py 10/10 pass; ruff F,E9 clean across all
src/*.py touched. Ready for Stage 1 (HF download walkindia-200k).
```
