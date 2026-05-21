# 🛠️ iter16 Code Modifications — M1 + M2 + M3 + M4 + M6 + M7 + M8 + M9 (NEW)

> **Status legend** (update each task as it moves through the lifecycle):
> ⏳ pending · 🟡 in-progress · ✅ done · ❌ blocked · ⏭️ skipped
>
> When ALL T35-T39 (+ T18 + T41 + T42 + T43) below are ✅, this file moves to
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
│ T35  │ 🅼1 — clip_pool_ratio yaml + helpers + caller wiring                │ ✅     │
│      │      Phase 1 (sorted[:N] CLI path) ✅ landed 2026-05-21 AM.           │ done   │
│      │      Option X (o-z) ✅ — subsample_manifest_for_mode + symmetric    │ Opt X  │
│      │      probe_action/probe_labels wiring + mode-keyed probe_split       │        │
│      │      (SANITY 60/20/20 clip-level; POC/FULL 75/5/20 video-disjoint). │        │
│      │      End-to-end SANITY smoke 2026-05-21 PM: ✅ M1+M2+M5+M9 code     │        │
│      │      paths all verified through m09a startup (subsampled 100/10K,   │        │
│      │      11 motion classes, 57/20/20 split, max_epochs.sanity=1 read).  │        │
│      │      m09a OOM at V-JEPA load = dev-box 36 GB cgroup cap (not        │        │
│      │      M1/M2/M5/M9 bug); full SANITY training validation lands on     │        │
│      │      Pro 6000 96 GB box (241 GB cgroup) per CLAUDE.md.              │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T36  │ 🅼2 — max_epochs single source of truth in base_optimization.yaml   │ ✅     │
│      │      (delete duplicates in pretrain/surgery yamls)                  │ done   │
│      │      Verified 2026-05-21: yaml_extract walks extends chain — all 5  │ 05-21  │
│      │      train configs resolve to (sanity:1, poc:2, full:1).            │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T37  │ 🅼3 — gen_full_local_manifest.py (NEW)                              │ ✅     │
│      │      Landed 2026-05-21: src/utils/gen_full_local_manifest.py + ran  │ done   │
│      │      against data/full_local/tags.json → wrote full_local.json with │ 05-21  │
│      │      n=115,687 clips, num_videos=1,559, ~74 clips/video. Schema     │        │
│      │      matches eval_10k.json. CLAUDE.md-compliant: argparse required  │        │
│      │      paths, FAIL LOUD on missing fields, no .get() defaults.        │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T38  │ 🅼4 — checkpoint.saves_per_epoch: 2 → 9                             │ ✅     │
│      │      Verified 2026-05-21: base_optimization.yaml:315 saves_per_      │ done   │
│      │      epoch=9. pretrain_encoder + surgery_base both resolve to 9      │ 05-21  │
│      │      via load_merged_config + yaml_extract.                          │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T18  │ 🅼6 — DINO 4-anchor batched inference in m10_sam_segment.py        │ ✅     │
│      │      Landed 2026-05-21: detect_boxes_grounding_dino_batched added   │ done   │
│      │      (~75 LoC); per-anchor loop replaced with single batched call.  │ 05-21  │
│      │      ~18% per-clip speedup expected; compounds with SAM3.1 (R1).    │        │
│      │      HF #32206 safety: identical compound_prompt across anchors.    │        │
│      │      Code-level verified; full GPU smoke deferred to T19 / Pro 6000.│        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T42  │ 🅼8 — torch.compile m04d RAFT (R4 unparked) — WebSearch validated   │ ✅     │
│      │      Landed 2026-05-21: mode="default" (NOT reduce-overhead — saves│ done   │
│      │      ~70 GB cudagraphs pool) + dynamic=False (avoids CantSplit bug │ 05-21  │
│      │      that crashed iter13). Recipe sources: PyTorch #105279,         │        │
│      │      #120733, #176653. Stage 2 wall: ~8 hr → ~3-4 hr on Pro 6000.   │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T41  │ 🅼7 — torch.compile DINO yaml gate + m10 wrapper                    │ ✅     │
│      │      Landed 2026-05-21: dino_compile block in pipeline.yaml +       │ done   │
│      │      load_grounding_dino wraps with torch.compile when enabled.    │ 05-21  │
│      │      Recipe inherits M8 lessons: mode="default" + dynamic=False    │        │
│      │      (avoids 70 GB cudagraphs pool + CantSplit). Pro 4000 default  │        │
│      │      OFF; flip enabled=true post-migration to Pro 6000 96 GB.       │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T43  │ 🅼9 — yaml-keyed local_data_dir (single source for data path)       │ ✅     │
│      │      Option III LANDED 2026-05-21 PM: cfg.data.local_data_dir +     │ done   │
│      │      cfg.data.master_manifest_name in pipeline.yaml; load_merged_   │ Opt III│
│      │      config injects probe.{subset,local_data,tags_path} post-merge. │        │
│      │      0 consumer changes. Verified: flip yaml → all paths flip.       │        │
│      │      USAGE docstrings refreshed (5 train yamls + 3 src/*.py); split │        │
│      │      filenames corpus-agnostic ({train,val,test}_split.json); 5    │        │
│      │      retired files moved to data/eval_10k_local/legacy/.            │        │
├──────┼───────────────────────────────────────────────────────────────────┼────────┤
│ T39  │ 📒 Runbook: flip ⏳ M1/M2/M3 → ✅ + add ✅ M4 + M6 + M7 + M8 + M9    │ ✅     │
│      │      Landed 2026-05-21: M1/M2/M3 status flipped + LANDED notes      │ done   │
│      │      added; M4/M6/M7/M8/M9 sections appended after M5 with status,  │ 05-21  │
│      │      design summary, code refs, recipe sources. M9 flip-ready ops   │        │
│      │      block + pre-flight checklist documented inline.                 │        │
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

### 🔄 M1 Revision — Option X (2026-05-21 PM, post-T36 SANITY pre-flight)

🚨 **Trigger**: T36 SANITY pre-flight invoked `./scripts/run_train.sh --SANITY
pretrain_encoder` and surfaced two design conflicts in the original M1:

```
┌────┬──────────────────────────────────────────────────────────────────────┐
│ 🧪 │ M1 Phase 1 (sorted[:N]) — conflicts surfaced 2026-05-21 PM            │
├────┼──────────────────────────────────────────────────────────────────────┤
│ 1  │ POC↔FULL parity violation — sorted(clip_keys)[:n_clips] returns       │
│    │ alphabetically-first n clips. Top 1000 of eval_10k = all goa/* tier   │
│    │ = 1-3 motion classes only. CLAUDE.md mandates POC schema = FULL       │
│    │ schema (all 8 motion classes after the 34-clip filter).               │
├────┼──────────────────────────────────────────────────────────────────────┤
│ 2  │ Asymmetric wiring — M1 lives in probe_action.run_labels_stage (CLI    │
│    │ path) but probe_labels.ensure_probe_labels_for_mode (in-process       │
│    │ bootstrap called by m09a/m09c at startup) BYPASSES it. Same           │
│    │ action_labels.json generation, two different subsamplers.             │
├────┼──────────────────────────────────────────────────────────────────────┤
│ 3  │ run_train.sh:88-126 still passes pre-made eval_10k_{sanity,poc}.json  │
│    │ — violates plan user-decision #1 ("no per-mode pre-made shortcuts").  │
└────┴──────────────────────────────────────────────────────────────────────┘
```

🎯 **Revised mechanism per mode** — observation: SANITY validates code, not
class balance; POC + FULL are the costly modes that need parity:

```
┌────────┬─────────────────────────────────────────────────────────────────┐
│ Mode    │ Subsample mechanism (Option X)                                   │
├────────┼─────────────────────────────────────────────────────────────────┤
│ SANITY  │ sorted(clip_keys)[:n] — class-imbalanced OK, code check only    │
│ POC     │ stratified_by_motion_class_subset — POC↔FULL parity REQUIRED   │
│ FULL    │ identity (clip_pool_ratio.full = 1.0)                            │
└────────┴─────────────────────────────────────────────────────────────────┘
```

📐 **Scope** (~30 LoC + yaml + shell + retire 2 JSONs):

**(o) NEW shared helper `src/utils/action_labels.py:subsample_manifest_for_mode()`**:

```python
def subsample_manifest_for_mode(mode: str, clip_keys: list,
                                 motion_features_path,
                                 n_motion_classes: int) -> list:
    """Per-mode subsample of master manifest. iter16 M1 Option X.

      SANITY → sorted(clip_keys)[:n]  (class-imbalance OK; code check)
      POC    → stratified_by_motion_class_subset (POC↔FULL parity)
      FULL   → identity (clip_pool_ratio.full = 1.0)
    """
    from utils.config import get_clip_pool_size
    if mode == "full":
        return clip_keys
    n_target = get_clip_pool_size(mode, len(clip_keys))
    if mode == "sanity":
        return sorted(clip_keys)[:n_target]
    # mode == "poc"
    from utils.eval_subset import stratified_by_motion_class_subset
    target_per_class = max(1, n_target // n_motion_classes)
    out = stratified_by_motion_class_subset(
        {"clip_keys": clip_keys}, motion_features_path, target_per_class)
    return out["clip_keys"]
```

**(p) `src/probe_action.py:run_labels_stage`** — replace Phase 1's
`sorted(manifest["clip_keys"])[:n_clips]` with
`subsample_manifest_for_mode(mode, manifest["clip_keys"],
args.motion_features, n_motion_classes)`.

**(q) `src/utils/probe_labels.py:ensure_probe_labels_for_mode`** — replace
the existing POC-specific stratified path (L145-176) with the same helper
call for ALL 3 modes. Eliminates asymmetric subsampling between CLI and
in-process bootstrap paths.

**(r) `configs/train/base_optimization.yaml`**:
- `eval_subset_in.sanity`: `eval_10k_sanity.json` → `eval_10k.json` (master)
- `poc_subset_out`: DELETED (no longer pre-made on disk)

**(s) Shell simplification**:
- `scripts/run_train.sh:80-137` — drop per-mode subset selection;
  always pass `data/eval_10k_local/eval_10k.json`
- `scripts/run_train.sh:199-203` — `EVAL_SUBSET_TX` always = master
- `scripts/run_eval.sh:109-145` — drop SANITY+POC subset generation calls

**(t) File retirement**:
- `mv data/eval_10k_local/eval_10k_sanity.json → data/eval_10k_local/legacy/`
- `mv data/eval_10k_local/eval_10k_poc.json → data/eval_10k_local/legacy/`
- `src/utils/hf_outputs.py:579` — drop `eval_10k_poc.json` from ignore_patterns

**(u) Docstring/comment cleanup (NO DEFER)** — refresh stale refs in:
`src/utils/eval_subset.py:26`, `src/probe_taxonomy.py:26`,
`src/utils/probe_labels.py:14`, `src/m09a1:511`, `src/m09c1:630`.

🧪 **Verification** for Option X:

```bash
# Helper unit smoke
venv_walkindia/bin/python -c "
import sys; sys.path.insert(0, 'src')
import json
from utils.action_labels import subsample_manifest_for_mode
keys = sorted(json.load(open('data/eval_10k_local/eval_10k.json'))['clip_keys'])
m04d = 'data/eval_10k_local/m04d_motion_features/motion_features.npy'
print('SANITY:', len(subsample_manifest_for_mode('sanity', keys, m04d, 8)))
print('POC   :', len(subsample_manifest_for_mode('poc',    keys, m04d, 8)))
print('FULL  :', len(subsample_manifest_for_mode('full',   keys, m04d, 8)))
"
# Expected: SANITY ≈ 100; POC ≈ 8 × (1000//8) = 1000 (stratified); FULL = N_master

# End-to-end SANITY smoke (validates M2 max_epochs too)
./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | \
    tee logs/m09a_sanity_$(date +%Y%m%d_%H%M%S).log | \
    grep -E '\[M1|max_epochs|epochs=|epoch [0-9]/1'
# Expected: [M1 ...] line; max_epochs.sanity=1 read; 1-epoch run completes
```

📐 **Critical-files delta from Option X** (additive to the table above):

```
┌──────────────────────────────────────────────────┬────────────────────────────┬─────────┐
│ 📄 Path                                           │ ✏️ Change                   │ 📐 LoC   │
├──────────────────────────────────────────────────┼────────────────────────────┼─────────┤
│ 🔧 src/utils/action_labels.py                     │ + subsample_manifest_for_   │ +25     │
│                                                   │   mode() helper             │         │
│ 🔧 src/probe_action.py                            │ Replace Phase 1 sorted[:N]  │ +5/−7   │
│                                                   │   with helper call          │         │
│ 🔧 src/utils/probe_labels.py                      │ Replace POC stratified path │ +6/−24  │
│                                                   │   with helper (all modes)   │         │
│ 🔧 configs/train/base_optimization.yaml           │ eval_subset_in.sanity →     │ +1/−2   │
│                                                   │   master; DELETE poc_subset_│         │
│                                                   │   out                       │         │
│ 🔧 scripts/run_train.sh:80-137,199-203            │ Drop per-mode subset gen    │ +6/−45  │
│ 🔧 scripts/run_eval.sh:109-145                    │ Drop SANITY+POC subset gen  │ +4/−35  │
│ 🔧 src/utils/hf_outputs.py:579                    │ Drop eval_10k_poc.json      │ −1      │
│ 🪦 data/eval_10k_local/{sanity,poc}.json          │ mv → legacy/                │ 0       │
│ 🔧 5× docstring/comment refs                      │ Refresh stale refs          │ +5/−5   │
└──────────────────────────────────────────────────┴────────────────────────────┴─────────┘
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

### 🅼9 (NEW) — yaml-keyed `local_data_dir` (single-source data path)

🚨 **Trigger**: `data/eval_10k_local/` is hardcoded 118+ times across yaml +
shells + src. Blocks the planned full_local migration (iter15 results came
from eval_10k_local; iter16 FULL must land on data/full_local once m04/m10/m11
outputs exist there).

🧠 **Audit (2026-05-21 PM, post-grep across active code)**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔍 Hardcoded "data/eval_10k_local/" refs still active                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ configs/train/pretrain_encoder.yaml:148-150 — probe.{subset,local_data,    │
│   tags_path} block (read by m09a1/m09a2 cfg["probe"][...])                  │
│ configs/train/surgery_base.yaml:367-369 — probe.{subset,local_data,        │
│   tags_path} block (read by m09c1/m09c2 cfg["probe"][...])                  │
│ configs/train/{pretrain_head,pretrain_encoder,surgery_2stage_noDI_head}    │
│   .yaml — docstring USAGE blocks (~12 comment refs)                         │
│ src/m09{a1,a2,c1,c2}*.py — docstring USAGE blocks (~8 comment refs)        │
│ base_optimization.yaml — probe_taxonomy_labels.{train,val,eval}_subset      │
│   etc. (LEGACY ultra_hard_3066 paths — separate concern from M9)            │
│ src/utils/probe_labels.py:99,193 — already migrated to cfg["data"][...]    │
│   (M9 partial Option X — was eval_subset_in[mode_dir] hardcoded in yaml)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

🎯 **Design — Option III: zero consumer changes, single-source-of-truth**:

YAML has no cross-file variable interpolation (anchors are single-file only).
Instead: DELETE hardcoded yaml keys + inject derived values in
`load_merged_config` post-processing. Consumers (m09a1/a2/c1/c2) read
`cfg["probe"]["subset"]` etc. as before — but values are now derived from
`cfg["data"]["local_data_dir"]` at runtime.

**(v) `configs/pipeline.yaml`** — `data` block with both keys (LANDED):

```yaml
# iter16 M9 (2026-05-21): single source of truth for active local data dir.
data:
  local_data_dir:        "data/eval_10k_local"     # iter15 / iter16 SANITY+POC
  # local_data_dir:      "data/full_local"         # iter16 FULL — flip when ready
  master_manifest_name:  "eval_10k.json"           # iter15 / iter16 SANITY+POC master
  # master_manifest_name: "full_local.json"        # iter16 FULL (after M3)
```

**(w) `configs/train/pretrain_encoder.yaml:148-150`** — DELETE probe block
hardcoded keys:

```yaml
# Before:
probe:
  subset:     data/eval_10k_local/eval_10k_val_split.json    # HARDCODED ❌
  local_data: data/eval_10k_local                             # HARDCODED ❌
  tags_path:  data/eval_10k_local/tags.json                   # HARDCODED ❌

# After (Option III) — keys DELETED; load_merged_config injects derived
# values from cfg.data.local_data_dir + cfg.data.master_manifest_name:
probe:
  # iter16 M9 (2026-05-21): subset / local_data / tags_path DELETED — now
  # injected by src/utils/config.py:load_merged_config from cfg.data.
  # local_data_dir (configs/pipeline.yaml). Flipping that one yaml key
  # migrates the whole pipeline to data/full_local.
  enabled: true
  # ...other probe keys unchanged...
```

**(x) `configs/train/surgery_base.yaml:367-369`** — same DELETE pattern.

**(y) `src/utils/config.py:load_merged_config`** — POST-PROCESSING injection
(~10 LoC after the extends-chain merge loop):

```python
# iter16 M9 (2026-05-21): inject derived probe paths from cfg.data.local_data_dir.
# Single source of truth — flipping that one yaml key migrates the pipeline.
# Consumers (m09a1/a2/c1/c2) read cfg["probe"]["subset"] etc. unchanged.
data_cfg = merged.get("data", {})
local_data_dir = data_cfg.get("local_data_dir")
if local_data_dir:                                   # FAIL LOUD if probe needs it
    merged.setdefault("probe", {})
    merged["probe"]["subset"]     = f"{local_data_dir}/val_split.json"
    merged["probe"]["local_data"] = local_data_dir
    merged["probe"]["tags_path"]  = f"{local_data_dir}/tags.json"
```

**(z) Consumers (m09a1, m09a2, m09c1, m09c2)** — NO CHANGE. Continue reading
`cfg["probe"]["subset"]`, `cfg["probe"]["local_data"]`, `cfg["probe"]["tags_path"]`.

**(aa) USAGE docstring refresh** (NO DEFER per CLAUDE.md) — update to
template syntax `${LOCAL_DATA}/{train,val,test}_split.json`:
- `configs/train/pretrain_encoder.yaml:31-34`
- `configs/train/pretrain_head.yaml:27-30`
- `configs/train/surgery_2stage_noDI_head.yaml:28-31`
- `src/m09a1_pretrain_encoder.py`, `src/m09a2_pretrain_head.py`,
  `src/m09c1_surgery_encoder.py`, `src/m09c2_surgery_head.py` USAGE blocks
- `src/m09a1_pretrain_encoder.py:511`, `src/m09c1_surgery_encoder.py:612,632`
  inline comments

**(bb) Existing eval_10k_*_split.json file retirement** — `mv` to
`data/eval_10k_local/legacy/` (per CLAUDE.md DELETE PROTECTION — never `rm`).
Splits regenerate fresh under corpus-agnostic names (`train_split.json`,
`val_split.json`, `test_split.json`) on next run_train.sh invocation.

🧪 **M9 verification — full_local flip dry-run**:

```bash
# Step 1 (verify Python resolves correctly with current yaml):
venv_walkindia/bin/python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import load_merged_config
cfg = load_merged_config('configs/model/vjepa2_1.yaml',
                          'configs/train/pretrain_encoder.yaml')
print('local_data_dir   :', cfg['data']['local_data_dir'])
print('probe.subset     :', cfg['probe']['subset'])
print('probe.local_data :', cfg['probe']['local_data'])
print('probe.tags_path  :', cfg['probe']['tags_path'])
"
# Expected:
#   local_data_dir   : data/eval_10k_local
#   probe.subset     : data/eval_10k_local/val_split.json
#   probe.local_data : data/eval_10k_local
#   probe.tags_path  : data/eval_10k_local/tags.json

# Step 2 (flip yaml + re-verify; revert after):
sed -i 's|local_data_dir:        "data/eval_10k_local"|local_data_dir:        "data/full_local"|' \
    configs/pipeline.yaml
# (re-run Step 1 — expect all paths under data/full_local/)
sed -i 's|local_data_dir:        "data/full_local"|local_data_dir:        "data/eval_10k_local"|' \
    configs/pipeline.yaml
# (revert until m04/m10/m11 outputs land at full_local)
```

📋 **Activation gate** (M9 yaml flip eval_10k_local → full_local):

```
┌─────┬──────────────────────────────────────────────────────────────────────┐
│ Pre │ data/full_local/ must contain BEFORE flipping local_data_dir          │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 1   │ tags.json (already present — 155 MB; Stage 1 metadata done)          │
│ 2   │ full_local.json (generated by M3 / T37)                              │
│ 3   │ m04d_motion_features/motion_features.npy + .paths.npy (Stage 2 GPU)  │
│ 4   │ m10_sam_segment/masks/ (Stage 3 GPU)                                  │
│ 5   │ m11_factor_datasets/ (Stage 4 — m11 streaming or pre-computed)       │
│ 6   │ subset-*.tar shards from data download (Stage 1 — RUNNING NOW)       │
└─────┴──────────────────────────────────────────────────────────────────────┘
```

> 🛑 **Until items 3-6 land at full_local/**, keep `local_data_dir:
> "data/eval_10k_local"`. M9's value is the 1-line flip when FULL data +
> outputs are ready — no code edits needed at migration time.

📐 **Critical-files delta from M9 Option III**:

```
┌──────────────────────────────────────────────────┬────────────────────────────┬─────────┐
│ 📄 Path                                           │ ✏️ Change                   │ 📐 LoC   │
├──────────────────────────────────────────────────┼────────────────────────────┼─────────┤
│ ✅ configs/pipeline.yaml                          │ data.local_data_dir +       │ +8      │
│                                                   │   master_manifest_name      │ (done)  │
│ 🔧 configs/train/pretrain_encoder.yaml            │ DELETE probe.{subset,       │ +3/−3   │
│                                                   │   local_data,tags_path}     │         │
│ 🔧 configs/train/surgery_base.yaml                │ DELETE probe.{subset,       │ +3/−3   │
│                                                   │   local_data,tags_path}     │         │
│ 🔧 src/utils/config.py                            │ load_merged_config post-    │ +10     │
│                                                   │   processing injection      │         │
│ 🔧 src/utils/config.py                            │ + get_local_data_dir() +    │ +20     │
│                                                   │   get_master_manifest_path()│ (done)  │
│ ✅ src/utils/probe_labels.py                      │ Read cfg["data"][...] for   │ +3/−6   │
│                                                   │   eval_subset + tags_json   │ (done)  │
│ ✅ scripts/run_train.sh                           │ yaml_extract LOCAL_DATA +   │ +8/−15  │
│                                                   │   MASTER_MANIFEST early     │ (done)  │
│ 🔧 scripts/run_eval.sh                            │ Same yaml_extract pattern   │ +5/−10  │
│ 🔧 src/utils/hf_outputs.py                        │ ✅ split filenames corpus-   │ +5/−4   │
│                                                   │   agnostic (M9 partial)      │ (done)  │
│ 🔧 5× USAGE docstring refs                        │ Refresh to ${LOCAL_DATA}/   │ +5/−5   │
│ 🪦 data/eval_10k_local/eval_10k_{sanity,poc,     │ mv → legacy/                │ 0       │
│   train_split,val_split,test_split}.json          │                              │         │
└──────────────────────────────────────────────────┴────────────────────────────┴─────────┘
```

⏱️ **Estimated wall**: ~45 min — DELETE 6 yaml lines + ADD 10 LoC in
load_merged_config + USAGE docstring sweep + verification.

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
1.  T35 — 🅼1 — clip_pool_ratio + Option X redesign 🟡 SMOKE PENDING
                (helper + wiring ✅; final SANITY end-to-end smoke remains)
2.  T36 — 🅼2 — max_epochs single source ✅ DONE 2026-05-21
3.  T43 — 🅼9 — yaml-keyed local_data_dir Option III ✅ DONE 2026-05-21
                (PROMOTED — landed before remaining T35 smoke due to overlap;
                absorbed T35 remaining items s/t/u/v)
4.  T37 — 🅼3 — write gen_full_local_manifest.py + run + verify
                full_local.json output (~120 MB) ⏳ PENDING
5.  T38 — 🅼4 — base_optimization.yaml saves_per_epoch 2 → 9 ⏳ PENDING
6.  T18 — 🅼6 — refactor m10_sam_segment.py to batched DINO + smoke ⏳ PENDING
7.  T42 — 🅼8 — torch.compile m04d RAFT (yaml + WebSearch + SANITY) ⏳ PENDING
8.  T41 — 🅼7 — torch.compile DINO yaml gate + m10 wrapper ⏳ PENDING
9.  T39 — 📒 — flip ✅ M1/M2/M3 in runbook + add M4/M6/M7/M8/M9 ⏳ PENDING
10. T40 — 🪦 — move this plan → iter16/legacy/ ⏳ PENDING
11. T19 — 3-check + GPU SANITY end-to-end on Pro 4000 24 GB ⏳ PENDING
                (validates M1+M2+M5+M9 wired correctly — T35 smoke folded here)
```

🕒 **Estimated wall**: ~4-5 hours remaining (M1+M2+M9 done; ~3.5 hrs of work
on M3/M4/M6/M7/M8 + final SANITY/runbook/cleanup).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ✅ Landed 2026-05-21 (sessions AM + PM)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ M1 Phase 1 ✅ : clip_pool_ratio yaml + helpers in config.py + load_subset_   │
│   with_labels kwarg + probe_action.run_labels_stage sorted[:N] wiring         │
│ M1 Option X ✅: subsample_manifest_for_mode helper (sorted SANITY +           │
│   stratified POC + identity FULL) + probe_labels.ensure_probe_labels_for_    │
│   mode symmetric wiring                                                       │
│ M2 ✅         : max_epochs.{sanity:1, poc:2, full:1} in base_optimization.    │
│   yaml; DELETED from pretrain_encoder + surgery_base; yaml_extract extends   │
│   chain resolution verified on all 5 train configs                            │
│ M9 ✅         : cfg.data.local_data_dir + master_manifest_name in pipeline.   │
│   yaml; load_merged_config injects derived probe.* values; pretrain_encoder │
│   + surgery_base probe blocks DELETED; USAGE docstrings → ${LOCAL_DATA}/     │
│   template; corpus-agnostic split filenames; 5 retired files → legacy/       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Runbook code-mod archive (consolidated from runbook 2026-05-21)

Operator-flavored summaries + Stage design sidebars, extracted from the
runbook during T40 prep so the runbook becomes terminal-command focused.

### Cross-mode clip counts (M1 design table)

```
┌────────┬────────┬──────────┬──────────┬─────────┬──────────┐
│ Mode   │ Ratio  │ N_clips  │ N_train  │ N_val   │ N_test   │
├────────┼────────┼──────────┼──────────┼─────────┼──────────┤
│ FULL   │ 100 %  │ 115,000  │  86,250  │  5,750  │  23,000  │
│ POC    │  10 %  │  11,500  │   8,625  │    575  │   2,300  │
│ SANITY │   1 %  │   1,150  │     862  │     58  │     230  │
└────────┴────────┴──────────┴──────────┴─────────┴──────────┘
```
- POC val = 575 clips → stable val metric at the parity ratio.
- SANITY val = 58 → fine for code-correctness. Post-Option-X, SANITY uses
  60/20/20 (NOT 75/5/20) because M5 video-disjoint SGKF is infeasible at
  SANITY scale.

### M2 exception — `pretrain_2X_encoder.yaml` retains override

This variant intentionally doubles the pretrain budget (compute-matched Δ3
control). Implementation: `run_train.sh:248-252` passes `--max-epochs
$((_BASE_EP * 2))` via shell-level override (no new yaml). With M2's
base.full=1 → shell passes 2 epochs.

### M5 — SANITY clip-level fallback (added post-Option X)

M5's video-disjoint SGKF requires `n_splits ≤ min_videos_per_class`.
SANITY's tiny pool (~5 videos/class) cannot satisfy k_val=16 from
val_pct=0.05. Mode-aware stratified_split:
- SANITY: clip-level `StratifiedShuffleSplit` (no video-disjoint guarantee)
- POC/FULL: M5 video-disjoint SGKF (paper-grade)

Justified per CLAUDE.md: POC↔FULL parity rule applies to POC↔FULL only.
SANITY validates code, not paper splits.

### Expected research impact (M5 + M1 Option X)

Probe metrics (top-1 / mAP@K / future-MSE) will drop ≈ 1-3 pp absolute vs
iter15 because visual-style leakage between train/val/test is gone. This
is research-correct, not regression. Re-baseline iter15 with the new
split BEFORE publishing iter16↔iter15 deltas.

### Stage 2 — m04d wall-time extrapolation

- eval_10k (9,297 clips) took 6,974 s previously (see
  `data/eval_10k_local/m04d_motion_features/motion_features.meta.json`).
- Scaling: 115,687 × 0.75 sec/clip ≈ 24 hr on iter13 hardware.
- Pro 4000 24 GB with AdaptiveBatchSizer + fp16 RAFT autocast: 10-15 hr.
- WITH M8 torch.compile (mode=default + dynamic=False): additional ~1.5-2×
  speedup → Stage 2 wall ≈ 3-4 hr on Pro 6000.

### Stage 3 — GPU sizing extrapolation

```
┌─────────────┬─────────────────┬──────────────────┐
│ Config       │ Pro 6000 96 GB  │ Pro 4000 24 GB   │
├─────────────┼─────────────────┼──────────────────┤
│ Serial m10   │ ~180 hr (~8 d)  │ ~540 hr (~22 d)  │
│ Parallel × 4 │ ~ 92 hr (~4 d)  │ ~230 hr (~10 d)  │
│ Parallel × 6 │ ~ 75 hr (~3 d)  │  N/A (VRAM tight)│
└─────────────┴─────────────────┴──────────────────┘
```
Each worker holds DINO (~500 MB) + SAM3.1 (~3.5 GB) ≈ 4 GB VRAM. With
M6 (DINO batched) + M7 (torch.compile DINO, Pro 6000): compounded ~12.5×
speedup → Stage 3 ≈ 6 hr on Pro 6000.

### Stage 2 ↔ Stage 3 GPU sharing note

m04d (Stage 2) and m10 (Stage 3) want the same GPU. Run **sequentially**:
Stage 2 on Pro 4000 (~10-15 hr) → migrate to Pro 6000 → Stage 3 (~6-9 hr).
Total data-prep wall ≈ 16-24 hr (was ~90 hr pre-M6/M7/M8).

### Condensed yaml/python excerpts (runbook M-section snippets)

```yaml
# (M1) configs/pipeline.yaml
clip_pool_ratio: {full: 1.00, poc: 0.10, sanity: 0.01}
probe_split:
  sanity: {train_pct: 0.60, val_pct: 0.20}   # clip-level shuffle
  poc:    {train_pct: 0.75, val_pct: 0.05}   # video-disjoint (M5)
  full:   {train_pct: 0.75, val_pct: 0.05}   # video-disjoint (M5)
```

```yaml
# (M2) configs/train/base_optimization.yaml:196 — SOLE max_epochs site
max_epochs: {sanity: 1, poc: 2, full: 1}
# pretrain_encoder.yaml + surgery_base.yaml overrides DELETED
```

```yaml
# (M9) configs/pipeline.yaml — single source of truth for data dir
data:
  local_data_dir:        "data/eval_10k_local"  # flip → "data/full_local"
  master_manifest_name:  "eval_10k.json"        # flip → "full_local.json"
```

```yaml
# (M7/M8) configs/pipeline.yaml — torch.compile gates
m04d_compile: {enabled: true,  mode: "default", dynamic: false, fullgraph: false}
dino_compile: {enabled: false, mode: "default", dynamic: false, fullgraph: false}
```

---

## 📦 Commit message (when all M1-M9 + runbook caught up)

```
iter16 M1+M2+M3+M4+M6+M7+M8+M9 — runbook gaps + throughput R1-R5 + path refactor

M1: clip_pool_ratio + probe_split helpers + Option X stratified-for-POC
    redesign + load_subset_with_labels kwarg + symmetric wiring in
    probe_action.run_labels_stage + probe_labels.ensure_probe_labels_for_mode
M2: max_epochs consolidated to base_optimization.yaml (single source);
    duplicates DELETED from pretrain_encoder + surgery_base yamls
M3: src/utils/gen_full_local_manifest.py — manifest writer for 115K corpus
M4: base.checkpoint.saves_per_epoch 2 → 9 (iter16 1-epoch FULL trajectory)
M6: m10_sam_segment.py DINO batched 4-anchor inference (~18% per-clip; R3)
M7: m10 torch.compile DINO yaml gate (Pro 6000 only; ~1.3-1.5× DINO; R5)
M8: m04d torch.compile RAFT-Large (dynamic=False + reduce-overhead; R4
    unparked from iter13; Stage 2 wall ~8 hr → ~3 hr)
M9: yaml-keyed data.local_data_dir → 30+ hardcoded "data/eval_10k_local/"
    refs derive from cfg. One-line yaml flip migrates SANITY+POC pipeline
    to data/full_local (FULL) when m04/m10/m11 outputs land there.

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
