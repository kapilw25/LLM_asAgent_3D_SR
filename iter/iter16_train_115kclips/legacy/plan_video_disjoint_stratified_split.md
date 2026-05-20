# 🎬 Video-Disjoint `stratified_split()` — iter16 M5 Plan

> **Status legend** (update each task as it moves through the lifecycle):
> ⏳ pending · 🟡 in-progress · ✅ done · ❌ blocked · ⏭️ skipped
>
> When ALL T27-T33 below are ✅, this file moves to
> `iter/iter16_train_115kclips/legacy/` per T34.

---

## 🎯 Why this change

`stratified_split()` at `src/utils/action_labels.py:234-277` is the **single
source of truth** for the train ↔ val ↔ test partition across the entire
factorjepa pipeline. Today it shuffles **clip-keys** within each motion class
and slices them 70/15/15 — so clips from the **same `video_id`** routinely
straddle splits.

```
┌─────────────────────────────────┬────────────────────────────────────┐
│ 🩺 Audit finding                 │ 🔍 Evidence                         │
├─────────────────────────────────┼────────────────────────────────────┤
│ ✅ Clip-level disjoint splits    │ lines 271-276 partition every clip │
│ ✅ Class-stratified (motion cls) │ per-class for-loop @ line 249      │
│ ❌ VIDEO-LEVEL DISJOINTNESS      │ ❌ NOT enforced anywhere           │
│ ❌ City / section diversity      │ ❌ NOT considered                  │
└─────────────────────────────────┴────────────────────────────────────┘
```

🚨 **Consequence**: visual style & background content from the same source
video bleed into train AND val AND test → probe-trio top-1 / mAP@K /
future-MSE numbers are **leakage-inflated** by an estimated **1-3 pp absolute**
(to be confirmed by the iter16 re-baseline).

🎯 **Goal**: every clip from a given `video_id` lands in **exactly one** of
{train, val, test}, while per-class proportions stay within ±5% of
`probe_split.train_pct / val_pct / test_pct`.

🔒 **User decisions captured in plan-mode Phase 3**:

```
┌────┬──────────────────────────────────────────────────────────────────┐
│ #  │ Decision                                                          │
├────┼──────────────────────────────────────────────────────────────────┤
│ 1  │ 🔨 HARD replace stratified_split body (no yaml A/B flag)          │
│ 2  │ 💥 FAIL LOUD on SANITY infeasibility (no clip-level fallback)      │
│ 3  │ 📒 Track as new M5 section in iter16 runbook                       │
└────┴──────────────────────────────────────────────────────────────────┘
```

---

## 📋 Execution checklist

```
┌────┬──────────────────────────────────────────────────────────────┬────────┐
│ #  │ Task                                                          │ Status │
├────┼──────────────────────────────────────────────────────────────┼────────┤
│ T27│ Add _extract_video_id helper + sklearn import                 │ ⏳     │
│ T28│ Rewrite stratified_split() body (hybrid SGKF + inner-split)   │ ⏳     │
│ T29│ Update self-test diagnostic (video-disjointness print)        │ ⏳     │
│ T30│ Add tests/test_action_labels.py (5 unit + 1 integration)      │ ⏳     │
│ T31│ Run pytest + py_compile + ruff + SANITY integration           │ ⏳     │
│ T32│ Update src/MEMORY.md action_labels entry                      │ ⏳     │
│ T33│ Add ⏳ M5 section to iter16 runbook                            │ ⏳     │
│ T34│ Move this plan file → iter16/legacy/ after all above ✅       │ ⏳     │
└────┴──────────────────────────────────────────────────────────────┴────────┘
```

---

## 🗂️ Critical files

```
┌──────────────────────────────────────────────────┬───────────────────────────┬──────────┐
│ 📄 Path                                           │ ✏️ Change                  │ 📐 ~LoC   │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 🔧 src/utils/action_labels.py                     │ + _extract_video_id helper │ +90/−45  │
│    (L234-277 = current body · L222 = record       │ + sklearn import           │          │
│     schema · L288 = JSON output)                   │ 🔨 REWRITE stratified_split│          │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 🎯 src/probe_action.py                            │ (optional) 1-line post-    │ +0 / +2  │
│    (L285-286 = call site)                          │ call sanity-print          │          │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 🪪 src/utils/probe_labels.py                      │ ✅ NO CHANGE                │ 0        │
│    (L186 = call site; signature unchanged)         │                            │          │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 🧪 src/utils/action_labels.py self-test (L322)    │ Add video-disjoint check   │ +5       │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 📒 src/MEMORY.md (L73)                            │ Note iter16 video-disjoint │ +2/−1    │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 📒 iter/iter16_train_115kclips/                   │ + ⏳ M5 section before     │ +35      │
│    runbook_train_115kclips.md                     │ Stage 1                    │          │
├──────────────────────────────────────────────────┼───────────────────────────┼──────────┤
│ 🧪 tests/test_action_labels.py (NEW)              │ 5 unit + 1 integration     │ +80      │
└──────────────────────────────────────────────────┴───────────────────────────┴──────────┘
```

🛑 **NO yaml changes** — `configs/pipeline.yaml > probe_split` already has
`train_pct: 0.75, val_pct: 0.05` (added in earlier iter16 M1 work).
🛑 **NO `action_labels.json` schema change** — output stays
`{clip_key: {class, class_id, split}}`; downstream readers untouched.

---

## ♻️ Existing helpers to reuse / consult

```
┌────────────────────────────────────────────────────────────┬─────────────────────────────────┐
│ 📦 Function (file:line)                                     │ 🪄 Role in this plan             │
├────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 📖 src/m00c_sample_subset.py:30-55 load_clips_by_video()   │ Reference pattern for video    │
│                                                             │ grouping (different schema; do  │
│                                                             │ NOT import — copy the idiom)   │
├────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 🧰 sklearn.model_selection.StratifiedGroupKFold            │ ✅ already pinned by            │
│    (`scikit-learn>=1.3.0` per requirements.txt:39)          │ requirements.txt — primary lib  │
├────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 💾 src/utils/action_labels.py:280 write_action_labels_json │ ✅ unchanged — output schema     │
│                                                             │ preserved                       │
├────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 📥 src/utils/action_labels.py:222 load_subset_with_labels  │ ✅ unchanged — record schema     │
│                                                             │ {clip_key, class, class_id}     │
└────────────────────────────────────────────────────────────┴─────────────────────────────────┘
```

---

## 🧠 Algorithm — hybrid sklearn `StratifiedGroupKFold` + inner split

### 🪜 Two phases (both video-disjoint, both class-stratified)

```
┌────────────────────────────────────────────────────────────────────────┐
│  📥 records (clip_key, class, class_id)                                │
│           │                                                             │
│           ▼                                                             │
│  🔍 _extract_video_id(clip_key)  → parallel arrays                     │
│           │                                                             │
│           ▼                                                             │
│  🎲 Phase 1 — SGKF #1 (random_state=seed)                              │
│      ✂️  carve out TEST  (k_test = round(1/test_pct))                   │
│           │                                                             │
│           ▼                                                             │
│  🎲 Phase 2 — SGKF #2 (random_state=seed+1) on TRAIN+VAL pool          │
│      ✂️  carve out VAL   (val_inner = val_pct / (train_pct+val_pct))   │
│           │                                                             │
│           ▼                                                             │
│  🔢 vectorised per-class per-split count                                │
│           │                                                             │
│           ▼                                                             │
│  💥 min_per_split assert  (FAIL LOUD with diagnostics)                 │
│           │                                                             │
│           ▼                                                             │
│  🛡️  defense-in-depth straddler assert  (BUG-only path)                │
│           │                                                             │
│           ▼                                                             │
│  📤 returns {clip_key: "train"|"val"|"test"}                            │
└────────────────────────────────────────────────────────────────────────┘
```

### 🪞 Why **hybrid** (not pure-sklearn-twice, not pure-custom)

```
┌─────────────────────────────────┬──────────────────────────────────────────────────┐
│ Option                          │ Verdict                                          │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
│ 🅰️ Pure sklearn × 2              │ ❌ Brittle when val_pct=0.05 → 1/0.05=20 folds;  │
│                                 │ inner split needs renormalisation anyway          │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
│ 🅱️ Pure custom (Sechidis 2011)   │ ❌ ~120 LoC + independent test surface; sklearn  │
│                                 │ reaches the same result for this corpus shape    │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
│ 🅲 ⭐ Hybrid (CHOSEN)            │ ✅ ~70 LoC; both phases lean on battle-tested    │
│                                 │ sklearn; train ↔ val renorm is one explicit line │
└─────────────────────────────────┴──────────────────────────────────────────────────┘
```

### 🧾 Pseudocode (full, with `_extract_video_id` helper)

```python
# top of src/utils/action_labels.py — new helper
def _extract_video_id(clip_key: str) -> str:
    """Extract video_id from canonical 5-part clip_key.

    Format: <section>/<city>/<action>/<video_id>/<video_id>-<clip_num>.mp4
    Verified across eval_10k_local, full_local, val_1k_local manifests
    (10K+ samples; 5-part depth is consistent).
    """
    parts = clip_key.split("/")
    if len(parts) < 5:
        raise ValueError(
            f"clip_key '{clip_key}' has {len(parts)} parts; need 5 "
            f"(<section>/<city>/<action>/<video_id>/<file>.mp4). "
            f"Fix upstream manifest schema; do not silently fall through."
        )
    return parts[-2]


# inside stratified_split(records, train_pct, val_pct, seed, *, min_per_split)

# 🅢tep 1 — build parallel arrays from records
clip_keys = np.array([r["clip_key"] for r in records])
class_ids = np.array([r["class_id"] for r in records])
video_ids = np.array([_extract_video_id(k) for k in clip_keys])

n_classes = int(class_ids.max() + 1)
n_videos  = len(np.unique(video_ids))

# 🅢tep 2 — outer SGKF peels off TEST
test_pct = 1.0 - train_pct - val_pct
k_test   = int(round(1.0 / test_pct))                # 0.20 → 5 ; 0.15 → 7
if abs(1.0 / k_test - test_pct) > 0.03:
    raise ValueError(
        f"test_pct={test_pct:.3f} maps to k={k_test} (~{1/k_test:.3f}); "
        f"choose a test_pct that is 1/k for integer k (0.10, 0.125, 0.15, 0.20, 0.25)"
    )
sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
trainval_idx, test_idx = next(sgkf.split(np.zeros_like(class_ids),
                                          class_ids, video_ids))

# 🅢tep 3 — inner SGKF on TRAIN+VAL pool with renormalised val ratio
val_inner_pct = val_pct / (train_pct + val_pct)
k_val = int(round(1.0 / val_inner_pct))
sgkf_inner = StratifiedGroupKFold(n_splits=k_val, shuffle=True,
                                   random_state=seed + 1)
inner_train, inner_val = next(sgkf_inner.split(
    np.zeros_like(class_ids[trainval_idx]),
    class_ids[trainval_idx],
    video_ids[trainval_idx],
))
train_idx = trainval_idx[inner_train]
val_idx   = trainval_idx[inner_val]

# 🅢tep 4 — build {clip_key: split} dict
splits = {}
for i in train_idx: splits[clip_keys[i]] = "train"
for i in val_idx:   splits[clip_keys[i]] = "val"
for i in test_idx:  splits[clip_keys[i]] = "test"

# 🅢tep 5 — vectorised per-class per-split counts + min_per_split assert
split_id = np.full(len(records), -1)
split_id[train_idx], split_id[val_idx], split_id[test_idx] = 0, 1, 2
counts = np.zeros((n_classes, 3), dtype=np.int64)
np.add.at(counts, (class_ids, split_id), 1)

infeasible = [(int(c), counts[c].tolist()) for c in range(n_classes)
              if counts[c].min() < min_per_split]
if infeasible:
    video_per_class = {int(c): len(np.unique(video_ids[class_ids == c]))
                       for c in range(n_classes)}
    raise ValueError(
        f"💥 video-disjoint stratified split failed min_per_split={min_per_split} "
        f"for {len(infeasible)} class(es). Per-class [train,val,test]: {infeasible}. "
        f"Videos-per-class: {video_per_class}. "
        f"Fix options: (a) raise sanity.default/poc.default_n in pipeline.yaml, "
        f"(b) raise min_clips_per_class so sparse classes drop earlier, "
        f"(c) lower min_per_split via CLI (SANITY only)."
    )

# 🅢tep 6 — defense-in-depth: NO video straddles splits
video_to_splits = defaultdict(set)
for k, s in splits.items():
    video_to_splits[_extract_video_id(k)].add(s)
straddlers = {v: sp for v, sp in video_to_splits.items() if len(sp) > 1}
assert not straddlers, f"🐞 BUG: videos straddling splits — {straddlers}"

print(f"[stratified_split] 🎬 video-disjoint iter16+ mode: "
      f"N_clips={len(records)}, N_videos={n_videos}, N_classes={n_classes}; "
      f"train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
return splits
```

### 🎲 Determinism / POC ↔ FULL parity

- 🌱 `random_state=seed` (outer) + `random_state=seed+1` (inner) — both
  sklearn-stable; `seed+1` decorrelates the two shuffles.
- 📏 Algorithm is **corpus-size invariant**: same `(records, seed, train_pct,
  val_pct)` produces identical split structure at SANITY / POC / FULL.
- ✅ Relative class proportions per split match within ~5% across all 3 modes
  (when corpus is large enough to be feasible).

---

## 📍 Iter16 runbook integration — new ⏳ M5 section

Insert between current **M4** (`full_local.json` generator) and **Stage 1**
(HF download). M5 MUST land before Stage 1 so the first FULL probe-labels run
writes a leakage-free `action_labels.json`.

```markdown
### ⏳ M5. Video-disjoint stratified_split (close the train↔val↔test leakage gap)

🩺 Per src/CLAUDE.md > NO DEFER, NO TECH DEBT — replace the clip-key-level
shuffle in stratified_split() with a video_id-grouped class-stratified split
so no clip from a given video_id straddles {train, val, test}.

🔒 Single source of truth: src/utils/action_labels.py.

🧪 Validation gate: SANITY may FAIL LOUD if a class has all clips in 1 video.
That is correct behavior — either raise sanity.default in pipeline.yaml or
relax min_per_split via CLI. Do NOT add a clip-level fallback at SANITY
(POC↔FULL parity rule).

📘 Full algorithm + rationale: iter/iter16_train_115kclips/legacy/
   plan_video_disjoint_stratified_split.md (after T34 move).
```

---

## 🧪 Verification (end-to-end)

```
┌─────────┬─────────────────────────────────────────────────────────────────┐
│ 🔬 Level│ Test                                                             │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Unit    │ 🧪 test_extract_video_id_{happy,short,empty}: 5-part → ID;       │
│         │    <5-part → ValueError; "" → ValueError                         │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Unit    │ 🎬 test_video_disjoint: synthetic 30v × 10c × 5cls → assert     │
│         │    NO video_id in >1 split (Step-6 invariant)                    │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Unit    │ ⚖️ test_class_proportions: per-class train/val/test counts      │
│         │    within ±5% of train/val/test_pct                              │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Unit    │ 🌱 test_deterministic: same seed → byte-identical splits dict   │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Unit    │ 💥 test_infeasible_fails_loud: class with all clips in 1 video  │
│         │    + min_per_split=5 → ValueError naming the class               │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Integ.  │ 🛠️ python -u src/probe_action.py --stage labels --SANITY \      │
│         │   --eval-subset data/eval_10k_local/eval_10k_sanity.json \       │
│         │   --motion-features data/eval_10k_local/m04d_motion_features/... │
│         │ then jq: assert NO video_id appears in >1 split                  │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Regress.│ 🔁 probe_train_subset.py × 3 → 3 non-empty JSONs, clip-key      │
│         │    disjoint across splits                                        │
│         │ 🔁 probe_action.py stages 2/3, probe_motion_cos, probe_future_*  │
│         │    no crash. Expect 1-3 pp absolute DROP vs iter15 probe acc.    │
│         │ ⚠️ If NO drop occurs → new code is silently buggy.                │
├─────────┼─────────────────────────────────────────────────────────────────┤
│ Lint    │ ⚙️ post-edit-lint.sh hook auto-runs py_compile + ruff F,E9       │
│         │    on every Edit/Write per CLAUDE.md.                            │
└─────────┴─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Risks & explicit non-decisions

```
┌──────────────────────────────────────────────────────────┬────────────────────────────┐
│ ⚠️ Risk                                                   │ 🛡️ Mitigation               │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ test_pct doesn't divide evenly into 1 (e.g. 0.13)        │ Step 2 raises ValueError    │
│                                                           │ pointing user to 1/k frac.  │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ One video has >25% of class C's clips → can't hit        │ sklearn approximates;       │
│ 75/5/20 at video level for that class                    │ min_per_split=5 check       │
│                                                           │ catches catastrophic skew.  │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ SANITY (1,150 clips × 16 classes ≈ 72/class) may have    │ ValueError is BY DESIGN at  │
│ 1-video-classes                                          │ SANITY. User raises         │
│                                                           │ sanity.default in yaml.     │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ iter15 published probe numbers were leakage-inflated     │ Re-baseline iter15 with the │
│ → unfair iter16 ↔ iter15 comparison                      │ new split before publishing │
│                                                           │ deltas. Note in M5.         │
├──────────────────────────────────────────────────────────┼────────────────────────────┤
│ Future maintainer changes train_pct to non-1/k value     │ Step 2 tolerance check      │
│ silently → silently wrong split                          │ FAILS LOUD.                 │
└──────────────────────────────────────────────────────────┴────────────────────────────┘
```

### 🚫 Explicit non-decisions

- 🚫 No yaml `video_disjoint: bool` flag (HARD replace).
- 🚫 No SANITY auto-drop / clip-level fallback (FAIL LOUD everywhere).
- 🚫 No `_old_stratified_split` parallel function (iter15 path is dormant;
  no consumer needs the old behavior).

---

## 📦 Commit message (when iter15+ run-train.sh has caught up)

```
iter16 M5 — video-disjoint stratified_split closes train↔val↔test leakage gap

Replaces clip-key-level stratification (lines 234-277 of utils/action_labels.py)
with sklearn StratifiedGroupKFold over video_id groups. No video straddles
splits anymore. Output schema, callers, downstream consumers unchanged.

Expect 1-3 pp absolute drop in raw probe accuracy vs iter15 — that's visual-
style leakage removal, not regression. Re-baseline before claiming iter16↔
iter15 deltas.

Plan archived at iter/iter16_train_115kclips/legacy/
plan_video_disjoint_stratified_split.md
```

🕒 Estimated work: **~3-4 hours** including unit tests + SANITY verification
on Pro 4000 24 GB.
