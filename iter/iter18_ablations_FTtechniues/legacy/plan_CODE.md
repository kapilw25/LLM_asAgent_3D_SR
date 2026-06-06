# 🩹 Probe-pool leak fix — 🅰 val-split probes everywhere + 🅱 FAIL-LOUD disjointness guard

## 📋 Task tracker

| # | 🧩 task | 📁 where | status |
|---|---------|----------|--------|
| A1 | 🅰 m09a1 probes the 451 val split (`subset_keys_override=val_key_set`) | `src/m09a1_pretrain_encoder.py:690` | ✅ |
| A2 | 🅰 m09a2 same fix (`subset_keys_override=val_keys`) | `src/m09a2_pretrain_head.py:303` | ✅ |
| A3 | 🅰 m09c2 same fix (`subset_keys_override=val_keys`) | `src/m09c2_surgery_head.py:487` | ✅ |
| A4 | 🧹 fix stale "external val_1k" comment in 5 encoder trainers | m09c1:888 · m09e:891 · m09b:902 · m09d:919 · m09f:889 | ✅ |
| A5 | 🔧 run_train.sh `--probe-subset` → `"$VAL_SPLIT"` for the 3 leaky arms | `scripts/run_train.sh:315,480,529` | ✅ |
| A6 | 📝 docstring USAGE lines + m09a1 probe print message | 3 trainer files | ✅ |
| B1 | 🅱 add required `train_pool_keys` + overlap-RAISE guard | `src/utils/training.py:2286` (`build_probe_clips`) | ✅ |
| B2 | 🅱 `setup_probe_pipeline` forwards `train_pool_keys` | `src/utils/m09_common.py:221` | ✅ |
| B3 | 🅱 all 8 call sites pass their train-pool keys | m09c1/e/b/d/f (`_pool_set`) + m09a1/a2/c2 (`train_keys`) | ✅ |
| V1 | ✅ 3-check gate + ruff on every touched file | auto hook + `venv_walkindia/bin/ruff` | ✅ |
| V2 | ✅ CPU unit: guard raises on overlap, passes on disjoint | python snippet, no GPU | ✅ |
| V3 | 🧪 USER: SANITY gate on 4× (`iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 2`) ~30-40 min | 4× box | ⬜ |
| V4 | 🚀 USER: POC (`--mode POC --gpus 4 --cache 2`) ~17-20 h → results ~Sat afternoon PDT | 4× box | ⬜ |
| K1 | 📓 runbook STATUS: leak found+fixed, 3rd restart on 4× | `runbook.md` | ✅ |
| K2 | 🧠 memory update (leak + guard + restart) | `project_iter18_b3_contssl_built.md` + MEMORY.md | ✅ |
| K3 | 📅 daily_progress Fri Jun 5 entry (paper-goal level) | `iter/daily_progress.md` | ✅ |

Legend: ⬜ pending · 🔄 in progress · ✅ done

---

## 🧠 Context — why this exists

The mid-training probe (5 metrics, now the **best-ckpt selection criterion** since the future_l1 switch) is computed on **different pools** per trainer family:

```text
m09c1/b/d/e/f (5 encoder arms)  → probe = 451 held-out val clips  (subset_keys_override=set(val_keys))  ✅
m09a1 pretrain                  → probe = 1000-subsample of action_labels.json → ~77% its OWN train clips ❌
m09a2 / m09c2 (head arms)       → same leaky 1000-pool as m09a1                                           ❌
```

🕰️ **Root cause:** the iter13 comment *"m09a uses external val_1k → no override needed"* was true when val_1k existed (disjoint pool, deleted long ago). It silently became a leak when run_train.sh started passing `action_labels.json` (overlaps the train pool), and became **selection-critical** when the probe became the checkpoint selector. No guard exists for probe∩train (the clip_splits guard only covers `train = universe − val − test`).

🎯 **Stakes:** POC-10k is the paper run (no FULL). Pretrain's selected ckpt (the init for all 12 arms) was picked on the leaky pool → **all 13 arms restart on the 4× box after this fix.**

---

## 🅰 Fix — every trainer probes the 451 val split

The mechanism already exists (`subset_keys_override` in `build_probe_clips`, `src/utils/training.py:2286`). Replicate the m09c1 pattern (`m09c1_surgery_encoder.py:890-892`) into the 3 leaky trainers — val keys + train keys are **already in scope** in all three:

| 📁 file | probe call | ➕ add | vars already in scope |
|---------|-----------|--------|----------------------|
| `src/m09a1_pretrain_encoder.py:690` | `build_probe_clips(...)` | `subset_keys_override=val_key_set` | `val_key_set` (l.447) · `train_keys` (l.454/456) |
| `src/m09a2_pretrain_head.py:303` | `build_probe_clips(...)` | `subset_keys_override=val_keys` | `val_keys` (l.234) · `train_keys` (l.233) |
| `src/m09c2_surgery_head.py:487` | `build_probe_clips(...)` | `subset_keys_override=val_keys` | `val_keys` (l.418) · `train_keys` (l.417) |

Also:

- 🧹 Fix the stale *"(m09a uses external val_1k → no override needed)"* comment in the 5 encoder trainers (m09c1:888 · m09e:891 · m09b:902 · m09d:919 · m09f:889) — it documents the leak as a feature.
- 🖨️ Update m09a1's `[probe] decoding clips from {probe_cfg['subset']}` print to name the val split.
- 🔧 `scripts/run_train.sh`: flip `--probe-subset` from `action_labels.json` → `"$VAL_SPLIT"` at lines 315 (m09a1), 480 (m09a2), 529 (m09c2) — matches m09c-family line 447, so the declared flag matches actual behavior.
- 📝 Docstring USAGE lines in the 3 trainers: `--probe-subset action_labels.json` → `val_split.json`.
- ♻️ `knn_probe_clips: 1000` cap stays (no-op at 451/20). `--probe-action-labels` unchanged — top1 class ids still come from `action_labels.json`.

---

## 🅱 Guard — probe ∩ train = ∅ enforced at the shared layer

`src/utils/training.py` → `build_probe_clips()` (l.2286):

- Add **required keyword-only** param `train_pool_keys: set` (no default — every caller must declare).
- After subset_keys resolution (override or JSON parse), **before** the `max_clips` subsample (~l.2330):

```python
if not isinstance(train_pool_keys, (set, frozenset)):
    raise ValueError("build_probe_clips: train_pool_keys must be the trainer's SSL-train key set "
                     "(pass set() ONLY for a trainer with literally no train pool)")
overlap = set(subset_keys) & train_pool_keys
if overlap:
    raise RuntimeError(
        f"[probe-leak guard] probe pool ∩ SSL-train pool = {len(overlap)} clips "
        f"(e.g. {sorted(overlap)[:5]}) — the probe is the best-ckpt selection metric and MUST be "
        f"disjoint from training data. Pass a held-out pool (val split). "
        f"(iter18 2026-06-06: m09a probed ~77% of its own train clips.)")
```

📞 Callers to update (8 sites — each passes its existing train-pool variable):

| caller | passes |
|--------|--------|
| `src/utils/m09_common.py:221` `setup_probe_pipeline` | new required param, forwards to `build_probe_clips` |
| m09c1:890 · m09e:893 · m09b:904 · m09d:921 · m09f:891 | `train_pool_keys=_pool_set` (l.610 in each) |
| `m09a1:690` | `train_pool_keys=train_keys` |
| `m09a2:303` | `train_pool_keys=train_keys` |
| `m09c2:487` | `train_pool_keys=train_keys` |

---

## ✅ Verification

1. 🔍 3-check gate (auto hook) + `venv_walkindia/bin/ruff check --select F,E9` on all touched files.
2. 🧪 CPU guard unit (no GPU): `build_probe_clips(subset_keys_override={"x.mp4"}, train_pool_keys={"x.mp4"}, ...)` → raises `RuntimeError`; the m09c-family case (override=val, train=pool, ∅ overlap) passes.
3. 🚦 **USER runs** SANITY gate on the 4× box: `python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 2` (~30-40 min). PASS signals per arm log:
   - m09a1/m09a2/m09c2 probe lines show the **val-split N** (20 at SANITY, 451 at POC) — not 97/1000
   - zero `[probe-leak guard]` raises · 13/13 promoted
4. 🚀 **USER runs** POC: `python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 2` (~17-20 h incl. per-encoder evals + m13 finale → results ~Sat afternoon PDT).

---

## 📚 Bookkeeping (same session)

- 📓 Runbook STATUS line: probe-leak found (selection-protocol asymmetry), fixed via 🅰+🅱; 06-05 POC arms invalid → 3rd restart, on 4× via ngpu; §1.2/§2.2 eval commands unchanged.
- 🧠 Memory: update `project_iter18_b3_contssl_built.md` (leak + guard + restart) + MEMORY.md pointer.
- 📅 `iter/daily_progress.md`: Fri Jun 5 entry (paper-goal level: checkpoint selection used a partially-seen quiz for the reference model; fixed; rerunning).
