# iter18 — Runbook · 🎯 cross-set 10k retest (`subset_10k`) — tighten the 17 trained 2B arms' CIs ~2×

**WHAT (1 line):** reuse every already-trained 2B arm; re-score on a *disjoint* 10k corpus; every clip → TEST
(n≈10k vs eval_10k's ≈1825) → 95% bands shrink ~2×. The 13 head-free metrics tighten on their own; the
2 head metrics (Action top-1, taxonomy F1) reuse eval_10k's heads.

**HARD RULE:** the 17-arm retest runs through `scripts/iter18_poc_ngpu.py` **ONLY** — `EVAL_CORPUS=subset_10k`
writes results to `<bb>_2B/eval/subset_10k/` (a separate corpus dir → eval_10k never clobbered; encoders reused
from `<bb>_2B/train/` so `--cache 1` skips all training).

| 🧩 box | hardware | DAG | what |
|---|---|---|---|
| 🟢 Box 1 | RTX 2060 8 GB | #3 #4 #6 #7 | code + CPU smoke — **✅ DONE this session** |
| 🟠 Box 2 | 1× RTX 6000 96 GB | #1 #2 | prep the disjoint corpus (m04d motion-features + disjointness) |
| 🔵 Box 3 | 4× RTX 6000 96 GB | #5 #8 #9 | regen 2 reusable eval_10k artifacts → **the retest** → pick 2 OURS |

---

## 🟢 Box 1 · RTX 2060 (8 GB) — code + CPU smoke  ✅ DONE

```bash
# all plan_eval_10k.md code edits are in; these are the green CPU checks (re-runnable anywhere)
python -m py_compile scripts/iter18_poc_ngpu.py src/m04e_action_labels.py src/m12a_action_top1.py src/m12c_taxonomy_f1.py src/utils/action_labels.py src/utils/audit_disjoint.py   # syntax gate
python src/utils/audit_disjoint.py --selftest                                  # disjointness helper self-test
python src/utils/output_paths.py selftest                                      # backbone-first path helper self-test
EVAL_CORPUS=subset_10k python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --dry-run \
  --skip-arms surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder   # MUST print "[resume --cache 1] skipping 16 already-trained arms" + 203 jobs

# m04e test-all label smoke (M04D_COMPILE=0 = eager, fits the 2060's 8 GB):
M04D_COMPILE=0 python -u src/m04e_action_labels.py --SANITY --probe-split test-all \
  --eval-subset data/eval_10k_local/eval_10k.json \
  --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
  --output-root outputs/sanity/_xset_smoke --cache-policy 2   # MUST print "every clip → TEST (train=0,val=0,test=N)"
```

---

## 🟠 Box 2 · 1× RTX 6000 — prep the disjoint corpus  (DAG #1 #2)

```bash
# subset_10k tars + manifests already present; ONLY m04d motion-features is missing (DAG #1)
ls data/subset_10k_local/subset_10k.json data/subset_10k_local/tags.json     # corpus manifest + VLM tags present
test -d data/subset_10k_local/m04d_motion_features || echo "m04d ABSENT → run #1 below"

# #1 — m04d RAFT motion-features for subset_10k (compile fits 96 GB; ~1:00–1:30 wall, decode-bound)
python -u src/m04d_motion_features.py --POC \
  --eval-subset data/subset_10k_local/subset_10k.json --local-data data/subset_10k_local \
  --cache-policy 1 2>&1 | tee logs/m04d_subset_10k_$(date +%Y%m%d_%H%M%S).log
ls -la data/subset_10k_local/m04d_motion_features/motion_features.npy        # success marker

# #2 — disjointness audit: eval_10k vs subset_10k MUST show 0 exact overlap (else the retest is invalid)
python -u src/utils/audit_disjoint.py \
  --set-a data/eval_10k_local/eval_10k.json   --keys-field-a clip_keys \
  --set-b data/subset_10k_local/subset_10k.json --keys-field-b clip_keys \
  --window-clips 1 2>&1 | tee logs/audit_disjoint_$(date +%Y%m%d_%H%M%S).log    # expect exact_overlap=0

# ship the prepped corpus to HF so Box 3 can pull it (SKIP if Box 2 == Box 3)
set -a; . .env; set +a
python -u src/utils/hf_outputs.py upload-data data/subset_10k_local 2>&1 | tee logs/upload_data_subset_10k_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔵 Box 3 · 4× RTX 6000 — regen artifacts, then the RETEST  (DAG #5 #8 #9)

```bash
# pull the prepped corpus (SKIP if Box 2 == Box 3)
python -u src/utils/hf_outputs.py download-data data/subset_10k_local 2>&1 | tee logs/download_data_subset_10k_$(date +%Y%m%d_%H%M%S).log
python -u src/utils/hf_outputs.py download-data outputs/poc --ext json,csv,pt 2>&1 | tee logs/download_outputs_poc_$(date +%Y%m%d_%H%M%S).log   # trained heads/ckpts + labels
```

### 3a · regen the 2 reusable eval_10k artifacts  (artifact PREP — NOT the retest)

```bash
# class_edges.json = eval_10k's motion-bin definition; scratch output-root → never touches the trained labels/heads
python -u src/m04e_action_labels.py --POC \
  --eval-subset data/eval_10k_local/eval_10k.json \
  --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
  --output-root outputs/poc/_xset_edges --cache-policy 2 2>&1 | tee logs/m04e_xset_edges_$(date +%Y%m%d_%H%M%S).log

# VERIFY the regen's class order == the order the action heads were trained on (must pass before the retest)
E10=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k)   # eval_10k root: migrated heads + labels
python - "$E10" <<'PY'
import json, sys
e=json.load(open("outputs/poc/_xset_edges/class_edges.json"))["class_names"]
a=json.load(open(f"{sys.argv[1]}/probe_action/action_labels.json"))
m={}; [m.setdefault(v["class"],v["class_id"]) for v in a.values()]
ref=[c for c,_ in sorted(m.items(),key=lambda x:x[1])]
assert e==ref, f"MISMATCH edges={e} head={ref}"; print("OK class_names match trained head:", e)
PY

# 17-encoder roster (7 competitors + 9 OURS + frozen), single-sourced from arm_registry minus the 7 non-roster arms
DROP="pretrain_2X_encoder pretrain_head surgical_3stage_DI_head surgical_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder"
ENCS="vjepa_2_1_frozen"; for t in $(python src/utils/arm_registry.py eval-tokens); do case " $DROP " in *" $t "*) :;; *) ENCS="$ENCS vjepa_2_1_$t";; esac; done; echo "$ENCS"   # expect 17 names

# regen taxonomy heads on eval_10k: stage 11 only, cache 2 (force re-train), KEEP_PROBE_HEADS=1 (persist for reuse)
KEEP_PROBE_HEADS=1 CACHE_POLICY_ALL=2 SKIP_STAGES="1,2,3,4,5,6,7,8,8b,8c,9,9b,9c,10,12,13" \
  bash scripts/run_eval.sh --POC --encoders "$ENCS" 2>&1 | tee logs/regen_taxheads_$(date +%Y%m%d_%H%M%S).log
ls "$E10"/probe_taxonomy/vjepa_2_1_pretrain_encoder/probe_*.pt             # per-dim heads now persisted (new tree)
```

### 3b · THE RETEST — `iter18_poc_ngpu.py` ONLY  (DAG #5 + #8, one combined run)

```bash
SKIP="surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder"   # → 16 arms + frozen = 17
```

**3b-i · SANITY-scale smoke** — POC mode (reuses the trained endpoints + heads), but eval only ~200 clips.
Note: NO `--mode SANITY` — that reads `outputs/sanity/` (no trained encoders/heads). POC mode reads `outputs/poc/`.

```bash
# build a 200-clip slice of subset_10k (POC clip_pool_ratio=1.0 = identity → EVAL_SUBSET size IS the clip count)
python - <<'PY'
import json
d=json.load(open("data/subset_10k_local/subset_10k.json"))
d["clip_keys"]=d["clip_keys"][:200]; d["n"]=len(d["clip_keys"])
json.dump(d, open("data/subset_10k_local/subset_10k_smoke.json","w")); print("smoke clips:", d["n"])
PY
# same recipe as the full retest but EVAL_SUBSET=smoke + EVAL_CORPUS=subset_10k_smoke (separate eval dir, no clobber) → ~15 min
ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k_smoke \
LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k_smoke.json \
PROBE_SPLIT=test-all CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json \
EVAL_HEAD_REUSE_ROOT=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k) \
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_xset_smoke_$(date +%Y%m%d_%H%M%S).log
# green (→ <bb>_2B/eval/subset_10k_smoke/) = full cross-set path validated on REAL endpoints → clean it, then run full
```

**3b-i.clean · wipe the throwaway smoke artifacts** so the full retest starts from scratch. Deletes the
`subset_10k_smoke` eval dir ONLY — never `eval/subset_10k/`, the trained encoders/heads, or the shared frame cache.

```bash
SMOKE=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG subset_10k_smoke)   # the smoke eval dir ONLY
ls -d "$SMOKE" && rm -rf "$SMOKE"                            # guard: rm fires ONLY if the smoke eval dir exists
rm -f data/subset_10k_local/subset_10k_smoke.json           # the tiny smoke manifest (frame cache is SHARED → kept)
```

**3b-ii · the full retest** — identical recipe, full corpus, `EVAL_CORPUS=subset_10k`.

```bash
# cross-set INPUTS — exported once; every run_eval.sh subprocess inherits them (Popen inherits env)
export ITER18_BACKBONE=vjepa_2_1_vitG EVAL_CORPUS=subset_10k        # eval corpus → writes eval/subset_10k/ (no clobber)
export LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k.json   # the disjoint corpus (9566)
export PROBE_SPLIT=test-all                                          # every clip → TEST → n_test≈10k
export EVAL_HEAD_REUSE_ROOT=$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitG eval_10k)   # eval_10k heads
export CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json          # reuse eval_10k motion-bin definition

# launch: trains skipped (--cache 1 finds them at train/), 13 head-free + 2 head metrics via reuse → eval/subset_10k/
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_xset_subset10k_$(date +%Y%m%d_%H%M%S).log
# banner MUST show: "[resume --cache 1] skipping 16 already-trained arms" → results under <bb>_2B/eval/subset_10k/
```

### 3c · refresh scorecard + pick 2 OURS  (DAG #9)

```bash
# the scheduler's §3 finale auto-writes the per-backbone scorecard when all jobs pass:
ls $(python src/utils/output_paths.py plot-dir poc vjepa_2_1_vitG subset_10k)/eval/   # m13 figures + eval_metrics.{json,csv}
# pick the 2 best OURS = surgery arms with future-frame MSE win whose tightened 95% band clears base surgery
# live monitor: set the SAME EVAL_CORPUS=subset_10k in the watch pane so it reads eval/subset_10k/ (not eval_10k)
```

---

## 🧰 Teardown · 3 backups before kill / recycle / destroy (NON-NEGOTIABLE)

```bash
bash git_push.sh "<commit message>"                                  # 1) code → GitHub (prompts Kapil/Gaytri)
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
  --include "outputs/poc/**" --exclude "**/.*" 2>&1 | tee logs/upload_large_folder_$(date +%Y%m%d_%H%M%S).log   # 2) outputs → HF (the new <bb>_2B tree)
# 3) sessions+memory → Mac (run ON THE MAC): bash claude_session.sh --download --host <vast_host_alias>
```

## ⏱️ Durations (estimates · test-all = ~5× the test clips of the old per-split eval)

| op | box | wall (h:mm) |
|---|---|---:|
| #1 m04d motion-features (subset_10k, compiled) | 1× RTX 6000 | ~1:00–1:30 |
| #2 disjointness audit (CPU) | any | ~0:02 |
| 3a class_edges regen (m04e, CPU) | any | ~0:03 |
| 3a taxonomy-head regen (17 enc, lazy-extract) | 4× RTX 6000 | ~1:30–3:00 |
| 3b cross-set retest (16 arms + frozen, eval-only) | 4× RTX 6000 | ~8:00–12:00 |

## 🔁 fresh-box once-only (run before any GPU eval on a new node)

```bash
# m12f (8c) self-smoke — confirms encoder-temporal metrics wire up on this box
SKIP_STAGES="1,2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
  bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen 2>&1 | tee logs/m12f_sanity_$(date +%Y%m%d_%H%M%S).log   # 4 aggregate_*.json
```
