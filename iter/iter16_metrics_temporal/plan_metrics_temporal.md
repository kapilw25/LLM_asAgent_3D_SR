# iter16 — Predictor-only Temporal Metrics + Eval Re-home (by function) · CODE plan

> Scope: add ≥5 **predictor-only, mutually-orthogonal** temporal eval metrics (none redundant
> with `future_mse` or each other), and re-home the `probe_*.py` family BY FUNCTION in STRICT
> pipeline direction (annotation→m04, evaluation→m12, visualization→m13-LAST) — not one m12 lump.
> Predictor-only is the highest-probability arm for `surgery ≫ pretrain` (it already CI-wins
> only on `future_mse`). CONFIRMED: future_regress→legacy; metrics = 6 util-fns + 1 orchestrator.

## 📊 PROGRESS — 2026-05-28 (eval DONE 02:29 PDT · 596 min wall · m12e+m12f GPU-validated)

### ✅ DONE — §2.1 predictor suite + §6 encoder suite, CPU-checked AND GPU-SANITY-validated
```text
✅  src/utils/predictor_eval.py        🧱 shared loaders + masked_predict_l1 + rollout helper
✅  src/utils/pt_rollout.py        ★   #1 free-run drift slope
✅  src/utils/pt_causal.py             #2 causal future-block L1
✅  src/utils/pt_tdist.py              #3 L1-vs-Δt slope
✅  src/utils/pt_teacher_free.py   ★   #4 free−teacher exposure-bias gap
✅  src/utils/pt_maskratio.py          #5 L1-vs-mask-ratio slope
✅  src/utils/pt_order.py              #6 shuffled−ordered ΔL1
✅  src/m12e_predictor_temporal.py  🎯 standalone orchestrator (forward + paired_per_variant)
✅  src/utils/per_frame_features.py 🧱 §6: forward_per_frame + reverse/permute/stride transforms
✅  src/utils/et_aot.py            ★   §6: Arrow-of-Time (Wei CVPR18) trainable binary head
✅  src/utils/et_tov.py            ★   §6: TOV/VCOP (Misra ECCV16 / Xu CVPR19) N-way head
✅  src/utils/et_pace.py           ★   §6: Pace (Wang ECCV20) rate head + oversample decode
✅  src/utils/et_tcc.py            ★   §6: TCC (Dwibedi CVPR19) training-free soft-NN cycle+τ
✅  src/m12f_encoder_temporal.py    🎯 §6 orchestrator (forward + paired_per_variant)
✅  🧪 CPU 3-check (py_compile + ruff F,E9) + import-smoke → ALL GREEN (14 files)
✅  🧪 GPU SANITY (Pro 6000): m12e --metric all (6 metrics, expected signs, 68s) +
       m12f --metric all (aot 0.975 / tov 0.788 / pace 0.750 / tcc τ 0.407, 154s) +
       both paired_per_variant stages → EXIT 0. §3.1 ✅ (rollout slope +ve → L1 grows w/ horizon)
✅  🐛 2 bugs caught by GPU smoke (CPU 3-check missed): (1) fp32-batch vs bf16-encoder dtype →
       to_pixel single-source; (2) eager dict eval of None pace_strides on aot path → lazy n_classes
✅  🔢 §1 strict-direction numbering defined · 🔎 predictor+encoder metrics web-searched + chosen
```

### ⏳ REMAINING — 🚦 strict §3 order (eval done → GPU free)
```text
#    action                                                              plan   task  status
1    🧪 GPU SANITY-smoke m12e/m12f (★ rollout/teacher_free monotonicity) §3.1   #15   ✅ DONE
2    🔀 re-home by function + dedup (m04e/f · m12a-f · m13 · legacy)     §3.2   #9    ✅ DONE
3a   🧹 m13: remove ALL line plots (bars-with-CI only)                  §3.3a  #19   ✅ DONE
3b   🔌 wire m12e+m12f → run_eval (St 8b/8c/9b/9c) + pipeline.yaml      §3.3b  #20   ✅ DONE
       + opt(c) load_encoder_only (m12f no predictor build) + loop-guard fix
3c   📊 m13: 14 bar-with-CI panels + HERO (table-CI + plot-CI) (§7)     §3.3c  #21   ✅ DONE
4    🧹 hardcodes: pt_*.py sweep (_DELTAS/_RATIOS/_SEED) → pipeline.yaml §3.4   #1    ✅ DONE
       probe.predictor_temporal; new utils clean; pre-existing flagged
5    📈 incremental per-encoder probe_plot → N-A/obsolete (§3.3a removed §3.5   #2    ✅ DONE
       line curves; m13 bars+hero are end-stage paired/aggregator)
```
🎉 ALL §3 (3.1–3.5) COMPLETE. iter16 metrics+eval pipeline built + validated end-to-end:
   m04e/m04f (annotation) · m12a-f + m13 (eval+viz) · run_eval Stages 1-13+8b/8c/9b/9c ·
   14 metrics → 14 bar-with-CI + HERO (table-CI + Δ-vs-frozen heatmap). Remaining iter16 work
   is the JEPA-variant roster wiring (Task #13, plan_model.md) — separate from this metrics plan.

§3.2 e2e: run_eval.sh --SANITY 2-enc PASS (8 min). §3.3b e2e:
run_eval_sanity_temporal2_20260528_053135.log (3 min) — St 8b/8c/9b/9c clean, fresh
predictor_temporal + encoder_temporal per-variant JSONs. opt(c): m12f drops the unused
predictor build (load_encoder_only; Stage 8c uses lighter encoder_ckpt_for) — identical outputs.

### 🧾 BUILD × VALIDATION MATRIX — 2026-05-28 (current snapshot)
```text
┌────────────────────────────────────────────┬─────────┬─────────┬─────────┬─────────┐
│ 📦 artifact (plan ref)                       │ 🔨 built │ 🔀 renam │ 🧪 CPU3 │ 🟢 GPU  │
│                                              │         │ e done  │ -check  │ SANITY  │
├────────────────────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ 🧱 §2.1 utils/predictor_eval.py              │   ✅    │   🚫    │   ✅    │   ✅    │
│ 🧱 §2.1 utils/pt_{6 files}.py                │   ✅    │   🚫    │   ✅    │   ✅    │
│ 🎯 §2.1 m12e_predictor_temporal.py           │   ✅    │   🚫    │   ✅    │   ✅    │
│ 🧱 §6  utils/per_frame_features.py           │   ✅    │   🚫    │   ✅    │   ✅    │
│ 🧱 §6  utils/et_{aot,tov,pace,tcc}.py        │   ✅    │   🚫    │   ✅    │   ✅    │
│ 🎯 §6  m12f_encoder_temporal.py              │   ✅    │   🚫    │   ✅    │   ✅    │
├────────────────────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ 🔀 §3.2 probe_action → m04e + m12_action     │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 🔀 §3.2 probe_taxonomy → m04f + m12c         │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 🔀 §3.2 probe_motion_cos → m12b_motion_cos   │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 🔀 §3.2 probe_future_mse → m12d_future_mse   │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 🔀 §3.2 probe_plot → m13_eval_plot           │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 📦 §3.2 probe_future_regress → src/legacy/   │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│ 🔁 §3.2 m12d_future_mse imports utils/       │   🚫    │   ⬜    │   ⬜    │   ⬜    │
│      predictor_eval (DEDUP)                  │         │         │         │         │
├────────────────────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ 🔌 §3.3 m12e+m12f wire → run_eval+yaml+m13   │   🚫    │   🚫    │   ⬜    │   ⬜    │
│ 🧹 §3.4 hardcode removal in utils            │   🚫    │   🚫    │   ⬜    │   ⬜    │
│ 📈 §3.5 incremental probe_plot               │   🚫    │   🚫    │   ⬜    │   ⬜    │
└────────────────────────────────────────────┴─────────┴─────────┴─────────┴─────────┘
   ✅ done   ⬜ TODO   🚫 N/A      🧱 util   🎯 orchestrator   🔀 rename   🔁 dedup
   🔨 built  🔀 renamed  🧪 py_compile+ast+ruff F,E9  🟢 smallest-SANITY on Pro 6000
```
🏁 GPU-SANITY results (code-correctness only, NOT performance): m12e 6/6 metrics expected-sign
   (68s) · m12f aot 0.975 / tov 0.788 / pace 0.750 / tcc τ 0.407 (154s) · both stages EXIT 0.
🐛 GPU smoke caught 2 bugs CPU-3-check missed → 🩹 fp32↔bf16 via to_pixel · lazy n_classes.

### 🅿️ PARKED · 🤔 DECIDE
```text
🅿️  iter18 — full m05-m11 strict renumber  (separate iter; blast radius §1)
🤔  skip iter17-115k → iter16-ablations on 10k?   (decide after full 10k results land)
✅  phase-2 encoder metrics: TOV / Arrow-of-Time / Pace / TCC — BUILT + GPU-validated (was OPEN)
```

## 0. HARD CONSTRAINT — a full eval is running

`logs/iter15_v2_poc_eval_20260527_163314.log` (`run_eval.sh --POC`, ~5 h, on the single Pro 6000).

```text
DO-NOT-TOUCH while the eval runs (any one breaks it):
  • rename/move any probe_*.py        → run_eval.sh spawns them by filename (probe_taxonomy.py LIVE now)
  • edit scripts/run_eval.sh           → live bash script; re-read mid-exec corrupts it (4h-burn incident)
  • edit src/probe_plot.py             → run_eval.sh calls it at the END stage
  • edit src/utils/*.py                → eval's future subprocesses import them
  • edit configs/pipeline.yaml         → a yaml typo crashes every future cfg load in the eval
  • add a GPU job (SANITY smoke)       → contends with eval + the live m09c2 SANITY job on one GPU
SAFE while the eval runs:
  • write a NEW .py file the eval never references (m12f_predictor_temporal.py)
  • CPU-only 3-check on that new file (py_compile / ast / ruff F,E9) + import smoke
```

## 1. Module numbering — STRICT pipeline direction

Canonical AI/ML direction: DATA-PROC → ANNOTATION → FACTOR-PREP → TRAINING → EVALUATION →
VISUALIZATION (viz LAST). Strict-order band table (✓ = number already direction-correct;
⚠ = chronological artifact, see renumber note):

```text
dir  band              FUNCTION                  modules                                    status
1    m00{,b-e}         DATA PROCESSING           data_prep, fetch_durations, sample_subset,  ✓
                                                 download_subset, difficulty_split
2    m01               DOWNLOAD                  download                                    ✓
3    m02{,b}           SCENE DETECT              scene_detect, scene_fetch_duration          ✓
4    m03               PACK shards               pack_shards                                 ✓
5    m04{,b-f}         ANNOTATION                vlm_tag, vlm_select, sanity_compare,        ✓
                                                 motion_features + NEW m04e/m04f labels
( )  m05{,b,c}         EMBEDDINGS (legacy        vjepa_embed, baselines, true_overlap        legacy
                       analysis branch)                                                     branch
6    m10 / m11         FACTOR-PREP (feeds        sam_segment ; factor_datasets              ⚠ numbered
                       surgery)                                                             AFTER train
7    m09{a1,a2,c1,c2}  TRAINING                  pretrain/surgery × encoder/head            ⚠ (dirn:
                                                                                            after fprep)
8    m12{a-f}          EVALUATION (NEW)          action, motion_cos, taxonomy, future_mse,  ✓ after
                                                 predictor_temporal, encoder_temporal       train
9    m13 (NEW)         VISUALIZATION (LAST)      eval_plot (was probe_plot)                 ✓ after eval
     m07 / m08         legacy viz                umap (m07) ; old plots (m08)               ⚠ pre-train
gap  m06 faiss → src/legacy/
```

### Task 2 fix — spread the probe family by FUNCTION, strict-direction

probe_*.py is MULTI-FUNCTION. action & taxonomy are DUAL (a `--stage labels` ANNOTATION step
that feeds TRAINING + a `--stage train` EVALUATION probe) → **SPLIT**. plot is the LAST stage
→ its own band AFTER eval (m13 — NOT the old m08 plot band). future_regress is dead. Re-homed:

```text
ANNOTATION  (m04 band — derived labels; consumed by m09 training AND m12 eval)
  m04e_action_labels.py    ← probe_action  --stage labels   (action_labels.json ← m04d motion-flow)
  m04f_taxonomy_labels.py  ← probe_taxonomy --stage labels   (taxonomy_labels.json ← m04 VLM tags)
        shared derive logic → utils/action_labels.py (exists) + utils/taxonomy_labels.py (NEW)
EVALUATION  (m12 band — after train + factor-prep; metric-suffixed names — user decision)
  m12a_action_top1.py        ← probe_action  train + paired_delta   (action top-1 acc)
  m12b_motion_cos.py         ← probe_motion_cos                       (motion cosine)
  m12c_taxonomy_f1.py        ← probe_taxonomy train + paired_delta    (per-dim top-1 + sample-F1)
  m12d_future_mse.py         ← probe_future_mse  (DEDUP → imports utils/predictor_eval)
  m12e_predictor_temporal.py ← NEW orchestrator (6 predictor metrics, §2.1)
  m12f_encoder_temporal.py   ← NEW orchestrator (4 encoder metrics: aot/tov/pace/tcc, §6)
VISUALIZATION  (m13 — LAST, after eval)
  m13_eval_plot.py         ← probe_plot   (strict direction: viz is the final stage — NOT m08d)
RETIRE   src/legacy/probe_future_regress.py   (dead — superseded by future_mse+motion_aux)
UTILS    utils/predictor_eval.py + utils/pt_{rollout,causal,tdist,teacher_free,maskratio,order}.py
         + utils/per_frame_features.py + utils/et_{aot,tov,pace,tcc}.py + utils/taxonomy_labels.py
```

WHY SPLIT action/taxonomy: `--stage labels` DERIVES labels consumed by TRAINING (annotation,
must precede m09) while `--stage train` is the downstream EVAL probe — two pipeline steps. The
shared derive logic lives in utils/ (rule-32-safe), so the split is clean.

### FULL strict renumber of EXISTING m05-m11 = SEPARATE high-cost iter (NOT bundled here)

Making EVERY existing number strict (factor-prep m10/m11 → before train m09; legacy viz m07/m08
→ after eval) means renumbering deeply-referenced bands. Blast radius:
```text
• ckpt/output paths: outputs/.../m09{a,c}_*/, m10_sam_segment/, m11_factor_datasets/  (on disk
  + on HF + the CURRENTLY-RUNNING eval reads them + v15a result-dir comparability)
• configs/*.yaml (data.masks_subdir / factor_subdir, train cfgs), scripts/*.sh, wandb run names,
  every iter*/ doc.  → ~hundreds of refs.
RECOMMEND a dedicated iter (e.g. iter18_renumber): git-mv + global grep-replace + output-dir
migration + full --SANITY regression. Do NOT bundle with the temporal-metrics work — it would
multiply this PR's risk for zero functional gain. The NEW bands above (m04e/f, m12*, m13) are
ALREADY placed strict-direction-correct, so iter16's additions need no renumber.
```

Decision flagged: for iter16 keep `--output-root outputs/.../probe_*/` dir names (renaming them
invalidates the running eval's cache + v15a comparability).

## 2. BUILD NOW (CPU-safe, zero eval impact)

### 2.1 Task 2 — architecture: 6 metric util-fns + 1 standalone orchestrator

Rule-32 (m*.py cannot import each other) + ViT-G load cost (~1-2 min/load) force this layout:
shared forward + the 6 metric computations are NON-STANDALONE → src/utils/; ONE standalone
orchestrator loads the model ONCE per variant and dispatches.

```text
file                                  kind         role
src/utils/predictor_eval.py   (NEW)   util         shared loaders (encoder hierarchical +
                                                   predictor + mask-gen) + masked encode→predict→
                                                   L1 primitive + bootstrap CI.
src/utils/pt_rollout.py       (NEW)   util         metric #1 compute fn (rollout drift)        ★
src/utils/pt_causal.py        (NEW)   util         metric #2 (causal future-block)
src/utils/pt_tdist.py         (NEW)   util         metric #3 (temporal-distance scaling)
src/utils/pt_teacher_free.py  (NEW)   util         metric #4 (teacher-vs-free gap)             ★
src/utils/pt_maskratio.py     (NEW)   util         metric #5 (mask-ratio robustness)
src/utils/pt_order.py         (NEW)   util         metric #6 (temporal-order sensitivity)
src/m12e_predictor_temporal.py(NEW)   STANDALONE   orchestrator: load once/variant, --metric
                                                   {one|all} → dispatch to the 6 fns, write
                                                   outputs + paired_delta. ONLY shell entry.
```

WHY not 6 standalone src/ scripts: reloads ViT-G per (encoder × metric) = 8×6 = 48 loads
(~+60-90 min POC, worse FULL) vs the orchestrator's 8. The 6 fns stay individually readable/
testable (one pure fn each) but live in utils so ONE process runs all six.
ALT (each independently runnable as its own m12 script): 6 standalone src/m12{e-j}_*.py + the
shared predictor_eval.py, accepting the 6× reload — §5 decision.

DEDUP NOTE: predictor_eval.py is written NOW as a SELF-CONTAINED new util (copies future_mse's
module-local primitives — `_load_vjepa_2_1_encoder_hierarchical`, `_load_predictor_2_1`,
`_build_mask_gen`, the `_forward_one_batch` L1 core). Temporary duplication is tolerated because
refactoring `probe_future_mse.py` to import predictor_eval would EDIT a live-invoked script →
that dedup is DEFERRED to §3.2 (post-eval). So BUILD-NOW touches only NEW files.

Each metric SWEEPS one axis future_mse holds fixed (1 / random / true / fixed-Δt / fixed-% /
ordered) → mutually orthogonal, pure predictor latent-L1, no trained probe:
```text
#1 rollout      drift slope d(L1)/dh over iterated horizon h=1,2,4,8 (feed preds back)   ★
#2 causal       mask FUTURE temporal half only (m_pred: t>T/2 ; m_enc: t<=T/2)
#3 tdist        single-shot L1 vs target offset Δt={1,2,4,8} → predictability-horizon slope
#4 teacher_free gap = freerun_L1 − teacherforced_L1  (exposure bias / error-recovery)    ★
#5 maskratio    L1 vs mask-ratio sweep (npred / spatial_pred_mask_scale) → degradation slope
#6 order        ΔL1 = shuffled-context − ordered-context (predictor's reliance on order)
```

predictor_eval.py public API (the core every pt_*.py calls):
- `load_encoder_predictor(ckpt, T)` → (encoder hier 6656-dim, predictor, mask_gen)
- `masked_predict_l1(encoder, predictor, mask_gen, batch, *, mask=None, ctx_override=None)`
  → per-clip L1 (the shared primitive; each metric passes a different mask / context manipulation)

Per-metric manipulation (each pt_*.py fn, all on the same core):
```text
#1 rollout:      split T into blocks; predict block k+1 from {1..k}; free-running feeds PREDICTED
                 tokens back as context for k+1; record L1 per h; fit drift slope.
#2 causal:       deterministic temporal-split mask (m_enc=past idxs, m_pred=future idxs); L1 future.
#3 tdist:        per Δt, m_pred=tokens Δt ahead of context; L1 per Δt; slope.
#4 teacher_free: run #1 twice (true vs own context); per-clip gap.
#5 maskratio:    loop mask ratios (npred ∈ {2,4,8,12}); L1 per ratio; slope.
#6 order:        permute temporal index order of context tokens before predictor; ΔL1.
```

Orchestrator CLI (mirrors probe_future_mse):
```text
--stage {forward, paired_per_variant}  --metric {rollout,causal,tdist,teacher_free,maskratio,order,all}
--variant <vjepa_*>  --encoder-ckpt <.pt>  --action-probe-root <dir>  --local-data <dir>
--output-root outputs/<mode>/predictor_temporal  --num-frames  --cache-policy {1,2}
--motion-aux-head <.pt> (head-cell symmetry)  --no-wandb
```
Outputs per variant: `per_clip_<metric>.npy` + `aggregate_<metric>.json` (mean/std/BCa ci/n_test)
— same schema family as `aggregate_mse.json`. `paired_per_variant` mirrors future_mse's pairwise
BCa Δ across discovered vjepa variants, per metric.

GPU-checklist (orchestrator, per src/CLAUDE.md): check_gpu, cleanup_temp, add_cache_policy_arg +
prompt, AdaptiveBatchSizer, save/load resume ckpt, iter_clips_parallel, make_pbar,
print_cgroup_header + start_oom_watchdog, --no-wandb, FAIL-LOUD (0-clip → sys.exit). Gold cites in
docstring (V-JEPA 2-AC; Scheduled-Sampling Bengio'15; VideoMAE; CPC '18; Shuffle&Learn '16; AoT '18).

Hyperparams (horizons, Δt set, mask-ratio set) → NEW `pipeline.yaml probe.predictor_temporal:`
block — DEFERRED (§3.3); module doesn't run until post-eval integration anyway.

### 2.2 Verify NOW (CPU only)
`python -m py_compile` + `ruff check --select F,E9` + `python -c "import ast; ast.parse(...)"`
+ an import-smoke that loads the module and asserts the 6 metric fns resolve. **No GPU smoke now.**

## 3. WAIT FOR EVAL TO FINISH — then, in this order

```text
3.1  GPU SANITY-smoke m12e_predictor_temporal.py  (smallest --SANITY, 1 variant, --metric all)
3.2  RE-HOME-BY-FUNCTION (strict direction) + DEDUP pass (atomic, one commit):
       SPLIT: extract --stage labels → src/m04e_action_labels.py + src/m04f_taxonomy_labels.py
         (annotation; derive logic → utils/) ; keep train+eval as m12a_action.py + m12c_taxonomy.py ;
       git mv probe_motion_cos.py→m12b_motion_cos.py ; probe_future_mse.py→m12d_future_mse.py ;
       git mv probe_plot.py→m13_eval_plot.py (VIZ, LAST) ; probe_future_regress.py→src/legacy/ ;
       refactor m12d_future_mse to import utils/predictor_eval (dedup) ;
       grep -rl 'probe_' scripts/ src/ configs/ iter/ ; rewire run_eval.sh (labels→m04e/m04f ;
         eval→m12*/m12b/m12c/m12d ; plot→m13) + run_train.sh (labels→m04e/m04f) + docstrings ;
       KEEP output-dir names ; smoke run_eval.sh --SANITY end-to-end.
3.3  INTEGRATE m12e into run_eval.sh (Task 8): per-encoder forward (--metric all) + per-metric
       paired_delta + NEW pipeline.yaml probe.predictor_temporal keys + m13_eval_plot rows
       (rollout drift-slope + teacher_free gap = the 2 headline panels).
3.4  Hardcode/fallback removal in src/utils (Task 1, audit done) — T1→T2→T3, SANITY-smoke each.
3.5  Incremental probe_plot per-encoder in run_eval.sh (Task 2).
```

## 4. Reference-update checklist for the rename (§3.2 grep targets)

```text
scripts/run_eval.sh           — python -u src/probe_*.py  (Stages 1/2/3/3.5/5/6/8/11-13)
scripts/run_train.sh          — probe_action + probe_taxonomy label-gen calls
scripts/legacy2/*.sh          — legacy eval refs (verify)
configs/eval/probe_encoders.yaml — verify it keys on encoder names, NOT script names
src/*/<docstrings>            — self-referential USAGE blocks
iter*/runbook.md, .claude/skills/preflight/SKILL.md — doc refs
NOT touched: outputs/.../probe_*/ dir names (decoupled — preserve cache + v15a comparability)
NO python cross-imports between probes exist (rule 32) — only PATH args (output dirs), unaffected.
```

## 5. Decisions
CONFIRMED:
- A · probe_future_regress → RETIRE to src/legacy/.
- B · metric layout → 6 util-fns + 1 orchestrator (8 loads).
- Re-home BY FUNCTION, STRICT direction: labels→m04e/m04f (annotation) ; train+eval→m12* ;
  plot→m13 (visualization, LAST — NOT m08d).
- FULL m05-m11 strict renumber → SEPARATE iter (iter18_renumber), NOT bundled (blast-radius §1).
OPEN:
- output-dir rename: KEEP (recommended) vs migrate-with-cache-wipe.
- headline plot panels: rollout drift-slope + teacher_free gap (the ★).

## 6. 📂 PHASE-2 — encoder-side temporal metrics (options kept OPEN, Tasks #6/#7)

Lower `surgery≫pretrain` probability than the §2.1 predictor metrics (they read the ENCODER's
frozen features, not the predictor surgery directly optimizes) — but cheap (reuse probe_action's
cached features) and give orthogonal temporal coverage. Keeping ALL options open per the user.

```text
┌──────────────┬─────────────────────────────────────────┬────────────────────────────────┬─────────┐
│ metric       │ definition (encoder-feature based)      │ gold + GitHub                  │ build   │
├──────────────┼─────────────────────────────────────────┼────────────────────────────────┼─────────┤
│ TOV / VCOP   │ shuffle frames/clips → tiny head        │ Shuffle&Learn (Misra ECCV16) ; │ LOW     │
│              │ classifies ordered-vs-shuffled OR       │ VCOP (Xu CVPR19)               │ reuse   │
│              │ predicts the permutation. DISCRIMINATIVE│ github.com/xudejing/video-     │ cached  │
│              │ temporal order.                         │ clip-order-prediction          │ feats   │
│ Arrow-of-Time│ forward vs time-REVERSED clip → binary  │ Wei CVPR18                     │ LOW     │
│              │ head. directional asymmetry; cheap (flip│ vcg.seas.harvard.edu/publica-  │ flip    │
│              │ the frame axis).                        │ tions/learning-…-arrow-of-time │ frames  │
│ Pace / Speed │ re-sample clip at strides 1x/2x/4x →    │ Pace (Wang ECCV20) ·           │ LOW-MED │
│              │ classify the rate. temporal-SCALE       │ github.com/laura-wang/video-   │ re-decode│
│              │ sensitivity.                            │ pace ; SpeedNet CVPR20 ; RSPNet│ strides │
│ TCC          │ per-FRAME embeddings → align same-action│ TCC (Dwibedi CVPR19)           │ MED     │
│              │ clip pairs by NN in embedding space;    │ github.com/google-research/    │ per-fr  │
│              │ metric = cycle-back err / Kendall's-τ.  │ google-research/tree/master/tcc│ feats,  │
│              │ NO training (pure geometry).            │ ; pytorch: github.com/June01/  │ no-train│
│              │                                         │ tcc_Temporal_Cycle_Consistency │         │
└──────────────┴─────────────────────────────────────────┴────────────────────────────────┴─────────┘
```

Integration when un-parked: TOV/AoT/Pace need a tiny head trained on the train split → EVAL band
(m12*), reuse probe_action's cached features. TCC is training-free (geometry) but needs a
PER-FRAME feature path (probe_action mean-pools tokens → a frame-resolved extract is required).
Build AFTER the predictor suite (Task #5 done) lands + is GPU-validated.
- interrupt the running eval to start the re-home now, OR wait for it to finish (§0 / below).

## 7. 📊 m13_eval_plot — ALL-14-METRIC viz: 14 individual bars + 1 HERO (Task #10 §3.3)

m13 MUST cover the full 14-metric suite (4 headline + 6 predictor §2.1 + 4 encoder §6) as:
(A) 14 individual **bar plots WITH CI**, and (B) ONE combined HERO = a **table WITH CI** + a
**plot WITH CI**. Reuses the existing `_bar_with_ci` + `_sort_by_metric` + `_emit_one` machinery
(N-encoder generic, BCa whiskers, ↑/↓ badge — already in m13).

### 7.0 ❌ REMOVE all LINE plots from m13 (user: "line plots seem useless")
```text
DELETE  plot_loss_curves()        → probe_action_loss.{png,pdf}   (train-loss-vs-step LINE)
DELETE  plot_acc_curves()         → probe_action_acc.{png,pdf}    (val-acc-vs-step LINE)
DELETE  run_training_side_compare() + --training-side / --training-root args
        → the 12 train-trajectory LINE PNGs (loss/grad/lr/drift vs % training)
        + helpers _read_csv_typed / _col / _probe_col / _derive_loss_total_m09c1 /
          _drift_mean_series / _load_training_run / _plot_training_metric /
          TRAINING_RUNS_CANONICAL / TRAINING_LINESTYLE_BY_FAMILY  (all line-plot-only)
KEEP    plot_encoder_comparison() machinery (_bar_with_ci / _sort_by_metric / _emit_one)
        — extend from 3 → 14 bar-with-CI panels (§7.2) + add the HERO (§7.3).
NET: m13 emits ONLY bar-with-CI (14) + hero-table-with-CI + hero-plot-with-CI. Zero line plots.
```

### 7.1 The 14 metrics, their source JSON, and DIRECTION (critical for the hero normalization)
```text
┌────┬──────────────┬─────────┬────────────────────────────────────────────┬───────────┐
│ #  │ metric       │ family  │ source JSON (per-variant)                   │ direction │
├────┼──────────────┼─────────┼────────────────────────────────────────────┼───────────┤
│ 1  │ action_top1  │ HEAD    │ m12a probe_paired_delta.json .acc_pct       │ ↑ higher  │
│ 2  │ motion_cos   │ HEAD    │ m12b probe_motion_cos_paired.json           │ ↑ higher  │
│ 3  │ taxonomy_f1  │ HEAD    │ m12c <enc>/test_metrics.json (per-dim mean) │ ↑ higher  │
│ 4  │ future_mse   │ HEAD    │ m12d probe_future_mse_per_variant.json      │ ↓ lower   │
│ 5  │ rollout    ★ │ PRED    │ m12e predictor_temporal_per_variant.json    │ ↓ lower   │
│ 6  │ causal       │ PRED    │   "  (per-metric block)                     │ ↓ lower   │
│ 7  │ tdist        │ PRED    │   "                                         │ ↓ lower   │
│ 8  │ teacher_free★│ PRED    │   "                                         │ ↓ lower   │
│ 9  │ maskratio    │ PRED    │   "                                         │ ↓ lower   │
│ 10 │ order        │ PRED    │   "                                         │ ± signed  │
│ 11 │ aot        ★ │ ENC     │ m12f encoder_temporal_per_variant.json      │ ↑ higher  │
│ 12 │ tov        ★ │ ENC     │   "                                         │ ↑ higher  │
│ 13 │ pace       ★ │ ENC     │   "                                         │ ↑ higher  │
│ 14 │ tcc (τ)    ★ │ ENC     │   "  kendalls_tau (cycle_back = appendix)   │ ↑ higher  │
└────┴──────────────┴─────────┴────────────────────────────────────────────┴───────────┘
  tcc contributes 2 raw numbers (cycle_back ↓ + Kendall's τ ↑); τ is the headline (14th)
  column, cycle_back is its appendix companion (a 15th panel, not in the hero grid).
  `order` is sign-interpreted (shuffled−ordered ΔL1) → excluded from the win-count tally,
  shown in the hero with a neutral diverging scale (not green/red-better).
```

### 7.2 Output A — 14 individual bar plots (one PNG+PDF each)
```text
Each: N-encoder bars sorted by metric value (desc for ↑-better, asc for ↓-better), BCa 95% CI
whiskers, ↑/↓ direction badge, N/A bars hatched + tail-positioned. Naming + machinery EXACTLY
mirror the current probe_{action_acc,motion_cos,future_mse}_compare plots:
  HEAD : m13_action_top1_compare · m13_motion_cos_compare · m13_taxonomy_f1_compare
         · m13_future_mse_compare
  PRED : m13_{rollout,causal,tdist,teacher_free,maskratio,order}_compare        (6)
  ENC  : m13_{aot,tov,pace,tcc_tau}_compare  (+ m13_tcc_cycle_compare appendix) (4 + 1)
Grouped into 3 figure-folders under <output-dir>/eval/{head,predictor,encoder}/ so a reader
opens one family at a time. KEEP probe_*_compare names for the 3 legacy headline panels
(v15a comparability §5); the 11 NEW panels take m13_ names.
```

### 7.3 Output B — the HERO = TABLE-with-CI (B1) + PLOT-with-CI (B2), all 14 metrics
**B1 · hero TABLE-with-CI** — `m13_hero_table.{png,pdf}` (rendered) + `m13_hero_table.csv` (machine):
```text
rows = encoders (frozen pinned TOP as BASE) ; cols = 14 metrics in HEAD | PRED | ENC blocks.
cell = "<value> ±<BCa ci_half>"  — each metric's per-variant aggregate mean ± 95% CI half-width.
★ on a cell where that metric's paired Δ-vs-frozen 95% BCa CI EXCLUDES 0 (significant win/loss).
final WINS col = k/13 metrics with a significant GOOD-direction win vs frozen (order excluded).
Rendered to PNG via matplotlib `ax.table` (mono font) — the NUMERIC scorecard companion to B2;
.csv is the same grid for the paper's appendix. Every number carries its CI (no point estimates).
```

**B2 · hero PLOT-with-CI** (WEBSEARCH-backed; "be creative" → the RIGHT tool, not radar):
WEBSEARCH verdict (Highcharts/Domo/Carbon + the ML radial-viz paper arXiv 2104.07377):
- ❌ RADAR/spider — degrades past ~10-12 axes (label crowding, adjacent-axis blur) and >4-5
  overlapping polygons; 14 axes × up to 8 encoders is unreadable. REJECTED.
- ✅ PRIMARY HERO = **"Surgery-vs-Frozen Δ scorecard" heatmap** (HELM/benchmark-leaderboard idiom).
  This is the single most thesis-aligned view — it renders the paper's win condition directly.
- ✅ SECONDARY = **parallel-coordinates** (14 direction-normalized axes, one polyline/encoder) —
  the websearch-preferred many-dimension alternative to radar (own scale per axis, 4-10+ dims).

```text
HERO  m13_hero_surgery_vs_frozen.{png,pdf}   —  direction-normalized Δ-vs-baseline heatmap
┌──────────────────────────┬─────HEAD(4)─────┬───────PRED(6)────────┬────ENC(4)────┬──────┐
│ encoder ↓  / metric →     │ a1  mc  tf  fm │ ro ca td tf mr or    │ ao tv pc tc  │ WINS │
├──────────────────────────┼────────────────┼──────────────────────┼──────────────┼──────┤
│ vjepa_2_1_frozen (BASE)   │  ·   ·   ·   ·  │  ·  ·  ·  ·  ·  ·     │  ·  ·  ·  ·   │  —   │
│ pretrain_encoder          │ [signed Δ vs frozen, color = GREEN better / RED worse after │      │
│ pretrain_2X_encoder       │  direction-flip so ↓-better metrics are inverted; ★/bold     │  k/14│
│ surgical_noDI_encoder     │  border where BCa 95% CI EXCLUDES 0 — the non-overlapping-CI │      │
│ surgical_3stage_DI_encoder│  win the paper needs]                                        │      │
│ pretrain_head             │                                                              │      │
│ surgical_noDI_head        │                                                              │      │
│ surgical_3stage_DI_head   │                                                              │      │
└──────────────────────────┴────────────────┴──────────────────────┴──────────────┴──────┘
  • cell value  = paired BCa Δ (variant − frozen) from each metric's per_variant pairwise block
  • cell color  = sign-corrected Δ (lower-better metrics negated) on a diverging RdYlGn colormap
  • ★ / bold cell border = 95% BCa CI excludes 0 (statistically-significant win/loss)
  • WINS column = count of metrics (of 13; `order` excluded) where CI-excludes-0 in the GOOD
    direction → the headline number: "surgery beats frozen on k/13 metrics, non-overlapping CI"
  • 3 vertical rules separate HEAD | PRED | ENC families; frozen row pinned top as the baseline.
SECONDARY  m13_hero_parallelcoords.{png,pdf} — 14 axes (each min-max normalized + direction-flipped
  so up=better), one polyline per encoder (ENCODER_COLORS), frozen dashed-grey as the reference.
```

### 7.4 Wiring (folds into §3.3 — Task #10, post §3.2)
```text
• m13 reads 6 per-variant JSONs: m12a/m12b/m12c/m12d (existing) + m12e predictor_temporal_
  per_variant.json + m12f encoder_temporal_per_variant.json (NEW — both already emitted by the
  orchestrators' paired_per_variant stage, GPU-validated §3.1).
• a NEW pipeline.yaml `probe.plot_hero:` block declares the 14-metric catalog + per-metric
  direction + family grouping (single source — m13 reads it, NO hardcoded metric list, mirrors
  m13's existing "auto-discover encoders from by_encoder" no-hardcode pattern).
• run_eval.sh Stage 10/13 already invoke m13; the per-metric + hero panels are added inside m13
  (no new shell stage). FAIL-LOUD if a per_variant JSON is missing (consistent with _load_json).
```

Sources (hero viz choice): Highcharts radar-limits, Domo parallel-coordinates, Carbon complex-charts,
arXiv 2104.07377 (radial ML-model-comparison viz), Datanovia/DataCamp (direction-normalized heatmaps).
