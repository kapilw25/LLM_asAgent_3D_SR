# 🚀 iter18 Eval Speedup — 3 Levers

> **#1** 🔀 metric-parallel split (~2.5×) · **#2** 📦 `bs 4→16` (✅ done, ~1.5–2×) · **#3** ♻️ h-memoization (~1.3×)

---

## 🎯 Context — why eval is slow

Each per-encoder eval is **~5–7 h**, dominated by **stage 8b** (`m12e_predictor_temporal`, 6 metrics over 1825 test clips) at **~2.2 h**. Evidence gathered this session:

- 🕰️ **iter15 evals were ~1h13/encoder** — because **stage 8b didn't exist yet** (added iter16) and they ran **solo**. iter18 adds 8b (**+2.2 h**) *and* runs evals concurrently with trains (**~1.7× contention** on stages 2–8).
- 💾 **The frame cache is NOT the bottleneck.** `data/eval_10k_local/m12_frame_cache` (159 G, **test 1825/1825 cached**, ONE shared dir used by all arms) already memoizes **decode** — but stage 8b is **100 % compute, 0 % decode**, so the cache can't touch it.
- 🐌 **Stage 8b wastes the GPU twice:** runs `bs=4` on a 96 G card (**~10 % util**) *and* recomputes the mask-independent `h = encoder(pixel)` **~14×/batch**.

### 🧰 The 3 levers (all verified safe this session)

| # | Lever | Win | Why it's safe |
|---|-------|-----|---------------|
| 1️⃣ | 🔀 **Metric-parallel split** — the 6 metrics are independent; `m12e` already has `--metric` + writes per-metric `per_clip_<m>.npy`/`aggregate_<m>.json`. Fan across 4 GPUs ⇒ stage-8b wall = **slowest single metric** (~teacher_free), not the sum of six. | **~2.5×** | outputs already per-metric & disjoint |
| 2️⃣ | 📦 **`bs 4→16`** — per-clip metrics **PROVEN batch-invariant** (`pt_order`/`pt_maskratio` seed a local `Generator(PT_SEED)`; causal/tdist/rollout deterministic; encoder/predictor `eval()`+`no_grad`). `safe_metric` OOM-subbatches→1. | **~1.5–2×** ✅ **DONE** | bit-identical + OOM-guarded |
| 3️⃣ | ♻️ **h-memoization** — V-JEPA's own *"amortize target computation"*: compute `h = encoder(pixel)` **once**, reuse across a metric's internal mask sweep. | **~1.3×** | bit-identical (deterministic eval forward) |

### 🧑‍⚖️ User decisions
- 🔁 **(#1) RESTART the live run** ⇒ the live status tool must handle the new jobs **on first try**.
- 🧪 **(#3) BUILD now, smoke on next free GPU** ⇒ **#3 ships flag-gated OFF** so the restart can't run unverified numerics; flip ON only **after** the parity smoke passes.

---

## 🧭 Design decisions

- 🔢 **6 single-metric jobs per encoder** (not buckets): needs **no** `m12e --metric` signature change → lowest risk.
  - 💸 *Cost:* each of the 6 processes reloads the model (~20–40 s) and recomputes `h` independently (cross-metric h-share lost) — paid in **GPU-time**, but they run in **parallel** so **wall-time wins**. Correct trade given idle tail GPUs + a wall-time deadline.
- 🖥️ **Rollup display:** keep **ONE eval cell per encoder** in the status table; that cell aggregates the encoder's `E:` job (stages 2–8) + its 6 `P:` jobs (stage 8b). Σ TOTAL / DAG sim / ETA use the full **84-job** set. Table width unchanged.
- 🚦 **#3 flag-gated** (`PT_H_MEMO`, default **off**): code lands inert (exact current behavior) until the GPU parity smoke passes, then flipped on for remaining evals.
- 🛡️ **Per-metric resume ckpt (RACE FIX):** 6 concurrent `m12e` for one encoder currently write the **same** `.m12e_ckpt.npz` → corruption. Key it by `args.metric`.

---

## 🔀 Part 1 — Metric-parallel split (the big one)

### 1️⃣a 🛡️ `src/m12e_predictor_temporal.py` — per-metric resume ckpt *(RACE FIX, required)*
- Line **120**: `ckpt = out_dir / ".m12e_ckpt.npz"` → `ckpt = out_dir / f".m12e_ckpt_{args.metric}.npz"`.
  - `--metric tdist` → `.m12e_ckpt_tdist.npz` · `--metric all` → `.m12e_ckpt_all.npz` ⇒ concurrent single-metric runs write **disjoint** files.
  - ✅ `per_clip_<m>.npy` / `aggregate_<m>.json` / `tmp_decode` are already collision-free.

### 1️⃣b 🐚 `scripts/run_eval.sh` — `PT_METRIC` env passthrough
- Near line **151** (where `SKIP_STAGES` is read): add `PT_METRIC="${PT_METRIC:-all}"` (mirrors the `${VAR:-default}` pattern at lines 111–116, 131–152).
- Line **976**: `--stage forward --metric all` → `--stage forward --metric "$PT_METRIC"`.
- Line **987**: tee log `..._forward_${ENC}.log` → `..._forward_${ENC}_${PT_METRIC}.log` ⚠️ (6 concurrent runs must not clobber one tee file).

### 1️⃣c 🗂️ `scripts/iter18_poc_ngpu.py` — split eval into `E:` (stages 2–8) + 6 `P:` (stage 8b ×metric)
- ➕ Constant `PT_METRICS = ["rollout","causal","tdist","teacher_free","maskratio","order"]` *(comment: mirrors `m12e.METRICS` keys — stable set)*.
- ➕ Stage strings:
  - `EVAL_SKIP_PERENC_NO8B = EVAL_SKIP_SHARED + ",8b"` — the **E:** job skips 8b.
  - `EVAL_SKIP_ONLY_8B = "1,2,3,11,5,6,8,4,12,13,7,9,9b,10"` — everything **except** 8b.
- 🛠️ `build_jobs()` (lines 140–148):
  - **E:** eval cmd → `SKIP_STAGES={EVAL_SKIP_PERENC_NO8B} ...` (otherwise unchanged).
  - 🆕 New loop, per eval-encoder × metric:
    - `jid = f"P:{enc_name}:{m}"` · `kind="eval"`
    - `deps = {T:...arm}` 👉 **gate on TRAIN, not on `E:`** so 8b runs concurrently with stages 2–8
    - `needs_labels=True`
    - `cmd = CUDA_VISIBLE_DEVICES={gpu} SKIP_STAGES={EVAL_SKIP_ONLY_8B} PT_METRIC={m} CACHE_POLICY_ALL={cache} {pin}./scripts/run_eval.sh {mflag} --encoders {enc_name}`
    - `log = logs/iter18_ngpu_{mtag}_pt_{enc}_{m}_{{ts}}.log`
- ♻️ **`--cache 1` resume** (lines 222–231): after the train loop, add a **P:** loop — mark `P:` done if `outputs/{mtag}/predictor_temporal/{enc}/aggregate_{m}.json` exists (avoids launching ~84 no-op jobs on resume). Leave **E:** as-is (re-run, internal cache-skip).
- ✂️ **`--skip-arms`** (lines 196–198): also drop the 6 `P:` jids per skipped arm — `| {f"P:vjepa_2_1_{ARM2ENC[a]}:{m}" for a in skip for m in PT_METRICS}`. **`--only`** already drops all non-`T:` jobs → `P:` dropped automatically.
- 🏁 **§3 finale:** **NO change** — it skips per-encoder stages and reads the cached `per_clip_<m>.npy` the `P:` jobs produce *(confirmed: `S3_SKIP_PERENC` is a static stage list; `iter18_poc_metrics.py` reads per-encoder aggregates regardless of producer)*.

### 1️⃣d 📟 `scripts/iter18_poc_status.py` — handle `E:`/`P:` split + rollup display
- 📋 `_eval_plan_for(E:)` → `EVAL_PLAN` **minus `"8b"`** (8b now lives in `P:` jobs); keep the existing drop-8/8b-if-no-ckpt branch.
- 📊 **P: jobs estimated from their single `m12e` clip bar** directly (`cur/tot/recent=`), **NOT** the stage ledger — each `P:` log is one metric's 0→1825 bar (this is the old single-bar estimator, always correct for one bar). Pending `P:` prior = median of completed same-metric `P:` jobs, else a per-metric share of the 8b prior.
- 🧮 **Rollup cell:** add `_eval_group(enc) = [E:enc] + [P:enc:*]`; the encoder's eval cell shows status = **done(all) / running(any) / pending**, remaining = **max finish** over the group (they run in parallel on the pool). Append a one-line note like `·8b 3/6✅`.
- ➕ Σ TOTAL eval column, DAG sim (`jobs[x]["deps"] <= done_set`), counts, `settled`/`s3` logic all **iterate `jobs`** → auto-include `P:` jobs. Mirror the `--skip-arms`/`--only` `P:` filtering already added in 1c.

---

## 📦 Part 2 — `bs 4→16`  ✅ DONE

`configs/pipeline.yaml` → `inference_predictor_temporal_bs: 16`. CPU-verified it resolves to **16**. Active on every new `m12e` launch (reads yaml fresh). **No further work.**

---

## ♻️ Part 3 — h-memoization 🚦 *(flag-gated OFF until GPU parity smoke)*

### 3️⃣a 🧠 `src/utils/predictor_eval.py`
- 🚦 Module flag: `_H_MEMO = os.environ.get("PT_H_MEMO", "0") == "1"` *(default OFF → exact current path)*.
- ➕ Helper `full_target_h(encoder, pixel)` → the concatenated full-forward `h` (the `encoder(pixel)` + `torch.cat` if hierarchical, factored out of the two call sites).
- `masked_predict_l1(..., h_full=None)`: line **223** becomes `h = h_full if h_full is not None else <current compute>`.
- `rollout_l1_per_horizon(..., h_full=None)`: line **256** same.
- ✅ Both new params optional → **backward-compatible**.

### 3️⃣b 🎛️ Metric fns — compute `h` once, thread it in *(only the 3 with internal redundancy)*
- `pt_tdist.compute`: `h_full = full_target_h(...) if _H_MEMO else None` **before** the deltas loop; pass to each `masked_predict_l1(..., h_full=h_full)` → **saves 3** forwards/batch.
- `pt_maskratio.compute`: same **before** the ratios loop → **saves 3**.
- `pt_teacher_free.compute`: `h_full` computed once, passed to **both** `rollout_l1_per_horizon` calls → **saves 1**.
- ⏭️ `pt_causal` / `pt_order` / `pt_rollout`: **unchanged** (1 call / different-pixel / 1 call — no gain).
- 🔒 `compute()` **signatures unchanged** ⇒ `src/utils/probe_trio.py` **untouched**; it calls `pt_maskratio`, which gets the speedup for free once `PT_H_MEMO=1`.

---

## ✅ Verification

### 🖥️ CPU-side *(no GPU; do before restart)*
1. 🔎 **3-check** on every edited `.py`: `py_compile` + `ast.parse` + `ruff check --select F,E9`.
2. 🌫️ `python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms cassle_encoder --dry-run` → prints dependency waves; confirm each non-frozen encoder shows **1 `E:` + 6 `P:`** jobs and the `P:` jobs depend on the train arm.
3. 📟 Status tool against the **current live log** with `maybe_backup`/`maybe_plot` stubbed (read-only) → confirm the rollup eval cell renders, Σ TOTAL/ETA include `P:` jobs, **no crash**.
4. 🧵 `PT_METRIC=tdist` resolves through `run_eval.sh` (grep the rendered cmd) and the ckpt name becomes `.m12e_ckpt_tdist.npz`.

### 🔬 GPU parity smoke for #3 *(on the first free GPU; gates `PT_H_MEMO=1`)*
- Pick a done encoder (e.g. **frozen**). Run `m12e --stage forward --metric tdist` with `PT_H_MEMO=0` → save `per_clip_tdist.npy` as baseline; rerun with `PT_H_MEMO=1` (cache-policy 2) → `np.allclose(base, new, atol=1e-4)` **MUST hold** (ideally exact). Repeat for `maskratio`, `teacher_free`.
- 🟢 Only after all three pass: set `PT_H_MEMO=1` in `${WORKSPACE}/.env` (or via the scheduler env) so remaining/new evals use it. Until then it's **inert**.

### 🔁 Restart procedure *(runbook §0.D — the user runs it)*
1. ⛔ Ctrl-C the scheduler tmux (hourly anchors bound train loss; trained arms skip on resume).
2. ▶️ Relaunch the **SAME** command:
   `python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms cassle_encoder`
   → first lines MUST print the cpuset pin, `skipping N already-trained arms`, and the new **P:-job count**.
3. 👀 Watch panes unchanged (`status` + `metrics`). Expect: pending encoders' 8b **fans out 6-way** across free GPUs; the 2 in-flight evals' 8b **restarts from scratch** (`m12e` writes aggregates only at the end, so mid-8b progress isn't resumable — acceptable, parallelized the 2nd time).

---

## 📁 Files touched

| File | Change |
|------|--------|
| 🛡️ `src/m12e_predictor_temporal.py` | per-metric ckpt name |
| 🐚 `scripts/run_eval.sh` | `PT_METRIC` env + per-metric tee log |
| 🗂️ `scripts/iter18_poc_ngpu.py` | `PT_METRICS`, skip strings, build_jobs `E:`/`P:` split, cache-1 + skip-arms `P:` handling |
| 📟 `scripts/iter18_poc_status.py` | `E:` plan minus 8b, `P:` single-bar estimate, rollup eval cell, `P:` in Σ/DAG |
| 🧠 `src/utils/predictor_eval.py` | `_H_MEMO` flag, `full_target_h`, optional `h_full` params |
| 🎛️ `src/utils/pt_tdist.py`, `pt_maskratio.py`, `pt_teacher_free.py` | compute `h` once |
| 📖 `iter/iter18_ablations_FTtechniues/runbook.md` | §0.D restart + #3 smoke steps |
| 📦 `configs/pipeline.yaml` | ✅ already done (#2) |

---

## ⚠️ Risks / notes

- 🔁 **#1 needs a scheduler RESTART** to take effect (running scheduler's job dict is fixed). User opted in.
- 💸 **6-way split multiplies eval GPU-time** (6× model reload + h recompute per encoder) but **cuts wall-time** — correct for a wall-time deadline with idle tail GPUs. **Log the trade** so it's not mistaken for a leak.
- 🚦 **#3 stays OFF** until the parity smoke passes; the restarted run runs **#1 + #2 only** until then.
- ✅ `iter18_poc_metrics.py` needs **NO change** (reads per-encoder `aggregate_<fam>.json`, producer-agnostic).
