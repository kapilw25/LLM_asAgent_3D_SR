# 🎯 Tighten the confidence bands — re-eval on a **fresh, disjoint 10k**  ·  2B (done) + **1B (cheap)**

---

## 🔵 1B VERSION (iter19, 2026-07-01) — **the heads are already saved → skip the ~$250 regen**

> 🟢 **The 1B disjoint-10k retest is MUCH cheaper than the 2B one.** The costly part of the 2B run was
> **re-training the flushed probe heads on `eval_10k`** (`--taxheads-only` ~15 h + `--etheads-only` ~5 h on
> 4× RTX 6000 ≈ **$250**). **The 1B eval already persisted every head we need** — verified on HF (2026-07-01):
> for **all 14 evaluated 1B encoders** the tree `outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/` already contains
> `probe_action/<enc>/probe.pt`, `probe_taxonomy/<enc>/probe_*.pt` (15 dims), and
> `encoder_temporal/<enc>/head_{aot,tov,pace}.pt`; `motion_aux_head.pt` is durable in `.../train/`.
> **⇒ the 1B retest needs NO head regeneration** — it reuses the saved 1B heads as `EVAL_HEAD_REUSE_ROOT`.

```text
   2B disjoint-10k (what it cost)                 1B disjoint-10k (this plan)
   ├─ m04d motion on subset_10k       ~1 h        ├─ m04d motion — ALREADY on HF ✅   download only
   ├─ --taxheads-only  regen  ~15 h  ⟶ $$$        ├─ (SKIP — heads already on HF) ✅   $0
   ├─ --etheads-only   regen  ~5 h×4 ⟶ ~$250      ├─ (SKIP — heads already on HF) ✅   $0
   └─ cross-set eval (test-all)  ~8–12 h          └─ cross-set eval (test-all)  ~8–12 h · reuse saved heads
```

### What changes vs the 2B doc below

| item | 2B (below) | **1B (this run)** |
|---|---|---|
| backbone / env | `ITER18_BACKBONE=vjepa_2_1_vitG` | **`ITER18_BACKBONE=vjepa_2_1_vitg`** |
| output tree | `outputs/poc/vjepa_2_1_vitG_2B/…` | **`outputs/poc/vjepa_2_1_vitg_1B/…`** |
| head regen (taxheads/etheads) | **required** (flushed in orig eval) | **SKIP** — 14 encoders' heads already on HF ✅ |
| subset_10k prep (`m04d`) | required (~1 h compute) | **already on HF** (verified 2026-07-01) → **download only** ✅ |
| `EVAL_HEAD_REUSE_ROOT` | rebuilt on eval_10k | point at the **existing** `…/vjepa_2_1_vitg_1B/eval/eval_10k` |
| roster | 17 arms | **14 evaluated 1B encoders** (the `arm_registry` eval-tokens present in the 1B tree) |
| disjointness proof | `audit_disjoint.py` (0 exact shared clips) | **same** — the data is backbone-independent |

### 1B runbook (the only steps needed)

```bash
export ITER18_BACKBONE=vjepa_2_1_vitg EVAL_CORPUS=subset_10k PROBE_SPLIT=test-all
export LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=data/subset_10k_local/subset_10k.json
export CLASS_EDGES=outputs/poc/_xset_edges/class_edges.json
# reuse the ALREADY-SAVED 1B eval_10k heads (NO regen):
export EVAL_HEAD_REUSE_ROOT="$(python src/utils/output_paths.py eval-root poc vjepa_2_1_vitg eval_10k)"

# 1) subset_10k motion features — ALREADY on HF (backbone-agnostic, built during the 2B prep) → DOWNLOAD ONLY.
#    Pulls m04d_motion_features/{motion_features.npy,.paths.npy,.meta.json} + tags.json + subset_10k.json.
#    (m10/m11 factor prep NOT needed for eval — grep-proven in the 2B doc below.) NO --ext / NO compute.
python -u src/utils/hf_outputs.py download-data data/subset_10k_local \
  2>&1 | tee logs/dl_subset_10k_local_$(date +%F_%H%M%S).log
#    [only if the npy were ever missing] recompute: src/m04d_motion_features.py --POC --subset …/subset_10k.json --local-data data/subset_10k_local --cache-policy 1

# 2) disjointness audit (CPU, ~5 s) — same clean-CI proof
python -u src/utils/audit_disjoint.py \
  --set-a data/eval_10k_local/eval_10k.json    --keys-field-a clip_keys --label-a eval_10k \
  --set-b data/subset_10k_local/subset_10k.json --keys-field-b clip_keys --label-b subset_10k --window-clips 1

# 3) cross-set retest — 14 encoders on the FULL 10k, REUSING the saved 1B heads (no taxheads/etheads regen)
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus <N> --cache 1   # ARM2ENC auto-covers the 1B roster
#    EVAL_CORPUS=subset_10k ⇒ the L: label-bootstrap runs Stage-1 once; every eval job reuses the saved heads.
```

> 📏 **Band-shrink (same math):** `n_test 1,825 → ~10,000` ⇒ CIs **~2.3× tighter** on the 1B, at ~**$0** of
> head-regen. Every new byte lands under a tagged tree; the 1B `eval_10k` results stay read-only (Task-1 contract).

---

*(The original **2B** plan follows unchanged — the reference for the head-regen mechanics + the Task-1/2/3 proofs.)*

---

> 💡 **One job, nothing else.** Our 2B arms are already trained + evaluated on `eval_10k` (a 75/5/20
> split). The close OURS arms have **overlapping 95% bands** (`n_test ≈ 1,825`), so we can't cleanly
> separate them. This run **re-uses the already-trained arms** and **tests them on the FULL, disjoint
> `subset_10k` (≈10k clips)** → `n_test ≈ 10k` → bands shrink ~**2.3×** → close arms can separate.
> **No re-training. No new arms. EVAL only.** *(The iter19 benchmark pivot — `plan_benchmark.md` — is PARKED.)*

```text
   data/eval_10k_local/   (10k clips, 75/5/20)        data/subset_10k_local/  (10k clips, DISJOINT)
   ├─ trained heads + arms  ──────── REUSE ──────────▶ test the SAME heads/arms on ALL 10k
   └─ results @ outputs/poc/...      (READ-only)        └─ NEW results @ outputs/poc/subset10k/...
        ▲ never touched (Task 1)                              ▲ tighter bands, lands beside the old
```

---

## ✅ Task checklist — ordered by ROI (highest first)

> Legend: ✅ done/proven this session · 🟢 zero-code (env only) · 🔧 small code to build · ⛔ GPU run (yours)

| # | task | what it buys | status |
|---|---|---|---|
| 1 | 🔒 **Task 1 — no previous artifact deleted/overwritten** (namespacing + guards) | safe to run | 🟢 zero-code (env block below) |
| 2 | 🧬 **Task 3 — factorization NOT needed for eval** (grep-proven) → prep = `m04d` only | skip m10 SAM (GPU-hrs) + m11 (~58 GB) | ✅ proven |
| 3 | 🧮 **Task 2 — disjointness audit** (`audit_disjoint.py`) | the clean-CI evidence | ✅ 0 exact clips · adjacency footnoted (proceed-as-is) |
| 4 | 🔧 **Stage A — `test-all` labels** → the **13** head-free metrics on the FULL 10k | settles the future-MSE arm-tie | 🔧 `m04e`/`m04f` small mode |
| 5 | ⚡ **Multi-GPU fan** the 17 encoders → `outputs/poc/subset10k/` | uses all GPUs | 🔧/⛔ |
| 6 | 🔧 **Stage B — head-reuse** (`--head-ckpt`) → the **2** probe metrics on the FULL 10k | tightens action-top1 + taxonomy-F1 | 🔧 `m12a`/`m12c` |

---

## 🧠 The ONE insight (why a naïve re-run does nothing)

```text
  point the eval at subset_10k  ──▶  m04e re-splits it 70/15/15  ──▶  TEST = 15% ≈ 1.5k  ──▶ SAME wide bars ❌
  the ONLY lever that shrinks bands:  make TEST = the FULL disjoint 10k  ──▶ n_test ≈ 10k  ──▶ ~2.3× tighter ✅
```

All 15 metrics read their **test clip keys from ONE file** (`action_labels.json`, written by `m04e`). So the
single change that tightens every metric is: emit that file with **test = every clip** (the `test-all` mode, item 4).

---

## 🗂️ The roster — **17 encoders** (frozen + 7 competitors + 9 OURS), all already trained on 2B

| 🥊 Competitors — non-ours (7) | 🏆 OURS — surgery family (9) | ⚓ anchor |
|---|---|---|
| `pretrain_encoder` · `surgical_autorgn_encoder` · `surgery_raw_encoder` · `full_ft_encoder` · `lpft_encoder` · `peft_lora_encoder` · `peft_dora_encoder` | `surgery_3stage_DI_encoder` · `surgery_noDI_encoder` · `…_replay25_encoder` · `…_diheavy_encoder` · `…_tccaux_encoder` · `…_intervene_encoder` · `surgical_intervene_wiseft_f30/f50/f70_encoder` | `vjepa_2_1_frozen` |

> 🧾 Eval-encoder name = `vjepa_2_1_<encoder-token>` (e.g. `surgery_3stage_DI_encoder` → `vjepa_2_1_surgical_3stage_DI_encoder`).
> **Excluded on purpose** (not in this roster): the `*_head` arms, `pretrain_2X`, `cassle`, `ewc`, the v1 `…_wiseft`.
> Tokens resolve via `configs/arm_registry.yaml` (the single source) — no hand-typed lists.

---

## 🧪 Which metrics tighten, and how (verified across `m12a`–`m12f`)

| group (count) | metrics | trained head? | to reach `n_test≈10k` it needs |
|---|---|---|---|
| 🟦 forward-only (8) | motion-cosine sep · future-frame MSE · rollout drift · causal L1 · L1-vs-Δt · exposure-bias gap · mask-ratio · frame-order | **none** (`m12b`/`m12d`/`m12e` forward passes) | **`test-all` only** (Stage A) |
| 🟩 self-contained (5) | Arrow-of-Time · frame-permutation · playback-pace · TCC Kendall τ · TCC cycle-back | trains its **own** read-out on the eval corpus (`m12f`) | **`test-all` only** (Stage A) |
| 🟥 transfer head (2) | Action top-1 · taxonomy F1 | **yes** — head trained on `eval_10k` (`m12a`/`m12c`) | **`test-all` + REUSE the `eval_10k` head** (Stage B) |

> 🎯 **Future-frame MSE — the metric that drives the arm-selection tie — is in the head-free group**, so the
> selection tightens in **Stage A alone** (no head-reuse, no label-binning subtlety). Stage B is a follow-on for the 2 probe metrics.

---

## 🔒 Task 1 — guarantee **nothing** from `eval_10k` is deleted or overwritten

**Two things must stay untouched:** (a) the `eval_10k` **data prep** in `data/eval_10k_local/`, and (b) the
`eval_10k` **results + trained arms** in `outputs/poc/` (incl. `…/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.{json,csv}`).

The contract is **env-only — zero code change** (verified against every write/cleanup path in `run_eval.sh`):

| risk path (in `run_eval.sh`) | what it touches by default | the guard that makes it safe |
|---|---|---|
| Stage 1–8 **result writes** (`OUTPUT_ACTION`, …, `OUTPUT_PLOTS`) | `outputs/poc/probe_action`, … (eval_10k) | **override the 7 `OUTPUT_*` → `outputs/poc/subset10k/*`** → new results land in the tag |
| **encoder/head ckpt resolvers** (`encoder_ckpt_for`, `motion_aux_head_for`) | read `${DEFAULT_OUTPUT_PREFIX}/…/student_encoder.pt` | **leave `DEFAULT_OUTPUT_PREFIX = outputs/poc`** → trained arms are **READ**, never written |
| pre-eval cleanup (L605–657) — `rm` `m09*_ckpt_latest/step.pt` | the **untagged** trained dir (eval_10k scratch) | **`EVAL_KEEP_LATEST=1`** → skips this block entirely |
| post-Stage-8 cleanup (L1069) — `rm features_test.npy` | `OUTPUT_ACTION` (now tagged) + only on `FULL`/`policy=2` | **`CACHE_POLICY_ALL=1`** at POC → `P_ACTION=1` → block skipped; even if run, it's the tagged dir |
| Stage 10 `m13` "wipes its own output_dir" | `OUTPUT_PLOTS` (now tagged) | tagged → wipes only `outputs/poc/subset10k/probe_plot` |
| frame cache (`EVAL_FRAME_CACHE_DIR`) | `${LOCAL_DATA}/m12_frame_cache` | **`LOCAL_DATA=data/subset_10k_local`** → cache lands under subset_10k, never eval_10k |

> 🧷 **Net:** the run **reads** trained arms/heads from the untagged `outputs/poc/…`, and **writes every new byte** under
> `outputs/poc/subset10k/…`. The `eval_10k` results, the trained `*.pt`, and all of `data/eval_10k_local/` are **physically out of every write/`rm` path.**
> *(Optional thin convenience: an `EVAL_TAG` env in `run_eval.sh` that sets the 7 `OUTPUT_*` in one place — same effect, fewer envs to type.)*

---

## 🧮 Task 2 — the disjointness proof (the clean-CI evidence)

The tighter bands are only honest if `eval_10k` (where heads were trained) and `subset_10k` (the new test set)
don't leak. Both were carved from the same WalkIndia parent. **Audited 2026-06-20:**

```text
  clip key  =  section/video_id/source_file   (e.g. tier2/bhopal/rain/--pBu8H35ro/--pBu8H35ro-004.mp4)
  [exact]    shared clip keys           :    0   ← THE leakage bar (iter15 was exact-clip). Encoders saw
                                                   ZERO subset_10k clips. ✅ CLEAN.
  [video]    shared source videos       :  681   ← NOT leakage: WalkIndia videos are ~160 clips each
                                                   (276 h / 714 vids) → non-adjacent same-video clips differ.
  [adjacent] same video within ±1 clips : 2996   ← near-duplicate consecutive-10s pairs (the only residual).
```

**Decision (2026-06-20, user): proceed as-is.** 0 exact overlap is the bar — the encoders trained on none of
subset_10k. The 2996 adjacency does **not** bias the *surgery-vs-baselines* claim: it's a *relative* comparison
(adjacency inflates every arm equally) and the CI is bootstrapped over subset_10k's own ~6-clips/video
(spread-out → independent) sample, not over the eval_10k pairing. **📌 Paper footnote:** ~31% of subset_10k
clips have a temporal neighbour in eval_10k → absolute held-out levels carry that mild caveat, the arm ranking
does not. (Root cause: subset_10k was built `disjoint_from: data/val_1k.json`, not eval_10k.)

- 🆕 **`src/utils/audit_disjoint.py`** (pure-CPU, built this session, self-test passes) checks three strengths:
  **exact** shared clip-key · **shared source-video** (same `video_id` in both) · **adjacent clips** (same video within ±N clip-indices = the ±30 s hard-mode rule, metadata-based — **not** frame-embedding cosine, which floods false dups on same-camera walking clips).
- 🔧 `m00d_download_subset.py` just **downloads** whatever keys are in `subset_10k.json`; the disjoint **selection** happened upstream — the audit is what **proves** it held. Put the printed verdict in the appendix.

```bash
# the audit (CPU, ~5 s) — run before trusting the tighter bands
python -u src/utils/audit_disjoint.py \
  --set-a data/eval_10k_local/eval_10k.json   --keys-field-a clip_keys  --label-a eval_10k \
  --set-b data/subset_10k_local/subset_10k.json --keys-field-b clip_keys --label-b subset_10k \
  --window-clips 1     # ±1 = consecutive-10s near-dups; the 2026-06-20 run → 0 exact / 681 vid / 2996 adj
```

---

## 🧬 Task 3 — does the EVAL need factorization? **No.**

Factorization = `m10_sam_segment/` (SAM masks) + `m11_factor_datasets/` (`D_L`/`D_A`/`D_I` factor tubes). These are
**training inputs consumed by `m09c` surgery** — not the eval. **Proven by grep** across every eval module:

```text
grep  m10_sam | m11_factor | D_L | D_A | D_I | masks | factor_manifest | sam_segment
   over  m04d, m04e, m04f, m12a, m12b, m12c, m12d, m12e, m12f, m13, run_eval.sh
   →  0 real reads   (every hit is a code COMMENT, or "head" = a probe-HEAD, unrelated)
```

> ✅ **Consequence:** for the `subset_10k` eval you build **only `m04d` motion-features** (needed by Stage-1
> action/motion-class labels). **Skip `m10` (SAM, GPU-hours) and `m11` (factor-gen, ~58 GB) entirely.**
> `tags.json` is already present in `subset_10k_local/` → the taxonomy stage can run too.

---

## 🔧 Code to build (small, grounded) — what exists vs what's new

| piece | exists? | change |
|---|---|---|
| route outputs to a tag | ✅ | **none** — override the 7 `OUTPUT_*` env (Task 1); `ITER18_EVAL_TAG` was only *proposed*, never built |
| **Stage A — `test-all` labels** | 🔧 | `m04e`/`m04f`: a `--probe-split test-all` mode → every clip → the **test** split (train/val empty), skipping the ≥5-per-split asserts. All 13 head-free metrics then test on the full 10k. `run_eval.sh` `SKIP_STAGES` drops the 2 probe-train stages in this mode |
| **Stage B — head-reuse** | 🔧 | `m12a`/`m12c`: a `--head-ckpt <eval_10k probe.pt>` → **load** the trained head, **skip training**, extract features for all 10k, infer + bootstrap-CI. ⚠️ the `subset_10k` **action labels must use `eval_10k`'s motion-class bin edges** (shared-derivation arg) — else the reused head is scored against differently-defined classes |
| **multi-GPU fan** | 🔧 | one `run_eval.sh` per encoder pinned to a GPU (the §-finale fan pattern), **or** thread the env block + the 17-roster into `iter18_poc_ngpu.py`'s eval jobs for the metric-parallel `P:`/`F:` fan |

---

## ▶️ Runbook (commands first; rationale as `#` comments)

```bash
# ── Step 0 · pull ONLY the trained arms + heads + result tables (READ-only source) ──
#    outputs/poc on HF is 829 GB, but 477 GB of that is *ckpt_latest/step/stage* resume anchors
#    (training scratch — the eval NEVER loads them) + 33 GB of regenerable .npy/.npz caches.
#    The eval needs ONLY ~297 GB: student_encoder.pt + *ckpt_best.pt + probe*.pt + motion_aux_head.pt
#    + the json/csv results. Do NOT `download-data outputs/poc` (full 829 GB) — include-filter instead:
HF_HUB_DISABLE_XET=1 hf download anonymousML123/factorjepa-outputs --repo-type dataset \
  --include "outputs/poc/**/student_encoder.pt" --include "outputs/poc/**/*ckpt_best.pt" \
  --include "outputs/poc/**/probe*.pt"          --include "outputs/poc/**/motion_aux_head.pt" \
  --include "outputs/poc/**/*.json"             --include "outputs/poc/**/*.csv" \
  --local-dir . 2>&1 | tee logs/download_outputs_poc_lean_$(date +%Y%m%d_%H%M%S).log
#    (if it aborts on a server-side xet-corrupt .pt blob, re-run — it's idempotent/skip-existing —
#     or use the non-fatal plain-HTTP path: `download-data outputs/poc --ext pt,json,csv`, which is
#     796 GB because --ext can't drop the latest/step anchors; only the include-filter hits 297 GB.)

# ── Step 1 · the ONLY prep subset_10k needs = m04d motion-features (Task 3: NO m10/m11) ──
#    ~30-60 min, 1 GPU. Writes data/subset_10k_local/m04d_motion_features/ (durable).
python -u src/m04d_motion_features.py --POC \
  --subset data/subset_10k_local/subset_10k.json \
  --local-data data/subset_10k_local --cache-policy 1

# ── Step 2 · disjointness audit (CPU, ~5 s) — the clean-CI evidence (Task 2) ──
python -u src/utils/audit_disjoint.py \
  --set-a data/eval_10k_local/eval_10k.json    --keys-field-a clip_keys --label-a eval_10k \
  --set-b data/subset_10k_local/subset_10k.json --keys-field-b clip_keys --label-b subset_10k \
  --window-clips 1     # ±1 = consecutive-10s near-dups; the 2026-06-20 run → 0 exact / 681 vid / 2996 adj

# ── Step 3 · STAGE A — the 13 head-free metrics on the FULL 10k (after the m04e test-all build) ──
#    Task-1 no-clobber env block: results → outputs/poc/subset10k/* ; trained arms READ from outputs/poc/.
#    SKIP the 2 probe-train + their paired stages (3,11,4,12,13). Fan one encoder per GPU.
ENCS="vjepa_2_1_frozen vjepa_2_1_pretrain_encoder vjepa_2_1_surgical_autorgn_encoder ..."   # the 17 (from arm_registry)
for ENC in $ENCS; do GPU=...; CUDA_VISIBLE_DEVICES=$GPU \
  LOCAL_DATA=data/subset_10k_local \
  EVAL_SUBSET=data/subset_10k_local/subset_10k.json \
  OUTPUT_ACTION=outputs/poc/subset10k/probe_action \
  OUTPUT_COS=outputs/poc/subset10k/probe_motion_cos \
  OUTPUT_MSE=outputs/poc/subset10k/probe_future_mse \
  OUTPUT_PREDTEMP=outputs/poc/subset10k/predictor_temporal \
  OUTPUT_ENCTEMP=outputs/poc/subset10k/encoder_temporal \
  OUTPUT_TAXONOMY=outputs/poc/subset10k/probe_taxonomy \
  OUTPUT_PLOTS=outputs/poc/subset10k/probe_plot \
  CACHE_POLICY_ALL=1 EVAL_KEEP_LATEST=1 PROBE_SPLIT=test-all SKIP_STAGES=3,11,4,12,13 \
  ./scripts/run_eval.sh --POC --encoders "$ENC" \
  2>&1 | tee logs/subset10k_eval_${ENC}_$(date +%Y%m%d_%H%M%S).log & done; wait

# ── Step 4 · STAGE B — action-top1 + taxonomy-F1 via head-reuse (after the m12a/m12c build) ──
#    same env block, plus --head-ckpt pointing at the UNTAGGED eval_10k heads (read-only).

# ── Step 5 · refresh the consolidated table under the tag ──
#    lands at outputs/poc/subset10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.{json,csv}
#    (the eval_10k one at outputs/poc/probe_plot/... stays intact → diff the two).
```

---

## 📐 The band-shrink math (honest)

```text
  CI half-width  ∝  1/√n_test
  eval_10k:  n_test ≈ 1,825        subset_10k (test-all):  n_test ≈ 10,000
  shrink factor = √(10000/1825) ≈ 2.3×     (top-1 ±2.3pp → ±1.0pp ;  future-MSE ±0.001 → ±0.0004)
```

> ⚠️ **Tightening separates arms whose true gap ≳ the new half-width.** The current 3-way future-MSE tie
> (`intervene 0.4955 · tccaux 0.4956 · diheavy 0.4959`, gaps 0.0001–0.0004) sits **at** the shrunk half-width —
> some pairs will separate, some may stay a **reported tie**. Report ties as **co-equal**; do not manufacture a winner.

---

## ⚠️ Honest caveats

- 🧊 **Temporal ties with Frozen are by construction.** WiSE-FT f70 *is* 70% Frozen → Arrow-of-Time / TCC τ will
  match Frozen at any `n`. The real claim is the **trade-off**: Frozen-level temporal structure **and** better future-frame prediction.
- 🟥 **Stage B label-binning.** Action-top1 head-reuse is only valid if `subset_10k`'s motion classes use **`eval_10k`'s
  bin edges** (shared-derivation). Taxonomy-F1 is safe (its labels come from `tags.json`, dataset-independent).
- 🔒 **Disjointness is verified, not assumed** (Task 2) — the tighter bands are clean **because** of the `audit_disjoint` proof.
- 📏 **Scale = 10k, declared.** This is the generalization-to-fresh-clips check **and** the band-tightening run — it does **not**
  replace still-owed training-seed variance (out of scope here).

---

## ✅ Bottom line

Re-use the **17 already-trained 2B arms**, test them on the **disjoint `subset_10k` (full 10k)** → bands ~**2.3×** tighter →
close OURS arms separate. **No training, no new arms, no factorization** (eval reads none — `m04d` is the only prep). Every new
byte lands under **`outputs/poc/subset10k/`**; the `eval_10k` results + trained arms are **read-only and physically untouched**
(Task 1). Disjointness is **proven** (Task 2: 0 shared clips). The only code to build is the **`test-all` labels** (Stage A → the
13 head-free metrics, incl. the selection-critical future-MSE) and **head-reuse** (Stage B → the 2 probe metrics).
