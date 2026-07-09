---
description: Forensically identify WHY a GPU job is idle/slow — measure the per-item pipeline stage by stage, classify the stall against known signatures, WEBSEARCH the gold-standard fix, verify the speedup from log traces
argument-hint: [optional job/log/module to diagnose; default = the currently-running GPU job showing idle time]
---

# /gpu-bottleneck — where is the GPU time going, and what is THE fix?

Target: $ARGUMENTS (if blank, the currently-running GPU job whose nvtop/status-pane shows idle or a regressed ETA).

**Portability**: the file paths, metric names and numbers below are THIS repo's measured precedents — in a
different project substitute the analogous stages (decode → your producer, clips → your items, `recent=` →
your progress bar's windowed-rate token). The signatures and fixes are generic; the examples are calibration.

## Hard rules (read before touching anything)

1. **MEASURE, don't assert.** Every claim needs a same-turn tool result: a log grep, a timed stage, an
   `nvidia-smi` read. Banned: "probably decode-bound", "should be the dataloader". 07-08 lesson: the
   asserted fix was "offload decode to the GPU"; measurement said the wall was elsewhere entirely
   (the numbers live in Step 3's traps and Step 4's rows — don't restate them, reuse them).
2. **Windowed rates only** — read the pbar's `recent=…s/clip`, never `total/elapsed` (resume-skips poison it).
3. **Accuracy-over-speed gate**: any fix that changes pixels, precision, masks, or eval math is NOT a
   speedup — it is a study change. See the validity box at the bottom.
4. Never run/kill the user's GPU jobs — diagnose from logs + `nvidia-smi`; standalone microbenches go in
   the scratchpad on CPU (or a few-second, <1 GB GPU op if the cards have headroom).

## Protocol

### Step 1 — OBSERVE the shape of the idleness (30 s, no code)
```bash
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader
```
```text
┌────────────────────────────────┬──────────────────────────────────────────────────────────┐
│ signature                       │ meaning                                                   │
├────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ square wave (100% ↔ 0%, ~10s)   │ PIPELINE STALL — GPU waits for the producer, not slow math│
│ flat 99% + high power           │ forward-bound — the GPU IS the wall (skip to step 4e)     │
│ flat low % + low power          │ starved producer OR tiny kernels (measure step 3)         │
│ one proc at 2000%+ CPU          │ internally-threaded serial producer (e.g. one ffmpeg      │
│                                 │ decode using 21 cores — still ONE item at a time)         │
│ 0% BUT the process is young     │ STARTUP TRANSIENT, not a stall — model load + resume-ckpt │
│ (`ps -o etime -p <pid>` ≤ ~10m) │ np.load + skip-scan run CPU-only. Wait, re-observe, THEN  │
│                                 │ diagnose (07-09 false alarm: a 4-min-old job at 0%)       │
└────────────────────────────────┴──────────────────────────────────────────────────────────┘
```

### Step 2 — LOCATE the stage from its log (rates, not vibes)
```bash
tr '\r' '\n' < logs/<job>.log | grep -aoE "recent=[0-9.]+s/clip|[0-9]+/[0-9]+ \[" | tail -3
```
Compare sibling jobs: a healthy sibling's rate (e.g. tov 0.9 s/clip @99%) prices what the sick one
(pace 1.4 s/clip @17%) should do — the gap is the recoverable time.
**ALSO grep `[resume]`/`discard` lines** — idle hunts surface CORRECTNESS bugs, which outrank
throughput: a progress bar restarting from ~0 after a resume is a shared-state collision smell
(07-09: four sibling jobs shared ONE ckpt filename → last writer wins, siblings discard hours).

### Step 3 — BREAK OPEN the per-item pipeline (the decisive step)
Write a scratchpad script that times EACH stage standalone on real data (real TAR clips, real shapes):
`read/IO → decode → preprocess (resize/normalize) → H2D copy → forward → save/ckpt`.
Traps that session-tested real: (a) time preprocess at BOTH `torch.set_num_threads(96)` and `(1)` — a
pool worker gets 1 thread, so the multi-thread number lies about worker cost (228 ms vs 1353 ms);
(b) H2D is almost never the wall (113 MB @ PCIe gen4 ≈ 5 ms) — don't "optimize" it first;
(c) a cumulative `np.savez` ckpt is O(n) and GROWS — time it at current n, not at n=0.

### Step 3b — AUDIT every remaining CPU op for CUDA-movability
Grep the hot path (consumer/main thread): `torch\.(stack|cat)|index_select|\.cpu\(\)|np\.` — then
apply the rubric per op: (1) per-batch or one-shot? one-shot ≈ never worth it; (2) measured share of
the batch wall — <1% skip; (3) VALUE-CHANGING (matmul/resize/interp/reductions → validity-gated,
clean-slate only) vs pure DATA MOVEMENT (stack/cat/flip/permute/index_select → byte-identical,
PROVE with `torch.equal(cpu_out, gpu_out.cpu())`); (4) the mover is usually ONE line at the dispatch
top — `batch = batch.to("cuda", non_blocking=True)` (pinned source → async DMA) — after which every
downstream gather runs on GPU with zero edits to the metric code. `.cpu()` on RESULTS stays: storage
needs CPU.

### Step 4 — CLASSIFY against known signatures → each maps to ONE fix
```text
┌───┬──────────────────────────────────┬───────────────────────────────────────────────────────┐
│ # │ diagnosis (evidence)              │ THE fix (in-repo precedent)                            │
├───┼──────────────────────────────────┼───────────────────────────────────────────────────────┤
│ a │ serial producer in the main loop  │ utils/decode_feeder.DecodeFeeder — N workers own       │
│   │ (one decode between forwards)     │ queue→decode→bounded ready_q (smoked: 7 ms wait vs     │
│   │                                   │ 2000 ms serial). frozen_features' decoded_q = template │
│ b │ refill-stall pool (producer only  │ SAME fix as (a) — the submit-refill ThreadPoolExecutor │
│   │ refilled between consumer pops;   │ shape is the trap: refill dies during every forward.   │
│   │ square wave SURVIVES a pool)      │ ready_depth ≥ 3 batches                                │
│ c │ blocking save on the hot path     │ background writer thread, ≤1 outstanding, snapshot =   │
│   │ (GB-scale cumulative savez)       │ shallow copies, atomic os.replace (m12f _save_ckpt)    │
│ d │ cache miss / recompute (cap hit,  │ check EVAL_FRAME_CACHE cap + policy BEFORE code: a 64f │
│   │ set too big to cache)             │ test-set ≈ 650G > 600G cap decodes cold FOREVER        │
│ e │ genuinely forward-bound (99%,     │ STOP optimizing I/O. Only levers left change the study │
│   │ feed ≪ forward)                   │ (batch/precision/model) → accuracy gate, usually NO    │
│ f │ "offload to GPU hardware" itch    │ measure FIRST: NVDEC on short/low-res clips was SLOWER │
│   │ (NVDEC, GPU resize)               │ (per-clip session init) + pixels differ (0.66/255 ≈    │
│   │                                   │ the bf16 step) → mid-study confound. Clean-slate only  │
│ g │ residual troughs at the BATCH     │ double-buffer the batch build: builder thread stacks + │
│   │ cadence after (a/b) — the CPU     │ pins batch k+1 while k forwards (BatchFeeder, depth 2);│
│   │ stack of GBs between flushes      │ OOM-backoff halves by SLICING views, not re-stacking   │
│ h │ CPU gather INSIDE the forward     │ upload-once per Step 3b's rubric — 07-09: one line     │
│   │ call (exactly 1 core at 100%      │ removed 5.3s/batch of 1-thread permute-gather (tov's   │
│   │ during each 0% GPU window)        │ 41% idle window)                                       │
└───┴──────────────────────────────────┴───────────────────────────────────────────────────────┘
```

### Step 5 — WEBSEARCH the gold standard (mandatory, ≥2 cites, BEFORE building)
Query patterns that landed: "GPU idle between batches prefetch queue producer consumer dataloader
prefetch_factor" · "torchcodec NVDEC decoder overhead short videos slower than CPU" · "pinned memory
non_blocking CUDA streams overlap copy compute". Verify the community fix matches your Step-4 class —
if the search says "hi-res long video" and your clips are 10 s 480p, the advice INVERTS.

### Step 6 — FIX at the shared layer
One util, every sibling module imports it (the m12e-got-it/m12f-didn't asymmetry burned a full day).
Test-first: CPU smoke of the new path on real data (items in == items out, exactly once + a starve
assert) BEFORE wiring; wire ALL siblings in the same pass; 3-check + `--help` yaml-resolution per module.
Smoke-rate trap: the fake consumer must run at the MEASURED production rate — a too-fast fake forward
false-FAILs the starve assert (07-09). Fix the test's realism, never the threshold.

### Step 7 — VERIFY the speedup from logs (the only acceptable proof)
Quote before/after `recent=` lines WITH their log filenames. Honest accounting is Amdahl-scaled:
`overall = 1 / ((1-share) + share/speedup)` — a 1.8× on ⅓ of the wall is 1.18×, and claiming
otherwise gets called "why are you lying". State what the ETA should now read and why the pane's
projection lags (it re-prices only when a job finishes at the new rate).

## Validity gates (a speedup that breaks these is a rejected paper, not a win)
```text
🔒 pixel source is part of the study — decoder/resize backend flips mid-study manufacture a confound
   (compare any pixel diff against the bf16 quantization step ≈ 4e-3 before even discussing it)
🔒 POC ↔ FULL parity — a FULL-only pipeline change that alters inputs breaks the scale-replication claim
🔒 never touch the novelty recipe's knobs to go faster; baselines' infra is fair game
```

**Distinction from sibling commands**: `/10x` names constraints to relax for a FUTURE 10× (aspirational,
any axis); `/gpu-bottleneck` is a forensic protocol for a LIVE job burning money right now — it ends
with a measured before/after trace, not a roadmap. `/brutal` critiques an artifact; this diagnoses a system.
