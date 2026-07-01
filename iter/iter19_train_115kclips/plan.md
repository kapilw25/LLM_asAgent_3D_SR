# 🚀 iter19 — Full-scale **115k** training + eval on the **1B** backbone (AAAI headline run)

> 🎯 **One decisive run.** Train the paper's headline trio on **116k clips** (75/5/20 split, `n_test ≈ 23k`),
> on the cheaper **1B** backbone (`vjepa_2_1_vitg`), then eval them + frozen on the **23k** test →
> **decisive CIs (~3.5× tighter than the 10k POC)** + a full-scale answer to "does it hold at scale?".
> Scope kept to the **3 arms that matter** (best-OUR + best-COMP + seed) → budget-feasible on personal funding.

---

## 🧭 Decisions locked (2026-07-01)

| axis | choice | why |
|---|---|---|
| 🧬 backbone | **1B** `vjepa_2_1_vitg` | ~½ the 2B cost; scale-replication ρ (0.94 fut · 0.98 causal · 0.95 mcos) shows the 1B ranking **transfers** to 2B |
| 🏆 Best-OUR | **`surgical_3stage_DI_diheavy`** | factor-surgery variant (novelty-consistent); best OUR on action-top1 + mask-ratio + teacher-free at 1B |
| 🥊 Best-COMP | **`peft_lora`** | wins future-frame/causal at 1B **and** trains few params → cheapest full-scale competitor |
| ⚓ anchor + seed | **`frozen`** (eval-only) + **`pretrain_encoder`** (SSL seed / init) | frozen = the safe-claim baseline; pretrain = the "just more SSL" baseline + every arm's init ckpt |
| ➕ post-hoc | **`surgical_diheavy_wiseft_{f50,f70}`** | eval-only WiSE-FT merges of diheavy × frozen (θ*=(1−α)·frozen+α·OURS) — registry rows added, **no training** |

**Paper claim this run defends** (calibrated, low-risk): *surgery ≫ frozen (decisive, replicates 2B→1B) and
**matches full fine-tuning-class baselines at a fraction of the trainable params** (efficiency, not raw dominance).*

---

## 📊 Grounded status (verified 2026-07-01, HF + code)

```text
✅ DATA-PREP DONE  — data/full_local on HF: 116 subset-*.tar (~120 GB) + tags.json + full_local.json
                     + m04d_motion_features + m10_sam_segment(masks) + m11_factor_datasets(D_L/D_A/D_I)
                     → the EXPENSIVE factor prep (m10 SAM + m11) is already computed & backbone-AGNOSTIC.
🚫 TRAINING = 0    — outputs/full/ is EMPTY on HF → iter19 trains the 3 arms from scratch on 1B.
🔒 m09 = SINGLE-GPU— src/m09c1_surgery_encoder.py: `device = torch.device("cuda")`; NO FSDP / DDP /
                     torch.distributed / torchrun anywhere in src/m09*.py. One arm = one GPU.
```

---

## ⚙️ Multi-GPU: the 1B needs **DDP, not FSDP** — grounded in measured wall-time

**Measured** from the v5_1B POC train logs (1B, single-GPU per arm, ~7.7k clips):

| arm (POC, 1 GPU) | steps | wall | s/step |
|---|---:|---:|---:|
| diheavy (Best-OUR) | 480 | 3:16 | ~25 |
| peft_lora (Best-COMP) | 438 | 3:14 | ~27 |
| surgery family | 480 | 3:05–4:06 | ~23–31 |
| FT baselines | 438–481 | 2:14–5:03 | ~18–42 |

**Extrapolated to FULL** (per-step time ~constant, so steps scale with clips×epochs).
**FULL = 1 epoch** (confirmed 2026-07-01): 116k clips need one pass — POC used 2 epochs *only because* it had 10k:

| | POC · 2 epochs (~7.7k train) | FULL · 1 epoch (~87k train) |
|---|---|---|
| one surgery arm | ~3:16 (480 steps) | **~19 h** (~2,700 steps) |

⇒ steps scale by `87k×1 / (7.7k×2) ≈ 5.6×` → 480 → ~2,700 steps × ~25 s ≈ **~19 h on one GPU**. With the DAG
(`pretrain → diheavy ∥ peft_lora`) plus eval, the wall is **~2–3 days, ~$150–250**. Real cost — earlier draft hand-waved it.

**FSDP vs DDP — the correction** (an earlier draft wrongly lumped them and dismissed both):

| | what it does | fit for our 1B |
|---|---|---|
| **FSDP** | shards params/grads/optimizer across GPUs — for models **too big** for one GPU | ❌ overkill — the 1B **fits** on one GPU (it trained fine at POC), so sharding buys ~nothing |
| **DDP** | replicates the 1B on N GPUs, **splits the batch** → near-linear speedup, no sharding | ✅ the right lever — diheavy on 4 GPUs ≈ **~6 h** vs ~24 h |

**Options (honest):**

| opt | wall (full run) | GPU $ | engineering | risk | when |
|---|---|---|---|---|---|
| **A · single-GPU per arm** | ~2–3 days | ~$150–250 | none | none | **one-shot full run** (recommended) |
| **B · add DDP to m09c1** | ~10–14 h | ~$80–130 | ~2–4 days | streaming-shard correctness | **multiple full runs** (2B, seeds) |
| C · FSDP / vjepa2.1 port | — | — | weeks | high | never (overkill for a fits-on-one-GPU model) |

> 🧭 **Recommendation:** for a **single** 1B full run, **A (single-GPU) clearly wins** — at 1 epoch the whole run
> is ~2–3 days, so DDP (~2–4 days to build + correctness risk) costs MORE than the ~1–1.5 days it would save.
> **B (DDP) pays off only if you'll do several full runs** (the 2B replication, multi-seed, more ablations).
> Either way it is **DDP, not FSDP** — vjepa2.1's FSDP (`app/vjepa_2_1`, `main_distributed.py`) is for its
> 1B/8B *pretraining*, a bigger regime than our fits-on-one-GPU surgery.

### 🔧 What adding DDP to `m09c1` actually touches (scoping for option B)

`m09c1` is a **4-stage progressive-unfreeze** loop feeding from a **streaming IterableDataset** of factor tubes —
that shape is what makes DDP moderate-hard rather than a one-line wrap:

| # | change | where in `m09c1` | effort | risk |
|---|---|---|---|---|
| 1 | torchrun launch + `init_process_group` + `set_device(local_rank)` | `train()` top; `device = torch.device("cuda")` (~L484) | 🟢 small | low |
| 2 | wrap student + predictor in `DistributedDataParallel`, **RE-WRAP each stage** (trainable prefix changes per stage) | stage loop + `build_optimizer` (~L1265, L1323–1331) | 🟡 moderate | med — per-stage re-wrap must stay in sync with the optimizer rebuild |
| 3 | **rank-aware sharding of the streaming factor loader** — an IterableDataset, so `DistributedSampler` does NOT apply; each rank must pull a **disjoint** clip shard | `stream_loader` / `_streaming_worker_init` (~L1396–1410) | 🔴 **hard** | **high — the crux**: wrong sharding ⇒ duplicated clips ⇒ wrong grads / inflated effective epoch |
| 4 | `model.no_sync()` around the grad-accum macro-step's intermediate backwards | grad-accum block (~L1518–1583) | 🟡 moderate | med — premature all-reduce corrupts accumulated grads |
| 5 | gate checkpoint / probe / live-plot / wandb writes to **rank 0** | `save_training_checkpoint`, `_run_probe_at_step`, `_render_live_plots`, `_log_step` | 🟢 mechanical | low (many call sites) |
| 6 | rank-consistent **resume** (every rank resumes the same stage/step) | resume block (~L868–871, L1360–1373) | 🟡 moderate | med |
| 7 | `UncertaintyWeights` module into the DDP/optimizer path | ~L726 | 🟢 small | low |

**Bottom line for B:** ~2–4 focused days **plus a parity run** (DDP result must match single-GPU within the CI).
The correctness risk is concentrated in **#3 (streaming shard)** and **#2 (per-stage re-wrap)** — that is why
this is genuine engineering, not a wrapper. If you only ever do this one 1B full run, **A wins**; if the 2B
replication + seeds are coming, **B amortizes**.

---

## 🗂️ Roster (6 eval encoders · 3 trained + frozen + 2 merges)

```text
TRAIN (1B, --FULL, single-GPU each):
  1. pretrain_encoder            (m09a1) — SSL seed; provides SURGERY_INIT + peft init            [raw clips]
  2. surgical_3stage_DI_diheavy  (m09c1) — Best-OUR; inits from pretrain                          [needs m11 factor tubes ✅ on HF]
  3. peft_lora_encoder           (m09b)  — Best-COMP; inits from pretrain                         [raw clips]
EVAL-ONLY:
  4. frozen                              — anchor (no train)
  5. surgical_diheavy_wiseft_f50         — post-hoc merge (α=0.5)   } kind: merge, built by
  6. surgical_diheavy_wiseft_f70         — post-hoc merge (α=0.3)   } wiseft_merge.py (no train)
```
DAG: `pretrain` (serial, ~GPU0) → then `diheavy` (GPU0) **∥** `peft_lora` (GPU1) in parallel → eval all 6.

---

## ▶️ Stages (commands are illustrative — the operator runs GPU jobs; I build code + this plan)

```bash
# ── Stage 0 · pull the prepped full data (~120 GB, one command) ──────────────────────────────
#    m04d + m10 SAM + m11 factor are ALREADY in the repo tree on HF (backbone-agnostic) → just download.
python -u src/utils/hf_outputs.py download-data data/full_local 2>&1 | tee logs/iter19_dl_full_local_$(date +%F_%H%M%S).log

# ── Stage 1 · point the pipeline at full_local + the 1B backbone ─────────────────────────────
#    (M9 yaml-keyed data dir — one flip; iter16 pre-flight gate). Backbone via env, per iter18.
sed -i -e 's|local_data_dir:.*|local_data_dir: "data/full_local"|' \
       -e 's|master_manifest_name:.*|master_manifest_name: "full_local.json"|' configs/pipeline.yaml
export ITER18_BACKBONE=vjepa_2_1_vitg     # 1B

# ── Stage 2 · TRAIN the seed, then Best-OUR ∥ Best-COMP (single-GPU each) ─────────────────────
#    2a) pretrain FIRST (serial — it is every arm's init ckpt).
CUDA_VISIBLE_DEVICES=0 CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --FULL \
  2>&1 | tee logs/iter19_full_pretrain_$(date +%F_%H%M%S).log
#    2b) diheavy (GPU0) ∥ peft_lora (GPU1) — both init from pretrain's m09a_ckpt_best.pt.
SURGERY_INIT=outputs/full/vjepa_2_1_vitg/m09a_pretrain_encoder/m09a_ckpt_best.pt \
CUDA_VISIBLE_DEVICES=0 CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_diheavy_encoder --FULL \
  2>&1 | tee logs/iter19_full_diheavy_$(date +%F_%H%M%S).log &
CUDA_VISIBLE_DEVICES=1 CACHE_POLICY_ALL=2 bash scripts/run_train.sh peft_lora_encoder --FULL \
  2>&1 | tee logs/iter19_full_peftlora_$(date +%F_%H%M%S).log &
wait

# ── Stage 3 · EVAL all 6 on the 23k test (produces the 15 metrics + the merges) ──────────────
#    run_eval builds the frozen anchor + the two diheavy WiSE-FT merges post-hoc (kind: merge).
EVAL_CORPUS=full CACHE_POLICY_ALL=1 bash scripts/run_eval.sh --FULL \
  --encoders "vjepa_2_1_vitg_frozen vjepa_2_1_vitg_pretrain_encoder vjepa_2_1_vitg_surgical_3stage_DI_diheavy_encoder vjepa_2_1_vitg_peft_lora_encoder vjepa_2_1_vitg_surgical_diheavy_wiseft_f50_encoder vjepa_2_1_vitg_surgical_diheavy_wiseft_f70_encoder" \
  2>&1 | tee logs/iter19_full_eval_$(date +%F_%H%M%S).log

# ── Stage 4 · cross-backbone + forest/scale plots + HF persist ───────────────────────────────
python -u src/m13_eval_plot.py --cross-plots --cross-mode full          # forest_{best,frozen}_{ci,mean} @ 23k CIs
python -u src/utils/hf_outputs.py upload-data data/full_local
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset --include "outputs/full/**" --exclude "**/.*"
```

---

## ⏱️ Wall-time & cost (grounded in the v5_1B POC train logs — see the multi-GPU section)

| step | GPUs | wall (option A, single-GPU) | note |
|---|---|---:|---|
| Stage 0 download | — | ~0:20–0:40 | ~120 GB |
| pretrain (seed) | 1 | ~19 h | FULL 1 epoch ≈ ~2,700 steps at ~25 s/step |
| diheavy ∥ peft_lora | 2 | ~19 h | parallel; each ~19 h (116k × 1 epoch) |
| eval (6 enc, 23k) | 1–N | ~1 day | ~2–3× the 10k eval; fan per-encoder across GPUs |
| **total wall (option A)** | 2–3 | **~2–3 days** | pretrain (serial) → diheavy ∥ peft_lora → eval |
| **budget** | | **~$150–250** (A) · **~$80–130** (B, DDP) | 1B on RTX 6000 @ ~$1.2/hr |

> ✅ **FULL = 1 epoch (confirmed 2026-07-01).** 116k clips need only one pass; POC used 2 epochs *because* it had
> just 10k clips. So per arm ≈ **~19 h** (not 48 h). Verify `max_epochs.full: 1` in
> `configs/train/base_optimization.yaml`. Everything else is measured (diheavy 480 steps / 3:16 ≈ 25 s/step).

---

## ⚠️ Risks / open items

- ✅ **Epochs at FULL = 1 (decided 2026-07-01).** 116k clips → one pass (POC used 2 only because it had 10k). Verify `max_epochs.full: 1` in `configs/train/base_optimization.yaml` (iter16 M2 single-source). No under-training worry: full-scale sees ~87k train-clip-passes (116k×1) vs POC's ~15k (7.7k×2) → ~5.6× more clip-exposure.
- 🔗 **1B init ckpt** — `pretrain_encoder --FULL` must write `outputs/full/vjepa_2_1_vitg/m09a_pretrain_encoder/m09a_ckpt_best.pt`; diheavy + peft_lora depend on it. Verify the path after Stage 2a.
- 🧊 **m11 factor tubes are backbone-agnostic** (built from clips, not the model) → the HF `m11_factor_datasets` is reusable for the 1B diheavy. Confirm the streaming loader resolves them under `data/full_local`.
- 💾 **Disk** — 116 tar (~120 GB) + m10/m11 (~60 GB) + 3 arm ckpts + eval artifacts → provision ≥ 350 GB.
- 🧵 **peft_lora at FULL** — confirm `m09b` FULL config exists (it was a POC baseline); if not, add the FULL mode block (copy-from-POC, per baseline convention).

---

## 🔁 Supersedes

This plan **replaces** `runbook_train_115kclips.md` (the iter16 2B-era runbook: Pro 4000/6000 migration, m04d
resume, the old 7-cell roster). Reuse from it only: the **M1–M9 code mods** (already landed) and the
**data-prep outputs** (done, on HF). Everything below Stage 1 here is the current iter19 path.
