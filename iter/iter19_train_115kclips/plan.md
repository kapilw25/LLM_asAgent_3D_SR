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

## ⚙️ The FSDP question — **decision: do NOT port FSDP for this run**

The V-JEPA 2 repo does ship FSDP (V-JEPA 2.1 pretraining lives in `app/vjepa_2_1/`, launched via
`app/main_distributed.py` on SLURM; configs under `configs/train_2_1/`). **But our surgery training loop
(`m09c1`) is our own code and is single-GPU.** Adopting FSDP means porting the entire surgery objective
(3-stage DI, factor tubes, uncertainty weighting, predictor blend) into an FSDP-wrapped loop — a large,
risky rewrite that is **not needed for the 1B**:

| option | fit? | verdict |
|---|---|---|
| **A · single-GPU per arm, fan arms across GPUs** | 1B (~1408-d ViT-g) trains comfortably on one 96 GB RTX 6000 | ✅ **chosen** — simplest, uses existing code, DAG-parallel across the 3 arms |
| B · add FSDP/DDP to `m09c1` | would speed a single arm ~linearly across GPUs | ⏭️ deferred — big port + solo-project risk; revisit only if wall-time is prohibitive or for a 2B run |
| C · port surgery into vjepa2.1's FSDP loop | maximal scale | ❌ out of scope — months of work; contradicts the budget goal |

> 🧷 **Net:** the 1B fits on one GPU, so we get multi-GPU speed by **fanning the 3 arms across GPUs**
> (data-parallel *across arms*, not within), respecting the `pretrain → {diheavy, peft_lora}` DAG.
> FSDP stays a documented future lever (ref: `facebookresearch/vjepa2` `app/vjepa_2_1` + `main_distributed.py`).

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

## ⏱️ Wall-time & cost (⚠️ estimate — needs a SANITY timing pass to firm up)

| step | GPUs | rough wall | note |
|---|---|---:|---|
| Stage 0 download | — | ~0:20–0:40 | ~120 GB |
| pretrain (seed) | 1 | measure | 115k clips × FULL epochs; **run one SANITY step to get s/step, then ×N** |
| diheavy ∥ peft_lora | 2 | measure | parallel; diheavy streams m11 factor tubes |
| eval (6 enc, 23k) | 1–N | measure | ~2–3× the 10k eval; fan per-encoder across GPUs |
| **budget** | | **~$150–250** | 1B on RTX 6000 @ ~$1.2/hr (per your 2B disjoint burn); firm up after the timing pass |

> 🚫 **No fabricated wall-times.** The POC per-arm 1B time was never isolated (the ngpu scheduler ran many
> arms concurrently). **First action on the box: a `--SANITY` timing run** of `pretrain` + `diheavy` to get
> s/step, then extrapolate `115,687 × epochs / (steps/s)`. Report as duration only (hh:mm).

---

## ⚠️ Risks / open items

- 🧮 **Epochs at FULL** — confirm `max_epochs.full` in `configs/train/base_optimization.yaml` (iter16 M2 = single-source `{…,full:1}`); 1 epoch over 115k may under-train diheavy vs the POC's 2 epochs over 7.7k. **Decide epochs before launch.**
- 🔗 **1B init ckpt** — `pretrain_encoder --FULL` must write `outputs/full/vjepa_2_1_vitg/m09a_pretrain_encoder/m09a_ckpt_best.pt`; diheavy + peft_lora depend on it. Verify the path after Stage 2a.
- 🧊 **m11 factor tubes are backbone-agnostic** (built from clips, not the model) → the HF `m11_factor_datasets` is reusable for the 1B diheavy. Confirm the streaming loader resolves them under `data/full_local`.
- 💾 **Disk** — 116 tar (~120 GB) + m10/m11 (~60 GB) + 3 arm ckpts + eval artifacts → provision ≥ 350 GB.
- 🧵 **peft_lora at FULL** — confirm `m09b` FULL config exists (it was a POC baseline); if not, add the FULL mode block (copy-from-POC, per baseline convention).

---

## 🔁 Supersedes

This plan **replaces** `runbook_train_115kclips.md` (the iter16 2B-era runbook: Pro 4000/6000 migration, m04d
resume, the old 7-cell roster). Reuse from it only: the **M1–M9 code mods** (already landed) and the
**data-prep outputs** (done, on HF). Everything below Stage 1 here is the current iter19 path.
