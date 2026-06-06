# iter18 · plan_baselines_CODE.md — DETAILED code plan per FT (Fine-Tuning) baseline

> Companion to `plan_baselines_roadmap.md`. **Every abbrev is written `abbrev (FULL FORM)` at every mention.**
> Reference modules RE-READ for this plan: `scripts/run_train.sh` (surgery dispatch 324-419),
> `src/m09c1_surgery_encoder.py` (THE COPY SOURCE — complete, tested trainer), `src/utils/training.py`
> (shared primitives), `configs/train/{base_optimization,surgery_base,surgical_autorgn_encoder}.yaml`.
> Official-code repos grounding every snippet → the `§ 2` "Official code" table in `plan_baselines_roadmap.md`
> (each repo's core file was READ, June 2026).

---

## 0 · ARCHITECTURE — each baseline is its OWN script (copy-first-then-factor)

```text
┌ rule ─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Each FT baseline B1-B4 gets its OWN full trainer script — NEVER an if/freeze_rule branch inside    │
│    m09c1_surgery_encoder.py. The paper NOVELTY (m09c1) stays un-polluted; the comparison stays        │
│    un-attackable ("did surgery win because of a shared code path?" → no, every arm is isolated).      │
│ 2. BUILD each by COPYING m09c1 VERBATIM FIRST (cp), THEN specialize its loop. m09c1 is the complete,  │
│    battle-tested trainer — copying it means the baseline inherits EVERY already-built function. Do    │
│    NOT build from scratch and do NOT revive a stale module.                                           │
│ 3. WE CAN AFFORD REDUNDANCY; WE CANNOT MISS AN ALREADY-BUILT FUNCTION. A helper duplicated across 4   │
│    copies costs a little disk; a MISSING helper costs a multi-error debug cycle — exactly the         │
│    m09a1↔m09c1 drift (functions built+tested in m09a1 were absent in m09c1) that motivated this rule. │
│ 4. Factoring genuinely-common helpers to src/utils/ is a SEPARATE post-copy phase (tracker #19),      │
│    done AFTER all four scripts exist + pass SANITY. utils/training.py stays technique-agnostic (#49). │
│ 5. EXCEPTION: surgery_raw IS surgery on raw clips → it stays a CONFIG of m09c1, not a new script.     │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```text
┌ baseline ─────────────────┬ own script (cp m09c1) ───────┬ status ──────────────┐
│ B2 Auto-RGN               │ src/m09e_autorgn_encoder.py  │  BUILT this iter     │
│ B1 LoRA → DoRA            │ src/m09b_peft_encoder.py     │ cp pending (#14)     │
│ B3 CaSSLe + EWC           │ src/m09d_contssl_encoder.py  │ cp pending (#15)     │
│ B4 Full-FT / LP-FT        │ src/m09f_naiveft_encoder.py  │ cp pending (#12)     │
│ surgery_raw (RAW control) │ m09c1 + config (IS surgery)  │ config pending (#16) │
│ surgery (OURS / novelty)  │ src/m09c1_surgery_encoder.py │ — untouched          │
└───────────────────────────┴──────────────────────────────┴──────────────────────┘
```

---

## 0.1 · Full forms used below (read them again)

```text
┌ abbrev ──┬ FULL FORM ─────────────────────────────────────────────────────────────┐
│ Auto-RGN │ Automatic Relative Gradient Norm   (Surgical-FT, Lee et al. ICLR'23)   │
│ EWC      │ Elastic Weight Consolidation       (Kirkpatrick et al. PNAS'17)        │
│ LoRA     │ Low-Rank Adaptation                (Hu et al. 2021)                    │
│ DoRA     │ Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024)                │
│ PEFT     │ Parameter-Efficient Fine-Tuning                                        │
│ LP-FT    │ Linear-Probing then Fine-Tuning    (Kumar et al. ICLR'22)              │
│ Full-FT  │ Full Fine-Tuning                                                       │
│ CaSSLe   │ continual self-supervised distillation (stylized; Fini et al. CVPR'22) │
│ SSL      │ Self-Supervised Learning                                               │
│ SPD      │ Selective Projection Decay         (Tian et al. NeurIPS'24)            │
│ SALT     │ Self-Anchored Latent Teacher       (Apple 2025)                        │
└──────────┴────────────────────────────────────────────────────────────────────────┘
```
> 🔒 **NAMING RULE (MANDATORY):** in every new `*.py` / `*.sh` / `*.yaml` / log string / docstring,
> write the abbrev as `abbrev (FULL FORM)` at EVERY mention — never the bare abbrev. Arm names always
> carry the `_encoder` suffix (`surgical_autorgn_encoder`, `peft_lora_encoder`, … — never bare).

---

## 0.2 · The CONTRACT every script must honor (so eval is FREE)

Every m09* trainer emits the SAME two artifacts; honor them and `run_eval.sh` runs the 9-metric suite
with zero new eval code. Because each baseline is a COPY of m09c1, it inherits these calls for free.

```text
┌ helper (src/utils/training.py) ──┬ writes ─────────────────────────────────────────────────────────┐
│ export_student_for_eval(student, │ student_encoder.pt (key "student_state_dict", encoder only).    │
│   path, explora_enabled=False)   │ explora_enabled=True FIRST merges LoRA/DoRA adapters into the   │
│                                  │ base → a PLAIN ViT (the PEFT export path).                      │
│ finalize_outputs(*, student,     │ BOTH student_encoder.pt + <ckpt_prefix>_ckpt_best.pt (keys      │
│   output_dir, ckpt_prefix, …)    │ "student"+"predictor"). m09a2 / m09c2 use it verbatim; every    │
│                                  │ copied baseline keeps its m09c1 call (just change ckpt_prefix). │
└──────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```
```text
┌ artifact ────────────┬ schema / why ───────────────────────────────────────────────────────┐
│ student_encoder.pt   │ key "student_state_dict" — encoder only. m12a/b/c (action/motion/   │
│                      │ taxonomy) load this.                                                │
│ m09X_ckpt_best.pt    │ keys "student" + "predictor" — Stage 8 future_mse (m12d) + Stage 8b │
│                      │ predictor-temporal (m12e) REQUIRE the "predictor" key. No predictor │
│                      │ key → m12d/m12e FATAL for that arm (run_eval.sh preflight).         │
│ configs/eval/        │ one row per arm {kind: vjepa, arch, crop, embed_dim} → run_eval.sh  │
│  probe_encoders.yaml │ ENCODERS=<arm> → m12a–m12e → 9 metrics + paired BCa 95% CI.         │
└──────────────────────┴─────────────────────────────────────────────────────────────────────┘
```
> ⚠ PEFT arms must STILL emit the `predictor` key (m12d/m12e). After `merge_and_unload()`, pass the
> (unchanged) JEPA predictor through `finalize_outputs(... ckpt_prefix="m09b")` so `m09b_ckpt_best.pt`
> carries both keys — else future_mse / predictor-temporal FATAL for that arm.

---

## 1 · Shared scaffolding — already in utils/training.py (every copy inherits it)

The copy-first rule means each baseline starts with m09c1's full set of calls into these tested
primitives. Specialize only the loop body; never re-implement these.

```text
┌ primitive (src/utils/training.py) ─────┬ role ─────────────────────────────────────────────────────────────┐
│ build_student_predictor(mcfg, dcfg)    │ :216 → (student ViT, predictor) — identical kwargs across m09*    │
│ build_optimizer(student, predictor,    │ :833 → param groups. init_params=θ* anchors θ→θ* (SPD slot);      │
│   cfg_opt, init_params=None)           │ Fisher-weight it → EWC (Elastic Weight Consolidation)             │
│ set_trainable_prefix(student, n)       │ :1626 → unfreeze a CONTIGUOUS prefix of n blocks. Auto-RGN /      │
│                                        │ Full-FT REPLACE this selection with their own.                    │
│ select_blocks_auto_rgn(...)            │ :1642 → B2 Auto-RGN primitive (OOM-retry + gradient-checkpoint    │
│                                        │ scoring built in). Called by m09e; technique-agnostic (#49).      │
│ build_scheduler(opt, opt_cfg, steps)   │ single front-loaded warmup (capped 10%)                           │
│ StreamingFactorDataset / FactorSampler │ FACTOR clips (surgery path) — OFF for raw baselines via config    │
│ enable_gradient_checkpointing(model)   │ :922 → ~2-4× activation-memory cut (idempotent)                   │
│ export_student_for_eval :1372 /        │ the export CONTRACT (§ 0.2) — student_encoder.pt + ckpt_best.pt   │
│   finalize_outputs :243                │                                                                   │
│ AdaptiveBatchSizer + cuda_cleanup      │ OOM safety (48 GB SANITY / 96 GB FULL) — sub-batch + on_oom retry │
└────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘
```
> m09a1 freeze knob = `layer_freeze.{enabled, freeze_below}`; pretrain_encoder.yaml `freeze_below: 20`.
> m09c1 freeze knob = per-stage `surgery.stages[i].unfreeze_below` → `set_trainable_prefix`. A copied
> baseline picks whichever knob its technique needs (Full-FT = freeze_below 0; LP-FT = lp_ft_stage0).

---

## 2 · Per-baseline code plan — copy m09c1, then the labelled delta

> Each delta is grounded in the official repo READ for it (links → roadmap § 2 table). Where our setup
> forces a deviation it is labelled **ADAPTATION** and must be stated as such in the paper.

### 🚩 B2 · Auto-RGN (Automatic Relative Gradient Norm) — ✅ AS-BUILT (the KILL-SHOT)

```text
SCRIPT        src/m09e_autorgn_encoder.py  =  cp of m09c1_surgery_encoder.py, docstring relabelled
              "iter18 B2 BASELINE … NOT the paper novelty". m09c1 itself is UNTOUCHED (pure surgery).
DELTA         the stage-loop reads cfg["surgery"]["freeze_rule"]==auto_rgn (from the yaml, no CLI arg)
              → calls utils.training.select_blocks_auto_rgn ONCE at stage 0 instead of set_trainable_prefix.
PRIMITIVE     select_blocks_auto_rgn lives in utils/training.py (:1642) — shared, technique-agnostic.
data          RAW clips: replay.raw_pretrain_pct=1.0 ; EMA teacher ; NO SALT/SPD/saliency.
budget-match  k = round(48 × surgery deepest unfreeze_below 0.167) = 8 blocks → EXACTLY surgery's
              trainable-param count (else the namesake comparison is attackable).
single-shot   Auto-RGN picks ONCE on n_score_batches warmup batches (single trainable phase).
```

**(a) OFFICIAL RGN scorer — faithful to `anniesch/surgical-finetuning` `main.py`** (`get_lr_weights` /
`get_grad_norms`): per-PARAMETER-TENSOR, L2/Frobenius, averaged over the first 5 batches, norm/LayerNorm
tensors excluded, gradients from a fwd+bwd with NO optimizer step. **AS-BUILT** this reuses the real
training step `_train_step_grad_accum` so masks / EMA-teacher / predictor / JEPA loss are bit-identical.

**(b) OFFICIAL selection = SOFT per-tensor LR — we DELIBERATELY do not use it.** The repo keeps every
tensor trainable and scales its LR `lr ∝ RGN/maxRGN`. That cannot hit a FIXED trainable-param budget,
which the comparison vs surgery REQUIRES → replaced by the budget-matched hard top-k.

**(c) OUR budget-matched ADAPTATION — hard top-k BLOCKS (state as an adaptation in the paper).**

```python
# src/utils/training.py :1642  (AS-BUILT — shared primitive; m09e calls it; m09c1 does NOT)
def select_blocks_auto_rgn(student, teacher, predictor, score_clips, mask_generators, cfg,
                           dtype, mp_cfg, scaler, sizer, loss_exp, init_params, depth, device, k):
    set_trainable_prefix(student, depth)        # ALL blocks trainable → grads flow everywhere
    enable_gradient_checkpointing(student)      # all-48-block backward → recompute acts (fit 48 GB)
    for clips in score_clips:                   # n_score_batches warmup batches
        while True:                             # OOM-retry: sizer halves micro-batch until it fits
            _bs = sizer.size
            try: _train_step_grad_accum(...); break          # fwd+bwd, NO opt.step
            except torch.cuda.OutOfMemoryError:
                if sizer.size >= _bs: raise RuntimeError("OOM at min sub-batch …")
        # RGN(tensor) = ||grad||₂ / ||θ||₂ ; skip "norm"/LayerNorm tensors (repo convention)
    block_rgn = per-block mean of the tensor RGNs           # aggregate tensor→block
    keep = top-k blocks by mean RGN ; p.requires_grad = (b in keep)   # hard freeze the rest
    print(f"[Auto-RGN (Automatic Relative Gradient Norm)] kept top-{k} = {sorted(keep)} (budget-matched)")
```

```text
CONFIG   configs/train/surgical_autorgn_encoder.yaml (extends surgery_base):
         surgery.freeze_rule: auto_rgn            # read straight from yaml (no CLI arg in m09e)
         surgery.auto_rgn: {k_blocks: 8, n_score_batches: 5}
         surgery.teacher_mode: EMA ; lp_ft_stage0: false ; spd.enabled: false ; saliency: false
         replay.raw_pretrain_pct: 1.0             # ALL raw clips = "surgery method on RAW"
RUN      REPLAY_OVERRIDE=off BACKBONE=vjepa_2_1_vitG ./scripts/run_train.sh surgical_autorgn_encoder --SANITY
         (run_train RUNNER var routes the SUBCMD → m09e; OUT_DIR = …/m09e_autorgn_encoder)
REGISTRY probe_encoders.yaml: vjepa_2_1_surgical_autorgn_encoder {kind: vjepa, arch: vit_gigantic_xformers,
         crop: 384, embed_dim: 1664}             # 2B ViT-G, depth 48
VERIFY   3-check ✅ ; SANITY on the REAL 2B ViT-G (assert "kept top-8", no OOM with the retry).
COMPARE  surgery > Auto-RGN? AND surgery > vanilla continual SSL 2×? → on vitG 2B (surgery's best 4/0/5).
```

### 🚩 B4 · Full-FT (Full Fine-Tuning) / LP-FT (Linear-Probing then Fine-Tuning)

```text
SCRIPT   src/m09f_naiveft_encoder.py = cp m09c1. ONE script, two configs (Full-FT vs LP-FT).
DELTA    factor curriculum OFF (raw clips, single stage). Full-FT: all 48 blocks trainable
         (unfreeze_below 0.0). LP-FT: surgery.lp_ft_stage0.enabled=true (the head-only warmup
         stage m09c1 already prepends) THEN unfreeze.
```
> **CRITICAL LP-FT detail (Kumar et al. ICLR'22, AnanyaKumar/transfer_learning):** phase-2 (encoder
> unfreeze) LR must be FAR LOWER than phase-1 (head probe) — head ≈ 1e-1, backbone ≈ 1e-5…1e-4
> (≈10³–10⁴× lower) so the already-accurate head does not distort pretrained features. So
> `lpft_encoder.yaml`: stage0 `head_lr_multiplier` HIGH, encoder stages `base_lr` LOW. Full-FT keeps
> ONE (un-warmed) LR for all blocks → that LR/distortion contrast IS the LP-FT-vs-Full-FT story.

```text
CONFIG   full_ft_encoder.yaml (freeze_below 0, one LR) ; lpft_encoder.yaml (lp_ft_stage0 on, low enc LR).
RUN      run_train.sh SUBCMDs `full_ft_encoder` / `lpft_encoder` → m09f_naiveft_encoder.py (RUNNER var).
REGISTRY vjepa_2_1_{full_ft_encoder, lpft_encoder} rows (2B ViT-G, vit_gigantic_xformers, embed_dim 1664).
```

### B1 · PEFT (Parameter-Efficient Fine-Tuning): LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed LoRA)

```text
SCRIPT   src/m09b_peft_encoder.py = cp m09c1 (NOT a revive of the stale m09b_explora — copy-first avoids
         its missing-function drift). Specialize: wrap the student in HuggingFace peft adapters; train
         ONLY the adapters (+ DoRA magnitude vector); MERGE before export → eval loads a PLAIN ViT.
```

```python
# src/m09b_peft_encoder.py — gold-standard PEFT via HuggingFace peft (LoRA + DoRA, one flag)
from peft import LoraConfig, get_peft_model
TARGETS = ["qkv", "proj", "fc1", "fc2"]          # timm ViT: block.attn.{qkv,proj} + block.mlp.{fc1,fc2}

def wrap_peft(student, r=16, alpha=32, use_dora=False):                 # use_dora=True ⇒ DoRA, one flag
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=TARGETS,
                     lora_dropout=0.0, bias="none", use_dora=use_dora)  # scaling = alpha/r
    return get_peft_model(student, cfg)          # base auto-FROZEN; only adapters (+DoRA m) require grad

# LoRA fwd:  y = W0·x + (B@A)·x·(alpha/r)   (A kaiming, B zeros ⇒ Δ=0 at init)
# DoRA fwd:  W = m ⊙_out (W0 + B@A·s) / ||W0 + B@A·s||_{dim=1}   (m = per-out-channel magnitude; denom detached)

def export_peft(peft_model, predictor, output_dir):                    # eval loads a PLAIN ViT, zero peft dep
    merged = peft_model.merge_and_unload()       # LoRA: W += B@A·s ; DoRA: W = (m/||·||)·(W0+B@A·s)
    finalize_outputs(student=merged, output_dir=output_dir, ckpt_prefix="m09b",   # keeps predictor key
                     ckpt_payload={"student": merged.state_dict(), "predictor": predictor.state_dict()})
```

```text
DEP      add `peft>=0.12` to setup_env_uv.sh + requirements_gpu.txt (use_dora needs ≥0.12). Cite in docstring.
CONFIG   peft_lora_encoder.yaml (use_dora:false) , peft_dora_encoder.yaml (use_dora:true).
RUN      run_train.sh SUBCMDs `peft_lora_encoder` / `peft_dora_encoder` → m09b_peft_encoder.py (RUNNER var).
REGISTRY vjepa_2_1_{peft_lora_encoder, peft_dora_encoder} rows (2B ViT-G).
NOTE     PEFT may LOOK competitive on action_top1 (capacity) but should LOSE on temporal/world-model
         metrics — that asymmetry IS the story (PEFT adapts features; surgery adapts dynamics).
```

### B3 · Continual-SSL: CaSSLe + EWC (Elastic Weight Consolidation)

```text
SCRIPT   src/m09d_contssl_encoder.py = cp m09c1. Specialize: 2 config-gated regularizers on RAW clips
         (factors off, all blocks trainable). CaSSLe vs EWC via config.
```

**CaSSLe** (Fini CVPR'22 · `DonkeyShot21/cassle`): a predictor `g` maps the CURRENT feature to predict the
FROZEN previous-model feature; distillation reuses the SSL loss; stop-grad on the frozen target. We
already hold a FROZEN teacher (the SALT slot in m09c1).

```python
# src/m09d_contssl_encoder.py — CaSSLe distillation (reuses the FROZEN teacher already in the m09c1 recipe)
g = nn.Sequential(nn.Linear(D, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, D))  # repo D→2048→D
def cassle_loss(z_student, z_frozen_teacher):                 # z_frozen from the FROZEN teacher (no_grad)
    return compute_jepa_loss(g(z_student), z_frozen_teacher.detach())   # reuse SSL loss; sg on target
# L_total = L_jepa + λ_cassle · cassle_loss(z_student, sg(teacher_frozen_feat))   # repo λ ≈ 1.0
```

**EWC** (Kirkpatrick'17 · `moskomule/ewc.pytorch`): a diagonal Fisher `F_i` weights an anchor-to-init
penalty. The anchor slot already exists — SPD via `build_optimizer(init_params=θ*)` anchors θ→θ* with
uniform weight 1; EWC = that slot with per-parameter weight `F_i`.

```python
# src/m09d_contssl_encoder.py — diagonal Fisher (SSL task-loss replaces the repo's classification NLL)
def estimate_fisher(student, predictor, subset_batches, jepa_loss_fn):     # N ≈ few-hundred clips, small bs
    F = {n: torch.zeros_like(p) for n, p in student.named_parameters() if p.requires_grad}
    for x in subset_batches:
        student.zero_grad(set_to_none=True)
        jepa_loss_fn(student, predictor, x).backward()        # ∂L_jepa/∂θ  (NOT NLL — we have no labels)
        for n, p in student.named_parameters():
            if p.requires_grad and p.grad is not None:
                F[n] += p.grad.detach() ** 2 / len(subset_batches)         # F_i = mean over N of (∂L/∂θ_i)²
    return F
# L_ewc = λ_ewc · Σ_i F_i · (θ_i − θ*_i)²   (θ* = pretrained init = the stored SPD anchor). ONE-time pass.
```

```text
CONFIG   cassle_encoder.yaml (λ_cassle on, EWC off) , ewc_encoder.yaml (EWC Fisher on, CaSSLe off).
         Keys: loss.cassle_lambda, optimization.ewc.{enabled, lambda, fisher_n_batches}.
RUN      run_train.sh SUBCMDs `cassle_encoder` / `ewc_encoder` → m09d_contssl_encoder.py (RUNNER var).
REGISTRY vjepa_2_1_{cassle_encoder, ewc_encoder} rows (2B ViT-G).
```

### RAW control · surgery_raw_encoder (factor OFF) — config-only on m09c1 (THE causal control)

```text
NO new script (it IS surgery). configs/train/surgery_raw_encoder.yaml = surgery on RAW clips
(replay.raw_pretrain_pct=1.0, the STRUCTURED 4/8/8 blocks, factors OFF) → m09c1_surgery_encoder.py.
THE single most important ablation — surgery changes TWO things vs the baselines: DATA (factor vs raw)
AND METHOD. surgery_raw isolates them:
  surgery − surgery_raw_encoder   = the FACTOR-curriculum effect (Figure-1, the headline causal claim)
  surgery_raw_encoder − Auto-RGN  = the METHOD effect (structured blocks vs gradient-heuristic blocks)
Pairs with vanilla continual SSL 2× (pretrain_2X_encoder) for the full RAW-vs-FACTOR control.
Registry row vjepa_2_1_surgery_raw_encoder (2B ViT-G). RUNNER stays m09c1 (default).
```

---

## 3 · run_train.sh wiring — RUNNER var routes each SUBCMD to its OWN script

The surgery dispatch (324-419) now sets, per SUBCMD, a `RUNNER` (which `*.py` to exec) + `MODULE_PREFIX`
(the OUT_DIR / log namespace), both defaulting to m09c1, then `python -u "$RUNNER" …` with the same
`--subset $TRAIN_POOL --output-dir $OUT_DIR --init-from-ckpt $SURGERY_INIT --cache-policy` plumbing.

```text
┌ SUBCMD ──────────────────┬ RUNNER (own script, cp m09c1) ─┬ train-config + key delta ───────────────┐
│ surgical_autorgn_encoder │ m09e_autorgn_encoder.py        │ surgical_autorgn_encoder.yaml           │
│                          │                                │   (freeze_rule: auto_rgn, from yaml)    │
│ full_ft_encoder          │ m09f_naiveft_encoder.py        │ full_ft_encoder.yaml (freeze_below 0)   │
│ lpft_encoder             │ m09f_naiveft_encoder.py        │ lpft_encoder.yaml (lp_ft_stage0 on)     │
│ peft_lora_encoder        │ m09b_peft_encoder.py           │ peft_lora_encoder.yaml (use_dora false) │
│ peft_dora_encoder        │ m09b_peft_encoder.py           │ peft_dora_encoder.yaml (use_dora true)  │
│ cassle_encoder           │ m09d_contssl_encoder.py        │ cassle_encoder.yaml (cassle_lambda)     │
│ ewc_encoder              │ m09d_contssl_encoder.py        │ ewc_encoder.yaml (ewc fisher)           │
│ surgery_raw_encoder      │ m09c1_surgery_encoder.py (def) │ surgery_raw_encoder.yaml (factors off)  │
└──────────────────────────┴────────────────────────────────┴─────────────────────────────────────────┘
```
- All eight slot into the EXISTING `surgery_*` dispatch case: add each to its inner `case "$SUBCMD"`
  that maps SUBCMD→`TRAIN_CFG`+`VARIANT_TAG`, and (for the non-default ones) override `RUNNER` +
  `MODULE_PREFIX` there. The default (set before the inner case) is m09c1 + `m09c_surgery`.
- The recipe knobs (`--teacher-mode … --replay …`) are resolved from the yaml via `_y` and passed in
  `RECIPE_V2_ARGS`. Each copied script accepts the SAME args as m09c1 (it's a copy), minus any it drops.
- **Also register each arm in `scripts/iter17_poc_ngpu.py` `ARM2ENC` + `ARM2DIR`** so the N-GPU scheduler
  fans them out (OUT_DIR = `outputs/<mode>/<backbone>/<MODULE_PREFIX>_<VARIANT_TAG>`).

---

## 4 · run_eval.sh / registry — each arm → 9 metrics, ZERO new eval code

```text
1. add the row to configs/eval/probe_encoders.yaml (encoder_ckpt_for() resolves student_encoder.pt;
   encoder_predictor_ckpt_for() resolves m09X_ckpt_best.pt for Stage 8/8b).
2. ENCODERS=<arm> ./scripts/run_eval.sh --POC  → m12a (action_top1) · m12b (motion_cos) · m12c
   (taxonomy_f1) · m12d (future_mse) · m12e (predictor-temporal ×6) = the 9-metric suite + paired BCa 95% CI.
3. §G aggregate (m13_eval_plot) renders the hero grid: surgery vs every baseline (the "vanilla
   continual SSL" anchor shows in every plot).
```

---

## 5 · Verification per arm (NEVER skip) + POC↔FULL parity

```text
┌ gate ──────────────────────────┬ command ─────────────────────────────────────────────────────────────┐
│ 3-check (after every src edit) │ py_compile + ast.parse + ruff check --select F,E9 (post-edit hook)   │
│ clean-copy check               │ new m09X has zero traces of OTHER techniques' knobs; m09c1 untouched │
│ smallest SANITY (per arm)      │ BACKBONE=vjepa_2_1_vitG run_train.sh <arm> --SANITY on the 2B ViT-G  │
│                                │ — catches FAIL-LOUD asserts / CLI wiring / dtype / OOM before POC    │
│ Auto-RGN selector unit         │ assert "kept top-8" logged AND trainable-param count == surgery's    │
│ PEFT merge round-trip          │ merge_and_unload() output loads as a PLAIN ViT (no peft) +           │
│                                │ student_encoder.pt has NO lora_/dora_ keys                           │
│ POC↔FULL parity                │ ONLY n_clips + max_epochs differ; every other yaml/CLI flag byte-    │
│                                │ identical (CLAUDE.md). No "disable feature X at POC".                │
└────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘
```

---

## 6 · Module map summary

```text
┌ technique ───────────────────────────────────────────────┬ own script (cp m09c1) ──────┬ status ───────────┐
│ B2 Auto-RGN (Automatic Relative Gradient Norm)           │ m09e_autorgn_encoder.py     │  BUILT            │
│ B4 Full-FT (Full FT) / LP-FT (Linear-Probe then FT)      │ m09f_naiveft_encoder.py     │ cp pending (#12)  │
│ B1 LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed) │ m09b_peft_encoder.py        │ cp + peft dep #14 │
│ B3 CaSSLe + EWC (Elastic Weight Consolidation)           │ m09d_contssl_encoder.py     │ cp + 2 losses #15 │
│ RAW control surgery_raw_encoder                          │ m09c1 + config (IS surgery) │ config-only  #16  │
│ surgery (OURS / novelty)                                 │ m09c1_surgery_encoder.py    │ — untouched       │
│ anchors: vanilla cont-SSL (m09a1) / 2× / frozen          │ m09a1 / — (have)            │ —                 │
└──────────────────────────────────────────────────────────┴─────────────────────────────┴───────────────────┘
```
> **NET code:** 4 baseline scripts, each a FULL copy of m09c1 + its labelled delta (REDUNDANCY ACCEPTED —
> never miss an already-built function), + surgery_raw as config-only on m09c1. The surgery novelty
> (m09c1) stays untouched. Shared helpers already live in utils/training.py; further dedup across the 4
> copies is the SEPARATE post-copy factor phase (#19). New deps: `peft>=0.12` (B1). Eval is FREE (§0.2).
