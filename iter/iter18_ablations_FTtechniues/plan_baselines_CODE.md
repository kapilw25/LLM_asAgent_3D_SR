# iter18 · plan_baselines_CODE.md — DETAILED code plan per FT (Fine-Tuning) baseline

> Companion to `plan_baselines_roadmap.md`. **Every abbrev is written `abbrev (FULL FORM)` at every mention.**
> Reference modules re-read for this plan: `scripts/run_train.sh`, `src/m09a1_pretrain_encoder.py`,
> `src/m09a2_pretrain_head.py`, `src/m09c1_surgery_encoder.py`, `src/m09c2_surgery_head.py`.

## 📖 Full forms used below (read them again)

```text
┌ abbrev ──┬ FULL FORM ──────────────────────────────────────────────────────────┐
│ Auto-RGN │ Automatic Relative Gradient Norm   (Surgical-FT, Lee et al. ICLR'23) │
│ EWC      │ Elastic Weight Consolidation       (Kirkpatrick et al. PNAS'17)      │
│ LoRA     │ Low-Rank Adaptation                (Hu et al. 2021)                  │
│ DoRA     │ Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024)              │
│ PEFT     │ Parameter-Efficient Fine-Tuning                                      │
│ LP-FT    │ Linear-Probing then Fine-Tuning    (Kumar et al. ICLR'22)            │
│ Full-FT  │ Full Fine-Tuning                                                     │
│ CaSSLe   │ continual self-supervised distillation (stylized; Fini et al. CVPR'22)│
│ SSL      │ Self-Supervised Learning                                             │
│ SPD      │ Selective Projection Decay         (Tian et al. NeurIPS'24)          │
│ SALT     │ Self-Anchored Latent Teacher       (Apple 2025)                      │
└──────────┴──────────────────────────────────────────────────────────────────────┘
```
> 🔒 **NAMING RULE (MANDATORY, enforced in review):** in every new `*.py` / `*.sh` / `*.yaml` / log
> string / docstring, write the abbrev as `abbrev (FULL FORM)` at EVERY mention — never the bare
> abbrev. Example config comment: `# freeze_rule: auto_rgn  → Auto-RGN (Automatic Relative Gradient
> Norm), Lee et al. ICLR'23`. Example log: `print("[Auto-RGN (Automatic Relative Gradient Norm)] kept blocks", keep)`.

---

## 0 · The CONTRACT every new trainer must honor (so eval is FREE)

Every m09* trainer in this repo emits the SAME two artifacts; honor them and `run_eval.sh` runs the
9-metric suite with zero new eval code (verified in `m09a2`/`m09c2` `finalize_outputs(...)`):

```text
┌ artifact ─────────────────┬ schema / why ───────────────────────────────────────────────────────┐
│ student_encoder.pt        │ key "student_state_dict" — encoder only. m12a/b/c (action/motion/    │
│                           │ taxonomy) load this.                                                 │
│ m09X_ckpt_best.pt         │ keys "student" + "predictor" — Stage 8 future_mse (m12d) + Stage 8b  │
│                           │ predictor-temporal (m12e) REQUIRE the "predictor" key. No predictor  │
│                           │ key → m12d/m12e FATAL for that arm (run_eval.sh:442-484 preflight).  │
│ configs/eval/             │ one row per arm {kind: vjepa, arch, crop, embed_dim} → run_eval.sh   │
│  probe_encoders.yaml      │ ENCODERS=<arm> → m12a–m12e → 9 metrics + paired BCa 95% CI.          │
└───────────────────────────┴──────────────────────────────────────────────────────────────────────┘
```
`finalize_outputs(student, output_dir, ckpt_prefix, ckpt_payload={...}, summary={...})` writes BOTH
files — reuse it verbatim (it is the single export path for m09a1/a2/c1/c2).

---

## 1 · Shared scaffolding — REUSE, do NOT rebuild (CLAUDE.md #49 isolation, but shared primitives)

Every technique below is a *small delta*; it reuses these `utils.training` primitives exactly as
m09a1/a2/c1/c2 do (confirmed in the re-read):

```text
┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
│ primitive                            │ role                                                           │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ build_student_predictor(mcfg, dcfg)  │ (student ViT, predictor) — identical kwargs across m09a/c      │
│ build_optimizer(student, predictor,  │ param groups; pass init_params=None (or SPD (Selective         │
│   opt_cfg, init_params=None)         │ Projection Decay) anchor for surgery / EWC (Elastic            │
│                                      │ Weight Consolidation))                                         │
│ build_scheduler(opt, opt_cfg, steps) │ single front-loaded warmup (capped 10%)                        │
│ producer_thread(cfg,q,…)             │ RAW-clip CPU decode → GPU (m09a1/a2 path)                      │
│ StreamingFactorDataset + _build_     │ on-the-fly D_L/D_A/D_I FACTOR clips (m09c1/c2 path)            │
│   factor_loader(...)                 │                                                                │
│ run_motion_aux_step(student, ma_head │ motion_aux head CE+MSE (head cells; also aux on encoder cells) │
│   , …)                               │                                                                │
│ assert_encoder_frozen / set_         │ freeze guards (head cells + Auto-RGN (Automatic                │
│   trainable_prefix(student, n)       │ Relative Gradient Norm) block selection)                       │
│ TrainLogWriter / render_val_plots /  │ crash-safe loss_log + 5 per-val plots + in-train probe-trio    │
│   run_trio_at_val                    │                                                                │
│ finalize_outputs(...)                │ export student_encoder.pt + m09X_ckpt_best.pt (the contract)   │
│ AdaptiveBatchSizer + cuda_cleanup    │ OOM safety (24 GB SANITY / 96 GB FULL)                         │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────┘
```

---

## 2 · Per-technique code plan — in BUILD ORDER (§ 0.5 of plan_baselines_roadmap.md)

### 🚩 WAVE 1 · B2 · Auto-RGN (Automatic Relative Gradient Norm) — the KILL-SHOT

```text
┌ field ──────────┬ plan ────────────────────────────────────────────────────────────────────────┐
│ WHERE           │ a freeze_rule BRANCH inside src/m09c1_surgery_encoder.py — NOT a new module    │
│ NEW module?     │ NO. Auto-RGN (Automatic Relative Gradient Norm) = surgery minus factors with   │
│                 │ gradient-picked blocks → m09c1 already has the staged-unfreeze loop + RAW path │
│ contract-clean? │ YES. freeze_rule is a PARAMETER {depth_fraction | auto_rgn}, not an            │
│                 │ `if technique==` branch (honors utils/training.py technique-agnostic contract) │
│ data            │ RAW clips: factor_streaming=false ; EMA teacher ; NO SALT/SPD/saliency/replay  │
│ budget-match    │ k = round(depth × surgery_trainable_frac) → SAME trainable-param count as       │
│                 │ surgery (else the namesake comparison is attackable)                          │
└─────────────────┴──────────────────────────────────────────────────────────────────────────────┘
```

The ~15-line selector (plugs into m09c1's per-stage block-unfreeze setup, replacing the
`unfreeze_below` depth-fraction when `freeze_rule == "auto_rgn"`):

```python
# src/m09c1_surgery_encoder.py — Auto-RGN (Automatic Relative Gradient Norm) block selection
def select_blocks_auto_rgn(model, batch, k):                       # Lee et al. ICLR'23
    model.zero_grad(set_to_none=True)
    loss = jepa_forward_loss(model, batch); loss.backward()        # ONE warmup pass, no optim step
    rgn = {b: blk_grad_norm(blk) / (blk_param_norm(blk) + 1e-8)    # RGN = Relative Gradient Norm
           for b, blk in enumerate(model.blocks)}                  #     = ||grad(theta_b)|| / ||theta_b||
    keep = sorted(rgn, key=rgn.get, reverse=True)[:k]              # unfreeze the top-k "loudest" blocks
    for b, blk in enumerate(model.blocks):
        for p in blk.parameters(): p.requires_grad = (b in keep)
    model.zero_grad(set_to_none=True)
    return sorted(keep)
```

```text
CONFIG   configs/train/surgical_autorgn.yaml  (clone surgery_3stage_DI_encoder.yaml; set
         freeze_rule: auto_rgn  # Auto-RGN (Automatic Relative Gradient Norm), Lee et al. ICLR'23
         factor_streaming.{sanity,poc,full}: false   # RAW clips, no factor curriculum
         surgery.teacher_mode: ema ; optimization.spd.enabled: false ; loss.saliency_weighting: false
         replay.raw_pretrain_pct: 0.0)
RUN      run_train.sh: add SUBCMD `surgical_autorgn` → dispatch m09c1 with surgical_autorgn.yaml.
         BACKBONE=vjepa_2_1_vitg ./scripts/run_train.sh surgical_autorgn --POC   (vitg 1B kill-shot)
REGISTRY probe_encoders.yaml: vjepa_2_1_vitg_surgical_autorgn {kind: vjepa, arch: vit_giant_xformers_2_1}
EVAL     ENCODERS=vjepa_2_1_vitg_surgical_autorgn ./scripts/run_eval.sh --POC  → 9 metrics, FREE
VERIFY   3-check (py_compile+ast+ruff F,E9) → smallest SANITY of m09c1 freeze_rule=auto_rgn on Pro 4000
COMPARE  surgery > Auto-RGN (Automatic Relative Gradient Norm)? AND surgery > pretrain_2X (RAW control)?
         → run on vitg 1B / vitG 2B, NEVER vJEPA 2.0 (false-kill, surgery loses there 0/5/4).
```

### 🚩 WAVE 1 · B4 · Full-FT (Full Fine-Tuning) / LP-FT (Linear-Probing then Fine-Tuning) — config-only

```text
NO new code.
Full-FT (Full Fine-Tuning)        configs/train/full_ft.yaml → m09a1_pretrain_encoder.py with
                                  layer_freeze.freeze_below: 0  (unfreeze_below=1.0 = all 40/48 blocks),
                                  RAW clips. Expected to FORGET temporal = the cautionary ceiling.
LP-FT (Linear-Probing then        configs/train/lpft.yaml → m09c1_surgery_encoder.py with
  Fine-Tuning)                    surgery.lp_ft_stage0.enabled: true, factor_streaming: false (RAW),
                                  NO factor curriculum. = surgery minus the factors.
RUN      run_train.sh SUBCMDs `full_ft` (→ m09a1) and `lpft` (→ m09c1) with those yamls.
REGISTRY vjepa_2_1_vitg_{full_ft, lpft} rows in probe_encoders.yaml.
```

### WAVE 2 · B1 · PEFT (Parameter-Efficient Fine-Tuning): LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed Low-Rank Adaptation)

```text
┌─────────────┬───────────────────────────────────────────────────────────────────────────────┐
│ field       │ plan                                                                          │
├─────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ WHERE       │ revive src/legacy/m09b_explora.py → src/m09b_peft.py (mv out of legacy)       │
│ NEW module? │ REVIVED (m09b already existed as ExPLoRA = LoRA (Low-Rank Adaptation)         │
│             │ rank-16). Its own loop per #49.                                               │
│ loop        │ mirrors m09a1 continual SSL (Self-Supervised Learning) on RAW, but trains     │
│             │ ONLY the LoRA (Low-Rank Adaptation) adapters on attn.qkv + mlp.fc1/fc2 (r=16) │
│ DoRA delta  │ DoRA (Weight-Decomposed Low-Rank Adaptation): decompose W0 into magnitude     │
│             │ m + direction; train m (per-output-dim) + the LoRA (Low-Rank Adaptation)      │
│             │ direction ΔV. ~0.3% params.                                                   │
│ export      │ MERGE adapters into base weights (finalize_outputs explora_enabled=True       │
│             │ path) → eval loads a PLAIN ViT student_encoder.pt (no PEFT (Parameter-        │
│             │ Efficient Fine-Tuning) dep at eval time)                                      │
└─────────────┴───────────────────────────────────────────────────────────────────────────────┘
```

```python
# src/m09b_peft.py — DoRA (Weight-Decomposed Low-Rank Adaptation) forward (custom, if not peft use_dora)
# W_dir = W0 + (B @ A) * scaling
# W     = m * W_dir / W_dir.norm(dim=0, keepdim=True)        # trainable: m (per-out-dim), A, B
# → peft>=0.12 LoraConfig(use_dora=True) is the gold-standard path; the 8-line wrapper is the fallback.
```

```text
CONFIG   configs/train/peft_lora.yaml , peft_dora.yaml   (clone configs/legacy2/explora.yaml)
RUN      run_train.sh SUBCMDs `peft_lora` / `peft_dora` → m09b_peft dispatch.
REGISTRY vjepa_2_1_vitg_{peft_lora, peft_dora} rows.
```

### WAVE 2 · B3 · Continual-SSL: CaSSLe + EWC (Elastic Weight Consolidation)

```text
┌ field ──────────┬ plan ────────────────────────────────────────────────────────────────────────┐
│ WHERE           │ NEW module src/m09d_contssl.py — sibling of m09a1 (continual SSL on RAW)        │
│ NEW module?     │ YES (own loop, #49 isolation) — keeps m09a1 clean; adds 2 config-gated losses  │
│ CaSSLe          │ L_cassle = jepa_loss( predictor_g(z_student), stop_grad(teacher_frozen_feat) ) │
│                 │ → REUSE the FROZEN teacher already held in the recipe (SALT (Self-Anchored      │
│                 │ Latent Teacher) slot) + a small projector g (2-layer MLP) + weight λ_cassle    │
│ EWC (Elastic    │ F_i = E[(∂L/∂θ_i)²] (one-epoch diagonal Fisher) ;                              │
│  Weight         │ L_ewc = λ_ewc · Σ_i F_i (θ_i − θ*_i)²   (θ* = pretrained init)                 │
│  Consolidation) │ → REUSE the SPD (Selective Projection Decay) anchor SLOT, Fisher-weight it      │
│ data            │ RAW clips (continual SSL on the new domain; no factors)                        │
└─────────────────┴──────────────────────────────────────────────────────────────────────────────┘
```

```text
CONFIG   configs/train/cassle.yaml (λ_cassle on, EWC off) , ewc.yaml (EWC (Elastic Weight
         Consolidation) Fisher reg on, CaSSLe off) — both clone pretrain_encoder.yaml.
RUN      run_train.sh SUBCMDs `cassle` / `ewc` → m09d_contssl dispatch.
REGISTRY vjepa_2_1_vitg_{cassle, ewc} rows.
```

### RAW control · surgery_raw (factor OFF) — disentangles BLOCKS vs DATA

```text
NO new code. configs/train/surgery_3stage_DI_encoder.yaml with factor_streaming: false → m09c1
trains the STRUCTURED 4/8/8 blocks on RAW clips. Lets you attribute an Auto-RGN (Automatic Relative
Gradient Norm) win to the factor DATA (surgery vs surgery_raw) vs the structured BLOCKS
(surgery_raw vs Auto-RGN (Automatic Relative Gradient Norm)). Pairs with pretrain_2X for the full
RAW-vs-FACTOR control (§3 of plan_baselines_roadmap.md).
```

---

## 3 · run_train.sh wiring (new SUBCMDs) — mirror the existing `case "$SUBCMD"` dispatch (lines 263-487)

```text
┌ new SUBCMD ──────────┬ dispatches → ─────────────────┬ train-config ─────────────────────────────┐
│ surgical_autorgn     │ m09c1_surgery_encoder.py      │ surgical_autorgn.yaml (freeze_rule=auto_rgn)│
│ full_ft              │ m09a1_pretrain_encoder.py     │ full_ft.yaml (unfreeze all)                │
│ lpft                 │ m09c1_surgery_encoder.py      │ lpft.yaml (lp_ft_stage0 on, factor off)    │
│ peft_lora / peft_dora│ m09b_peft.py                  │ peft_lora.yaml / peft_dora.yaml            │
│ cassle / ewc         │ m09d_contssl.py               │ cassle.yaml / ewc.yaml                     │
└──────────────────────┴───────────────────────────────┴────────────────────────────────────────────┘
```
Each reuses the SAME leakage-safe `--subset $TRAIN_POOL`, `--val-subset $VAL_SPLIT`, `--init-from-ckpt
$SURGERY_INIT` (for c1/c2/b/d arms), per-backbone `outputs/<mode>/<backbone>/<arm>/` namespace, and the
`CACHE_POLICY_ALL` / `--no-wandb` plumbing already in run_train.sh. **Also add each arm to**
`scripts/iter17_poc_ngpu.py` `ARM2ENC` + `ARM2DIR` so the N-GPU scheduler fans them out.

---

## 4 · run_eval.sh / registry — each arm → 9 metrics, ZERO new eval code

```text
1. add the row to configs/eval/probe_encoders.yaml (encoder_ckpt_for() resolves student_encoder.pt;
   encoder_predictor_ckpt_for() resolves m09X_ckpt_best.pt for Stage 8/8b).
2. ENCODERS=<arm> ./scripts/run_eval.sh --POC  → m12a (action_top1) · m12b (motion_cos) · m12c
   (taxonomy_f1) · m12d (future_mse) · m12e (predictor-temporal ×6) = the 9-metric suite + paired BCa 95% CI.
3. §G aggregate (m13_eval_plot) renders the hero grid with surgery vs every baseline.
```

---

## 5 · Verification per arm (NEVER skip) + POC↔FULL parity

```text
┌────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│ gate                           │ command                                                            │
├────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ 3-check (after every src edit) │ py_compile + ast.parse + ruff check --select F,E9 (post-edit hook) │
│ smallest SANITY (per arm)      │ BACKBONE=vjepa_2_1_vitg run_train.sh <arm> --SANITY on Pro 4000    │
│                                │ — catches FAIL-LOUD asserts / CLI wiring / dtype before POC spend  │
│ POC↔FULL parity                │ ONLY n_clips + max_epochs differ; every other yaml/CLI flag byte-  │
│                                │ identical (CLAUDE.md). No "disable feature X at POC".              │
│ budget-match (Auto-RGN =       │ assert trainable-param count == surgery's, logged at startup       │
│ Automatic Relative Gradient    │                                                                    │
│ Norm — only)                   │                                                                    │
└────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

---

## 6 · Module map summary

```text
┌ technique ───────────────────────────────────────────────┬ module ───────────────┬ new? ─────────┐
│ B2 Auto-RGN (Automatic Relative Gradient Norm)           │ m09c1 freeze_rule branch│ delta (~15 ln)│
│ B4 Full-FT (Full Fine-Tuning)                            │ m09a1 + config         │ config-only   │
│ B4 LP-FT (Linear-Probing then Fine-Tuning)               │ m09c1 + config         │ config-only   │
│ B1 LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed  │ m09b_peft.py           │ REVIVE legacy │
│    Low-Rank Adaptation)                                  │                        │               │
│ B3 CaSSLe + EWC (Elastic Weight Consolidation)           │ m09d_contssl.py        │ NEW (sibling) │
│ RAW control surgery_raw                                  │ m09c1 + config         │ config-only   │
│ surgery (OURS)                                           │ m09c1 (have)           │ —             │
│ pretrain / pretrain_2X / frozen (anchors)                │ m09a1 / — (have)       │ —             │
└──────────────────────────────────────────────────────────┴────────────────────────┴───────────────┘
Net NEW code: 1 revived module (m09b_peft.py) + 1 new module (m09d_contssl.py) + 1 ~15-line branch
in m09c1 (Auto-RGN (Automatic Relative Gradient Norm)) + 6 config files. NO new training loop beyond m09b/m09d.
```
