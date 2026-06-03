# iter18 · plan_baselines_CODE.md — DETAILED code plan per FT (Fine-Tuning) baseline

> Companion to `plan_baselines_roadmap.md`. **Every abbrev is written `abbrev (FULL FORM)` at every mention.**
> Reference modules RE-READ for this plan (line numbers below are from these): `scripts/run_train.sh`
> (SUBCMD dispatch 263-488), `src/m09a1_pretrain_encoder.py`, `src/m09c1_surgery_encoder.py`,
> `src/utils/training.py` (shared primitives), `configs/train/{base_optimization,surgery_base,
> pretrain_encoder,surgery_3stage_DI_encoder}.yaml`.
> Official-code repos that ground every snippet → the `§ 2` "Official code" table in
> `plan_baselines_roadmap.md` (each repo's core file was READ, June 2026).

## 📖 Full forms used below (read them again)

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
> 🔒 **NAMING RULE (MANDATORY, enforced in review):** in every new `*.py` / `*.sh` / `*.yaml` / log
> string / docstring, write the abbrev as `abbrev (FULL FORM)` at EVERY mention — never the bare
> abbrev. Example config comment: `# freeze_rule: auto_rgn  → Auto-RGN (Automatic Relative Gradient
> Norm), Lee et al. ICLR'23`. Example log: `print("[Auto-RGN (Automatic Relative Gradient Norm)] kept blocks", keep)`.

---

## 0 · The CONTRACT every new trainer must honor (so eval is FREE)

Every m09* trainer in this repo emits the SAME two artifacts; honor them and `run_eval.sh` runs the
9-metric suite with zero new eval code. The export helpers (REAL names, verified in `src/utils/training.py`):

```text
┌ helper (src/utils/training.py) ──┬ writes ──────────────────────────────────────────────────────────┐
│ export_student_for_eval(student, │ student_encoder.pt (key "student_state_dict", encoder only).     │
│   path, explora_enabled=False)   │ explora_enabled=True FIRST merges LoRA/DoRA adapters into the    │
│   :1372                          │ base → a PLAIN ViT (the PEFT export path). m09a1 calls it :1339. │
│ finalize_outputs(*, student,     │ BOTH student_encoder.pt + <ckpt_prefix>_ckpt_best.pt (keys       │
│   output_dir, ckpt_prefix, …)    │ "student"+"predictor"). The single keyword-only export path —    │
│   :243                           │ m09a2 / m09c2 use it verbatim; reuse it in m09b/m09d.            │
└──────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```
```text
┌ artifact ────────────┬ schema / why ───────────────────────────────────────────────────────┐
│ student_encoder.pt   │ key "student_state_dict" — encoder only. m12a/b/c (action/motion/   │
│                      │ taxonomy) load this.                                                │
│ m09X_ckpt_best.pt    │ keys "student" + "predictor" — Stage 8 future_mse (m12d) + Stage 8b │
│                      │ predictor-temporal (m12e) REQUIRE the "predictor" key. No predictor │
│                      │ key → m12d/m12e FATAL for that arm (run_eval.sh:442-484 preflight). │
│ configs/eval/        │ one row per arm {kind: vjepa, arch, crop, embed_dim} → run_eval.sh  │
│  probe_encoders.yaml │ ENCODERS=<arm> → m12a–m12e → 9 metrics + paired BCa 95% CI.         │
└──────────────────────┴─────────────────────────────────────────────────────────────────────┘
```
> ⚠ PEFT (Parameter-Efficient Fine-Tuning) arms must STILL emit the `predictor` key (m12d/m12e). After
> `merge_and_unload()`, pass the (unchanged) JEPA predictor through `finalize_outputs(... ckpt_prefix=…)`
> so `m09b_ckpt_best.pt` carries both keys — else future_mse / predictor-temporal FATAL for that arm.

---

## 1 · Shared scaffolding — REUSE, do NOT rebuild (CLAUDE.md #49 isolation, but shared primitives)

Every technique below is a *small delta*; it reuses these `src/utils/training.py` primitives exactly as
m09a1/a2/c1/c2 do (line numbers verified in the re-read):

```text
┌ primitive (src/utils/training.py) ───┬ role ────────────────────────────────────────────────────────────┐
│ build_student_predictor(mcfg, dcfg)  │ :216 → (student ViT, predictor) — identical kwargs across m09a/c │
│ build_optimizer(student, predictor,  │ :833 → param groups. init_params=θ* anchors θ→θ* (SPD =          │
│   cfg_opt, init_params=None)         │ Selective Projection Decay slot); Fisher-weight it → EWC         │
│                                      │ (Elastic Weight Consolidation)                                   │
│ set_trainable_prefix(student, n)     │ :1626 → unfreeze a CONTIGUOUS prefix of n blocks (surgery's      │
│                                      │ per-stage unfreeze_below). Auto-RGN REPLACES this selection.     │
│ build_scheduler(opt, opt_cfg, steps) │ single front-loaded warmup (capped 10%)                          │
│ producer_thread(cfg,q,…)             │ RAW-clip CPU decode → GPU (m09a1/a2 path)                        │
│ StreamingFactorDataset + _build_     │ on-the-fly D_L/D_A/D_I FACTOR clips (m09c1/c2 path)              │
│   factor_loader(...)                 │                                                                  │
│ run_motion_aux_step(student, ma_head │ motion_aux head CE+MSE (head cells; also aux on encoder cells)   │
│   , …)                               │                                                                  │
│ assert_encoder_frozen(student) :1389 │ freeze guard (head cells + Auto-RGN sanity)                      │
│ export_student_for_eval :1372 /      │ the export CONTRACT (§ 0) — student_encoder.pt + ckpt_best.pt    │
│   finalize_outputs :243              │                                                                  │
│ AdaptiveBatchSizer + cuda_cleanup    │ OOM safety (24 GB SANITY / 96 GB FULL)                           │
└──────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```
> m09a1 freeze knob = `layer_freeze.{enabled, freeze_below}` (m09a1:280-313): freeze blocks `[0, freeze_below)`,
> train `[freeze_below, n_blocks)`. pretrain_encoder.yaml = `freeze_below: 20` (train 20-48). m09c1 freeze
> knob = per-stage `surgery.stages[i].unfreeze_below` (depth fraction) → `set_trainable_prefix`.

---

## 2 · Per-technique code plan — in BUILD ORDER (§ 0.5 of plan_baselines_roadmap.md)

> Each snippet is grounded in the official repo READ for it (links → roadmap § 2 table). Where our
> setup forces a deviation from the original, it is labelled **ADAPTATION** and must be stated as such
> in the paper (CLAUDE.md NO-LAZY / FAIL-LOUD: never present an adaptation as the original method).

### 🚩 WAVE 1 · B2 · Auto-RGN (Automatic Relative Gradient Norm) — the KILL-SHOT

```text
WHERE            a freeze_rule BRANCH inside src/m09c1_surgery_encoder.py — NOT a new module.
NEW module?      NO. Auto-RGN = surgery minus factors with gradient-picked blocks; m09c1 already has
                 the per-stage block-unfreeze loop + the RAW path.
contract-clean?  YES. freeze_rule is a PARAMETER {depth_fraction (current) | auto_rgn}, resolved from
                 yaml + a new --freeze-rule CLI arg (required=True, argparse choices=[...]) — NOT an
                 `if technique==` branch (honors utils/training.py technique-agnostic contract #49).
data             RAW clips: factor_streaming=false ; EMA teacher ; NO SALT/SPD/saliency/replay.
budget-match     k = round(n_blocks × surgery_deepest_unfreeze_below) = round(48 × 0.167) = 8 blocks →
                 EXACTLY surgery's trainable-param count (else the namesake comparison is attackable).
single-shot      Auto-RGN picks ONCE on a warmup batch (single trainable phase, NOT 3 stages).
```

**(a) OFFICIAL RGN scorer — faithful to `anniesch/surgical-finetuning` `main.py`**
(`get_lr_weights` / `get_grad_norms`): per-PARAMETER-TENSOR, L2/Frobenius, averaged over the first
5 batches, recomputed each epoch, norm/LayerNorm tensors excluded, gradients from `autograd.grad`
with NO optimizer step.

```python
# src/m09c1_surgery_encoder.py — RGN (Relative Gradient Norm) scoring (faithful)
def rgn_scores(student, predictor, batches5, jepa_loss_fn):        # repo: itertools.islice(loader, 5)
    acc = {}                                                       # name -> [per-batch RGN]
    for x in batches5:
        student.zero_grad(set_to_none=True)
        loss  = jepa_loss_fn(student, predictor, x)                # repo uses F.cross_entropy; SSL → JEPA L1
        names = [n for n, _ in student.named_parameters()]
        grads = torch.autograd.grad(loss, [p for _, p in student.named_parameters()],
                                    retain_graph=False, allow_unused=True)   # NO opt.step() during scoring
        for (n, p), g in zip(student.named_parameters(), grads):
            if g is None or "norm" in n.lower(): continue          # repo skips "bn"/LayerNorm tensors
            acc.setdefault(n, []).append((g.norm() / (p.norm() + 1e-12)).item())   # RGN = ||g||₂ / ||θ||₂
    return {n: sum(v) / len(v) for n, v in acc.items()}            # repo: mean over the 5 batches
```

**(b) OFFICIAL selection = SOFT per-tensor LR — we DELIBERATELY do not use it.** The repo keeps EVERY
tensor trainable and scales its LR: `lr_tensor = (RGN / max RGN) · base_lr` (tiny RGN → lr≈0 = soft
freeze). That cannot hit a FIXED trainable-param budget, which the namesake comparison vs surgery
REQUIRES. So we replace it with the budget-matched hard top-k below.

**(c) OUR budget-matched ADAPTATION — hard top-k BLOCKS (state as an adaptation in the paper).**
Aggregate the per-tensor RGN to a per-block mean and unfreeze the top-k blocks via `requires_grad`,
so the trainable-param count EXACTLY equals surgery's deepest stage (k=8 of 48).

```python
# src/m09c1_surgery_encoder.py — Auto-RGN block selection (budget-matched ADAPTATION of the soft-LR original)
def select_blocks_auto_rgn(student, predictor, batches5, jepa_loss_fn, k):
    rgn = rgn_scores(student, predictor, batches5, jepa_loss_fn)          # faithful RGN (a)
    per_block = {}                                                        # block idx -> [tensor RGNs]
    for name, score in rgn.items():
        if name.startswith("blocks."):
            b = int(name.split(".")[1]); per_block.setdefault(b, []).append(score)
    block_rgn = {b: sum(v) / len(v) for b, v in per_block.items()}        # per-block mean (repo-consistent reduction)
    keep = set(sorted(block_rgn, key=block_rgn.get, reverse=True)[:k])    # top-k loudest blocks
    for b, blk in enumerate(student.blocks):
        for p in blk.parameters(): p.requires_grad = (b in keep)          # hard freeze (vs set_trainable_prefix)
    student.zero_grad(set_to_none=True)
    print(f"[Auto-RGN (Automatic Relative Gradient Norm)] kept {sorted(keep)} (k={k}, budget-matched)")
    return sorted(keep)
# WIRE: in m09c1 train(), when freeze_rule=="auto_rgn" → call this ONCE before the (single) trainable
# phase, INSTEAD of the per-stage set_trainable_prefix(student, n) depth-fraction path.
```

```text
CONFIG   configs/train/surgical_autorgn.yaml  (clone surgery_3stage_DI_encoder.yaml; set
         freeze_rule: auto_rgn        # Auto-RGN (Automatic Relative Gradient Norm), Lee et al. ICLR'23
         auto_rgn: {k_blocks: 8, n_score_batches: 5}   # budget-match k=8/48 ; 5-batch RGN avg (repo)
         factor_streaming.{sanity,poc,full}: false     # RAW clips, no factor curriculum
         surgery.teacher_mode: ema ; optimization.spd.enabled: false ; optimization.loss.saliency_weighting: false
         replay.raw_pretrain_pct: 0.0
         surgery.stages: [ single stage, unfreeze_below ignored when freeze_rule==auto_rgn ])
CLI      m09c1 add --freeze-rule {depth_fraction,auto_rgn} (required=True, argparse choices) →
         cfg["surgery"]["freeze_rule"]. run_train.sh resolves it from yaml via _y (single source).
RUN      BACKBONE=vjepa_2_1_vitg ./scripts/run_train.sh surgical_autorgn --POC   (vitg 1B kill-shot)
REGISTRY probe_encoders.yaml: vjepa_2_1_vitg_surgical_autorgn {kind: vjepa, arch: vit_giant_xformers_2_1}
EVAL     ENCODERS=vjepa_2_1_vitg_surgical_autorgn ./scripts/run_eval.sh --POC  → 9 metrics, FREE
VERIFY   3-check → smallest SANITY of m09c1 --freeze-rule auto_rgn on Pro 5000 (assert len(keep)==k)
COMPARE  surgery > Auto-RGN? AND surgery > vanilla continual SSL 2× (pretrain_2X, RAW control)?
         → run on vitg 1B / vitG 2B, NEVER vJEPA 2.0 (false-kill, surgery loses there 0/5/4).
```

### 🚩 WAVE 1 · B4 · Full-FT (Full Fine-Tuning) / LP-FT (Linear-Probing then Fine-Tuning) — config-only

```text
NO new code. Both reuse existing trainers + freeze knobs.

B4(a) Full-FT (Full Fine-Tuning)   configs/train/full_ft.yaml → m09a1_pretrain_encoder.py with
                                   layer_freeze.freeze_below: 0   (= all 48 blocks trainable; m09a1
                                   :289-307 trains [0,n_blocks)), RAW clips, ONE base_lr for every
                                   block. Standard end-to-end FT — the "forgetting ceiling" expected to
                                   distort temporal features (Kumar ICLR'22: full-FT underperforms OOD).
B4(b) LP-FT (Linear-Probing then   configs/train/lpft.yaml → m09c1_surgery_encoder.py with
  Fine-Tuning, Kumar ICLR'22)      surgery.lp_ft_stage0.enabled: true (m09c1:685-701 prepends a head-
                                   only warmup stage, unfreeze_below=0.0) THEN unfreeze; factor_
                                   streaming:false (RAW); NO factor curriculum → = surgery minus factors.
```
> **CRITICAL LP-FT detail (Kumar et al. ICLR'22, AnanyaKumar/transfer_learning `run_adaptation_experiments.py`):**
> phase-2 (encoder unfreeze) LR must be FAR LOWER than phase-1 (head probe) — the paper warms the head
> at LR ≈ 1e-1 then fine-tunes the backbone at ≈ 1e-5…1e-4 (≈10³–10⁴× lower) so the already-accurate head
> does not distort pretrained features. So `lpft.yaml`: stage0 `head_lr_multiplier` HIGH for the probe,
> encoder stages `optimization.base_lr` LOW (≪ the probe LR). Full-FT keeps ONE (un-warmed) LR for all
> blocks → that LR/distortion contrast IS the LP-FT-vs-Full-FT story.

```text
RUN      run_train.sh SUBCMDs `full_ft` (→ m09a1) and `lpft` (→ m09c1) with those yamls.
REGISTRY vjepa_2_1_vitg_{full_ft, lpft} rows in probe_encoders.yaml.
```

### WAVE 2 · B1 · PEFT (Parameter-Efficient Fine-Tuning): LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed Low-Rank Adaptation)

```text
WHERE        revive src/legacy/m09b_explora.py → src/m09b_peft.py (mv out of legacy). Its own loop (#49).
NEW module?  REVIVED. m09b already injected LoRA rank-16 on attn.qkv + mlp.fc1/fc2 (ExPLoRA). Swap the
             hand-rolled injection for HuggingFace peft (gold standard) so DoRA is a one-flag delta.
loop         mirrors m09a1 continual SSL on RAW, but trains ONLY the adapters (+ DoRA magnitude vector).
export       MERGE adapters into base → eval loads a PLAIN ViT (no PEFT dep at eval).
```

```python
# src/m09b_peft.py — gold-standard PEFT via HuggingFace peft (microsoft/LoRA + NVlabs/DoRA mechanism)
from peft import LoraConfig, get_peft_model
TARGETS = ["qkv", "proj", "fc1", "fc2"]          # timm ViT: block.attn.{qkv,proj} + block.mlp.{fc1,fc2}

def wrap_peft(student, r=16, alpha=32, use_dora=False):                 # use_dora=True ⇒ DoRA, one flag
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=TARGETS,
                     lora_dropout=0.0, bias="none", use_dora=use_dora)  # scaling = alpha/r
    return get_peft_model(student, cfg)          # base auto-FROZEN; only adapters (+DoRA m) require grad

# LoRA forward (peft lora/layer.py):  y = W0·x + (B @ A)·x · (alpha/r)        # A kaiming, B zeros ⇒ Δ=0 at init
# DoRA forward (peft lora/dora.py · NVlabs dora.py):  W = m ⊙_out (W0 + B@A·s) / ||W0 + B@A·s||_{dim=1}
#   m  = per-OUTPUT-channel magnitude, init = ||W0||_{dim=1}  (DoraLinearLayer.weight / weight_m_wdecomp)
#   the direction-norm denominator is DETACHED (dora_simple) ⇒ grad flows only to m and to A,B.

# EXPORT — eval loads a PLAIN ViT, zero peft dep:
def export_peft(peft_model, predictor, output_dir):
    merged = peft_model.merge_and_unload()       # LoRA: W += B@A·s ;  DoRA: W = (m/||·||)·(W0+B@A·s)
    finalize_outputs(student=merged, output_dir=output_dir, ckpt_prefix="m09b",   # keeps predictor key
                     ckpt_payload={"student": merged.state_dict(), "predictor": predictor.state_dict()})
    # (or export_student_for_eval(merged, output_dir/"student_encoder.pt", explora_enabled=True) for the encoder-only file)
```

```text
DEP      add `peft>=0.12` to setup_env_uv.sh + requirements_gpu.txt (use_dora needs ≥0.12). Cite in docstring.
CONFIG   configs/train/peft_lora.yaml (use_dora:false) , peft_dora.yaml (use_dora:true)  (clone explora.yaml)
RUN      run_train.sh SUBCMDs `peft_lora` / `peft_dora` → m09b_peft dispatch.
REGISTRY vjepa_2_1_vitg_{peft_lora, peft_dora} rows.
NOTE     PEFT may LOOK competitive on action_top1 (capacity) but should LOSE on temporal/world-model
         metrics — that asymmetry IS the story (PEFT adapts features; surgery adapts dynamics).
```

### WAVE 2 · B3 · Continual-SSL: CaSSLe + EWC (Elastic Weight Consolidation)

```text
WHERE        NEW module src/m09d_contssl.py — sibling of m09a1 (continual SSL on RAW). Own loop (#49).
NEW module?  YES. Keeps m09a1 clean; adds 2 config-gated losses on top of the m09a1 JEPA loop.
data         RAW clips (continual SSL on the new domain; no factors).
```

**CaSSLe** (Fini CVPR'22 · `DonkeyShot21/cassle` `distillers/predictive.py`, `losses/byol.py`): a
predictor `g` maps the CURRENT feature to predict the FROZEN previous-model feature; distillation
REUSES the SSL loss; stop-grad on the frozen target. We already hold a FROZEN teacher (the SALT slot).

```python
# src/m09d_contssl.py — CaSSLe distillation (reuses the FROZEN SALT teacher already in the recipe)
g = nn.Sequential(nn.Linear(D, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, D))  # repo: D→2048→D BN-MLP
def cassle_loss(z_student, z_frozen_teacher):                 # z_frozen from the FROZEN teacher (no_grad)
    return compute_jepa_loss(g(z_student), z_frozen_teacher.detach())   # reuse SSL loss; sg on target
# L_total = L_jepa  +  λ_cassle · cassle_loss(z_student, sg(teacher_frozen_feat))    # repo λ ≈ 1.0
#   (g plays the SAME role as the JEPA predictor — you may clone the predictor head instead of a fresh MLP.)
```

**EWC** (Kirkpatrick'17 · `moskomule/ewc.pytorch` `utils.py` · `GMvandeVen/continual-learning`): a
diagonal Fisher `F_i` weights an anchor-to-init penalty. Our anchor slot already exists — SPD
(Selective Projection Decay) via `build_optimizer(init_params=θ*)` anchors θ→θ* with uniform weight 1;
EWC = that slot with per-parameter weight `F_i`.

```python
# src/m09d_contssl.py — diagonal Fisher (SSL task-loss replaces the classification NLL of the repo)
def estimate_fisher(student, predictor, subset_batches, jepa_loss_fn):     # N ≈ few-hundred clips, small bs
    F = {n: torch.zeros_like(p) for n, p in student.named_parameters() if p.requires_grad}
    for x in subset_batches:                                  # repo: model.eval(); 1 sample at a time
        student.zero_grad(set_to_none=True)
        jepa_loss_fn(student, predictor, x).backward()        # ∂L_jepa/∂θ  (NOT NLL — we have no labels)
        for n, p in student.named_parameters():
            if p.requires_grad and p.grad is not None:
                F[n] += p.grad.detach() ** 2 / len(subset_batches)         # F_i = mean over N of (∂L/∂θ_i)²
    return F
# L_ewc = λ_ewc · Σ_i  F_i · (θ_i − θ*_i)²        (θ* = pretrained init = the stored SPD anchor)
# WIRE: build_optimizer(student, predictor, cfg_opt, init_params={"theta_star": θ*, "fisher": F}) →
#       the SPD anchor term multiplies (θ−θ*)² by F_i instead of 1.0. ONE-time Fisher pass before training.
```

```text
CONFIG   configs/train/cassle.yaml (λ_cassle on, EWC off) , ewc.yaml (EWC Fisher reg on, CaSSLe off)
         — both clone pretrain_encoder.yaml (RAW continual SSL base). Keys: loss.cassle_lambda,
         optimization.ewc.{enabled, lambda, fisher_n_batches}.
RUN      run_train.sh SUBCMDs `cassle` / `ewc` → m09d_contssl dispatch.
REGISTRY vjepa_2_1_vitg_{cassle, ewc} rows.
```

### RAW control · surgery_raw (factor OFF) — disentangles BLOCKS vs DATA

```text
NO new code. configs/train/surgery_3stage_DI_encoder.yaml with factor_streaming: false → m09c1 trains
the STRUCTURED 4/8/8 blocks on RAW clips. Lets you attribute an Auto-RGN (Automatic Relative Gradient
Norm) win to the factor DATA (surgery vs surgery_raw) vs the structured BLOCKS (surgery_raw vs
Auto-RGN). Pairs with vanilla continual SSL 2× (pretrain_2X) for the full RAW-vs-FACTOR control
(§3 of plan_baselines_roadmap.md).
```

---

## 3 · run_train.sh wiring (new SUBCMDs) — mirror the existing `case "$SUBCMD"` dispatch (263-488)

The existing dispatch resolves a `TRAIN_CFG`, builds an `OUT_DIR=outputs/${mode_dir}/${BACKBONE}/<arm>`,
reads recipe knobs from the yaml via `_y() { yaml_extract.py "$TRAIN_CFG" "$1"; }`, then calls the
trainer with `--subset $TRAIN_POOL --val-subset $VAL_SPLIT --output-dir $OUT_DIR` (+ `--init-from-ckpt
$SURGERY_INIT` for c1-derived arms). Add these branches:

```text
┌ new SUBCMD ───────────┬ dispatches → ─────────────┬ train-config + key flag ──────────────────────┐
│ surgical_autorgn      │ m09c1_surgery_encoder.py  │ surgical_autorgn.yaml  --freeze-rule auto_rgn │
│ full_ft               │ m09a1_pretrain_encoder.py │ full_ft.yaml (layer_freeze.freeze_below=0)    │
│ lpft                  │ m09c1_surgery_encoder.py  │ lpft.yaml  --lp-ft-stage0 on  (factor off)    │
│ peft_lora / peft_dora │ m09b_peft.py              │ peft_lora.yaml / peft_dora.yaml               │
│ cassle / ewc          │ m09d_contssl.py           │ cassle.yaml / ewc.yaml                        │
│ surgery_raw           │ m09c1_surgery_encoder.py  │ surgery_3stage_DI_encoder.yaml (factor off)   │
└───────────────────────┴───────────────────────────┴───────────────────────────────────────────────┘
```
- `surgical_autorgn` / `lpft` / `surgery_raw` slot into the EXISTING `surgery_*` case (264-? → 324-407):
  add to its inner `case "$SUBCMD"` that maps SUBCMD→`TRAIN_CFG`+`VARIANT_TAG`, and pass the new
  `--freeze-rule $(_y surgery.freeze_rule)` in `RECIPE_V2_ARGS` (resolve from yaml, single source).
- `full_ft` mirrors the `pretrain_encoder` case (264-323) → m09a1, with its own `TRAIN_CFG`.
- `peft_lora` / `peft_dora` / `cassle` / `ewc` are NEW cases dispatching to the new modules, reusing the
  SAME `--subset $TRAIN_POOL --val-subset $VAL_SPLIT --init-from-ckpt $SURGERY_INIT --cache-policy`
  plumbing + `--no-wandb` + the `outputs/<mode>/<backbone>/<arm>/` namespace.
- **Also register each arm in `scripts/iter17_poc_ngpu.py` `ARM2ENC` + `ARM2DIR`** so the N-GPU scheduler fans them out.

---

## 4 · run_eval.sh / registry — each arm → 9 metrics, ZERO new eval code

```text
1. add the row to configs/eval/probe_encoders.yaml (encoder_ckpt_for() resolves student_encoder.pt;
   encoder_predictor_ckpt_for() resolves m09X_ckpt_best.pt for Stage 8/8b).
2. ENCODERS=<arm> ./scripts/run_eval.sh --POC  → m12a (action_top1) · m12b (motion_cos) · m12c
   (taxonomy_f1) · m12d (future_mse) · m12e (predictor-temporal ×6) = the 9-metric suite + paired BCa 95% CI.
3. §G aggregate (m13_eval_plot) renders the hero grid with surgery vs every baseline (the relabelled
   "vanilla continual SSL" anchor shows in every plot).
```

---

## 5 · Verification per arm (NEVER skip) + POC↔FULL parity

```text
┌────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
│ gate                           │ command                                                             │
├────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ 3-check (after every src edit) │ py_compile + ast.parse + ruff check --select F,E9 (post-edit hook)  │
│ smallest SANITY (per arm)      │ BACKBONE=vjepa_2_1_vitg run_train.sh <arm> --SANITY on Pro 5000     │
│                                │ — catches FAIL-LOUD asserts / CLI wiring / dtype before POC spend   │
│ Auto-RGN selector unit         │ assert len(select_blocks_auto_rgn(...))==k AND trainable-param      │
│                                │ count == surgery's (logged at startup) — else comparison attackable │
│ PEFT merge round-trip          │ assert merge_and_unload() output loads as a PLAIN ViT (no peft) +   │
│                                │ student_encoder.pt has NO lora_/dora_ keys                          │
│ POC↔FULL parity                │ ONLY n_clips + max_epochs differ; every other yaml/CLI flag byte-   │
│                                │ identical (CLAUDE.md). No "disable feature X at POC".               │
└────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘
```

---

## 6 · Module map summary

```text
┌ technique ──────────────────────────────────────────────┬ module ──────────────────┬ new? ──────────┐
│ B2 Auto-RGN (Automatic Relative Gradient Norm)          │ m09c1 freeze_rule branch │ delta (~25 ln) │
│ B4(a) Full-FT (Full Fine-Tuning)                        │ m09a1 + config           │ config-only    │
│ B4(b) LP-FT (Linear-Probing then Fine-Tuning)           │ m09c1 + config           │ config-only    │
│ B1 LoRA (Low-Rank Adaptation) → DoRA (Weight-Decomposed │ m09b_peft.py             │ REVIVE legacy  │
│    Low-Rank Adaptation)                                 │ (HF peft, use_dora flag) │ + peft dep     │
│ B3 CaSSLe + EWC (Elastic Weight Consolidation)          │ m09d_contssl.py          │ NEW (sibling)  │
│ RAW control surgery_raw                                 │ m09c1 + config           │ config-only    │
│ surgery (OURS)                                          │ m09c1 (have)             │ —              │
│ vanilla continual SSL (m09a1) / 2× / frozen (anchors)   │ m09a1 / — (have)         │ —              │
└─────────────────────────────────────────────────────────┴──────────────────────────┴────────────────┘
Net NEW code: 1 revived module (m09b_peft.py, + peft>=0.12 dep) + 1 new module (m09d_contssl.py) +
1 ~25-line freeze_rule branch in m09c1 (Auto-RGN scorer + budget-matched selector) + 6 config files +
1 new --freeze-rule CLI arg on m09c1. NO new training LOOP beyond m09b/m09d. Eval is FREE (the contract).
```
