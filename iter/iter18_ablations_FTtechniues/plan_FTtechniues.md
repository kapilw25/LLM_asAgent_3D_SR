# iter16 ablations — Continual-FT TECHNIQUE baselines: SAFE · SEEKR · SSIAT · SAPT

> Focused (2026-05-28) on the four continual-fine-tuning SOTA methods we benchmark our
> surgery/pretrain against — and the PET blocks they build on (Adapter, SSF, VPT, LoRA).
> JEPA model variants → `plan_model.md`. Temporal metrics → `plan_metrics_temporal.md`.
> The v15a recipe + execution-ops (former §1, §4–§12, §15) → `plan_code.md` + retired
> `legacy/plan_model_FTtechniues.md`.
> Source: iter/utils/teams_work/FactorJEPA-Alternatives_to_Vanilla_Continual_Finetuning.md ·
> motivation: CITA ARR-review ask #4 ("run SAFE/SEEKR/SSIAT/SAPT baselines") — now IN SCOPE.

═══════════════════════════════════════════════════════════════════════════════
§1 · Why these four (the gap they fill)
═══════════════════════════════════════════════════════════════════════════════

Our two anchors are vanilla continual FT: m09a1 (full-block partial-unfreeze SSL pretrain) and
m09c1 (staged-unfreeze factor surgery). Reviewers will ask: "vs the PET/continual-learning SOTA?"
None of our anchors cover **PET subspaces, shared-adapter reuse, selective distillation, or
input-conditioned routing**. SAFE/SEEKR/SSIAT/SAPT each add exactly one of those axes — so they
are the right "did you beat the obvious alternatives?" baselines.

═══════════════════════════════════════════════════════════════════════════════
§1.5 · Reviewer-pull RE-ASSESSMENT (websearch June 2026) — venue-specific, NOT AAAI-mandatory
═══════════════════════════════════════════════════════════════════════════════

> venue/CITA = these 4 are in scope ONLY because of CITA's ARR-review (ACL Rolling Review) ask #4
> ("run SAFE/SEEKR/SSIAT/SAPT") — NOT because general AAAI/A* reviewers demand them. All four target a
> DIFFERENT setting: class-incremental image classification (SAFE/SSIAT) or LLM continual learning
> (SAPT/SEEKR), all MULTI-session/task with a classification or language-model head. Our paper is a
> SINGLE-session continual-SSL adaptation of a VIDEO world-model (no classes, no sessions, JEPA
> predictor) → the obvious-competitor set is Full-FT / LoRA / Surgical-FT / CaSSLe+EWC, not these four.

```text
┌────────┬───────────────────────────────────────┬─────────────────────────────────────────┬──────────────────────────────────┐
│ method │ FULL FORM · venue · arXiv             │ built FOR (domain it targets)           │ re-assessed AAAI pull            │
├────────┼───────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────┤
│ SAFE   │ Slow-and-Fast Parameter-Efficient     │ class-incremental image classification  │ LOW · top of 4 (vision ViT;      │
│        │ tuning · NeurIPS'24 · 2411.02175      │ (ViT); needs SESSIONS + a class head    │ slow/fast ≈ our staged-unfreeze) │
│ SSIAT  │ Semantically-Shifted Incremental      │ class-incremental classification (ViT); │ LOW · vision ViT, cheapest       │
│        │ Adapter-Tuning · CVPR'24 · 2403.19979 │ PROTOTYPES + semantic-shift / SESSIONS  │ (lowest PET capacity)            │
│ SAPT   │ Shared Attention fwk for Parameter-   │ continual INSTRUCTION tuning of LLMs    │ LOWEST · LLM (wrong domain)      │
│        │ efficient CL · ACL'24 · 2401.08295    │ (T5/LLaMA); per-input routing / TASKS   │                                  │
│ SEEKR  │ Selective attEntion-guided Knowledge  │ continual learning of LLMs; attn-head   │ LOWEST · LLM (wrong domain)      │
│        │ Retention · EMNLP'24 · 2411.06171     │ selective KD + replay over a TASK seq   │                                  │
└────────┴───────────────────────────────────────┴─────────────────────────────────────────┴──────────────────────────────────┘
```
> Re-assessed pull order: **SAFE > SSIAT > SAPT ≈ SEEKR** (vision-ViT nearer to V-JEPA than LLM-CL). Build
> ONE only if a CL-leaning reviewer appears: SAFE (most pull) or SSIAT (cheapest); otherwise defer all four.

═══════════════════════════════════════════════════════════════════════════════
§2 · The four techniques — definition · mechanism · vs ours
═══════════════════════════════════════════════════════════════════════════════

```text
┌──────────┬────────────────────────────────────────────┬───────────────────────────────────────┐
│ method   │ core idea (what it adapts + how)            │ vs our m09a1 / m09c1                  │
├──────────┼────────────────────────────────────────────┼───────────────────────────────────────┤
│ 🐢⚡ SAFE │ Slow+Fast PET: two PET branches (Adapter/  │ NEW dual-pathway PET. Ours = full-     │
│          │ SSF/LoRA/VPT). "Slow" captures pretrained  │ block fine-tune. m09c1's staged        │
│          │ general knowledge then FREEZES after       │ unfreeze + EMA anchor mirrors the      │
│          │ session-1; "Fast" keeps updating. Transfer │ slow/fast idea at the BLOCK level, not │
│          │ loss aligns PET-model ↔ pretrained model.  │ via PET modules.                       │
│          │ → stability(slow) + plasticity(fast).      │                                        │
│ 📼🎯 SEEKR│ Replay + SELECTIVE knowledge-distillation │ NEW: adds a replay buffer + selective  │
│          │ on the top-K "retention-critical" units    │ KD. Ours has raw-replay 50% (m09c1)    │
│          │ (attention heads). KD only where forgetting│ but NO selective-KD component.         │
│          │ would hurt most → cheap, targeted anti-fgt.│                                        │
│ 🧠 SSIAT │ ONE shared adapter reused across ALL       │ NEW: 1 reusable PET, backbone frozen,  │
│          │ sessions (no per-session expansion);       │ updates confined to a low-dim subspace.│
│          │ backbone frozen; semantic-shift estimation │ Ours updates FULL ViT blocks.          │
│          │ to keep old-class prototypes valid.        │                                        │
│ 🧠🎯 SAPT │ Shared PET + per-INPUT attentive routing  │ NEW: input-conditioned PET routing     │
│          │ α_k(x)·Δ_k; a Shared Attentive Learning &  │ α_k(x). Ours has no adapter routing —  │
│          │ Selection module co-trains "which PET to   │ a single monolithic update.            │
│          │ learn" and "which to select" jointly.      │                                        │
└──────────┴────────────────────────────────────────────┴───────────────────────────────────────┘
```

PET building blocks the four share (insert points, all on the frozen V-JEPA ViT-G blocks):
```text
Adapter  bottleneck MLP inserted after attn/MLP sublayers      LoRA  low-rank ΔW = BA on q/k/v/proj
SSF      per-feature scale+shift (γ⊙x+β) after each sublayer   VPT   learnable prompt tokens prepended
```

Citations (ALL verified — websearch June 2026): SAFE — [arXiv 2411.02175](https://arxiv.org/abs/2411.02175)
(NeurIPS 2024) · SSIAT — [arXiv 2403.19979](https://arxiv.org/abs/2403.19979) (CVPR 2024) ·
SAPT — [arXiv 2401.08295](https://arxiv.org/abs/2401.08295) (ACL 2024) ·
SEEKR — [arXiv 2411.06171](https://arxiv.org/abs/2411.06171) (EMNLP 2024 — NOT CVPR/vision as the landscape
doc guessed). All four target class-incremental classification (SAFE/SSIAT) or LLM continual learning
(SAPT/SEEKR) — see § 1.5 for why their AAAI reviewer-pull is LOW.

═══════════════════════════════════════════════════════════════════════════════
§3 · Full landscape table (vs the vanilla anchors)
═══════════════════════════════════════════════════════════════════════════════

```text
┌────────────────┬──────────────────┬───────────────────────┬─────────┬─────────────────────┬────────┬─────────────────┐
│ Technique       │ Adapt subspace    │ Loss                  │ Stages   │ Anti-drift           │ Replay │ Factor data mix │
├────────────────┼──────────────────┼───────────────────────┼─────────┼─────────────────────┼────────┼─────────────────┤
│ 🐢⚡ SAFE       │ slow PET + fast   │ JEPA (+ align term)    │ 2 (S→F) │ slow branch frozen   │ ❌     │ ❌              │
│                │ PET               │                        │         │ after session 1      │        │                 │
│ 📼🎯 SEEKR     │ any (PET or full) │ JEPA + selective KD on │ 1+      │ replay + targeted KD │ ✅     │ ❌ (replay only)│
│                │                  │ top-K retention units  │         │ on critical units    │        │                 │
│ 🧠 SSIAT       │ 1 shared adapter  │ JEPA                  │ 1+      │ backbone frozen +    │ ❌     │ ❌              │
│                │ (reused)         │                        │         │ low-dim subspace     │        │                 │
│ 🧠🎯 SAPT      │ shared PET + per- │ JEPA + routing-aligned │ 1+      │ SSIAT + coordinated  │ ❌     │ ❌              │
│                │ input routing    │ select                 │         │ learn-and-select     │        │                 │
│ 🔥 m09a1 (ours)│ full ViT blocks   │ JEPA L1 + motion_aux   │ 1       │ ❌ vanilla           │ ❌     │ ❌              │
│ 🔥🎯 m09c1     │ progressive       │ JEPA L1 + motion_aux + │ 3-4     │ EMA teacher + SPD    │ ✅ 50% │ ✅ {D_L,D_A,D_I} │
│ (ours) surgery │ block unfreeze   │ deep supervision       │         │ anchor + DI drift    │ (raw)  │ stream mixing   │
└────────────────┴──────────────────┴───────────────────────┴─────────┴─────────────────────┴────────┴─────────────────┘
```
Closest overlap: m09c1's staged unfreeze + EMA anchor ≈ SAFE's slow/fast at the block level;
raw-replay 50% ≈ SEEKR's replay minus the selective-KD. None of ours does PET / shared-adapter /
routing — so the four are genuinely orthogonal baselines, not re-skins of surgery.

═══════════════════════════════════════════════════════════════════════════════
§4 · How we'd integrate them (benchmark wiring — brief)
═══════════════════════════════════════════════════════════════════════════════

```text
• Each technique = a NEW trainer in the TRAINING band (sibling of m09a1/m09c1), e.g.
  m09d_safe.py / m09d_seekr.py / m09d_ssiat.py / m09d_sapt.py — all reuse utils/training.py
  primitives (build_student_predictor, the JEPA loss, the streaming dataset) so only the
  PET-insertion + per-method loss/anti-drift differs.
• PET blocks (Adapter/SSF/VPT/LoRA) → utils/pet.py (shared insertion helpers on V-JEPA ViT-G).
• Each emits the SAME student_encoder.pt + (where applicable) m09*_ckpt_best.pt contract → they
  drop into run_eval.sh exactly like the surgery/pretrain arms (register in probe_encoders.yaml,
  add an encoder_ckpt_for() case). So eval is FREE once the trainer respects the ckpt contract.
• Scope/cost: these are iter16/17 "beyond-vanilla" baselines (CITA #4). Each ≈ one m09c1-class
  training run × the chosen backbone(s). Land AFTER the JEPA-variant loop (plan_model.md) so they
  reuse the same eval harness. DO NOT bundle into the current PR (GATE: live eval running).
```
