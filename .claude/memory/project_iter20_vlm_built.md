---
name: project-iter20-vlm-built
description: iter20 VLM-head (App A) is BUILT + 3060-verified — the OURS-VLM vs FROZEN-VLM demo_cosmos path; runs on 96GB box; early gate is the OOD-transfer make-or-break
metadata:
  type: project
---
**Built 2026-07-19** after the cheap probe/retrieval demos all hit the wall (motion invisible on WalkIndia
radial flow — see [[project_iter20_demo_cosmos_impossible]]). User chose App A (the VLM) knowing the OOD risk.

**Code (751 LoC, all 3-check clean, 3060-verified where possible):**
- `configs/vlm.yaml` — single source (LLM id, arms→ckpt, pool, projector, LoRA, 2 stages, gate thresholds).
- `src/utils/vlm_model.py` — `SpatialPoolProjector` + `VJepaLlavaVLM`. V-JEPA 2.1 ViT-G (frozen OR ours, native
  `load_encoder_only`, 5632 = 1408×4 concat) → spatial-pool 24²→8² (512 tokens, temporal kept) → MLP 5632→3584
  GELU → Qwen3-8B (hidden 4096, LoRA r32/α64 stage-2, NON-thinking gate; projector out-dim is DYNAMIC from the
  loaded LLM → swapping the LLM is a 1-config-line change). LLaVA merge for the `<video>` token. **Verified on 3060:
  encoder→pool→projector both arms → 512×3584 finite.**
- `src/m18_vlm_data.py` — builds instruction JSONL. Sources = ONLY HF-reachable: `lmms-lab/TempCompass` (gate:
  4033 temporal MC-QA — action/direction/speed/order/attr) + `lmms-lab/LLaVA-Video-178K` (align captions +
  instruct motion-QA, scene-QA filtered out). **SSv2/EK100 are gated/manual-DL → excluded.** Gate adapter tested.
- `src/m18_vlm_train.py` — LLaVA 2-stage (align=projector, instruct=projector+LoRA), `--arm {frozen,ours}` the
  ONLY diff, `--max-samples` cap. **dataset/collator verified on 3060** (`<video>` placed once, only answer supervised).
- `src/m18_vlm_eval.py` — greedy MC decode → acc + vectorized bootstrap 95% CI → gate + `heroes_vlm.json`. Logic verified.

**EARLY GATE (my addition to plan_E1, cheap de-risk BEFORE the multi-day pretrain):** instruct-only capped
(3000 samples, projector from scratch) both arms → eval a 60-video TempCompass subset → PASS iff OURS−FROZEN ≥2pp.
**CRUX:** TempCompass = general video = OOD; Diving48 (App B) showed OURS's edge fades OOD (−1pp). So the early
gate IS the make-or-break OOD-transfer test — if OURS≈FROZEN there, STOP (days saved) → forest plots.

**Runs on the 96GB box** (Qwen3-8B won't fit the 3060 12GB). Deps already present (transformers 5.5.4 / peft 0.19
/ accelerate / datasets) — verify on `/venv/main`. Full commands: `runbook.md` §E1 (early-gate-first). I build +
CPU-test; the USER runs the training/gate on the big box ([[feedback_never_run_training_commands]]).
