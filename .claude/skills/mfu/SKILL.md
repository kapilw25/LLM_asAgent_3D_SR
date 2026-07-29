---
name: mfu
description: Wire real-time Model FLOPs Utilization (MFU) into m09* training and m12* eval scripts via src/utils/mfu_calculator.py. Use when adding/auditing throughput or MFU logging.
---

# MFU (Model FLOPs Utilization) integration

Single source: `src/utils/mfu_calculator.py`. Config knobs: `configs/pipeline.yaml > mfu`.
Peak FLOPs is **measured at runtime** (dense BF16 GEMM micro-benchmark on the actual GPU) —
no hardcoded per-card number, no 2:4-sparse-vs-dense ambiguity.

## Formulas (gold standard: PaLM App. B / nanoGPT `estimate_mfu` / stas00 ml-engineering)
```
training  flops/token = 6N + 12*L*E*T        (E = embed_dim = n_heads*head_dim)
forward   flops/token = 2N + 4*L*E*T         (inference; = 1/3 of training)
MFU = flops_per_token * tokens_per_sec / peak_flops_dense
```
Frozen-aware (surgery/LoRA freeze most params) — we log BOTH:
```
honest   = 2*N_fwd + 4*N_train + 12*L*E*T    (== 6N when N_train == N_fwd)
dense-6N = 6*N_fwd            + 12*L*E*T
```
`tokens/clip T = (num_frames/tubelet) * (crop/patch)^2`. MFU > 1 signals an unrealistic
step time or wrong peak — treat it as a bug, not a result.

## Training scripts (m09*) — the 6-edit pattern (see m09a1 as the reference)
```python
# 1) import
from utils.mfu_calculator import build_calculator, measured_peak_flops
# 2) after batch_size is known + model built (student/predictor, device):
mfu_calc = build_calculator(
    forward_modules=[student, predictor],
    trainable_modules=[student, predictor],   # only requires_grad params -> N_train
    model_cfg=cfg["model"], num_frames=cfg["data"]["num_frames"],
    peak_flops=measured_peak_flops(get_pipeline_config(), device))
# 3) in the per-step log block, right after `throughput` (steps/sec) is computed:
mfu_rep = mfu_calc.report(mfu_calc.tokens_per_sec_from_steps(throughput, batch_size), "train")
# 4) step_record.update(**mfu_rep)            -> JSONL   (mfu, mfu_6n, tokens_per_sec)
# 5) wb_metrics: "mfu", "mfu_6n", "tokens_per_s"   -> wandb
# 6) pbar.set_postfix_str(f"... mfu={mfu_rep['mfu']:.3f}")   -> live terminal
```
Head trainers (m09a2/m09c2) have no per-step `throughput`; add a windowed timer or
skip (heads are cheap).

## Eval scripts (m12*) — integrate in the SHARED forward utils, not per-script
- Feature-extraction evals (m12a/b/c) share `utils/frozen_features.extract_features_for_keys`
  -> per-pass inference MFU is wired there (best-effort try/except so it never breaks eval).
- Predictor evals (m12d/e) share `utils/predictor_eval` -> wire the same
  `build_inference_calculator(...)` + `inference_mfu` there.
```python
from utils.mfu_calculator import build_inference_calculator, measured_peak_flops
mfu_calc = build_inference_calculator(          # N + depth introspected at runtime
    encoder=model, embed_dim=embed_dim, num_frames=args.num_frames,
    crop_size=crop, patch_size=PATCH_SIZE, tubelet_size=args.tubelet_size,
    peak_flops=measured_peak_flops(get_pipeline_config(), torch.device("cuda")))
# at end-of-pass (elapsed known):
tps = n_clips * mfu_calc.seq_len / elapsed
print(f"[mfu] inference_mfu={mfu_calc.inference_mfu(tps):.3f} ({tps:,.0f} tok/s)")
```
Token count is V-JEPA-accurate; approximate for non-V-JEPA baselines (DINOv2 patch 14,
no tubelet) — fine for instrumentation. Eval MFU is best-effort: a failure logs and disables,
it never crashes the eval.

## Rules honored
- No hardcodes: peak is measured; arch from model cfg / runtime introspection; benchmark
  knobs in `pipeline.yaml > mfu`.
- FAIL LOUD in training (config always present); best-effort try/except ONLY in eval
  (instrumentation must not break the research pass).
- CPU self-test (torch-free): `python3 -m utils.mfu_calculator --selftest`.

## Status (2026-07-28)
- Done: util + config, m09a1 (training template), frozen_features (m12a/b/c eval).
- TODO fan-out: m09{a2,b,c1,c2,d,e,f} (per-file anchors), predictor_eval (m12d/e).
