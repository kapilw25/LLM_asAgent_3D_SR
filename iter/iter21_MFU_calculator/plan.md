# iter21 MFU calculator — resume plan (remaining work)

## 1. Status

| # | Component | Path | Status | Verified (Mac) |
|---|-----------|------|--------|----------------|
| 1 | websearch (formulas + RTX PRO 6000 dense peak ~500 TFLOPS) | n/a | ✅ DONE | sourced |
| 2 | MFU util (train+inference, honest+6N, runtime GEMM peak, FAIL-LOUD) | src/utils/mfu_calculator.py | ✅ DONE | ruff+py_compile+selftest |
| 3 | MFU config (gemm_bench_size/iters/warmup) | configs/pipeline.yaml `mfu:` | ✅ DONE | parses |
| 4 | training template | src/m09a1_pretrain_encoder.py | ✅ DONE | ruff+py_compile |
| 5 | eval extraction (covers m12a/b/c) | src/utils/frozen_features.py | ✅ DONE | ruff+py_compile |
| 6 | skill doc | .claude/skills/mfu/SKILL.md | ✅ DONE | n/a |
| 7 | fan-out m09 encoders | src/m09{b,c1,d,e,f}_*.py | ⏳ TODO | - |
| 8 | fan-out m09 heads | src/m09{a2,c2}_*.py | ⏳ TODO | - |
| 9 | eval predictor (covers m12d/e) | src/utils/predictor_eval.py | ⏳ TODO | - |
| 10 | verify m12f coverage | src/m12f_encoder_temporal.py | ⏳ TODO | - |

## 2. m09 encoder fan-out — 4 identical edits per file (m09b, m09c1, m09d, m09e, m09f)

| Edit | old_string (exact substring, unique/file) | op | new_string |
|------|--------------------------------------------|----|------------|
| E1 import | `from utils.progress import make_pbar` | append line | `from utils.mfu_calculator import build_calculator, measured_peak_flops` |
| E2 init | `    predictor = models["predictor"]` (indent 4) | append line | `    mfu_calc = None` |
| E3 compute | `throughput = window_steps / window_elapsed if window_elapsed > 0 else 0` (indent 16) | append block E3B | see section 3 |
| E4 postfix | `f"{throughput:.2f} step/s")` | replace inline | `f"{throughput:.2f} step/s " f"mfu={mfu_rep['mfu']:.3f}")` |

## 3. E3B block (append after the E3 throughput line, indent 16, verbatim)

```python
                if mfu_calc is None:   # lazy: N_train counted AFTER each arm's freezing
                    mfu_calc = build_calculator(
                        forward_modules=[student, predictor],
                        trainable_modules=[student, predictor],
                        model_cfg=cfg["model"], num_frames=cfg["data"]["num_frames"],
                        peak_flops=measured_peak_flops(get_pipeline_config(), device))
                mfu_rep = mfu_calc.report(
                    mfu_calc.tokens_per_sec_from_steps(throughput, batch_size), "train")
```

## 4. m09 encoder anchor lines (indents uniform: predictor=4, throughput=16, postfix=24)

| File | predictor L | throughput L | postfix L | batch_size def L | notes |
|------|-------------|--------------|-----------|------------------|-------|
| m09b_peft_encoder | 541 | 1701 | 1707 | 792 | get_pipeline_config imported (4x); vars student/predictor/cfg/device/batch_size in scope |
| m09c1_surgery_encoder | 539 | 1683 | 1689 | 778 | same; do NOT touch surgery recipe knobs |
| m09d_contssl_encoder | 546 | 1745 | 1751 | 809 | same |
| m09e_autorgn_encoder | 542 | 1711 | 1717 | 781 | same |
| m09f_naiveft_encoder | 540 | 1678 | 1684 | 779 | same |
| all | - | - | - | - | line numbers shift +1 per insert; anchor by STRING not line |

## 5. m09 head trainers (no per-step throughput var)

| File | batch_size L | option A | option B (preferred if time-short) |
|------|--------------|----------|------------------------------------|
| m09a2_pretrain_head | grep `batch_size =` | add windowed timer (time.time delta over N steps) then E1/E2/E3/E4 with mode="train" | skip: head FLOPs negligible vs encoder; MFU on encoder arms suffices |
| m09c2_surgery_head | 332 | same | same |

## 6. eval predictor (m12d future_mse, m12e predictor_temporal)

| Item | Detail |
|------|--------|
| file | src/utils/predictor_eval.py |
| forward fns | safe_metric L212, masked_predict_l1 L260, rollout_l1_per_horizon L288 |
| pattern | build once before pass: `from utils.mfu_calculator import build_inference_calculator, measured_peak_flops`; `_mfu=build_inference_calculator(encoder=encoder, embed_dim=<cfg>, num_frames=num_frames, crop_size=<cfg>, patch_size=<cfg patch>, tubelet_size=<cfg>, peak_flops=measured_peak_flops(get_pipeline_config(), torch.device("cuda")))` |
| print | at per-pass elapsed: `print(f"[mfu] inference_mfu={_mfu.inference_mfu(n_clips*_mfu.seq_len/elapsed):.3f}")` |
| robustness | wrap build in try/except (best-effort; must NOT break eval), like frozen_features |
| caveat | encoder-only N; predictor forward adds FLOPs not counted -> approx; acceptable for instrumentation, or extend util to sum [encoder,predictor] params |
| m12f_encoder_temporal | verify: if it calls extract_features_for_keys then ALREADY covered; else add end-of-pass inference_mfu via build_inference_calculator |

## 7. verification per file (Mac; GPU SANITY = user runs on Blackwell box)

| Step | Command |
|------|---------|
| lint | `source venv_walkindia/bin/activate; python3 -m py_compile src/<file>.py; ruff check --select F,E9 src/<file>.py` |
| util selftest | `PYTHONPATH=src python3 -m utils.mfu_calculator --selftest` |
| GPU smoke (user) | smallest `run_train.sh --SANITY <arm>`; expect startup `[mfu] peak=... TFLOPS` + pbar `mfu=...`; eval expects `[mfu] inference_mfu=...` |

## 8. already-applied edits (reference for the pattern)

| File | Edits (current line numbers) |
|------|------------------------------|
| src/m09a1_pretrain_encoder.py | import L85; build_calculator L534-541; compute L1015-1016; step_record `**mfu_rep` L1036; wb_metrics L1066-1067; pbar postfix L1087 |
| src/utils/frozen_features.py | import L41; build_inference_calculator L557-563 (try/except); per-pass print L659-662 |
| src/utils/mfu_calculator.py | MFUCalculator(embed_dim), build_calculator, build_inference_calculator (introspects depth via .blocks / .encoder.layer), measure_peak_flops (GEMM), tokens_per_clip, report() |
| configs/pipeline.yaml | `mfu:` gemm_bench_size 8192 / iters 30 / warmup 10 |

## 9. gotchas

| Item | Note |
|------|------|
| MFU > 1 | bug signal: unrealistic step time or wrong peak; not a valid result |
| peak | measured dense BF16 GEMM at runtime; NOT NVIDIA 1 PFLOPS (that is 2:4 sparse) |
| honest vs 6N | honest = 2*N_fwd+4*N_train (frozen-aware); 6N = all-trainable; both logged |
| no hardcode | peak measured; arch from model_cfg or runtime introspection; bench knobs in yaml |
| eval best-effort | try/except in eval utils only; training path FAIL-LOUD |
