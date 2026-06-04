# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

**STATUS 2026-06-03:** all 9 arms BUILT (own scripts). 48 GB only CODE-SMOKED them — autorgn passed, the other 8 hit OOM there (code clean up to the wall, but **OOM is NOT a pass**). **NEXT → re-run §1 SANITY on the 96 GB box; every arm must COMPLETE real training (steps + ckpt, no "0 successful") BEFORE §2 POC.** Drift-audit done: autorgn/full_ft/lpft now declare drift-off; surgery+raw spd+drift double-anchor LEFT as-is (a knob for its own ablation — do NOT flip).

## 1 · SANITY — must PASS every arm on 96 GB (the gate before POC)

```bash
# GATE semantics: && stops at the FIRST failing arm. set -o pipefail is MANDATORY for that — each arm is
# `run_train.sh | tee`, and without pipefail the pipeline's exit code is tee's (always 0), so && would
# silently sail past a FATAL exit=1. Verified 2026-06-04: `false | tee f` → exit 0 default, 1 under pipefail.
# (`set +o pipefail` to undo after. §2 POC deliberately keeps `;` — independent hours-long arms, one failure
#  must NOT cancel the overnight queue.)
set -o pipefail && export BACKBONE=vjepa_2_1_vitG && \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | tee logs/sanity_pretrain_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --SANITY 2>&1 | tee logs/sanity_factor_3stage_DI_enc_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --SANITY 2>&1 | tee logs/sanity_factor_noDI_enc_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head    --SANITY 2>&1 | tee logs/sanity_factor_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head         --SANITY 2>&1 | tee logs/sanity_factor_noDI_head_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgical_autorgn_encoder --SANITY 2>&1 | tee logs/sanity_b2_autorgn_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh full_ft_encoder     --SANITY 2>&1 | tee logs/sanity_b4a_full_ft_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh lpft_encoder        --SANITY 2>&1 | tee logs/sanity_b4b_lpft_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_raw_encoder --SANITY 2>&1 | tee logs/sanity_ctrl_surgery_raw_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_lora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_lora_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_dora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_dora_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh cassle_encoder      --SANITY 2>&1 | tee logs/sanity_b3_cassle_$(date +%Y%m%d_%H%M%S).log && sleep 10 && \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh ewc_encoder         --SANITY 2>&1 | tee logs/sanity_b3_ewc_$(date +%Y%m%d_%H%M%S).log
```

## 1.1 · verify — every arm PASSED on 96 GB (real training, no OOM)

```bash
# PASS = the run FINISHED with real optimizer steps. The step-0 probe writes a frozen-init student_best.pt
# even on an OOM run, so ckpt-exists is NOT proof — the "0 successful" FATAL is the fail signal.
grep -lE "SURGERY FAILED: 0 successful|OutOfMemory" logs/sanity_*.log   # MUST be EMPTY — any file listed did NOT pass
grep -iE "FATAL|Traceback|KeyError|invalid choice" logs/sanity_*.log    # MUST be EMPTY
grep -hE "kept top-8" logs/sanity_b2_*.log                              # autorgn picked its blocks
find outputs/sanity/vjepa_2_1_vitG -maxdepth 2 -name '*_ckpt_best.pt' | sort
```

## 1.2 · run_eval (--SANITY) — eval code-path smoke on the 8 baseline encoders

```bash
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --SANITY --encoders "vjepa_2_1_surgical_3stage_DI_encoder vjepa_2_1_surgical_noDI_encoder vjepa_2_1_surgical_3stage_DI_head vjepa_2_1_surgical_noDI_head vjepa_2_1_surgical_autorgn_encoder vjepa_2_1_surgery_raw_encoder vjepa_2_1_full_ft_encoder vjepa_2_1_lpft_encoder vjepa_2_1_peft_lora_encoder vjepa_2_1_peft_dora_encoder vjepa_2_1_cassle_encoder vjepa_2_1_ewc_encoder" 2>&1 | tee logs/sanity_eval_$(date +%Y%m%d_%H%M%S).log
# CACHE_POLICY_ALL=1 here: KEEP already-evaluated encoders, compute only the missing ones.
# (§2.2 POC eval stays =2: outputs/poc holds STALE iter17 jsons — those must NOT be reused.)
# ^ ONE physical line (env var + command): a lost "\" continuation silently drops ENCODERS and run_eval
#   falls back to its DEFAULT list — exactly what happened in iter18_sanity_eval_20260604_052206.log.
grep -iE "FATAL|Traceback|not found|missing predictor" logs/sanity_eval_*.log   # MUST be EMPTY
```

## 2 · POC on 96 GB — only AFTER §1 SANITY passes for every arm

```bash
# MULTI-GPU ALTERNATIVE (2×/4× box): python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1
#   (--cache 1 = RESUME: skips arms whose student_encoder.pt already exists — the migration flow below
#    carries the 1×-box overnight progress over. DAG: pretrain → 12 arms fan out → per-encoder evals
#    pipeline → §3 paired-Δ+m13 finale. Wall from scratch: 1 GPU ≈ 50 h · 2 GPU ≈ 28 h · 4 GPU ≈ 17 h.)
# MIGRATION 1×→N× box (FULL fidelity — ckpt_best/motion_aux/npy included, nothing skipped):
#   on 1× box  : python -u src/utils/hf_outputs.py upload-full outputs/poc     # per-dir _full-*.tar shards
#   on N× box  : python -u src/utils/hf_outputs.py download-full outputs/poc   # pulls + auto-unpacks
#   (the light `upload` command DROPS m09*_ckpt_best.pt — every arm's --init-from-ckpt — do NOT use it here)
export BACKBONE=vjepa_2_1_vitG
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder         --POC 2>&1 | tee logs/poc_pretrain_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --POC 2>&1 | tee logs/poc_factor_3stage_DI_enc_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --POC 2>&1 | tee logs/poc_factor_noDI_enc_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head    --POC 2>&1 | tee logs/poc_factor_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head         --POC 2>&1 | tee logs/poc_factor_noDI_head_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgical_autorgn_encoder --POC 2>&1 | tee logs/poc_b2_autorgn_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_raw_encoder      --POC 2>&1 | tee logs/poc_ctrl_surgery_raw_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh full_ft_encoder          --POC 2>&1 | tee logs/poc_b4a_full_ft_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh lpft_encoder             --POC 2>&1 | tee logs/poc_b4b_lpft_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_lora_encoder        --POC 2>&1 | tee logs/poc_b1_lora_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_dora_encoder        --POC 2>&1 | tee logs/poc_b1_dora_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh cassle_encoder           --POC 2>&1 | tee logs/poc_b3_cassle_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh ewc_encoder              --POC 2>&1 | tee logs/poc_b3_ewc_$(date +%Y%m%d_%H%M%S).log
```

## 2.1 · verify (--POC)

```bash
grep -iE "FATAL|Traceback|OutOfMemory|invalid choice" logs/poc_*.log   # MUST be EMPTY
find outputs/poc/vjepa_2_1_vitG -maxdepth 2 \( -name student_encoder.pt -o -name '*_ckpt_best.pt' \) | sort
```

## 2.2 · run_eval (--POC)

```bash
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC --encoders "vjepa_2_1_surgical_3stage_DI_encoder vjepa_2_1_surgical_noDI_encoder vjepa_2_1_surgical_3stage_DI_head vjepa_2_1_surgical_noDI_head vjepa_2_1_surgical_autorgn_encoder vjepa_2_1_surgery_raw_encoder vjepa_2_1_full_ft_encoder vjepa_2_1_lpft_encoder vjepa_2_1_peft_lora_encoder vjepa_2_1_peft_dora_encoder vjepa_2_1_cassle_encoder vjepa_2_1_ewc_encoder" 2>&1 | tee logs/poc_eval_$(date +%Y%m%d_%H%M%S).log
# ^ ONE physical line — same lost-continuation guard as §1.2.
grep -iE "FATAL|Traceback|not found|missing predictor" logs/poc_eval_*.log   # MUST be EMPTY
```
