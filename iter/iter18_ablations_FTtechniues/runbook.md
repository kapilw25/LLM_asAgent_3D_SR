# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

**STATUS 2026-06-06 (probe-leak fix → 3rd restart, on 4×):** m09a1/a2/c2 probed a 1000-subsample of
action_labels.json — **6,854/9,130 (75%) inside the train pool** — while the m09c-family probed the
held-out 451 val split; since the probe is the best-ckpt selector (future_l1), pretrain's kept ckpt
(init for all 12 arms) was picked on partially-seen data. FIXED: (🅰) all trainers now probe the val
split via `subset_keys_override`; run_train.sh `--probe-subset "$VAL_SPLIT"` everywhere; (🅱)
`build_probe_clips` requires `train_pool_keys` and **RAISES `[probe-leak guard]`** on probe∩train ≠ ∅
(verified: overlap/missing/None raise; real val∩train = 0). See `plan_CODE.md` tracker. The 06-05 POC
arms (pretrain, 3stage_DI, noDI) are INVALID. **NEXT → on the 4× box: download data+ckpts, then
`iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 2` (~30-40 min gate; m09a probe lines must show
val-split N=20, not 97) → `--mode POC --gpus 4 --cache 2` (~17-20 h incl. evals + m13).**

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
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY --encoders "vjepa_2_1_frozen vjepa_2_1_pretrain_encoder vjepa_2_1_surgical_3stage_DI_encoder vjepa_2_1_surgical_noDI_encoder vjepa_2_1_surgical_3stage_DI_head vjepa_2_1_surgical_noDI_head vjepa_2_1_surgical_autorgn_encoder vjepa_2_1_surgery_raw_encoder vjepa_2_1_full_ft_encoder vjepa_2_1_lpft_encoder vjepa_2_1_peft_lora_encoder vjepa_2_1_peft_dora_encoder vjepa_2_1_cassle_encoder vjepa_2_1_ewc_encoder" 2>&1 | tee logs/sanity_eval_$(date +%Y%m%d_%H%M%S).log
# CACHE_POLICY_ALL=2 + frozen/pretrain INCLUDED (2026-06-05): after ANY retrain the eval caches are stale —
# training wipes clear outputs/*/m09* only, NOT outputs/*/probe_* (723 stale 06-04 .npy survived; Stage 2
# printed "Resume: 20 clips already cached" and served YESTERDAY's encoders' features → killed + rerun).
# pretrain was retrained too → its caches + every paired-Δ against it must recompute. Use =1 ONLY when no
# encoder in the list changed since its last eval; kill signal = "already cached" on a just-retrained arm.
# ^ ONE physical line (env var + command): a lost "\" continuation silently drops ENCODERS and run_eval
#   falls back to its DEFAULT list — exactly what happened in iter18_sanity_eval_20260604_052206.log.
grep -iE "FATAL|Traceback|not found|missing predictor" logs/sanity_eval_*.log   # MUST be EMPTY
```

## 2 · POC on 96 GB — only AFTER §1 SANITY passes for every arm

```bash
# MULTI-GPU ALTERNATIVE (2×/4× box): 
`python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1`
#   (--cache 1 = RESUME: skips arms whose student_encoder.pt already exists — the migration flow below
#    carries the 1×-box overnight progress over. DAG: pretrain → 12 arms fan out → per-encoder evals
#    pipeline → §3 paired-Δ+m13 finale. Wall from scratch: 1 GPU ≈ 50 h · 2 GPU ≈ 28 h · 4 GPU ≈ 17 h.)
#   ⚠ ONE-TIME purge first: outputs/poc/probe_* holds stale 06-04 eval caches (train wipes never touch them):
#     find outputs/poc -maxdepth 1 -type d -name 'probe_*' -exec rm -rf {} +
#     rm -rf outputs/poc/m12e_predictor_temporal outputs/poc/m13_eval_plot 2>/dev/null
# MIGRATION 1×→N× box (FULL fidelity — ckpt_best/motion_aux/npy included, nothing skipped):
#   on 1× box  : python -u src/utils/hf_outputs.py upload-full outputs/poc     # per-dir _full-*.tar shards
#   on N× box  : python -u src/utils/hf_outputs.py download-full outputs/poc   # pulls + auto-unpacks
#   (the light `upload` command DROPS m09*_ckpt_best.pt — every arm's --init-from-ckpt — do NOT use it here)
export BACKBONE=vjepa_2_1_vitG ; \
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
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC --encoders "vjepa_2_1_frozen vjepa_2_1_pretrain_encoder vjepa_2_1_surgical_3stage_DI_encoder vjepa_2_1_surgical_noDI_encoder vjepa_2_1_surgical_3stage_DI_head vjepa_2_1_surgical_noDI_head vjepa_2_1_surgical_autorgn_encoder vjepa_2_1_surgery_raw_encoder vjepa_2_1_full_ft_encoder vjepa_2_1_lpft_encoder vjepa_2_1_peft_lora_encoder vjepa_2_1_peft_dora_encoder vjepa_2_1_cassle_encoder vjepa_2_1_ewc_encoder" 2>&1 | tee logs/poc_eval_$(date +%Y%m%d_%H%M%S).log
# ^ ONE physical line — same lost-continuation guard as §1.2.
# frozen/pretrain INCLUDED + =2 (2026-06-05, same trap as §1.2): outputs/poc/probe_* holds 06-04 POC eval
# caches computed against the OLD top1-selected pretrain + stale iter17 jsons; pretrain is retrained at §2
# under future_l1, and every paired-Δ keys on pretrain/frozen artifacts → all 14 must recompute fresh.
grep -iE "FATAL|Traceback|not found|missing predictor" logs/poc_eval_*.log   # MUST be EMPTY
```
