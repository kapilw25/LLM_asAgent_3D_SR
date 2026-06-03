# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

**STATUS 2026-06-03:** all 9 arms BUILT (own scripts). 48 GB only CODE-SMOKED them — autorgn passed, the other 8 hit OOM there (code clean up to the wall, but **OOM is NOT a pass**). **NEXT → re-run §1 SANITY on the 96 GB box; every arm must COMPLETE real training (steps + ckpt, no "0 successful") BEFORE §2 POC.** Drift-audit done: autorgn/full_ft/lpft now declare drift-off; surgery+raw spd+drift double-anchor LEFT as-is (a knob for its own ablation — do NOT flip).

## 1 · SANITY — must PASS every arm on 96 GB (the gate before POC)

```bash
export BACKBONE=vjepa_2_1_vitG
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | tee logs/sanity_pretrain_$(date +%Y%m%d_%H%M%S).log
# Run ALL on the 96 GB box so every arm COMPLETES. On 48 GB only autorgn fit; the other 8 OOM'd there
# (full_ft/lpft = 48-block AdamW · surgery_raw = SPD anchor +7.4 GB · peft = all-48-block activations ·
# cassle/ewc = all-48-block + Fisher) — code-clean to the wall, but NOT a pass until they finish on 96 GB.
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgical_autorgn_encoder --SANITY 2>&1 | tee logs/sanity_b2_autorgn_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh full_ft_encoder     --SANITY 2>&1 | tee logs/sanity_b4a_full_ft_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh lpft_encoder        --SANITY 2>&1 | tee logs/sanity_b4b_lpft_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_raw_encoder --SANITY 2>&1 | tee logs/sanity_ctrl_surgery_raw_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_lora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_lora_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_dora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_dora_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh cassle_encoder      --SANITY 2>&1 | tee logs/sanity_b3_cassle_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
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

## 2 · POC on 96 GB — only AFTER §1 SANITY passes for every arm

```bash
export BACKBONE=vjepa_2_1_vitG
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder         --POC 2>&1 | tee logs/poc_pretrain_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
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
ENCODERS="vjepa_2_1_surgical_autorgn_encoder vjepa_2_1_surgery_raw_encoder vjepa_2_1_full_ft_encoder vjepa_2_1_lpft_encoder vjepa_2_1_peft_lora_encoder vjepa_2_1_peft_dora_encoder vjepa_2_1_cassle_encoder vjepa_2_1_ewc_encoder" \
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC 2>&1 | tee logs/iter18_poc_eval_$(date +%Y%m%d_%H%M%S).log
grep -iE "FATAL|Traceback|not found|missing predictor" logs/iter18_poc_eval_*.log   # MUST be EMPTY
```
