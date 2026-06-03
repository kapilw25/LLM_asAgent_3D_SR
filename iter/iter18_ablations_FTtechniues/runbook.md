# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

**STATUS 2026-06-03:** all 9 arms BUILT (own scripts) + 48 GB SANITY clean — autorgn fits; the other 8 reach a clean OOM (all-48-block / SPD-anchor / Fisher), no other error. **NEXT → §2 POC on the 96 GB box.** Drift-audit done: autorgn/full_ft/lpft now declare drift-off; surgery+raw spd+drift double-anchor LEFT as-is (a knob for its own ablation — do NOT flip).

## 1 · SANITY on 48 GB — ✅ COMPLETE

```bash
export BACKBONE=vjepa_2_1_vitG
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | tee logs/sanity_pretrain_$(date +%Y%m%d_%H%M%S).log
# fit 48 GB → PASS (autorgn only: ≤8 trainable blocks, no SPD anchor):
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgical_autorgn_encoder --SANITY 2>&1 | tee logs/sanity_b2_autorgn_$(date +%Y%m%d_%H%M%S).log
# OOM expected on 48 GB (code validated → real run on 96 GB): full_ft/lpft = 48-block AdamW; surgery_raw
# = SPD anchor +7.4 GB; peft = all-48-block activations; cassle/ewc = all-48-block + (cassle distill 2nd
# forward / ewc 7.4 GB diagonal Fisher):
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh full_ft_encoder     --SANITY 2>&1 | tee logs/sanity_b4a_full_ft_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh lpft_encoder        --SANITY 2>&1 | tee logs/sanity_b4b_lpft_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_raw_encoder --SANITY 2>&1 | tee logs/sanity_ctrl_surgery_raw_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_lora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_lora_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh peft_dora_encoder   --SANITY 2>&1 | tee logs/sanity_b1_dora_$(date +%Y%m%d_%H%M%S).log
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh cassle_encoder      --SANITY 2>&1 | tee logs/sanity_b3_cassle_$(date +%Y%m%d_%H%M%S).log ; sleep 10 ; \
REPLAY_OVERRIDE=off CACHE_POLICY_ALL=2 ./scripts/run_train.sh ewc_encoder         --SANITY 2>&1 | tee logs/sanity_b3_ewc_$(date +%Y%m%d_%H%M%S).log
```

## 1.1 · verify (only OOM acceptable)

```bash
grep -iE "FATAL|Traceback|KeyError|invalid choice" logs/sanity_*.log | grep -ivE "OutOfMemory|SURGERY FAILED: 0 successful|run_train.sh aborted"   # MUST be EMPTY
grep -hE "kept top-8" logs/sanity_b2_*.log
grep -hE "48/48 blocks|SURGERY FAILED: 0 successful.*OOMed" logs/sanity_b4*.log
grep -hE "CaSSLe. distill predictor|EWC. online diagonal Fisher|PEFT: (Lo|Do)RA" logs/sanity_b1_*.log logs/sanity_b3_*.log   # B1/B3 new code built before the OOM
find outputs/sanity/vjepa_2_1_vitG -maxdepth 2 \( -name student_encoder.pt -o -name '*_ckpt_best.pt' \) | sort
```

## 2 · POC on 96 GB — ▶ NEXT STEP (run this on the 96 GB box)

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
