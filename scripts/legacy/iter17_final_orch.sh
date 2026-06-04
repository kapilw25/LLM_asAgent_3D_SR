#!/bin/bash
cd /workspace/factorjepa
source venv_walkindia/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUM=logs/iter17_final_summary.log
echo "=== final orch start $(date +%H:%M:%S) ===" >> "$SUM"
# 1. regression: 2.1_vitg sanity-train (post training.py edits — 2.1 gates must still pass)
RLOG=logs/iter17_regress_2_1_vitg_$(date +%H%M%S).log
BACKBONE=vjepa_2_1_vitg CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --SANITY > "$RLOG" 2>&1
if [ -f outputs/sanity/vjepa_2_1_vitg/m09a_pretrain_encoder/student_encoder.pt ]; then
  echo "REGRESSION 2.1_vitg train: PASS (2.1 deep-sup path intact post-edit) | $RLOG" >> "$SUM"
else
  echo "REGRESSION 2.1_vitg train: FAIL | $RLOG" >> "$SUM"
fi
# 2. ssv2 frozen SANITY (kind=hf_vjepa2)
SLOG=logs/iter17_sanity_ssv2_$(date +%H%M%S).log
ENCODERS=vjepa_2_0_vitg_ssv2 SKIP_STAGES="3.5,4,7,8,8b,8c,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY > "$SLOG" 2>&1
if [ -f outputs/sanity/probe_action/vjepa_2_0_vitg_ssv2/test_metrics.json ]; then
  echo "ssv2 frozen (hf_vjepa2): PASS | $SLOG" >> "$SUM"
else
  echo "ssv2 frozen (hf_vjepa2): FAIL | $SLOG" >> "$SUM"
fi
echo "=== final orch done $(date +%H:%M:%S) ===" >> "$SUM"
