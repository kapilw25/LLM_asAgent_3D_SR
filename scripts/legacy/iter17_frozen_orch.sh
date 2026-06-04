#!/bin/bash
# iter17 frozen-sweep orchestrator: serially SANITY-infer the remaining 2.x frozen baselines
# as each weight download completes. Gates on log DONE marker + ckpt size-stability (no pgrep
# -f self-match; no || swallows per fail-hard hook). Encoder-feature stages only.
cd /workspace/factorjepa
source venv_walkindia/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SKIP="3.5,4,7,8,8b,8c,9,9b,9c,10,11,12,13"
SUM=logs/iter17_frozen_sweep_summary.log
echo "=== frozen sweep orchestrator start $(date '+%H:%M:%S') ===" >> "$SUM"

wait_done() {                       # $1=logfile → 0 done, 1 failed, 2 timeout
  for _ in $(seq 1 80); do
    if grep -q "DONE . total" "$1" 2>/dev/null; then return 0; fi
    if grep -qE "FATAL|Traceback \(most recent" "$1" 2>/dev/null; then return 1; fi
    sleep 15
  done
  return 2
}
wait_ckpt() {                       # $1=path → 0 when size stable & >1GB, 1 timeout
  local prev=-1 cur
  for _ in $(seq 1 150); do
    cur=$(stat -c%s "$1" 2>/dev/null); cur=${cur:-0}
    if [ "$cur" = "$prev" ] && [ "$cur" -gt 1000000000 ]; then return 0; fi
    prev=$cur; sleep 20
  done
  return 1
}
result_of() {                       # $1=encoder → PASS/FAIL by metrics file presence
  if [ -f "outputs/sanity/probe_action/$1/test_metrics.json" ]; then echo PASS; else echo FAIL; fi
}

# 1) let the currently-running v1_vitH free the GPU
VLOG=$(ls -t logs/iter17_sanity_vjepa1_vitH_2026*.log | head -1)
wait_done "$VLOG"
echo "vjepa_1_vitH_frozen: $(result_of vjepa_1_vitH_frozen)" >> "$SUM"

# 2) remaining 2.x frozen baselines (serial; wait for each ckpt download)
for spec in vjepa_2_1_vitg_frozen:checkpoints/vjepa2_1_vitg_384.pt \
            vjepa_2_1_vitL_frozen:checkpoints/vjepa2_1_vitl_dist_vitG_384.pt \
            vjepa_2_vitL_256_frozen:checkpoints/vjepa2_0_vitl_256.pt; do
  enc=${spec%%:*}; ck=${spec##*:}
  if ! wait_ckpt "$ck"; then
    echo "$enc: CKPT-TIMEOUT $ck" >> "$SUM"
    continue
  fi
  L="logs/iter17_sanity_${enc}_$(date +%H%M%S).log"
  ENCODERS="$enc" SKIP_STAGES="$SKIP" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY > "$L" 2>&1
  load=$(grep -oE "Loaded [0-9]+/[0-9]+ params|only [0-9]+/[0-9]+" "$L" | head -1)
  echo "$enc: $(result_of "$enc") | $load | $L" >> "$SUM"
done
echo "=== frozen sweep orchestrator done $(date '+%H:%M:%S') ===" >> "$SUM"
