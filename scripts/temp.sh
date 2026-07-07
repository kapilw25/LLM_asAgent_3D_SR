#!/bin/bash
# Cache warm-up monitor — refreshes every 10s like `watch -n 10`. Run: bash scripts/temp.sh   (Ctrl-C to stop)
EVAL_LOG=logs/ngpu_run_full_eval_vjepa_2_1_vitg_frozen_20260707_060206.log
while true; do
  clear
  echo "═══ cache warm-up monitor · $(date -u '+%H:%M:%S') UTC · refresh 10s · Ctrl-C to stop ═══"
  echo
  echo "① frame-cache .npy count  (climbs → plateaus ~23,000 = full test set cached):"
  find data/full_local/m12_frame_cache -name "*.npy" 2>/dev/null | wc -l
  echo
  echo "② cache enabled @600G  (proves the 200→600 cap change is live):"
  grep -a "eval frame cache" "$EVAL_LOG" 2>/dev/null
  echo
  echo "③ maskratio per-clip rate  (~0.2s = cache HIT · appears after teacher_free; t-free stays ~0.7 = compute-bound):"
  grep -aoE "recent=[0-9.]+s/clip" logs/ngpu_run_full_pt_vjepa_2_1_vitg_frozen_maskratio_*.log 2>/dev/null | tail -3
  echo
  echo "④ store failures / ENOSPC  (expect EMPTY):"
  grep -aE "frame-cache store failed|ENOSPC|No space" logs/ngpu_run_full_*20260707_06*.log 2>/dev/null
  sleep 10
done
