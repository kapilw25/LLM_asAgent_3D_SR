#!/usr/bin/env bash
# SANITY→FULL overnight chain. Run detached:
#   nohup bash scripts/temp.sh > logs/chain_$(date +%F_%H%M%S).log 2>&1 &
set -u
trap '' HUP                                    # survive terminal/SSH hangup (children inherit SIG_IGN)
cd /workspace/factorjepa || exit 1
source venv_walkindia/bin/activate || { echo "[chain] venv activate FAILED — abort."; exit 1; }

export ITER18_BACKBONE=vjepa_2_1_vitg
export EVAL_CORPUS=full
export SKIP="surgery_3stage_DI_encoder surgery_noDI_encoder surgery_3stage_DI_head surgery_noDI_head surgical_autorgn_encoder surgery_raw_encoder full_ft_encoder lpft_encoder peft_dora_encoder cassle_encoder ewc_encoder surgery_3stage_DI_replay25_encoder surgery_3stage_DI_tccaux_encoder surgery_3stage_DI_intervene_encoder surgical_3stage_DI_wiseft_encoder surgical_intervene_wiseft_f30_encoder surgical_intervene_wiseft_f50_encoder surgical_intervene_wiseft_f70_encoder"
SLOG="$(ls -t logs/iter19_sanity_rest_*.log 2>/dev/null | head -1)"
SEED=outputs/full/vjepa_2_1_vitg_1B/train/m09a_pretrain_encoder/m09a_ckpt_best.pt

# 1) Wait (zero CPU) for the SANITY scheduler to exit; settle so tee flushes the final line.
SPID="$(pgrep -f 'ngpu_run.py --mode SANITY' | head -1)"
if [ -n "$SPID" ]; then
  echo "[chain $(date +%T)] waiting on SANITY pid=$SPID ..."
  tail --pid="$SPID" -f /dev/null
fi
sleep 3
echo "[chain $(date +%T)] SANITY process has exited."

# 2) Double-run guard: never launch a second FULL on top of one already running.
if pgrep -f 'ngpu_run.py --mode FULL' >/dev/null; then
  echo "[chain $(date +%T)] a FULL run is ALREADY running — not launching another. Exit."
  exit 0
fi

# 3) Success gate. FAIL-FAST aborts BEFORE 72/72, so '72/72 done' + no abort marker == clean pass.
if [ -n "$SLOG" ] \
   && grep -q "72/72 done" "$SLOG" \
   && ! grep -qE "FAIL-FAST|Traceback \(most recent|^FATAL" "$SLOG"; then
  if [ ! -f "$SEED" ]; then
    echo "[chain $(date +%T)] ABORT: seed missing ($SEED) — --cache 1 would retrain 18h. Not launching."
    exit 1
  fi
  echo "[chain $(date +%T)] SANITY GREEN ($SLOG) → cleaning outputs/sanity + launching FULL"
  rm -rf outputs/sanity/
  ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode FULL --gpus 2 --cache 1 --skip-arms $SKIP \
    2>&1 | tee "logs/iter19_full_rest_$(date +%F_%H%M%S).log"
  echo "[chain $(date +%T)] FULL scheduler exited (rc=${PIPESTATUS[0]})."
else
  echo "[chain $(date +%T)] SANITY did NOT finish clean (no 72/72, or FAIL-FAST/FATAL/Traceback)."
  echo "[chain $(date +%T)] FULL NOT started — safe. Inspect: ${SLOG:-<no sanity log found>}"
fi
