# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

## 0 · POC restart + eval speedup (2026-06-07) — §0.C park-CaSSLe restart · §0.E multi-backbone runs

### 0.C · mid-POC restart (2026-06-07) — park CaSSLe + EWC (both solo stragglers ≈ 17-21 h), CPU-set pinning ON

```bash
# C1 · interrupt: Ctrl-C the scheduler tmux. Hourly anchors bound the loss:
#      dora resumes from m09c_ckpt_latest.pt (≤1 h redo); 10 ✅ arms skip entirely.
# C2 · relaunch with h-memo ON (PT_H_MEMO=1 — verified bit-identical, scheduler env → all m12e
#      inherit it), without cassle + ewc (anchors stay parked on disk):
PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms cassle_encoder ewc_encoder 2>&1 | tee logs/iter18_ngpu_poc_$(date +%Y%m%d_%H%M%S).log
# first lines MUST print:  [--skip-arms] dropped ['cassle_encoder', 'ewc_encoder'] (train+eval+8b-metrics)
#                          [cpuset] GPU slots pinned: GPU0→0..79(32t) ...
#                          [resume --cache 1] skipping 10 already-trained arms
#                          ═══ ... 95 jobs (11 train + 84 eval) ═══   # 84 = 12 enc × (1 E: + 6 P:)
# resumed arm logs MUST print:  "Resumed from step N"  +  "cores=32 (pinned cpuset)"
# 8b metric (P:) logs MUST print:  "h-memoization (PT_H_MEMO): ON"

# C3 · watch (separate panes; status = state/ETA + 45-min HF backup, metrics = numbers + graphs)
watch -n60  'python -u scripts/iter18_poc_status.py'
watch -n300 'python -u scripts/iter18_poc_metrics.py'

# C4 · LATER (post-finale, off the deadline): finish cassle + ewc + rebuild the full 14-encoder finale.
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 2>&1 | tee logs/iter18_ngpu_poc_$(date +%Y%m%d_%H%M%S).log
```

### 0.E · multi-backbone runs (ITER18_BACKBONE switch) — 2B vitG champion + 1B vitg scale-axis → combined 2B+1B hero; skip vJEPA-2.0

```bash
# set ONE backbone for the whole run — the SAME value in the launch AND every watch pane:
#   BB=vjepa_2_1_vitG   # 2B champion (default) · ckpt checkpoints/vjepa2_1_vitG_384.pt (~30 GB)
#   BB=vjepa_2_1_vitg   # 1B scale-axis         · ckpt checkpoints/vjepa2_1_vitg_384.pt (~16 GB)
BB=vjepa_2_1_vitg
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head"   # +2 HEADs: predictor metrics = pretrain (frozen predictor) → 0 new info, win nothing; saves ~240 min (90m train + 30m eval × 2)
ls -lh "checkpoints/$(echo "$BB" | sed 's/vjepa_2_1_/vjepa2_1_/')_384.pt"   # pre-req present?
# 1) quick SANITY (code-path validator, ~minutes) — catches a build/dim/OOM crash cheaply:
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_sanity_${BB}_$(date +%Y%m%d_%H%M%S).log
# 2) then the full POC (only after SANITY is green):
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_${BB}_$(date +%Y%m%d_%H%M%S).log
# banner MUST show: backbone=$BB · 79 jobs (9 train + 70 eval) · ids E:${BB}_<arm>
# watch panes — $BB MUST match the run (status warns "BACKBONE MISMATCH" otherwise); metrics writes outputs/poc/probe_plot/metrics_watch/$BB/
# ITER18_SKIP_ARMS=$SKIP HIDES the 4 skipped arms from every table/graph + the status tool's live m13 preview hero;
#   {train,eval}_metrics.{json,csv} still store ALL arms. \"$SKIP\" must stay QUOTED — it has spaces, else sh -c reads only arm #1.

BB=vjepa_2_1_vitG # or "vjepa_2_1_vitg"
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" 
ITER18_BACKBONE=$BB ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_metrics.py

watch -n60  "ITER18_BACKBONE=$BB ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_status.py"
watch -n300 "ITER18_BACKBONE=$BB ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_metrics.py"
```

### 0.F · backfill 2B autorgn action+taxonomy (the two `—` cells) — TARGETED, do NOT re-run the whole scheduler

```bash
# WHY missing (traced 2026-06-08): during the 2B eval, autorgn's action-feature stage hit a CPU-RAM OOM
#   scare (oom-watchdog: memory 483.2/483.4 GB = 100%); the operator dropped stages 3 (action-train) + 11
#   (taxonomy) via logs/.eval_extra_skip → autorgn ALONE lost act_top1 + tax_F1. Stage-2 action FEATURES
#   ARE cached (features_test.npy, 100 MB) → backfill is cheap (stage 3 reuses them; stage 11 re-extracts
#   taxonomy features + trains 16 dims). ~20-40 min on ONE idle GPU.

# pre-req: the runtime skip-file MUST be absent (else it re-skips 3,11 — and would silently break the 1B too):
test ! -e logs/.eval_extra_skip && echo "OK: no extra-skip" || { echo "found — removing:"; cat logs/.eval_extra_skip; rm -f logs/.eval_extra_skip; }

# ⛔ what if you "simply execute" the §0.E scheduler command (BB=vjepa_2_1_vitG, --cache 1)? It WOULD backfill
#   autorgn (cache-1 re-runs E: jobs, missing stages compute) BUT: (1) it grabs all 4 GPUs → collides with the
#   LIVE 1B run → the fail-fast scheduler SIGTERMs BOTH; (2) re-launches ~70 no-op 2B eval jobs; (3) as written
#   the runbook sets BB=vjepa_2_1_vitg (1B) — you'd target the wrong backbone. Use the TARGETED command instead:
CUDA_VISIBLE_DEVICES=3 ITER18_BACKBONE=vjepa_2_1_vitG \
  SKIP_STAGES="1,2,4,5,6,7,8,8b,9,9b,10,12,13" CACHE_POLICY_ALL=1 \
  ./scripts/run_eval.sh --POC --encoders vjepa_2_1_surgical_autorgn_encoder \
  2>&1 | tee logs/gapfill_autorgn_2B_action_tax_$(date +%Y%m%d_%H%M%S).log
# keeps ONLY stage 3 (action-train → probe_action/.../test_metrics.json) + stage 11 (taxonomy → probe_taxonomy/
#   .../test_metrics.json). Watch CPU RAM during stage 11's feature extract — if it nears 483 GB, run it when
#   the 1B is idle (pretrain-seed phase), NOT during the 1B fan-out. Finishes inside the ~2.5 h seed window.

# after it lands: refresh the table/scorecard (autorgn act/tax now filled), then re-upload (06-08 14:46 upload predates this):
ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="$SKIP" python -u scripts/iter18_poc_metrics.py
```
