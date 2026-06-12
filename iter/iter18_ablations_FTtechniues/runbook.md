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

### 0.E · ⏸️ PARKED (2026-06-12, user order) — 1B vitg scale-axis runs · resume only if a reviewer demands the scale ablation

```bash
# ⏸️ 1B (vjepa_2_1_vitg) train+eval PARKED — focus = §0.H (complete ALL 2B predictor+encoder evals).
#    1B state preserved on disk/HF: pretrain trained; 4 surgery arms mid-train (resume anchors in
#    outputs/poc/vjepa_2_1_vitg/, 91 GB); only vitg_frozen evaluated. To resume later: run the
#    commands below unchanged (--cache 1 picks up the anchors).
# set ONE backbone for the whole run — the SAME value in the launch AND every watch pane:
#   BB=vjepa_2_1_vitG   # 2B champion (default) · ckpt checkpoints/vjepa2_1_vitG_384.pt (~30 GB)
#   BB=vjepa_2_1_vitg   # 1B scale-axis         · ckpt checkpoints/vjepa2_1_vitg_384.pt (~16 GB)
BB=vjepa_2_1_vitG
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head"   # +2 HEADs: predictor metrics = pretrain (frozen predictor) → 0 new info, win nothing; saves ~240 min (90m train + 30m eval × 2)
ls -lh "checkpoints/$(echo "$BB" | sed 's/vjepa_2_1_/vjepa2_1_/')_384.pt"   # pre-req present?
# 1) quick SANITY (code-path validator, ~minutes) — catches a build/dim/OOM crash cheaply:
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_sanity_${BB}_$(date +%Y%m%d_%H%M%S).log
# 2) then the full POC (only after SANITY is green):
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 1 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_${BB}_$(date +%Y%m%d_%H%M%S).log
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

### 0.G · m12f encoder-temporal SANITY smoke (2026-06-12 revival + speedups #1-#8) — run BEFORE any POC with 8c

```bash
# m12f (aot/tov/pace/tcc) revived from legacy into run_eval STAGE 8c (+9c paired); speedups landed:
#   F:-job metric fan · variant-batched forwards · OOM-halving · .m12f_ckpt resume ·
#   share-features (AoT-fwd/TOV-identity from probe_action cache) · same-T single-pass decode ·
#   batched TCC w/ VRAM-auto pair-chunk. CPU+dummy-GPU parity all PASS — this is the REAL-encoder smoke.
# SKIP_STAGES = every OTHER stage except 1+8c (2/3 action · 4 action-paired · 5/6 motion · 7 motion-paired ·
#   8 future · 8b pred-temporal · 9/9b/9c paired · 10 plots · 11-13 taxonomy) → run_eval executes 1 then 8c.
#   stage 1 KEPT: outputs/sanity/ was cleared 2026-06-12 → m12f needs action_labels.json rebuilt (~2 min CPU).
#   encoder = vjepa_2_1_frozen (the 2B champion ckpt — 1B is PARKED per §0.E).
SKIP_STAGES="2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen \
2>&1 | tee logs/m12f_sanity_smoke_$(date +%Y%m%d_%H%M%S).log
# log MUST show:  "[share-features] test: ON"   (or an explicit off-reason — never silent)
#                 "[tcc] pair-chunk auto → N (free=…G × target=0.85 …)"   # VRAM-scaled (24/48/96 GB)
#                 4 files written: outputs/sanity/encoder_temporal/<enc>/aggregate_{aot,tov,pace,tcc}.json
# crash mid-run? just re-run the same command — .m12f_ckpt_{split}.npz resumes the extraction.
# after green: POC schedulers pick up 8c automatically (167 jobs = 13 T + 14 E + 84 P + 56 F).
```

### 0.H · 🎯 CURRENT FOCUS (2026-06-12) — complete ALL 2B (vjepa_2_1_vitG) evals: predictor ✅ done → encoder (8c) + autorgn gap

```bash
# STATE: 2B predictor metrics (8b) COMPLETE for all 11 evaluated encoders; encoder metrics (8c,
#   m12f aot/tov/pace/tcc) = 0 done; autorgn still missing act_top1+tax_F1 (§0.F).
# SKIP: cassle+ewc (never trained) + 2 HEADs — head arms FREEZE the encoder at pretrain-init, so
#   their ENCODER == pretrain's ⇒ m12f (encoder-only metrics) would duplicate the pretrain row,
#   the same 0-new-info logic that skipped their predictor metrics in §0.E.
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head"

# pre-reqs (one-time on a fresh box):
test ! -e logs/.eval_extra_skip && echo "OK: no extra-skip" || { cat logs/.eval_extra_skip; rm -f logs/.eval_extra_skip; }
ls data/eval_10k_local/test_split.json data/eval_10k_local/m00d_download_subset >/dev/null && echo "OK: eval data present"
ls checkpoints/vjepa2_1_vitG_384.pt >/dev/null && echo "OK: 2B ckpt present"

# THE run — ONE resumed 2B scheduler does everything (run §0.G smoke green FIRST):
#   --cache 1 ⇒ 9 trains skipped (student_encoder.pt on disk) · P: jobs skipped (8b aggregates on
#   disk) · E: jobs re-run as cheap cache-skims EXCEPT autorgn, whose missing stages 3+11 COMPUTE
#   → the §0.F backfill is SUBSUMED here · 40 F: jobs (10 enc × 4 metrics) do the real 8c work ·
#   §3 finale then runs 9c (m12f paired) + m13 with the 5 new hero rows (aot/tov/pace/tcc_τ/tcc_cycle).
ITER18_BACKBONE=vjepa_2_1_vitG PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC \
  --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_vitG_8c_$(date +%Y%m%d_%H%M%S).log
# (--gpus 1 on the 24 GB smoke box — works, just serial; rent the 4× box for the 4-way F: fan)
# banner MUST show: backbone=vjepa_2_1_vitG · [--skip-arms] dropped [...] (train+eval+8b/8c-metrics)
#                   [resume --cache 1] skipping 9 already-trained arms
#                   [resume --cache 1] skipping ~66 already-done Stage-8b metric jobs
#                   ids F:vjepa_2_1_<arm>:<aot|tov|pace|tcc> launching as GPUs free up

# watch + refresh (8c column appears in the status eval cells as `·8c d✓r▶/4`):
watch -n60  "ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_status.py"
watch -n300 "ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_metrics.py"

# after the finale: upload the completed 2B (encoder_temporal/ + refreshed plots) to HF:
# measured: 338 GB ≈ 2 hours 28 mins wall
python -u src/utils/hf_outputs.py upload-full outputs/ 2>&1 | tee logs/upload_full_outputs_$(date +%Y%m%d_%H%M%S).log
```
