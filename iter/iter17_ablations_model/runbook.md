# iter17 — Runbook: cross-arch model ablation (frozen baselines → trainable backbones)

Order: §1 frozen baselines (eval-only, NO training, ready now) → §2 trainable cross-arch
(vjepa_2_1_vitg + vjepa_2_0_vitg). Each block runs SANITY (validate code paths, ~tiny data)
then POC (real 10k numbers). `sleep 10` between train cells lets the prior GPU process release
VRAM. Stage map: 1=labels 2=feat 3=action-probe 4=action-Δ 5/6=motion_cos 7=motion-Δ
8/8b/8c=predictor-fwd 9/9b/9c=predictor-Δ 10=action-plot 11=taxonomy-probe 12=taxonomy-Δ 13=taxonomy-plot.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ roster                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ trainable (predictor → surgery)  vjepa_2_1_vitG (done iter16) · vjepa_2_1_vitg ·        │
│                                  vjepa_2_0_vitg                                         │
│ frozen baselines (encoder mx · §G predictor cols N/A):                                  │
│   dinov2 · ijepa_vitH14 · ijepa_vitG16 · lejepa_vitL   (image JEPA / non-JEPA)          │
│   vjepa_2_0_vitg_ssv2 · vjepa_1_vitL · vjepa_1_vitH · vjepa_2_vitL_256 · vjepa_2_1_vitL │
│ NOT trainable                    vjepa_2_1_vitL = distilled, no predictor → frozen      │
│ blocked (no public weights)      mc_jepa · d_jepa                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1 · Frozen baselines — eval only (START HERE; zero training)

```bash
# 9 frozen baselines (native .pt verified present; ijepa/dinov2/ssv2 pull from HF; lejepa = local timm load).
FROZEN="dinov2 ijepa_vitH14 ijepa_vitG16 vjepa_2_0_vitg_ssv2 vjepa_1_vitL_frozen vjepa_1_vitH_frozen vjepa_2_vitL_256_frozen vjepa_2_1_vitL_frozen lejepa_vitL_frozen"
# run 1/2/3/5/6/11 = action + motion_cos + taxonomy. SKIP: paired-Δ (no arm pairs here),
# predictor 8/8b/8c+9 (frozen baselines have no usable predictor → N/A), plots 10/13 (build combined §G later).
SKIP_FROZEN="4,7,8,8b,8c,9,9b,9c,10,12,13"
```

```bash
# ── 1a · SANITY (combined smoke — proves the 9-in-one-command path, ~10 min, tiny data) ──
ENCODERS="$FROZEN" SKIP_STAGES="$SKIP_FROZEN" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY 2>&1 | tee logs/iter17_sanity_frozen9_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── 1b · POC (real 10k §G numbers) ──
ENCODERS="$FROZEN" SKIP_STAGES="$SKIP_FROZEN" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC    2>&1 | tee logs/iter17_poc_frozen9_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── 1c · OVERNIGHT (sleepy): SANITY then POC, `;` not `&&` so POC runs even if SANITY hiccups ──
FROZEN="dinov2 ijepa_vitH14 ijepa_vitG16 vjepa_2_0_vitg_ssv2 vjepa_1_vitL_frozen vjepa_1_vitH_frozen vjepa_2_vitL_256_frozen vjepa_2_1_vitL_frozen lejepa_vitL_frozen" ; SKIP_FROZEN="4,7,8,8b,8c,9,9b,9c,10,12,13" ; \
ENCODERS="$FROZEN" SKIP_STAGES="$SKIP_FROZEN" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC    2>&1 | tee logs/iter17_poc_frozen9_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── VERIFY (frozen) ──
grep -iE "FATAL|Traceback|KeyError|invalid choice" logs/iter17_*frozen9*.log   # MUST be EMPTY
ls outputs/poc/probe_action/{dinov2,ijepa_vitH14,ijepa_vitG16,vjepa_2_0_vitg_ssv2,vjepa_1_vitL_frozen,vjepa_1_vitH_frozen,vjepa_2_vitL_256_frozen,vjepa_2_1_vitL_frozen,lejepa_vitL_frozen}/probe.pt  # 9 action probes
grep -hE "Test top-1 acc|score_mean=" logs/iter17_poc_frozen9_*.log   # per-encoder action top1 + motion_cos
```

## 2 · Trainable cross-arch — vjepa_2_1_vitg + vjepa_2_0_vitg (train + eval)

```text
WS-B3 DONE: predictor_eval is arch-aware (run_eval passes --model-config per backbone) → vitg/2.0_vitg
get ALL 10 metrics; predictor stages (8/8b/8c/9*) NO LONGER skipped. Surgery inits from THIS backbone's
m09a_pretrain_encoder ckpt (pipeline.yaml surgery_init, namespaced outputs/<mode>/<BACKBONE>/) →
pretrain_encoder MUST be the FIRST arm. Both backbones are 1B (40-blk/1408) — ~0.6× the vitG (2B) wall.
PLAN:  §2.1 SANITY both backbones on 1× node (sequential, validate code) → §2.2 POC on 2× GPU.
```

### 2.1 · SANITY — BOTH backbones, sequential on 1× node (validate first)

```bash
# vjepa_2_1_vitg SANITY (7 arms + eval). ✅ ALREADY PASSED 2026-05-30 (0 errors). Re-run only if code changed.
# vjepa_2_0_vitg SANITY (7 arms + eval) — version axis; FIRST GPU smoke of the 2.0 deep-sup-gated path on THIS box.
for BACKBONE in vjepa_2_1_vitg vjepa_2_0_vitg ; do \
  ARMS_EVAL="${BACKBONE}_frozen ${BACKBONE}_pretrain_encoder ${BACKBONE}_pretrain_2X_encoder ${BACKBONE}_pretrain_head ${BACKBONE}_surgical_3stage_DI_encoder ${BACKBONE}_surgical_noDI_encoder ${BACKBONE}_surgical_3stage_DI_head ${BACKBONE}_surgical_noDI_head" ; \
  for ARM in pretrain_encoder pretrain_2X_encoder surgery_3stage_DI_encoder surgery_noDI_encoder pretrain_head surgery_3stage_DI_head surgery_noDI_head ; do \
    BACKBONE=$BACKBONE CACHE_POLICY_ALL=2 ./scripts/run_train.sh "$ARM" --SANITY 2>&1 | tee "logs/iter17_sanity_${BACKBONE}_${ARM}_$(date +%Y%m%d_%H%M%S).log" ; \
    sleep 10 ; \
  done ; \
  ENCODERS="$ARMS_EVAL" SKIP_STAGES="" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY 2>&1 | tee "logs/iter17_sanity_${BACKBONE}_eval_$(date +%Y%m%d_%H%M%S).log" ; \
done
```

```bash
# ── VERIFY (SANITY both) — MUST be empty; then proceed to §2.2 POC ──
grep -iE "FATAL|Traceback|KeyError|RuntimeError|invalid choice" logs/iter17_sanity_vjepa_2_1_vitg_*.log logs/iter17_sanity_vjepa_2_0_vitg_*.log   # MUST be EMPTY
# 2.0 deep-sup gated OFF (n_output_distillation=1) — must NOT show the 2.1 'training='/'mod=' kwargs error:
grep -hE "n_output_distillation|deep-sup|Predictor:|Student loaded" logs/iter17_sanity_vjepa_2_0_vitg_a1_pretrain_encoder_*.log | head
```

### 2.2 · POC — 2× GPU (one backbone per GPU, parallel). RTX Pro 6000 96 GB each.

```text
SAFE for shared disk: training writes per-backbone outputs/poc/<BACKBONE>/ (distinct); the shared
data-derivation (train_pool/splits, regenerated every arm) is now ATOMIC-written (clip_splits.py +
probe_train_subset.py, git iter17) → concurrent identical-content rewrites are race-free. Each lane's
EVAL runs per-encoder stages only and SKIPS the shared paired-Δ aggregate (4,7,9,9b,9c,12) + plots
(10,13); the combined paired-Δ + m13 plots run ONCE in §3 over ALL encoders. Orchestrator pins each
backbone to a GPU via CUDA_VISIBLE_DEVICES and runs both lanes in the background, then waits.
```

```bash
# ── 2.2 · launch the 2× GPU orchestrator (backbone 0→GPU0, backbone 1→GPU1, parallel). Takes {SANITY|POC}. ──
# OPTIONAL: validate the 2× loop on a 2-GPU node first (~minutes), then the real POC:
./scripts/iter17_poc_2gpu.sh SANITY 2>&1 | tee logs/iter17_2gpu_orch_sanity_$(date +%Y%m%d_%H%M%S).log ; \
./scripts/iter17_poc_2gpu.sh POC    2>&1 | tee logs/iter17_2gpu_orch_poc_$(date +%Y%m%d_%H%M%S).log
# default (no arg) = POC. Per-lane logs: logs/iter17_<mode>_<BACKBONE>_<ARM>_*.log + ..._eval_*.log
# POC wall ≈ ~15h train (both lanes parallel) + ~6h eval (parallel) ≈ ~21h.  Watch live: nvidia-smi -l 5
```

```bash
# ── ALT: manual single-GPU POC (no 2× node) — one backbone fully, then the other (~42h total) ──
for BACKBONE in vjepa_2_1_vitg vjepa_2_0_vitg ; do \
  ARMS_EVAL="${BACKBONE}_frozen ${BACKBONE}_pretrain_encoder ${BACKBONE}_pretrain_2X_encoder ${BACKBONE}_pretrain_head ${BACKBONE}_surgical_3stage_DI_encoder ${BACKBONE}_surgical_noDI_encoder ${BACKBONE}_surgical_3stage_DI_head ${BACKBONE}_surgical_noDI_head" ; \
  for ARM in pretrain_encoder pretrain_2X_encoder surgery_3stage_DI_encoder surgery_noDI_encoder pretrain_head surgery_3stage_DI_head surgery_noDI_head ; do \
    BACKBONE=$BACKBONE CACHE_POLICY_ALL=2 ./scripts/run_train.sh "$ARM" --POC 2>&1 | tee "logs/iter17_poc_${BACKBONE}_${ARM}_$(date +%Y%m%d_%H%M%S).log" ; sleep 10 ; \
  done ; \
  ENCODERS="$ARMS_EVAL" SKIP_STAGES="4,7,9,9b,9c,12,13,10" CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC 2>&1 | tee "logs/iter17_poc_${BACKBONE}_eval_$(date +%Y%m%d_%H%M%S).log" ; \
done
# then run §3 (combined paired-Δ + plots) ONCE.
```

```bash
# ── VERIFY (trained backbone) ──
grep -iE "FATAL|Traceback|KeyError|invalid choice" logs/iter17_poc_vjepa_2_1_vitg_*.log   # MUST be EMPTY
ls outputs/poc/vjepa_2_1_vitg/m09a_pretrain_encoder/m09a_ckpt_best.pt \
   outputs/poc/vjepa_2_1_vitg/m09c_surgery_3stage_DI_encoder/m09c_ckpt_best.pt   # arm ckpts landed under the backbone ns
grep -hE "depth=|n_trainable=" logs/iter17_poc_vjepa_2_1_vitg_c1_*.log   # surgery froze int(40*frac) blocks (40-blk auto-scale)
# headline Δ (surgery_encoder − pretrain_encoder) for this backbone:
python -c "import json; d=json.load(open('outputs/poc/probe_action/probe_paired_delta.json'))['iter14_paper_deltas']; print({k:v.get('delta_mean') for k,v in d.items() if v and not v.get('skipped')})"
```

## 3 · §G aggregate — combined verdict plots across ALL encoders (run LAST)

```bash
# STEP 1 (MUST run FIRST) — rebuild the by_encoder AGGREGATES over ALL encoders. m13 reads the
# paired-Δ JSONs (probe_paired_delta / probe_motion_cos_paired / per_dim_acc), and the paired_delta
# stage is the ONLY place by_encoder is built — but §1 frozen SKIPs stages 4/7/12, so the 9 frozen
# baselines never landed there → they'd be INVISIBLE in §G. These 3 CPU aggregators auto-discover
# every encoder subdir present (arms + 9 baselines) and rebuild by_encoder from the on-disk
# per-encoder test_metrics.json — no GPU, no feature recompute, ~15 min total.
source venv_walkindia/bin/activate ; export PYTHONPATH=src ; \
python -u src/m12a_action_top1.py --POC --stage paired_delta --output-root outputs/poc/probe_action     --cache-policy 1 --no-wandb ; \
python -u src/m12b_motion_cos.py  --POC --stage paired_delta --output-root outputs/poc/probe_motion_cos --cache-policy 1 --no-wandb ; \
python -u src/m12c_taxonomy_f1.py --POC --stage paired_delta --features-root outputs/poc/probe_action --output-root outputs/poc/probe_taxonomy --cache-policy 1 --no-wandb
```

```bash
# STEP 2 — build the COMBINED §G plots. m13 re-reads the SHARED probe roots (now carrying EVERY
# encoder) → one hero_table / hero_heatmap / scoreboard / grouped spanning frozen + baselines + arms.
# Verdict is single-sourced via _family_verdict (champion duel): scoreboard == grouped tally always.
# Frozen reference auto-derives to the arms' same-backbone frozen (vjepa_2_1_frozen), NOT the
# alphabetically-first 'frozen' baseline. Baselines carry head metrics only (predictor cols N/A).
source venv_walkindia/bin/activate ; export PYTHONPATH=src ; \
python -u src/m13_eval_plot.py --POC \
--action-probe-root       outputs/poc/probe_action \
--motion-cos-root         outputs/poc/probe_motion_cos \
--future-mse-root         outputs/poc/probe_future_mse \
--taxonomy-root           outputs/poc/probe_taxonomy \
--predictor-temporal-root outputs/poc/predictor_temporal \
--encoder-temporal-root   outputs/poc/encoder_temporal \
--output-dir              outputs/poc/probe_plot \
--no-wandb 2>&1 | tee logs/iter17_poc_m13_plots_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── VERIFY (plots) ──
# STEP 1 landed: by_encoder must cover the baselines (frozen-9 → 17 with the iter16 arms), not just 8
python -c "import json; print('action by_encoder:', len(json.load(open('outputs/poc/probe_action/probe_paired_delta.json'))['by_encoder']))"   # expect 17 (8 arms + 9 baselines)
grep -E "\[hero-table\]|\[hero-heatmap\]" logs/iter17_poc_m13_plots_*.log   # cols/rows must reflect ALL encoders (e.g. 18 cols, 16 rows)
grep -iE "FATAL|Traceback" logs/iter17_poc_m13_plots_*.log                 # MUST be EMPTY (absent temporal "[skip]" lines are EXPECTED, not errors)
grep -E "\[scoreboard\]|\[grouped\]" logs/iter17_poc_m13_plots_*.log       # champion-duel tally — scoreboard & grouped MUST agree: surgery N · pretrain N · tie N
ls outputs/poc/probe_plot/eval/{m13_hero_table,m13_hero_surgery_vs_frozen,m13_scoreboard_surgery_vs_pretrain,m13_grouped_winner_surgery_vs_pretrain}.{png,pdf}
```
