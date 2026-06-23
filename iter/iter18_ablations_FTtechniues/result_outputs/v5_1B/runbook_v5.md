# iter18 — Runbook v5 · 🎯 REPLICATE on V-JEPA 2.1 **1B** (ViT-g) — full train + eval

> **Goal:** reproduce the full 2B pipeline on the **1B** backbone → produce
> `outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/`
> `{eval_metrics.csv, eval_metrics.json, eval_scorecard.png, eval_scorecard_paper.png, kept_scorecard.png,`
> `tcc_comparison.png, train_metrics.csv, train_metrics.json, train_trajectories.pdf, wiseft_sweep_table.png}`
> — the SAME file set as the 2B reference (`…/vjepa_2_1_vitG_2B/…/metrics_watch/vjepa_2_1_vitG/`).
>
> **The infra is fully backbone-parameterized** — same scheduler/run_train/run_eval/m13; one env var flips it:
> `ITER18_BACKBONE=vjepa_2_1_vitg` (1B, ViT-g, 1408-dim) vs `vjepa_2_1_vitG` (2B, 1664-dim). Verified wiring:
> `backbone_model_configs.vjepa_2_1_vitg → configs/model/vjepa2_1_vitg.yaml` · `backbone_size_labels=1B` ·
> ckpt `checkpoints/vjepa2_1_vitg_384.pt` · `output_paths bb_dir → outputs/poc/vjepa_2_1_vitg_1B`.
> ⚠️ inner metrics_watch dir is `vjepa_2_1_vitg` (lowercase g), not `vitG` — the goal's capital-G was a copy-paste.

> **Cost-optimized box split.** pretrain is the SINGLE seed (all 12 other arms init from its
> `m09a_ckpt_best.pt`). ✅ **The 1B pretrain seed ALREADY EXISTS** — on HF + in
> `outputs/poc/vjepa_2_1_vitg_1B/train/m09a_pretrain_encoder/` (m09a_ckpt_best.pt 4.0G + student_encoder.pt
> 3.8G), trained at the SAME POC recipe (verified: epochs=2 == `max_epochs.poc`; best_ckpt_metric=future_l1).
> So **Box B is SKIPPABLE** — reuse the seed and let Box C's `--cache 1` resume-skip pretrain.
> Box A = code (cheap 3060) · ~~Box B = pretrain seed~~ (SKIP — seed exists & matches) · Box C = train rest + eval (4× RTX 6000).

---

## 🟢 Box A · 1× RTX 3060 12 GB — CODE MODIFICATION ONLY (no training)

```bash
# This box is CODE-ONLY. (Its FA2 wheel is the wrong arch — setup ran --from-wheels which pulls the
# Blackwell sm_120 wheel onto this sm_86 card — but no GPU forward runs here, so it's irrelevant.)

# [done] 1B checkpoint registry: configs/checkpoints_download.yaml `trainable` group now lists
#        vjepa2_1_vitg_384.pt (was commented) → setup_env on Box-B/C will fetch the 1B backbone.

# audit: confirm nothing hardcodes the 2B dim (1664) on a path the 1B (1408) walks — dims must come
# from the model config, not literals. Expect ZERO hits in src/ + arm configs (matches are OK in
# comments / the 2B model yaml only):
grep -rnE "1664|1408" src/ configs/train/ configs/model/ | grep -viE "#|vjepa2_1\.yaml|vjepa2_1_vitg\.yaml"

# code gate (CPU — works despite the broken FA2 here): compile + lint touched files
python -m py_compile $(git ls-files 'src/**/*.py' 'scripts/**/*.py')
ruff check --select F,E9 src/ scripts/

# push the registry edit + any 1B fixes to GitHub (you run git_push.sh)
bash git_push.sh "iter18 1B replication: add vjepa2_1_vitg to trainable ckpt group + runbook_v5 (3-box plan)"
```

---

## 🟠 Box B · 1× RTX 6000 96 GB — pretrain SEED  ·  ⏭️ SKIP by default (seed already exists & matches)

```bash
# ⏭️ SKIP THIS WHOLE BOX by default. The 1B pretrain seed already exists and matches the POC recipe
# (epochs=2 == max_epochs.poc, best_ckpt_metric=future_l1) at
# outputs/poc/vjepa_2_1_vitg_1B/train/m09a_pretrain_encoder/ → Box C's --cache 1 reuses it, and the
# SANITY runs on Box C (step 1). Run Box B ONLY to re-train the seed from scratch (e.g. recipe changed).
#
# env — Blackwell, so --from-wheels (sm_120 prebuilt FA2/FAISS) IS correct here
bash setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu_$(date +%Y%m%d_%H%M%S).log
source venv_walkindia/bin/activate

# data (~21 GB) + the 1B ckpt (now in 'trainable' → setup already fetched it; this is a safety re-check)
python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 | tee logs/download_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log
ls -la checkpoints/vjepa2_1_vitg_384.pt

# (1) SANITY — validate the WHOLE 1B code path on a tiny subsample (~5-10 min). Green → proceed.
ITER18_BACKBONE=vjepa_2_1_vitg \
python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 1 --cache 2 \
2>&1 | tee logs/iter18_ngpu_sanity_1B_$(date +%Y%m%d_%H%M%S).log

# (2) pretrain SEED — POC train ONLY pretrain_encoder (--only skips eval + §3 finale). ~2-2.5 h on 1×
# (1B ≈ half the 2B's ~4.5 h). Produces outputs/poc/vjepa_2_1_vitg_1B/train/<pretrain>/{student_encoder,m09a_ckpt_best}.pt
ITER18_BACKBONE=vjepa_2_1_vitg \
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 1 --cache 1 --only pretrain_encoder \
2>&1 | tee logs/iter18_ngpu_poc_1B_pretrain_$(date +%Y%m%d_%H%M%S).log

# upload the seed so Box-C can resume-skip pretrain (additive, whole-folder)
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/vjepa_2_1_vitg_1B/**" --exclude "**/.*" \
2>&1 | tee logs/upload_outputs_poc_1B_seed_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔵 Box C · 4× RTX 6000 96 GB (Blackwell) — train kept arms + eval kept encoders
<!-- full DAG = 263 jobs (21 train + 242 eval, 22 enc). MONEY-SAVER (hero-covering, NOT overtuned) → 167 jobs -->
<!-- (13 train + 154 eval, 14 enc): keep flagship base + intervene (future-MSE/causal hero, wiseft base) + diheavy -->
<!-- (mask-ratio hero); wiseft f30/f50/f70 stay (eval-only, free). --skip-arms drops the 5 always-skip + noDI/tccaux/replay25. See jobs.md. -->


```bash
bash setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu_$(date +%Y%m%d_%H%M%S).log
source venv_walkindia/bin/activate

# data + 1B ckpt + the EXISTING pretrain seed (just the 1B tree ~22 GB — NOT the 491 GB 2B outputs/poc).
# --cache 1 (step 2) resume-skips pretrain (seed present + matches) + any surgery arms whose .pt downloaded.
python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 | tee logs/download_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log
python -u src/utils/hf_outputs.py download-data outputs/poc/vjepa_2_1_vitg_1B 2>&1 | tee logs/download_outputs_poc_1B_$(date +%Y%m%d_%H%M%S).log
ls -la checkpoints/vjepa2_1_vitg_384.pt

# MONEY-SAVER --skip-arms (hero-covering, NOT overtuned): drop the 5 always-skip arms + the 3 non-hero
# surgery variants (noDI/tccaux/replay25). KEEP flagship + intervene (future-MSE/causal hero + wiseft base)
# + diheavy (mask-ratio hero); wiseft f30/f50/f70 stay (eval-only, free). → 167 jobs (was 263). See jobs.md.
SKIP="surgery_3stage_DI_head surgery_noDI_head cassle_encoder ewc_encoder surgical_3stage_DI_wiseft_encoder surgery_noDI_encoder surgery_3stage_DI_tccaux_encoder surgery_3stage_DI_replay25_encoder"

# (1) SANITY — fresh-node re-validate (multi-GPU wiring), ~5-10 min
ITER18_BACKBONE=vjepa_2_1_vitg \
python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 2 --skip-arms $SKIP \
2>&1 | tee logs/iter18_ngpu_sanity_1B_$(date +%Y%m%d_%H%M%S).log

# (2) POC — --cache 1 resume-skips the pretrain seed → trains the kept arms (init from pretrain's
# m09a_ckpt_best.pt) + evals all kept encoders + §3 finale. EVAL_CORPUS defaults to eval_10k.
# ~0.3-0.6 day on 4× (reduced roster; 1B ≈ half the 2B wall).
ITER18_BACKBONE=vjepa_2_1_vitg \
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP \
2>&1 | tee logs/iter18_ngpu_poc_1B_$(date +%Y%m%d_%H%M%S).log

# live status (separate pane — ITER18_BACKBONE must match or all cells read pending)
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/iter18_poc_status.py --log logs/iter18_ngpu_poc_1B_<ts>.log

# heads persist INLINE — run_eval.sh defaults KEEP_PROBE_HEADS=1 (since 2026-06-20), so the normal
# eval above SAVES every read-off head as it scores: action probe.pt + taxonomy probe_<dim>.pt
# (Stage 11, --keep-probe-heads) + encoder-temporal head_{aot,tov,pace}.pt (Stage 8c, --keep-heads).
# => NO separate --taxheads-only / --etheads-only regen pass. That v4 retrofit (16h+12h) was ONLY
# because the June-14 2B run predated the default-ON flip and finished without saving the heads.
# verify the heads landed (per-encoder):
find outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_taxonomy -name 'probe_*.pt' | head
find outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/encoder_temporal -name 'head_*.pt' | head

# (3) generate the metrics_watch GOAL files (--plots → m13 --metrics-watch under the 1B backbone).
# ITER18_SKIP_ARMS="$SKIP" → m13 HIDES the 8 non-roster arms from the FIGURES (so the 1B eval_scorecard
# has NO N/A bars + the clean 14-encoder roster); the csv/json still keep every arm. The scheduler's
# --skip-arms drops them from the DAG but does NOT export this env, so it must be set here for the plots.
ITER18_BACKBONE=vjepa_2_1_vitg ITER18_SKIP_ARMS="$SKIP" python -u scripts/iter18_poc_status.py --plots --log logs/iter18_ngpu_poc_1B_<ts>.log
ls outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/   # ← the goal

# (4) back up to HF (additive, whole-folder) + verify
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/vjepa_2_1_vitg_1B/**" --exclude "**/.*" \
2>&1 | tee logs/upload_outputs_poc_1B_$(date +%Y%m%d_%H%M%S).log
python -u src/utils/hf_outputs.py verify outputs/poc 2>&1 | tee logs/verify_outputs_poc_$(date +%Y%m%d_%H%M%S).log
```

---

## ⏱️ Durations / hardware

```text
┌────┬──────────────────────────────┬──────────────────────────────────────────────┬──────────────┐
│ box│ GPU                          │ work                                          │ wall          │
├────┼──────────────────────────────┼──────────────────────────────────────────────┼──────────────┤
│ A  │ 1× RTX 3060 12 GB (cheap)    │ code mod + audit + push (NO training; FA2 N/A)│ minutes       │
│ B  │ 1× RTX 6000 96 GB (SKIPPED)  │ SKIP — 1B pretrain seed exists & matches POC  │ —            │
│ C  │ 4× RTX 6000 96 GB (Blackwell)│ SANITY → POC (--cache 1 reuses seed; trains   │ ~0:10+~0.3-0.6d│
│    │                              │ kept arms + 14-encoder eval) → metrics_watch  │              │
└────┴──────────────────────────────┴──────────────────────────────────────────────┴──────────────┘
# Box B (the serial-seed box) is SKIPPED: the 1B pretrain seed already exists at the matching POC recipe
# (epochs=2) → Box C reuses it via --cache 1 and fans the remaining arms + eval 4-wide. Run Box B ONLY
# to re-train the seed from scratch (e.g. the pretrain recipe changed).
```
