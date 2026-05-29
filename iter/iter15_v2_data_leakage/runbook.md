# iter15-v2 — Runbook: 7-cell paired-Δ + eval (SANITY → POC)

Order: SANITY (validate code paths) → POC (paper-proxy numbers). Each block runs 7
training cells then `run_eval`. Surgery cells init from the LOCAL per-mode
`m09a_pretrain_encoder` ckpt (pipeline.yaml `surgery_init`), so `pretrain_encoder`
MUST run first. `sleep 10` between cells lets the prior GPU process release VRAM/RAM.

## 1 · SANITY

```bash
# ── 7 training + 1 eval (--SANITY) ──
# ENCODER
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder          --SANITY 2>&1 | tee logs/sanity_a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X_encoder       --SANITY 2>&1 | tee logs/sanity_a1_pretrain_2X_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --SANITY 2>&1 | tee logs/sanity_c1_surgery_3stage_DI_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --SANITY 2>&1 | tee logs/sanity_c1_surgery_noDI_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
# HEAD
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_head             --SANITY 2>&1 | tee logs/sanity_a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head    --SANITY 2>&1 | tee logs/sanity_c2_surgery_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head         --SANITY 2>&1 | tee logs/sanity_c2_surgery_noDI_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
# eval / TEST of all
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --SANITY 2>&1 | tee logs/iter15_v2_sanity_eval_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── VERIFY (--SANITY) — all green before POC ──
grep -iE "FATAL|Traceback|KeyError|AttributeError|invalid choice" logs/sanity_*.log logs/iter15_v2_sanity_eval_*.log   # MUST be EMPTY
grep -hE "recipe-v3 receipts"            logs/sanity_c1*.log                       # surgery encoders: recipe_v3 (DI+noDI)
grep -hE "leakage-guard.*train pool"     logs/sanity_c2*.log                       # surgery heads: streaming + leakage filter
grep -hE "variant=(3stage_DI|noDI)_head" logs/sanity_c2*.log                       # head data.variant_tag read
grep -hE "val_loss=|val_jepa="           logs/sanity_a2*.log logs/sanity_c2*.log   # >=1 val cycle ran (shared compute_val_motion_aux_loss)
```

## 2 · POC

```bash
# ── 7 training + 1 eval (--POC) ──
# ENCODER
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder          --POC 2>&1 | tee logs/poc_a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X_encoder       --POC 2>&1 | tee logs/poc_a1_pretrain_2X_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --POC 2>&1 | tee logs/poc_c1_surgery_3stage_DI_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --POC 2>&1 | tee logs/poc_c1_surgery_noDI_encoder_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
# HEAD
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_head             --POC 2>&1 | tee logs/poc_a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head    --POC 2>&1 | tee logs/poc_c2_surgery_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head         --POC 2>&1 | tee logs/poc_c2_surgery_noDI_head_$(date +%Y%m%d_%H%M%S).log ; \
sleep 10 ; \
# eval / TEST of all
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC 2>&1 | tee logs/iter15_v2_poc_eval_$(date +%Y%m%d_%H%M%S).log
```

```bash
# ── VERIFY (--POC) — leakage-safe + paper Δ5 ──
source venv_walkindia/bin/activate
LD=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)
# (a) zero errors across all 7 training + eval
grep -iE "FATAL|Traceback|KeyError|AttributeError|invalid choice" logs/poc_*.log logs/iter15_v2_poc_eval_*.log   # MUST be EMPTY
# (b) train pool leakage-free — zero val/test clips in the training pool (the reason v2 exists)
python -c "import json; P=lambda f: set(json.load(open(f))['clip_keys']); pool=P('$LD/train_pool.json'); v=P('$LD/val_split.json'); t=P('$LD/test_split.json'); assert not(pool&v) and not(pool&t), 'LEAK'; print(f'leakage-free: pool={len(pool)} val_in_pool=0 test_in_pool=0')"
# (c) universe-symmetry — pretrain (m09a) + surgery (m09c) train on the SAME pool size
grep -hE "Train clips:|train/val split:|universe=broad_manifest" logs/poc_*.log
# (d) all 7 cells produced their best ckpts
ls outputs/poc/m09a_pretrain_encoder/m09a_ckpt_best.pt \
   outputs/poc/m09a_pretrain_2X_encoder/m09a_ckpt_best.pt \
   outputs/poc/m09a_pretrain_head/m09a_ckpt_best.pt \
   outputs/poc/m09c_surgery_3stage_DI_encoder/m09c_ckpt_best.pt \
   outputs/poc/m09c_surgery_noDI_encoder/m09c_ckpt_best.pt \
   outputs/poc/m09c_surgery_3stage_DI_head/m09c_ckpt_best.pt \
   outputs/poc/m09c_surgery_noDI_head/m09c_ckpt_best.pt
# (e) headline Δ5 = surgery_3stage_DI_encoder − surgery_3stage_DI_head (want non-overlapping CI)
python -c "import json; d=json.load(open('outputs/poc/probe_action/probe_paired_delta.json'))['iter14_paper_deltas'].get('delta_5_surgical_vs_surgical_head'); print('Δ5 unavailable — cells incomplete') if not d or d.get('skipped') else print(f'Δ5 {d[\"delta_mean\"]:+.4f}  95% CI [{d[\"delta_ci_lo\"]:+.4f},{d[\"delta_ci_hi\"]:+.4f}]  p={d[\"p_value\"]:.4f}')"
```
