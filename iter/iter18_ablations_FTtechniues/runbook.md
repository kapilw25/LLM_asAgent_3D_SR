# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

## 0 · 2-box pipeline (2026-06-06) — §0.A on the 1× box NOW · §0.B on the 4× box in PARALLEL · §0.C handoff

### 0.A · 1× box (this box) — kill invalid queue → pin code → POC the pretrain root

```bash
# A1 · SANITY only the arm THIS box will POC (~6 min; the other 12 arms gate on the 4× box via §0.B ngpu SANITY)
set -o pipefail && export BACKBONE=vjepa_2_1_vitG && \
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | tee logs/sanity_pretrain_$(date +%Y%m%d_%H%M%S).log

## verification
grep -iE "FATAL|Traceback|KeyError|OutOfMemory" logs/sanity_pretrain_202606*.log   # MUST be EMPTY

# A2 · POC the DAG root (~5.6 h measured 06-05) — all 12 other arms init from its m09a_ckpt_best.pt.
#      --only pretrain_encoder = train job only, no eval jobs, no §3 finale (evals run on the 4× box)
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 1 --cache 2 --only pretrain_encoder 2>&1 | tee logs/iter18_ngpu_poc_only_pretrain_$(date +%Y%m%d_%H%M%S).log
# arm output streams to the per-job log the scheduler prints at launch:
grep -E "probe-trio" logs/iter18_ngpu_poc_train_pretrain_encoder_*.log | tail -6   # MUST show (N=451), NOT (N=1000)

# A3 · ship m09a (14 G) + label jsons to HF. upload-full, NOT `upload` — light upload DROPS m09a_ckpt_best.pt
python -u src/utils/hf_outputs.py upload-full outputs/ 2>&1 | tee logs/upload_full_outputs_sanity_poc_$(date +%Y%m%d_%H%M%S).log

# A4 · verify EVERY file (sanity + poc) made it — compares the local inventory file-by-file against the
#      uploaded _full-manifest.json + checks every _full-*.tar shard exists on HF byte-identical;
#      exits 1 and lists the exact missing files on ANY gap
python -u src/utils/hf_outputs.py verify-full outputs/ 2>&1 | tee logs/verify_full_outputs_$(date +%Y%m%d_%H%M%S).log
grep "VERIFY-FULL: PASS" logs/verify_full_outputs_*.log   # MUST print PASS (file + shard counts)
```

### 0.B · 4× box — provision + gate NOW, in parallel with 0.A (do NOT wait for A5)

```bash
# B1 · provision: 4× RTX PRO 6000 96 GB class · ≥24 CPU cores (ngpu cpu-preflight wants gpus×6) · ≥300 G free
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa        # same commit as A2
mkdir -p logs && bash setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu_$(date +%Y%m%d_%H%M%S).log

# B2 · data (22 G)
python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 | tee logs/hf_outputs_download_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log



# B3 · pull the fresh pretrain seed; --cache 1 will then skip ONLY pretrain (student_encoder.pt marker)
python -u src/utils/hf_outputs.py download-full outputs/ 2>&1 | tee logs/hf_outputs_download_full_outputs_sanity_poc_$(date +%Y%m%d_%H%M%S).log
ls outputs/poc/vjepa_2_1_vitG/m09a_pretrain_encoder/student_encoder.pt \
   outputs/poc/vjepa_2_1_vitG/m09a_pretrain_encoder/m09a_ckpt_best.pt \
   outputs/poc/probe_action/action_labels.json outputs/poc/probe_taxonomy/taxonomy_labels.json   # all 4 MUST exist

# B4 · final SANITY re-gate of the 13-arm DAG immediately before the ~18 h POC commit (~40 min on 4 GPUs;
#        cache 2 forces real re-trains — cache 1 would skip-as-done off the §0.B SANITY ckpts. Includes the
#        6-min pretrain SANITY: ngpu has no arm-subset flag, and a fresh root re-gate is harmless.)
python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 2 2>&1 | tee logs/iter18_ngpu_sanity_regate_$(date +%Y%m%d_%H%M%S).log
grep -iE "FATAL|Traceback|KeyError|OutOfMemory|invalid choice" logs/iter18_ngpu_sanity_regate_*.log   # MUST be EMPTY

# B5 · POC: 12 arms (3 waves ≈ 15.6 h measured) + per-encoder evals pipelined onto freed GPUs + §3 m13 finale
#      ≈ 18 h → finale ≈ Sat 21:30-23:30 PDT. On any arm failure: fix → re-run SAME command (--cache 1 resumes survivors).
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 2>&1 | tee logs/iter18_ngpu_poc_$(date +%Y%m%d_%H%M%S).log
# first lines MUST print:  [resume --cache 1] skipping 1 already-trained arms: ['pretrain_encoder']

# B6 · verify
grep -iE "FATAL|Traceback|OutOfMemory|invalid choice" logs/iter18_ngpu_*.log    # MUST be EMPTY
find outputs/poc/vjepa_2_1_vitG -maxdepth 2 -name student_encoder.pt | wc -l    # = 13
```
