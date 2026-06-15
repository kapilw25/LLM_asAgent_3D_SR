# iter18 — Runbook · 2B ViT-G (1B ⏸️ PARKED 2026-06-12; resume = same commands with BB=vjepa_2_1_vitg)

## 1 · main run — SANITY → POC (trains skipped on resume; remaining work = 40 F: 8c jobs + autorgn 3+11 + finale)

```bash
BB=vjepa_2_1_vitG
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head"   # cassle/ewc never trained; heads = pretrain's frozen encoder/predictor → 0 new info

# pre-reqs
ls -lh "checkpoints/$(echo "$BB" | sed 's/vjepa_2_1_/vjepa2_1_/')_384.pt"
test ! -e logs/.eval_extra_skip && echo "OK: no extra-skip" || rm -f logs/.eval_extra_skip
ls data/eval_10k_local/test_split.json >/dev/null && echo "OK: eval data"

# 1) SANITY (code-path validator, ~minutes)
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_sanity_${BB}_$(date +%Y%m%d_%H%M%S).log

# 2) POC (--gpus 4 on the 96 GB box · --gpus 1 works serially on the 24 GB box)
# Confirm the 9 POC-skip arms are all present:
find outputs/poc -name student_encoder.pt | wc -l    # expect 12
rm -rf outputs/sanity
# If all 9 print a path, you're clear to start POC the moment SANITY goes green.

BB=vjepa_2_1_vitG
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" 
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_${BB}_$(date +%Y%m%d_%H%M%S).log
# banner MUST show: backbone=$BB · [resume --cache 1] skipping 9 already-trained arms + ~60 Stage-8b jobs



# watch panes (8c shows as ·8c d✓r▶/4 in the eval cells)
watch -n60 'ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_status.py'
# CONSOLIDATED refresh — status + EVERY metrics_watch figure (3 base + WiSE-FT sweep + paper scorecard + TCC) + {train,eval}_metrics.{json,csv} in ONE command:
ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_status.py --plots
# (legacy figure-only refresh — retired after the --plots path is verified; m13 now owns these plots:)
ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_metrics.py
```

## 1b · repeat for the 1B (ViT-g) — SAME §1 commands, just `BB=vjepa_2_1_vitg`

```bash
BB=vjepa_2_1_vitg     # ✅ lowercase 'g' = 1B ViT-g  (vitG = 2B). Then run §1 verbatim: pre-reqs → SANITY → POC.
# pre-req base ckpt: checkpoints/vjepa2_1_vitg_384.pt   (the §1 ls line already derives this from $BB)
# resume state: the 1B already has pretrain + surgery_3stage_DI/noDI/raw + autorgn (5 arms) → --cache 1 SKIPS them;
#   it TRAINS the 8 missing (full_ft, lpft, peft_lora, peft_dora + replay25, diheavy, tccaux, intervene),
#   merges wiseft, evals all 13.
SKIP="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head"
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_sanity_${BB}_$(date +%Y%m%d_%H%M%S).log
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC    --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_${BB}_$(date +%Y%m%d_%H%M%S).log
# watch — note the lowercase g in the env:
watch -n60 'ITER18_BACKBONE=vjepa_2_1_vitg ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_status.py'

# ✅ RUN THE FULL GRID — do NOT skip the 5 improvement arms. At 2B they are NOT null: CI-CLEAR wins over base
#    surgery — intervene/tccaux on future-MSE, diheavy/tccaux on mask-ratio, wiseft on aot + tcc_cycle (recovers
#    frozen's coherence = its design goal). The 1B tests whether these gains GENERALIZE across scale = the
#    stronger paper claim, so the marginal extra compute is worth it.
```

## 2 · m12f (8c) SANITY smoke — run once per fresh box, BEFORE the POC

```bash
SKIP_STAGES="2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen \
2>&1 | tee logs/m12f_sanity_smoke_$(date +%Y%m%d_%H%M%S).log
# MUST show: "[tcc] pair-chunk auto → N" + 4 files outputs/sanity/encoder_temporal/<enc>/aggregate_{aot,tov,pace,tcc}.json
# crash → re-run same command (.m12f_ckpt resumes)
```

## 3 · upload to HF — light mirror (run it DURING the POC, then once more after the finale)

```bash
# Mirror upload (deletes files on HF which do not exist on disk)
HF_UPLOAD_MODE=reuse python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log

# raw additive → overwrites same-path files, keeps remote-not-local (your ckpts survive). Neither shrinks the 3 TB history.
# upload-large-folder is parallel + truly resumable (re-run on any drop = it picks up where it left off).
set -a; . .env; set +a
hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
--include "outputs/poc/**" --exclude "**/.*"   2>&1 | tee logs/upload_large_folder_outputs_poc_$(date +%Y%m%d_%H%M%S).log

# light mirror: every file incl. resume anchors · no tars · xet dedups against the tar shards already on HF, so much less than 338G actually transfers
# run #1 mid-POC (overlaps the run = $0 extra) · run #2 after the finale = delta only, minutes → kill the box right after
# before killing: one last delta pass (~3-5 min, mostly dedup) for the finale's last files
HF_UPLOAD_MODE=reuse python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 | tail -5
# prints "Upload complete" → kill the box (verify-full FAIL vs the tar manifest = expected; the light mirror uploads loose files it doesn't count)

# One caveat tied to your actual goal: additive does NOT shrink the 3 TB — it adds a commit, so history keeps growing. To reclaim the 3 TB you still need to collapse history after the upload:

python -c "from huggingface_hub import HfApi, os; 
HfApi(token=os.environ['HF_TOKEN']).super_squash_history('anonymousML123/factorjepa-outputs', repo_type='dataset')"

# super_squash keeps the current tree (every file now in the repo, including your ckpt) and discards only the old commit versions — that's what frees the storage.
```

## 4 · ⏱️ measured durations (2026-06-12 unless noted)

| op | wall in (hours:min) |
|---|---:|
| download-data eval_10k_local (20.9 GB) | 0:05 |
| download-full outputs 497 GB (run-1 crashed 125/222 + run-2 resume/unpack) | 1:29 |
| upload-full outputs (338 GB) | 2:28 |
| m12f SANITY smoke | 0:07 |
| 11 HF model-repo pushes (xet dedup) | 0:09 |
| E: per-encoder eval, 4×96 GB (06-08): median | 1:40 |
| · raw 2:27 · full_ft 1:54 · dora 1:45 · noDI 1:42 | |
| · lora 1:39 · 3DI_head 1:35 · lpft 1:30 · noDI_head 1:11 | |
| · frozen 3:21 (monolithic) · autorgn 0:17 (truncated) | |
| F: 8c job, 1×24 GB (aot measured; pace ≈ 2-3×) | 1:30–2:15 |
| REMAINING: 40 F: + autorgn 3+11 + finale — 4×96 GB | ~10:00 |
| REMAINING: same on 1×24 GB serial | ~72:00 |

### 4b · ⏱️ TRAIN durations (measured 2026-06-07 · steps = training_summary.json · s/step = log progress bars)

| arm (m09 module) | steps | s/step | train wall |
|---|---:|---:|---:|
| pretrain (m09a · 2 ep · serial prefix, runs solo) | 482 | ~27 | ~3:30 |
| surgery 3stage_DI / noDI / raw (m09c1) | 480 | ~65 | ~8:40 |
| full_ft (m09f) | 438 | ~52 | ~6:20 |
| peft_lora (m09b) | 438 | ~72 | ~8:45 |
| peft_dora (m09b) · lpft (m09f) | 438 / 481 | restart-inflated | ~6–9 (≈ recipe) |

- s/step is CONTENTION-bound: ~27 solo → ~52–70 at 4-way → 100–140 at 6-arm peaks (the NGPU_CONCURRENCY tax). So MORE concurrent arms ⇒ slower per-step ⇒ wall scales sub-linearly with GPU count.
- dora/lpft raw log-spans read 12–18h, but that's restart + peak-contention idle gaps; same recipe as surgery ⇒ real ≈ 6–9h.
- 5 NEW arms: replay25 / diheavy = 480 steps (≈ OURS ~8:40 at 4-way) · tccaux ≈ +5% · intervene ≈ ×1.3 (3rd mask) ≈ ~11h · wiseft = post-hoc merge, ~10 min (no training).

