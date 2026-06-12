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
ITER18_BACKBONE=$BB PT_H_MEMO=1 python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms $SKIP 2>&1 | tee logs/iter18_ngpu_poc_${BB}_$(date +%Y%m%d_%H%M%S).log
# banner MUST show: backbone=$BB · [resume --cache 1] skipping 9 already-trained arms + ~60 Stage-8b jobs

# watch panes (8c shows as ·8c d✓r▶/4 in the eval cells)
watch -n60  "ITER18_BACKBONE=$BB ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_status.py"
watch -n300 "ITER18_BACKBONE=$BB ITER18_SKIP_ARMS=\"$SKIP\" python -u scripts/iter18_poc_metrics.py"
```

## 2 · m12f (8c) SANITY smoke — run once per fresh box, BEFORE the POC

```bash
SKIP_STAGES="2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13" CACHE_POLICY_ALL=1 \
bash scripts/run_eval.sh --sanity --encoders vjepa_2_1_frozen \
2>&1 | tee logs/m12f_sanity_smoke_$(date +%Y%m%d_%H%M%S).log
# MUST show: "[tcc] pair-chunk auto → N" + 4 files outputs/sanity/encoder_temporal/<enc>/aggregate_{aot,tov,pace,tcc}.json
# crash → re-run same command (.m12f_ckpt resumes)
```

## 3 · after the finale — upload to HF

```bash
python -u src/utils/hf_outputs.py upload-full outputs/ 2>&1 | tee logs/upload_full_outputs_$(date +%Y%m%d_%H%M%S).log
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
