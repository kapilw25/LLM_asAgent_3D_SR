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
# iter18 (2026-06-13): the 5 NEW improvement arms (replay25 · diheavy · tccaux · intervene · wiseft) are now in
# ARM2ENC, so this SAME command ALSO trains + evals them (the 9 old encoder arms skip on --cache 1); the §3
# finale + scorecard include them automatically. No command change needed — only the new arms actually run.

# watch panes (8c shows as ·8c d✓r▶/4 in the eval cells)
watch -n60 'ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_status.py'
ITER18_BACKBONE=vjepa_2_1_vitG ITER18_SKIP_ARMS="cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head" python -u scripts/iter18_poc_metrics.py
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
HF_UPLOAD_MODE=reuse python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
# light mirror: every file incl. resume anchors · no tars · xet dedups against the tar
# shards already on HF, so much less than 338G actually transfers
# run #1 mid-POC (overlaps the run = $0 extra) · run #2 after the finale = delta only,
# minutes → kill the box right after
# NOT upload-full here: it packs ~330G of shards BEFORE uploading (252G free → disk-full
# crashes the live run) and the 08:58 tar snapshot already covers the final train arms

# before killing: one last delta pass (~3-5 min, mostly dedup) for the finale's last files
HF_UPLOAD_MODE=reuse python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 | tail -5
# prints "Upload complete" → kill the box (verify-full FAIL vs the tar manifest = expected;
# the light mirror uploads loose files it doesn't count)
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

## 5 · move outputs/poc/ instance→instance (skip the slow/costly HF download on a $$$ box)

```bash
# ⚠️ DON'T run this on your Mac: rsync rejects remote→remote, and routing 123 GB THROUGH the Mac's
# home uplink is slow + double-transfers. Orchestrate FROM the Mac but make data flow DIRECTLY box→box:
#   ssh-add ~/.ssh/id_ed25519              # load the key into the Mac's ssh-agent
#   ssh -A vast_RTXpro_4X_96gb             # land on the DEST (4X) with agent-FORWARDING (-A)
# The two instances DON'T hold each other's keys (each only has YOUR Mac pubkey), so without -A the
# pull below fails auth. -A lets the dest authenticate to the source using your forwarded agent.
#
# SMART subset (~123 GB = what the 5-new-arm run needs) — then run this PULL on the DEST (4X):
# keeps m09a_ckpt_best.pt (pretrain init) + every student_encoder.pt + eval caches; drops 211 GB of resume anchors.
SRC_IP=<source PUBLIC_IPADDR>;  SRC_SSH=<source VAST_TCP_PORT_22>   # the 5000 source = 75.63.212.140 / 42229
rsync -a --info=progress2 --partial \
  --exclude='*_ckpt_latest.pt' --exclude='*_ckpt_stage*.pt' --exclude='*_ckpt_step*.pt' \
  --exclude='m09c_ckpt_best.pt' --exclude='m09b_ckpt_best.pt' --exclude='m09d_ckpt_best.pt' \
  --exclude='m09e_ckpt_best.pt' --exclude='m09f_ckpt_best.pt' \
  -e "ssh -p $SRC_SSH -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com" \
  root@$SRC_IP:/workspace/factorjepa/outputs/poc/ /workspace/factorjepa/outputs/poc/

# FULL copy (332 GB) — drop the --exclude lines above.
# FASTEST streaming (no per-file overhead, NOT resumable) — run on the SOURCE, push to dest:
#   tar -C outputs/poc --exclude='*_ckpt_latest.pt' --exclude='*_ckpt_stage*.pt' -cf - . | \
#   ssh -p $DEST_SSH -c aes128-gcm@openssh.com root@$DEST_IP "tar -C /workspace/factorjepa/outputs/poc -xf -"

# data/ for POC = data/eval_10k_local (20 GB) — IS BOTH the eval data AND the factor-streaming source
# (m10 masks 5.6G + m11 factor sets 4.5G + m00d raw clips + m04d motion_features + splits). Transfer it too:
rsync -a --info=progress2 --partial \
  -e "ssh -p $SRC_SSH -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com" \
  root@$SRC_IP:/workspace/factorjepa/data/eval_10k_local/ /workspace/factorjepa/data/eval_10k_local/
# (m10/m11 hold many small .npz/.npy → rsync per-file overhead; for THIS dir a tar-pipe is faster)

# duration (single ssh stream, aes128-gcm): outputs/poc 123 GB ≈ 8–16 min  +  data 20 GB ≈ 2–5 min  (vs HF ~47 min + CDN stalls)
```
