# iter18 — Runbook · baseline SANITY (48 GB) → POC/FULL (96 GB) · 2B ViT-G

## 0 · POC restart + eval speedup (2026-06-07) — §0.C park-CaSSLe restart · §0.D h-memo smoke

### 0.C · mid-POC restart (2026-06-07) — park CaSSLe (213.8 s/step measured ≈ 21 h solo straggler), CPU-set pinning ON

```bash
# C1 · interrupt: Ctrl-C the scheduler tmux. Hourly anchors bound the loss:
#      dora/ewc resume from m09c_ckpt_latest.pt (≤1 h redo each); 10 ✅ arms skip entirely.
# C2 · relaunch without cassle — its hourly anchor (m09d_cassle_encoder/m09c_ckpt_latest.pt) stays parked:
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 --skip-arms cassle_encoder 2>&1 | tee logs/iter18_ngpu_poc_$(date +%Y%m%d_%H%M%S).log
# first lines MUST print:  [--skip-arms] dropped ['cassle_encoder'] (train+eval+8b-metrics)
#                          [cpuset] GPU slots pinned: GPU0→0..79(32t) ...
#                          [resume --cache 1] skipping 10 already-trained arms
#                          ═══ ... 103 jobs (12 train + 91 eval) ═══   # 91 = 13 enc × (1 E: + 6 P:)
# resumed arm logs MUST print:  "Resumed from step N"  +  "cores=32 (pinned cpuset)"

# C3 · watch (separate panes; status = state/ETA + 45-min HF backup, metrics = numbers + graphs)
watch -n60  'python -u scripts/iter18_poc_status.py'
watch -n300 'python -u scripts/iter18_poc_metrics.py'

# C4 · LATER (post-finale, off the deadline): finish cassle + rebuild the full 14-encoder finale.
python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 2>&1 | tee logs/iter18_ngpu_poc_$(date +%Y%m%d_%H%M%S).log
```

### 0.D · #3 h-memo parity smoke (first free GPU; gates PT_H_MEMO on — #1/#2 already live via §0.C restart)

```bash
# D1 · frozen tdist, off vs on, throwaway dirs → MUST allclose (exit 0)
for m in 0 1; do PT_H_MEMO=$m CUDA_VISIBLE_DEVICES=0 python -u src/m12e_predictor_temporal.py --POC --stage forward --metric tdist --variant vjepa_2_1_frozen --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt --action-probe-root outputs/poc/probe_action --local-data data/eval_10k_local --output-root outputs/poc/_hmemo_smoke_$m --cache-policy 2 --no-wandb 2>&1 | tee logs/hmemo_smoke_tdist_${m}_$(date +%Y%m%d_%H%M%S).log; done
python -c "import numpy as np,sys; a=np.load('outputs/poc/_hmemo_smoke_0/vjepa_2_1_frozen/per_clip_tdist.npy'); b=np.load('outputs/poc/_hmemo_smoke_1/vjepa_2_1_frozen/per_clip_tdist.npy'); print('max|Δ|=',abs(a-b).max()); sys.exit(0 if np.allclose(a,b,atol=1e-4) else 1)"
# D2 · only if D1 exits 0 → enable h-memo for new evals (next P: launches pick it up; no restart)
echo 'PT_H_MEMO=1' >> ${WORKSPACE}/.env
```
