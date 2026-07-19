# 🎬 iter20 runbook

> 👉 **RUN ONLY the ACTIVE section below** — that is the current path: the **VLM-head build (§E1)**, on the **96 GB box**.
> Everything under **🗄️ ARCHIVE** is from two abandoned approaches (pixel-decoder triptych · YouTube-clip VQA).
> Those commands are **commented out (`#`) so they cannot run** — kept only for reference, delete when you're sure.

---

# ══ ▶️ ACTIVE · §E1 — VLM-head build (OURS-VLM vs FROZEN-VLM) · EARLY-GATE FIRST ═══════════════
# RUNS ON THE 96 GB BOX (Qwen3-8B + V-JEPA-G) — NOT the 3060. Precondition: §E0 passed (it did).
# spec: plan_E1_vlm_build.md · CODE BUILT + 3060-verified (2026-07-19): configs/vlm.yaml +
#   src/utils/vlm_model.py + src/m18_vlm_{data,train,eval}.py. Deps already present on the demo box
#   (transformers 5.5.4 / peft 0.19 / accelerate / datasets) — VERIFY they exist on the 96GB /venv/main;
#   if not: add to setup_env_uv.sh + requirements_gpu.txt, install via setup_env_uv.sh ONLY.
# DATA = only HF-reachable: lmms-lab/TempCompass (gate, 4033 temporal MC-QA) + lmms-lab/LLaVA-Video-178K
#   (align captions + instruct motion-QA). SSv2/EK100 are gated/manual → excluded. TempCompass is GENERAL
#   video → the early gate IS the OOD-transfer test (the crux risk).
```bash
source /venv/main/bin/activate ; set -a; source ${WORKSPACE}/.env; set +a ; set -o pipefail

# 0 · DATA — build JSONL + download the videos (heavy: LLaVA-Video tars + TempCompass zip)
for S in gate instruct align; do
  python src/m18_vlm_data.py --config configs/vlm.yaml --stage $S --download-videos --cache-policy 1 \
    2>&1 | tee logs/m18_data_${S}_$(date +%Y%m%d_%H%M%S).log
done

# 1 · EARLY GATE (cheap de-risk BEFORE the multi-day pretrain): instruct-only, capped, both arms →
#     eval a 60-video TempCompass subset. NO align needed (projector from scratch is fine to read the gap).
for ARM in frozen ours; do
  python src/m18_vlm_train.py --config configs/vlm.yaml --arm $ARM --stage instruct --max-samples 3000 --no-wandb \
    2>&1 | tee logs/m18_early_train_${ARM}_$(date +%Y%m%d_%H%M%S).log
  python src/m18_vlm_eval.py  --config configs/vlm.yaml --stage eval --arm $ARM --early \
    2>&1 | tee logs/m18_early_eval_${ARM}_$(date +%Y%m%d_%H%M%S).log
done
python src/m18_vlm_eval.py --config configs/vlm.yaml --stage gate --early \
  2>&1 | tee logs/m18_early_gate_$(date +%Y%m%d_%H%M%S).log
# → outputs/demo/vlm/gate_report_early.json. ⛔ FAIL (OURS not > FROZEN by ≥2pp) → STOP. Truthful negative.
#    Do NOT run step 2. ✅ PASS → the edge transfers to visible temporal QA → proceed.

# 2 · FULL 2-stage (ONLY if early gate PASSED) — align (projector) → instruct (projector+LoRA), both arms
for ARM in frozen ours; do
  python src/m18_vlm_train.py --config configs/vlm.yaml --arm $ARM --stage align    --no-wandb \
    2>&1 | tee logs/m18_align_${ARM}_$(date +%Y%m%d_%H%M%S).log
  python src/m18_vlm_train.py --config configs/vlm.yaml --arm $ARM --stage instruct --no-wandb \
    2>&1 | tee logs/m18_instruct_${ARM}_$(date +%Y%m%d_%H%M%S).log
done

# 3 · FULL GATE — non-overlapping 95% CIs on the full TempCompass temporal suite
for ARM in frozen ours; do
  python src/m18_vlm_eval.py --config configs/vlm.yaml --stage eval --arm $ARM \
    2>&1 | tee logs/m18_eval_${ARM}_$(date +%Y%m%d_%H%M%S).log
done
python src/m18_vlm_eval.py --config configs/vlm.yaml --stage gate \
  2>&1 | tee logs/m18_gate_$(date +%Y%m%d_%H%M%S).log
# → gate_report_full.json + heroes_vlm_full.json. ⛔ FAIL → forest plots (do NOT render).

# 4 · RENDER (only if FULL gate PASSED) — reuse m17 with the VLM heroes
python src/m17_vqa_demo.py --heroes outputs/demo/vlm/heroes_vlm_full.json \
  --config configs/demo.yaml --output-dir outputs/demo/vlm \
  2>&1 | tee logs/m17_vlm_$(date +%Y%m%d_%H%M%S).log
# → outputs/demo/vlm/demo_vqa_vlm.mp4 ; run the visual-audit agent (C10 gate-backed) before showing it
```

---

# ══ 🗄️ ARCHIVE — SUPERSEDED · DO NOT RUN (commands commented out; kept for reference) ══════════
#
# ── A · m15 pixel-decoder + triptych (m15/m14) ──────────────────────────────────────────────────
#   ABANDONED: decoded V-JEPA latents came out as fog → OURS-vs-FROZEN sub-perceptual (VM29).
#   Findings preserved in plan_v2 App D + memory. The Prof.'s pixel track (App D) uses SDXL/Cosmos, NOT this.
#
# cd /workspace/factorjepa && source venv_walkindia/bin/activate ; set -o pipefail
# # 1 · FROZEN feature precompute:
# PYTHONPATH=src python -u src/m15_pixel_decoder.py --stage precompute \
#   --ckpt checkpoints/vjepa2_1_vitg_384.pt \
#   --model-config configs/model/vjepa2_1_vitg.yaml --demo-config configs/demo.yaml \
#   --tars data/demo_src/data/train-00000.tar data/demo_src/data/train-00025.tar \
#   --work-dir outputs/demo/m15_frozen 2>&1 | tee logs/m15_precompute_frozen_$(date +%Y%m%d_%H%M%S).log
# # 2 · train FROZEN decoder + decode-sanity GATE:
# PYTHONPATH=src python -u src/m15_pixel_decoder.py --stage train      ...  --work-dir outputs/demo/m15_frozen
# PYTHONPATH=src python -u src/m15_pixel_decoder.py --stage decode-sanity ... --work-dir outputs/demo/m15_frozen
# # 3 · same 3 steps for OURS diheavy (ckpt = m09c_ckpt_best.pt, --work-dir outputs/demo/m15_ours)
# # 4 · triptych: PYTHONPATH=src python -u src/m14_metric_demo.py --scenes W \
# #     --ckpt "FROZEN 2.1=..." --ckpt "OURS diheavy=..." --decoder "FROZEN 2.1=.../decoder.pt" \
# #     --decoder "OURS diheavy=.../decoder.pt" --clips-dir data/demo_clips_humans --output-dir outputs/demo/metric_visual
#
# ── B · YouTube-clip VQA (cheap-probe path) ─────────────────────────────────────────────────────
#   ABANDONED: the cheap-probe/VQA demo is EXHAUSTED (plan_v2 §2 — motion not eye-verifiable on WalkIndia).
#   The VLM (§E1) sources its videos from HF (TempCompass + LLaVA-Video), NOT YouTube. This is dead.
#
# source venv_walkindia/bin/activate ; set -o pipefail ; mkdir -p data/youtube_demo
# yt-dlp --cookies data/youtube_demo/cookies.txt "https://www.youtube.com/watch?v=<VIDEO_ID>" \
#   -f "bv*[height<=720]+ba/b[height<=720]" --download-sections "*0:05-0:20" --force-keyframes-at-cuts \
#   --no-playlist -o "data/youtube_demo/q<NN>_%(id)s.%(ext)s" 2>&1 | tee logs/ytdl_q<NN>_$(date +%Y%m%d_%H%M%S).log
# # (on your OWN machine: swap --cookies for --cookies-from-browser chrome)
