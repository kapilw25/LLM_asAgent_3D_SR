# Instruction: Date & daily udpate in 2 line for daily progress && 1 line for next step. Each line covers max 15 words
# AUDIENCE: Research leads focussed on > ## 🎯 Paper goal:  `vjepa_surgery` [X_epochs(surgery) +X_epochs(pretrain)] ≫ `vjepa_pretrain` [2X epochs] ≫ `vjepa_frozen` on motion / temporal features
# AUDIENCE has knowledge about ONLY plots from @iter/iter15_trainHead_freezeEncoder/result_outputs/v15a/poc/probe_plot/eval/
## Note: do NOT write pipeline jargon (SANITY/POC/eval_10k/Δ5/cells),


## Wed, June 3, 2026
- Built and did SANITY testing on all eight finetuning baselines (surgical_autorgn, surgery_raw, full_ft, lpft, peft_lora, peft_dora, CaSSLe, EWC); every arm trains and evaluates cleanly.
- All  PREVIOUS four surgery variants join the same run — identical data, identical starting weights, fair duel.
- Next: full 10k-clip training of all thirteen arms; first surgery-vs-baseline numbers follow.

## Tue, June 2, 2026
- Surgery's edge holds on modern 2.1 bases; older 2.0's fragile pretrained dynamics explain its loss.
- Built FT baselines [ vjepa_2_1_encoders: surgical_autorgn, surgery_raw, full_ft, lpft, peft_lora, peft_dora, CaSSLe, EWC]
- Next: run those baselines, starting with the closest competitor, to lock the best-adaptation claim.

## Mon, June 1, 2026
- Compared surgery, pretrain, and frozen across three V-JEPA backbones (2B, newer 1B, older 1B).
- Surgery never loses to pretrain on 2B, leads newer 1B; older 2.0 favors pretrain.
- Next: Surgery vs finetuning baselines → best-adaptation claim

## Fri, May 29, 2026
- Ranked ten frozen encoders; no single one wins all three capabilities (action, motion, scene).
- V-JEPA 2.1 best action; DINOv2 best scene; iJEPA best motion — frozen bar set.
- Next: train smaller V-JEPA 2.1 and older V-JEPA 2.0; test surgery-beats-pretrain across scale/version.

## Thurs, May 28, 2026
- All six temporal-prediction metrics now in; surgery beats pretrain on four (motion, future-frame, causal, mask-robustness).
- Pretrain leads only weak action-classification; four metrics tie — surgery is the clear overall winner.
- Next: adding older V-JEPA and image baselines (DINOv2, iJEPA) as frozen references.

## Wed, May 27, 2026
- Full-scale 116k-clip surgery data prepared: ~125 GPU-hr, ~$150.
- Fixed test-clip leakage + 8x larger eval (n=1825); CIs tightened ~2.8x, conclusions held.
- Surgery beats frozen on motion + future-frame prediction; future-frame error is the cleanest CI-separated win.
- Next: 6 predictor-temporal metrics (rollout, causal, order, etc.) running; tightened CIs allows ablation (LeJEPA, iJEPA, vJEPA2.0, vJEPA1.0) at 10k not 115k.

## Tue, May 26, 2026
- First clean surgery-vs-pretrain numbers in (leakage-safe re-run).
- Surgery wins temporal (future-L1); pretrain wins action top1.
- Next: confirm on held-out test with CIs; add frozen.

## Mon, May 25, 2026
- iter15 headline INVALID: test-leakage + 8× data-asymmetry found, fixed.
- Re-running leakage-safe paired-Δ; SANITY validates pipeline before FULL.
- Next: pass SANITY → FULL, test surgery≫pretrain≫frozen on motion.
