# Instruction: Date & daily udpate in 2 line for daily progress && 1 line for next step. Each line covers max 25 words
# AUDIENCE: Research leads focussed on > ## 🎯 Paper goal:  `vjepa_surgery` outperforming ≫ `vjepa_frozen` and `other FT techniques` on motion / temporal features
# AUDIENCE has knowledge about ONLY plots from @iter/iter15_trainHead_freezeEncoder/result_outputs/v15a/poc/probe_plot/eval/
## Note: do NOT write pipeline jargon (SANITY/POC/eval_10k/Δ5/cells), 
## Note: but, /eli5 all technical failures faced and solutions built 



## Wed, July 1, 2026
- Confirmed the wins reproduce on the smaller 1B model: the method ranking survives 2B-to-1B on twelve of fifteen skill-scores.
- Against the strongest fine-tuner, surgery cleanly separates only on prediction (future-frame, causal); it ties or trails elsewhere.
- Next: full 116k-clip run — best surgery arm versus best baseline — to tighten bands and decide the rest.

## Wed, June 24, 2026
- The 1B model's training and evaluation finished; built the side-by-side 1B-versus-2B scorecard to compare scales at a glance.
- Fixed two glitches: one stuck job had cancelled the whole run (added an auto-retry); the live charts silently stopped updating (wrong tool).
- Next: cross-scale analysis — does the surgery-versus-baseline ranking hold from 2B down to 1B?

## Tue, June 23, 2026
- Trimmed the 1B run to only the models that carry the paper's story — the surgery winners plus three cheap weight-blends.
- Dropped redundant variants to roughly halve the compute bill; infrastructure day, no new numbers.
- Next: run the trimmed 1B training and evaluation.

## Mon, June 22, 2026
- Set up the smaller 1B model to reproduce the surgery wins at half the cost, testing whether they hold across model size.
- Reused the existing 1B starting weights (recipe matched) — no wasted retraining; reorganised the results archive.
- Next: trim the model roster to the essential arms, then train.

## Sun, June 21, 2026
- Retraining each trained encoder's read-off heads, which the first run never saved: scene-attribute ones (taxonomy) done in ~16h on 1X 96GB vRAM GPU.
- Time-ordering scorers (arrow-of-time, frame-order, pace) now retraining, ~12h on 4X 96GB vRAM GPU; every trained model reused, none retrained.
- Next (advisor pivot): stop the 10k ablation; restart full training + evaluation on the smaller V-JEPA 2.1 1B model, reproduce the wins.

## Sat, June 20, 2026
- Re-testing every model on a fresh, disjoint 10k-clip batch — 5× the old 2k — to tighten the confidence bands and prove the wins aren't noise.
- Infrastructure day; no new model numbers — readied the retest to score beside the old, not over.
- Next: run the retest; confirm surgery's future-frame win holds with tighter bands.

## Fri, June 19, 2026
- Built the fresh-batch retest: reuse every trained model, test on the full unseen batch.
- Hardened the result backups, then wired the retest so confidence bands tighten about twofold.
- Next: run the retest; confirm surgery's future-frame win and pick the two best variants.

## Thurs, June 18, 2026
- Weighed the open-benchmark detour against the core claim; chose to harden the surgery wins first.
- Planning day; no new model numbers.
- Next: confirm the surgery wins on a fresh, unseen batch of clips.

## Wed, June 17, 2026
- Built a study testing whether the fifteen skill-scores measure distinct abilities or overlap.
- Early signal weak on our models alone; needs outside encoders to conclude.
- Next: decide between chasing the open benchmark and hardening the surgery wins.

## Tue, June 16, 2026
- Began reframing the work: a benchmark scoring any open world-model, our arms as references.
- Started checking the fifteen metrics measure distinct skills, not redundant numbers (early, inconclusive).
- Next: score outside models (Cosmos, VideoMAE, DINO-WM) on the same fifteen-metric benchmark.

## Mon, June 15, 2026
- Blending surgery with frozen recovers frozen-level temporal while keeping surgery's future-frame win.
- All seventeen models now scored on every metric; the trade-off win is clean.
- Next: confirm the wins on a fresh, unseen batch of clips.

## Sun, June 14, 2026
- Built blend variants mixing surgery and frozen weights at fixed ratios (30/50/70%).
- Consolidated the model roster to one source; refreshed the results deck.
- Next: score the blend variants on prediction and temporal tests.

## Sat, June 13, 2026
- Built the all-metric head-to-head scorecard and the frame-alignment comparison chart.
- Designed blend variants to recover frozen's temporal edge without losing prediction.
- Next: build and evaluate the surgery-plus-frozen blends.

## Fri, June 12, 2026
- Five new temporal tests in (arrow-of-time, frame-order, pace, frame-alignment) for all models.
- Frozen keeps the strongest time-ordering; surgery trades it for future-frame prediction.
- Next: find a blend keeping prediction win plus frozen's temporal strength.

## Thu, June 11, 2026
- Sped up frame decoding so every model could afford the new temporal tests.
- Infrastructure day; no new model numbers, temporal results land tomorrow.
- Next: run all encoders through the five temporal tests.

## Wed, June 10, 2026
- Built five new temporal tests measuring whether encoders preserve frame order and timing.
- Extends the comparison beyond prediction into the pure time-structure of features.
- Next: speed up the pipeline, then score every model.


## Tue, June 9, 2026
- Surgery-on-raw matches surgery-on-factor on future-frame and causal — surgery's only clean wins over frozen.
- Factorized data adds no edge over the surgery technique; frozen/vanilla still lead the semantic metrics.
- Next: does factorization beat raw at full scale? else reposition the contribution to the technique.
- Head variants removed: frozen predictor makes temporal scores identical to vanilla (0.0077/0.558/0.0065) — no new signal.

## Mon, June 8, 2026
- Surgery-encoder wins future-frame and causal prediction; full fine-tuning leads motion-cosine — surgery not best everywhere.
- Frozen tops taxonomy (fine-tuning distorts general features); head variant only ties vanilla on rollout, no gain.
- Next: repeat the full surgery-vs-baselines comparison on the smaller 1B backbone (scale ablation).

## Sun, June 7, 2026
- Built faster evaluation and live charts tracking every model's scores as they finish.
- Infrastructure day; foundation for the held-out comparisons, no new model numbers.
- Next: run held-out evaluations; surgery-vs-baseline numbers follow.

## Sat, June 6, 2026
- Nine of thirteen arms trained; surgery leads future-frame and causal error at selected checkpoints.
- Full fine-tuning leads action/motion recognition; semantic gaps still within noise at current scale.
- Next: held-out test evaluations plus paired comparisons for all thirteen arms complete Sunday.

## Fri, June 5, 2026
- Switched checkpoint selection to future-frame error — the metric our method actually claims.
- Found the reference model's selection quiz overlapped its training clips (75%); fixed, guarded.
- Next: rerun all thirteen arms on a 4-GPU node; full comparison Saturday.

## Thurs, June 4, 2026
- First seven of thirteen arms trained on 10k clips; zero crashes, full trajectories recorded.
- Surgery leads future-frame error; Auto-RGN baseline leads motion/action; raw-data control surprisingly close.
- Next: switched model-selection to future-frame error (was action top-1); rerunning all arms.

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
