QQ0 — what an AAAI reviewer will bite (beyond the CI overlaps)

gap in the 2B experiment                          reviewer's line                                   does eval-on-10k fix it?
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SINGLE SEED — CIs are eval-bootstrap over test    "report ≥3 seeds; your bands don't capture        ✗ needs RE-TRAIN ×seeds
clips, NOT training-seed variance                training stochasticity"                          (the biggest one)
POC, not FULL — trained on ~1.8k clips            "show the full-scale result, not a 1.8k POC"      ✗ needs the FULL run
14 metrics × ~18 arms, NO multiple-comparison     "without Bonferroni/FDR, some 'wins' are          ✗ stats fix (free, do it)
correction                                       false positives at α=0.05"
in-domain only (all WalkIndia)                    "does it hold off your own dataset?"              ~ only if 10k is held-out
the improvement/wiseft arms are marginal/ties     "the 'improvements' aren't significant"            partially
Top-3 to fix before submission: multiple seeds, FULL-scale, multiple-comparison correction. Eval-on-10k addresses none of those — see the correction under Task 2.

Task 1 — download command

python -u src/utils/hf_outputs.py download-data data/subset_10k_local 2>&1 | tee logs/download_subset_10k_$(date +%Y%m%d_%H%M%S).log
(download-data <path> lists+fetches that subfolder via 8-way HTTP; uses HF_TOKEN from .env automatically.)

Task 1.2 — subset_10k_local vs eval_10k_local + post-processing

                        subset_10k_local (HF, raw)         eval_10k_local (current, full)
clips (tars)            ✅ subset-0000{0..9}.tar            ✅ m00d_download_subset
master manifest         ✅ subset_10k.json                 ✅ eval_10k.json
VLM tags                ✅ tags.json                       ✅ tags.json
motion features (m04d)  ❌ MISSING                          ✅ m04d_motion_features
action/taxonomy labels  ❌ (derived at eval time)           ✅ (already built)
splits / m10 / m11      ❌ (eval doesn't need them)          ✅ (training-only)
Post-processing needed = ONE GPU step: m04d motion features (~30-60 min). Labels are auto-derived by run_eval Stage 1 from tags.json; m10/m11/splits are training-only and the eval never reads them (verified
in m12c/e/f).

QQ1.2 — disk: YES, free space first 

You have 206G free; the fresh eval rebuilds a ~423G frame cache for the 10k clips (same scale as the existing one). Delete the now-stale data/eval_10k_local/m12_frame_cache (423G, regenerable — the 2B eval 
there is done) → ample room. (The 242G of outputs/poc/*.pt can also go after the HF backup, Task #6.)

Task 2 — code plan (run_eval directly, NOT the scheduler)

# 1. download (above) → 2. motion features:
python -u src/m04d_motion_features.py --POC --subset data/subset_10k_local/subset_10k.json \
    --local-data data/subset_10k_local --cache-policy 1
# 3. flip the 2 keys in configs/pipeline.yaml:
#    local_data_dir: "data/subset_10k_local"   ·   master_manifest_name: "subset_10k.json"
# 4. eval the EXISTING 2B ckpts → a SEPARATE output root (env overrides; no code edit):
SUB=outputs/poc_subset10k
CACHE_POLICY_ALL=2 OUTPUT_ACTION=$SUB/probe_action OUTPUT_COS=$SUB/probe_motion_cos \
OUTPUT_MSE=$SUB/probe_future_mse OUTPUT_TAXONOMY=$SUB/probe_taxonomy \
OUTPUT_PREDTEMP=$SUB/predictor_temporal OUTPUT_ENCTEMP=$SUB/encoder_temporal OUTPUT_PLOTS=$SUB/probe_plot \
./scripts/run_eval.sh --POC --encoders "vjepa_2_1_frozen vjepa_2_1_pretrain_encoder vjepa_2_1_surgical_intervene_wiseft_f30_encoder vjepa_2_1_surgical_intervene_wiseft_f50_encoder 
vjepa_2_1_surgical_intervene_wiseft_f70_encoder"

⚠️  Two gotchas the subagent caught:
1. CLOBBER — outputs/poc is keyed on mode, not the data dir, so a plain run would overwrite your eval_10k 2B numbers. The OUTPUT_* env overrides above redirect every product to $SUB while the ckpt inputs
still read the real 2B dirs. Keep both.
2. 🚨 The CI correction (matters for your whole reason to do this): run_eval builds its own 75/5/20 split inside Stage 1, so the test set is ~2k clips, NOT 10k (the other 80% trains the probe heads). ~2k ≈ 
your current 1,825 → the confidence bands barely shrink. So eval-on-subset_10k tests generalization (do the wins hold on fresh clips?) — it does NOT resolve the CI-overlap significance question. For that you
need multiple seeds or FULL-scale, not a second 10k at the same test-N.

So, honestly: run it for the generalization check (cheap, valuable), but don't expect it to turn the temporal-metric ties into significant wins — that was my earlier overstatement.
