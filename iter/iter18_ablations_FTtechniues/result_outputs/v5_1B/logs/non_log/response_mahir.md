---
Hey! Glad the codebase helped — you've got the right scripts (m12d_future_mse.py = future-frame MSE, m12e_predictor_temporal.py = the 6
temporal metrics; both share the utils/predictor_eval.py primitive). Answers:

1. Model yaml → configs/model/vjepa2_1.yaml (2B ViT-G). The predictor/arch keys predictor_eval reads:
arch: vit_gigantic_xformers   # 26 heads, 48 blocks
embed_dim: 1664 · depth: 48 · num_heads: 26
# predictor
pred_depth: 24 · pred_embed_dim: 384 · pred_num_heads: 12 · num_mask_tokens: 2
predict_all: true             # DENSE loss — L1 on ALL tokens, not just masked (the 2.1 win)
n_output_distillation: 4      # layers [11,23,37,47]
crop_size: 384 · patch_size: 16 · tubelet_size: 2
checkpoint_path: checkpoints/vjepa2_1_vitG_384.pt
If you want the smaller model for Colab, configs/model/vjepa2_1_vitg.yaml is the 1B (ViT-g, embed_dim 1408, vit_giant_xformers) — same
predictor block.

2. Checkpoint with the predictor → checkpoints/vjepa2_1_vitG_384.pt (the base V-JEPA 2.1 release) holds both encoder + predictor. The
eval output literally records "predictor_loaded_from": "checkpoints/vjepa2_1_vitG_384.pt". Don't split it by hand — call
utils/predictor_eval.load_encoder_predictor(ckpt_path, num_frames) → (encoder, predictor, embed_dim_concat); it resolves both state
dicts internally (resolve_encoder_state_dict / resolve_predictor_state_dict).
⚠️  For a fine-tuned arm, use that arm's *_ckpt_best.pt (~8 GB, tuned encoder+predictor). The student_encoder.pt (~3.8 GB) is
encoder-only — no predictor, so it won't work for future-frame.

3. Imports → Yes, all from the one vendored shim src/utils/vjepa2_imports.py: get_vit_predictor_2_1, get_apply_masks, get_mask_generator
(+ get_vit_by_arch) — never import from models… directly. Cleaner still: go through utils/predictor_eval.py (load_encoder_predictor /
build_mask_gen / masked_predict_l1) — it's the single-source primitive shared by m12d + m12e and wraps the shim for you.

4. Has it run clean on GPU? → Yes — that "CPU-checked, not GPU-smoked" comment is stale. The whole predictor suite has run clean on GPU
across the full 2B eval, and I have a 1B run producing these numbers right now. Frozen baseline reference: future-MSE mean = 0.557351
(std 0.018, n=1825, num_frames=16). One gotcha worth flagging (we hit it on a fresh node this week): it FAILs-loud without --local-data 
<clip TARs> and --action-probe-root (it needs the test split from action_labels.json) — make sure both are passed.

Run command (frozen, the simplest path):
python -u src/m12d_future_mse.py --FULL --stage forward --variant vjepa_2_1_frozen \
  --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \
  --action-probe-root outputs/full/probe_action \
  --local-data data/eval_10k_local \
  --output-root outputs/full/probe_future_mse --cache-policy 1

Known-good clip + expected output: run frozen over the eval_10k test split (1825 clips) → you should land future-MSE ≈ 0.557 (±0.001
CI). For a single-clip Colab smoke, masked_predict_l1(...) returns per-clip L1 (B,) — per-clip varies (std ~0.018), so use a small batch
and check the mean sits near 0.557. Happy to zip up the eval_10k clip TARs + the frozen aggregate_mse.json + per-clip outputs so you
have an exact reference — just say the word.

Shout if the predictor load throws a shape mismatch — 9 times out of 10 it's a student_encoder.pt (encoder-only) being passed where
*_ckpt_best.pt / the base ckpt is needed. Good luck with the demo! 🚀

---