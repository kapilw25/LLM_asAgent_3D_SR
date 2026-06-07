"""Metric #5 — Mask-Ratio Robustness (predictor latent-L1).

Sweep the fraction of tokens hidden (mask ratio r ∈ {0.3,0.5,0.7,0.9}); per-clip OLS slope of
L1-vs-r = graceful degradation under sparse context (orthogonal to horizon: it's how MUCH
context, not how far ahead). A deterministic random partition (fixed seed) is reused across all
clips/batches so the only thing varying is r. Gold: VideoMAE high-masking-ratio robustness
(Tong 2022); MAE-ST (Feichtenhofer 2022).

GPU-VALIDATION REQUIRED post-eval (§3.1).
"""
import numpy as np
import torch

from utils.predictor_eval import (
    PT_MASK_RATIOS, PT_SEED, expand_mask, full_target_h, masked_predict_l1, perclip_slope,
    to_pixel, token_grid,
)


def compute(encoder, predictor, batch, num_frames) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip slope of L1 vs mask-ratio (B,)."""
    Tp, _, _, S = token_grid(num_frames)
    n_tok = Tp * S
    pixel = to_pixel(batch)
    b = pixel.shape[0]
    g = torch.Generator().manual_seed(PT_SEED)
    perm = torch.randperm(n_tok, generator=g)        # fixed partition, shared across clips
    h_full = full_target_h(encoder, pixel)           # h-memo: one target encode reused across all r (None when off)
    cols = []
    for r in PT_MASK_RATIOS:
        k = max(1, int(round(n_tok * r)))            # # predicted (masked) tokens
        pred_idx = perm[:k].sort().values
        enc_idx = perm[k:].sort().values
        m_pred = expand_mask(pred_idx, b)
        m_enc = expand_mask(enc_idx, b)
        l1, _, _ = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred, h_full=h_full)
        cols.append(l1)
    L = np.stack(cols, axis=1)                       # (B, n_ratios)
    return perclip_slope(L, PT_MASK_RATIOS)                 # (B,)
