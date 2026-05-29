"""Metric #3 — Temporal-Distance scaling (predictor latent-L1).

SINGLE-SHOT (not iterated): from a fixed 1-slot context (slot 0), predict the token slot at
increasing temporal offset Δt ∈ {1,2,4,8}; per-clip OLS slope of L1-vs-Δt = the predictability
horizon (how fast single-shot prediction degrades with temporal distance). Orthogonal to #1
rollout (which iterates / compounds). Gold: CPC long-horizon dependency (van den Oord 2018).

GPU-VALIDATION REQUIRED post-eval (§3.1).
"""
import numpy as np

from utils.predictor_eval import (
    PT_DELTAS, expand_mask, masked_predict_l1, perclip_slope, temporal_token_idx, to_pixel, token_grid,
)


def compute(encoder, predictor, batch, num_frames) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip slope of L1 vs Δt (B,). Higher slope = faster decay."""
    Tp, _, _, _ = token_grid(num_frames)
    ctx = 0
    deltas = [d for d in PT_DELTAS if ctx + d < Tp]
    if len(deltas) < 2:
        raise ValueError(f"tdist needs ≥2 valid Δt for a slope; Tp={Tp} gave {deltas}")
    pixel = to_pixel(batch)
    b = pixel.shape[0]
    m_enc = expand_mask(temporal_token_idx(num_frames, [ctx]), b)
    cols = []
    for d in deltas:
        m_pred = expand_mask(temporal_token_idx(num_frames, [ctx + d]), b)
        l1, _, _ = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred)
        cols.append(l1)
    L = np.stack(cols, axis=1)                       # (B, n_deltas)
    return perclip_slope(L, deltas)                  # (B,)
