"""Metric #1 — Multi-step Rollout Drift (predictor latent-L1). ★ top surgery-win bet.

Free-running iterated rollout: predict slot k from {0..k-1}, feeding the PREDICTED features
back as context so errors COMPOUND. Per-clip OLS slope of L1 vs horizon h = drift slope (how
fast the world-model's error grows over the rollout). LOWER slope = more stable dynamics.
Orthogonal to future_mse (single masked step) and tdist (single-shot, no feedback).
Gold: V-JEPA 2-AC rollout loss (Meta 2025); LIVE / Rolling-Forcing drift (2025).

GPU-VALIDATION REQUIRED post-eval (§3.1) — confirm L1 grows monotonically with horizon.
"""
import numpy as np

from utils.predictor_eval import perclip_slope, rollout_l1_per_horizon, to_pixel


def compute(encoder, predictor, batch, num_frames) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip drift slope (B,)."""
    pixel = to_pixel(batch)
    L = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=True)  # (B, Tp-1)
    horizons = list(range(1, L.shape[1] + 1))
    return perclip_slope(L, horizons)
