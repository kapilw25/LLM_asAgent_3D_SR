"""Metric #4 — Teacher-vs-Free-run Gap (predictor latent-L1). ★ top surgery-win bet.

Run the rollout twice — teacher-forced (REAL context re-encoded each step) vs free-running
(the predictor's OWN predicted context) — and take the per-clip gap (free − teacher) averaged
over horizons = EXPOSURE BIAS: how much error inflates when the predictor must consume its own
mistakes. LOWER gap = recovers better / less compounding. Orthogonal to raw drift (#1 measures
the free-run growth rate; this ISOLATES the free-vs-teacher difference).
Gold: Scheduled Sampling (Bengio '15); BAgger (2025).

GPU-VALIDATION REQUIRED post-eval (§3.1).
"""
import numpy as np

from utils.predictor_eval import rollout_l1_per_horizon, to_pixel


def compute(encoder, predictor, batch, num_frames) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip mean(free − teacher) gap over horizons (B,)."""
    pixel = to_pixel(batch)
    free = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=True)
    teach = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=False)
    return (free - teach).mean(axis=1)
