"""Temporal Cycle-Consistency encoder-temporal metric (Dwibedi et al. CVPR19). TRAINING-FREE.

Mechanism: per-frame embeddings from each encoder → soft-NN alignment between same-action
clip pairs → cycle-back error (A→B→A) + Kendall's τ on hard-NN frame ordering. NO trainable
head — pure geometry. surgery≫pretrain hypothesis: lower cycle-back error and higher Kendall's
τ = encoder produces feature trajectories where corresponding frames of the same action align
across clip pairs, indicating temporal-structure awareness.

Pairing strategy (orchestrator m12f):
  - Pair test clips that share the same action label (loaded from action_labels.json).
  - Skip singleton classes (need ≥2 clips per action to form a pair).
  - For each pair (A, B): compute_pair returns (cycle_back_err, kendalls_tau). Per-clip score
    = mean over all pairs that clip participates in.

Outputs (m12f, --metric tcc):
  per_clip_tcc_cycle.npy   ← per-test-clip mean cycle-back error (lower = better)
  per_clip_tcc_tau.npy     ← per-test-clip mean Kendall's τ      (higher = better)
  aggregate_tcc.json       ← both metrics' mean/std/BCa CI + n_test + n_pairs + temperature

Gold:
  - Dwibedi et al. "Temporal Cycle-Consistency Learning" CVPR 2019
    github.com/google-research/google-research/tree/master/tcc (TF original)
  - June01/tcc_Temporal_Cycle_Consistency (PyTorch port)
  - Kendall, M.G. "A New Measure of Rank Correlation" Biometrika 1938

GPU-VALIDATION REQUIRED post-eval. CPU-checked (py_compile + ruff F,E9) only.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from utils.per_frame_features import forward_per_frame


@torch.no_grad()
def compute_per_frame(encoder, batch, num_frames, tubelet_size):
    """(B, T, C, H, W) → (B, T_eff, D) per-tubelet-step embeddings."""
    return forward_per_frame(encoder, batch, num_frames, tubelet_size).float().cpu()


def _soft_nn_indices(feats_a, feats_b, *, temperature):
    """For each frame i in A, return the soft-NN frame index in B.
    soft_idx_i = Σ_j softmax_j(<a_i, b_j> / τ) · j   (Dwibedi eq. 1).
    feats_a, feats_b: (T_eff, D). Returns (T_eff,) soft indices."""
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise RuntimeError(
            f"_soft_nn_indices expects 2-D per-clip; got A {feats_a.shape}, B {feats_b.shape}")
    if temperature <= 0:
        raise RuntimeError(f"temperature must be > 0; got {temperature}")
    sim = feats_a @ feats_b.T / temperature                   # (T_a, T_b)
    w = F.softmax(sim, dim=1)                                 # (T_a, T_b)
    j_range = torch.arange(feats_b.shape[0], device=feats_a.device, dtype=feats_a.dtype)
    return (w * j_range.unsqueeze(0)).sum(dim=1)              # (T_a,)


def cycle_back_error(feats_a, feats_b, *, temperature):
    """A→B→A cycle: for each frame i in A, soft-NN to B → soft-NN back to A. Mean |i - cycle_i|.
    Returns a scalar (Python float). Lower = better temporal correspondence."""
    soft_b_idx = _soft_nn_indices(feats_a, feats_b, temperature=temperature)
    # Round to nearest B-frame index for the A-back step (Dwibedi uses soft both ways but the
    # straightforward variant is fine for the scaffold; promotion to full-soft is a §3.3 knob).
    b_idx_hard = soft_b_idx.round().long().clamp_(0, feats_b.shape[0] - 1)
    selected_b = feats_b[b_idx_hard]                          # (T_a, D)
    soft_a_idx = _soft_nn_indices(selected_b, feats_a, temperature=temperature)
    i_range = torch.arange(feats_a.shape[0], device=feats_a.device, dtype=feats_a.dtype)
    return float((soft_a_idx - i_range).abs().mean().item())


def kendalls_tau_alignment(feats_a, feats_b):
    """Hard-NN alignment from A → B (one B-index per A-frame), then Kendall's τ vs the
    monotonic identity ordering. Higher τ (range [-1, 1]) = more order-preserving alignment.
    Returns a Python float; FAIL LOUD if the hard-NN alignment is constant (degenerate)."""
    sim = feats_a @ feats_b.T                                # (T_a, T_b)
    hard_b_idx = sim.argmax(dim=1).cpu().numpy()
    if np.unique(hard_b_idx).size < 2:
        raise RuntimeError(
            f"kendalls_tau_alignment: degenerate hard-NN alignment ({hard_b_idx.tolist()}); "
            "encoder produced collinear per-frame features for this clip pair")
    tau, _p = kendalltau(np.arange(hard_b_idx.size), hard_b_idx)
    return float(tau)


def compute_pair(feats_a, feats_b, *, temperature):
    """One pair → (cycle_back_err, kendalls_tau). Both metrics from the same (A, B) features."""
    if feats_a.shape != feats_b.shape:
        raise RuntimeError(
            f"compute_pair: feats_a {feats_a.shape} vs feats_b {feats_b.shape} shape mismatch")
    return cycle_back_error(feats_a, feats_b, temperature=temperature), \
           kendalls_tau_alignment(feats_a, feats_b)


__all__ = [
    "compute_per_frame", "cycle_back_error", "kendalls_tau_alignment", "compute_pair",
]
