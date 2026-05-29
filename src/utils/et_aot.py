"""Arrow-of-Time encoder-temporal metric (Wei et al. CVPR18). Trainable binary head.

Mechanism: forward vs time-REVERSED clip → linear classifier on mean-pooled encoder features.
Per-clip score on test split = 0/1 correctness on the (forward, reversed) pair (two examples
per clip, averaged). surgery≫pretrain hypothesis: AoT-acc higher = encoder preserves the
temporal arrow.

Outputs from the orchestrator (m12f, --metric aot):
  per_clip_aot.npy        ← per-test-clip 0/1 (avg of 2 directions)
  aggregate_aot.json      ← mean / std / BCa 95% CI / n_test (same schema as aggregate_mse.json)

Gold:
  - Wei et al. "Learning and Using the Arrow of Time" CVPR18
    vcg.seas.harvard.edu/publications/learning-and-using-the-arrow-of-time
  - Pickup et al. "Seeing the Arrow of Time" CVPR14 (earlier formulation)

GPU-VALIDATION REQUIRED post-eval. CPU-checked (py_compile + ruff F,E9) only.
"""
import numpy as np
import torch
import torch.nn as nn

from utils.per_frame_features import forward_per_frame, reverse_frames


@torch.no_grad()
def compute_features(encoder, batch, num_frames, tubelet_size):
    """Returns (feat_fwd, feat_rev) — each (B, D), mean-pooled across T_eff. Two examples per
    input clip; orchestrator stacks them with labels {0=fwd, 1=rev} for head training/eval."""
    feat_fwd = forward_per_frame(encoder, batch, num_frames, tubelet_size).mean(dim=1)
    feat_rev = forward_per_frame(encoder, reverse_frames(batch), num_frames, tubelet_size).mean(dim=1)
    return feat_fwd.float().cpu(), feat_rev.float().cpu()


def train_head(train_feats, train_labels, *, embed_dim, lr, epochs, weight_decay, batch_size,
               device, seed):
    """Train a linear head (D → 2). FAIL LOUD on empty/degenerate input. Deterministic seed."""
    if train_feats.ndim != 2 or train_feats.shape[1] != embed_dim:
        raise RuntimeError(
            f"train_feats shape {tuple(train_feats.shape)} mismatches embed_dim={embed_dim}")
    if train_labels.ndim != 1 or train_labels.numel() != train_feats.shape[0]:
        raise RuntimeError(
            f"train_labels shape {tuple(train_labels.shape)} doesn't match feats N={train_feats.shape[0]}")
    if train_feats.shape[0] == 0:
        raise RuntimeError("train_head: 0 training examples — pipeline failure")
    unique = torch.unique(train_labels)
    if unique.numel() < 2:
        raise RuntimeError(f"train_head: only 1 class in labels ({unique.tolist()}) — need both 0 and 1")
    g = torch.Generator(device="cpu").manual_seed(seed)
    head = nn.Linear(embed_dim, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = train_feats.shape[0]
    feats_d = train_feats.to(device)
    labels_d = train_labels.to(device)
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = head(feats_d[idx])
            loss = loss_fn(logits, labels_d[idx])
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def eval_head(head, test_feats, test_labels, *, device, batch_size):
    """Per-example 0/1 correctness. Orchestrator averages the two-direction pair per clip
    BEFORE writing per_clip_aot.npy."""
    if test_feats.shape[0] != test_labels.shape[0]:
        raise RuntimeError(
            f"test_feats N={test_feats.shape[0]} != test_labels N={test_labels.shape[0]}")
    out = np.empty(test_feats.shape[0], dtype=np.float32)
    feats_d = test_feats.to(device)
    labels_d = test_labels.to(device)
    for s in range(0, feats_d.shape[0], batch_size):
        logits = head(feats_d[s:s + batch_size])
        pred = logits.argmax(dim=1)
        out[s:s + batch_size] = (pred == labels_d[s:s + batch_size]).float().cpu().numpy()
    return out


__all__ = ["compute_features", "train_head", "eval_head"]
