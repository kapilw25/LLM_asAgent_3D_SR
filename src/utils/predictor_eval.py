"""Shared V-JEPA encoder+predictor eval primitive for the m12e predictor-temporal suite.

NON-STANDALONE helper (imported by src/utils/pt_*.py + src/m12e_predictor_temporal.py).
Self-contained NOW (copies probe_future_mse.py's module-local loaders + masked-predict L1
core) — the dedup refactor of probe_future_mse → import this is DEFERRED post-eval (editing a
live-invoked script). See iter/iter16_ablations/plan_metrics_temporal.md §2.1.

Gold standard: facebookresearch/vjepa2/app/vjepa_2_1 (encoder hierarchical + predictor).
GPU-VALIDATION REQUIRED post-eval (§3.1) — written but not yet GPU-smoked (eval holds the GPU).

PUBLIC API
    load_encoder_predictor(ckpt_path, num_frames) -> (encoder, predictor, embed_dim_concat)
    build_mask_gen(num_frames, *, npred=None, spatial_scale=None, ...) -> _MaskGenerator
    token_grid(num_frames) -> (Tp, Hp, Wp, spatial_per_frame)
    temporal_token_idx(num_frames, t_slots) -> LongTensor of token indices for those temporal slots
    to_pixel(batch) -> (B,3,T,H,W) bf16 cuda
    masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred) -> (per_clip_l1 (B,), out, h_target)
    bootstrap_ci  (re-exported from utils.bootstrap)
"""
import contextlib
import sys

import numpy as np
import torch

from utils.bootstrap import bootstrap_ci  # noqa: F401  (re-export for the pt_*.py + orchestrator)
from utils.config import (
    get_model_config,
    get_pipeline_config,
    load_train_config_with_extends,
)
from utils.frozen_features import (
    resolve_encoder_state_dict,
    resolve_predictor_state_dict,
)
from utils.vjepa2_imports import (
    get_apply_masks,
    get_mask_generator,
    get_vit_gigantic_xformers,
    get_vit_predictor_2_1,
)

# ── Constants (single-sourced from yaml — copied from probe_future_mse.py) ──
_PCFG = get_pipeline_config()
_MODEL_CFG = get_model_config(None)["model"]               # None → default vjepa2_1.yaml
NUM_FRAMES_DEFAULT = _PCFG["probe"]["num_frames"]
PATCH_SIZE = _MODEL_CFG["patch_size"]
TUBELET_SIZE = _MODEL_CFG["tubelet_size"]
CROP = _MODEL_CFG["crop_size"]
PRED_EMBED_DIM = _MODEL_CFG["pred_embed_dim"]
PRED_DEPTH = _MODEL_CFG["pred_depth"]
PRED_NUM_HEADS = _MODEL_CFG["pred_num_heads"]
NUM_MASK_TOKENS = _MODEL_CFG["num_mask_tokens"]
ENCODER_EMBED_DIM = _MODEL_CFG["embed_dim"]                # 1664 (ViT-G last-layer)

_MASK0 = load_train_config_with_extends("configs/train/base_optimization.yaml")["mask"][0]
DEFAULT_SPATIAL_SCALE = tuple(_MASK0["spatial_scale"])
DEFAULT_TEMPORAL_SCALE = tuple(_MASK0["temporal_scale"])
DEFAULT_ASPECT_RATIO = tuple(_MASK0["aspect_ratio"])
DEFAULT_NUM_BLOCKS = _MASK0["num_blocks"]

# iter16 §3.4: m12e predictor-temporal sweep params single-sourced from pipeline.yaml
# (were hardcoded _DELTAS/_RATIOS/_SEED in pt_tdist/pt_maskratio/pt_order).
_PT = _PCFG["probe"]["predictor_temporal"]
PT_DELTAS = tuple(_PT["deltas"])              # pt_tdist Δt sweep
PT_MASK_RATIOS = tuple(_PT["mask_ratios"])    # pt_maskratio sweep
PT_SEED = _PT["seed"]                         # pt_maskratio + pt_order deterministic-partition seed


# ── Loaders (encoder hierarchical + predictor) ─────────────────────────
def load_encoder_only(ckpt_path, num_frames):
    """Load ONLY the V-JEPA 2.1 ViT-G hierarchical encoder (NO predictor build) from a .pt.

    Returns (encoder, ckpt, embed_dim_concat=6656). For encoder-temporal metrics (m12f) that
    never use the predictor — iter16 §3.3 R2: skips the ~60M predictor construction + state
    load that load_encoder_predictor does but m12f discards. `ckpt` is returned so callers that
    DO need the predictor (load_encoder_predictor) reuse this single torch.load (no double read).
    """
    from pathlib import Path
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        sys.exit(f"FATAL: ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)

    # --- encoder (hierarchical: returns 4-layer concat = 6656-dim) ---
    enc_sd = resolve_encoder_state_dict(ckpt)
    enc_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_sd.items()}
    encoder = get_vit_gigantic_xformers()(
        img_size=(CROP, CROP), patch_size=PATCH_SIZE, num_frames=num_frames,
        tubelet_size=TUBELET_SIZE, use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    msg = encoder.load_state_dict(enc_sd, strict=False)
    loaded = len(enc_sd) - len(msg.unexpected_keys)
    total = len(list(encoder.state_dict().keys()))
    if loaded < total * 0.9:
        sys.exit(f"FATAL: only {loaded}/{total} encoder params loaded — key mismatch")
    encoder.return_hierarchical = True
    encoder = encoder.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.backends.cuda.sdp_kernel = contextlib.nullcontext
    embed_dim_concat = encoder.embed_dim * len(encoder.hierarchical_layers)
    if embed_dim_concat != ENCODER_EMBED_DIM * 4:
        sys.exit(f"FATAL: hierarchical concat dim {embed_dim_concat} != {ENCODER_EMBED_DIM}*4")
    return encoder, ckpt, embed_dim_concat


def load_encoder_predictor(ckpt_path, num_frames):
    """Load V-JEPA 2.1 ViT-G hierarchical encoder + predictor from one .pt.

    Returns (encoder, predictor, embed_dim_concat=6656). Mirrors
    probe_future_mse._load_vjepa_2_1_encoder_hierarchical + _load_predictor_2_1.
    iter16 §3.3: encoder half delegated to load_encoder_only (single source; one torch.load).
    """
    encoder, ckpt, embed_dim_concat = load_encoder_only(ckpt_path, num_frames)

    # --- predictor (same .pt) ---
    pred_sd = resolve_predictor_state_dict(ckpt)
    if pred_sd is None:
        sys.exit(f"FATAL: ckpt has no predictor key. top-level: {list(ckpt.keys())[:6]}")
    predictor = get_vit_predictor_2_1()(
        img_size=(CROP, CROP), patch_size=PATCH_SIZE, num_frames=num_frames,
        tubelet_size=TUBELET_SIZE, embed_dim=ENCODER_EMBED_DIM,
        predictor_embed_dim=PRED_EMBED_DIM, depth=PRED_DEPTH, num_heads=PRED_NUM_HEADS,
        use_mask_tokens=True, num_mask_tokens=NUM_MASK_TOKENS, zero_init_mask_tokens=True,
        use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=False, return_all_tokens=True,
    )
    pred_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pred_sd.items()}
    pmsg = predictor.load_state_dict(pred_sd, strict=False)
    p_total = len(list(predictor.state_dict().keys()))
    p_loaded = p_total - len(pmsg.missing_keys)
    if p_loaded / max(p_total, 1) < 0.5:
        sys.exit(f"FATAL: predictor only {p_loaded}/{p_total} loaded — random init = garbage")
    predictor = predictor.to(device="cuda", dtype=torch.bfloat16).eval()
    return encoder, predictor, embed_dim_concat


def build_mask_gen(num_frames, *, npred=None, spatial_scale=None,
                   temporal_scale=None, aspect_ratio=None):
    """_MaskGenerator with V-JEPA 2.1 small-block defaults; npred/scales overridable
    (the #5 mask-ratio metric sweeps npred)."""
    return get_mask_generator()(
        crop_size=(CROP, CROP), num_frames=num_frames,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE), temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=spatial_scale or DEFAULT_SPATIAL_SCALE,
        temporal_pred_mask_scale=temporal_scale or DEFAULT_TEMPORAL_SCALE,
        aspect_ratio=aspect_ratio or DEFAULT_ASPECT_RATIO,
        npred=npred if npred is not None else DEFAULT_NUM_BLOCKS,
    )


# ── Token-grid helpers (temporal-slot ↔ token-index mapping) ───────────
def token_grid(num_frames):
    """Token layout after patch_embed: Tp temporal × (Hp×Wp) spatial. Index order is
    t*(Hp*Wp) + h*Wp + w, so temporal slot t owns indices [t*S, (t+1)*S), S=Hp*Wp."""
    Tp = num_frames // TUBELET_SIZE
    Hp = CROP // PATCH_SIZE
    Wp = CROP // PATCH_SIZE
    return Tp, Hp, Wp, Hp * Wp


def temporal_token_idx(num_frames, t_slots) -> torch.Tensor:
    """LongTensor of token indices belonging to the given temporal slot ids."""
    _, _, _, S = token_grid(num_frames)
    return torch.tensor([t * S + j for t in t_slots for j in range(S)], dtype=torch.long)


def to_pixel(batch: torch.Tensor) -> torch.Tensor:
    """(B,T,3,H,W) cpu → (B,3,T,H,W) bf16 cuda (matches probe_future_mse)."""
    return batch.to("cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous()


# ── The shared masked-predict L1 core (from probe_future_mse._forward_one_batch) ──
@torch.no_grad()
def masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred):
    """pixel (B,3,T,H,W) bf16 cuda ; m_enc/m_pred (B,n) long cuda (custom per-metric).
    Returns (per_clip_l1 (B,) np.float32, out, h_target). Same path as future_mse.
    """
    apply_masks = get_apply_masks()
    z = encoder(pixel, masks=[m_enc])
    if isinstance(z, (list, tuple)):
        z = torch.cat(list(z), dim=-1)
    h = encoder(pixel)
    if isinstance(h, (list, tuple)):
        h = torch.cat(list(h), dim=-1)
    h_target = apply_masks(h, [m_pred])
    out = predictor(z, [m_enc], [m_pred], mask_index=0)
    if isinstance(out, tuple) and len(out) == 2:
        out = out[0]
    if out.shape != h_target.shape:
        sys.exit(f"FATAL: predictor out {out.shape} != h_target {h_target.shape}")
    per_clip_l1 = (out.float() - h_target.float()).abs().mean(dim=(1, 2))
    return per_clip_l1.cpu().numpy().astype(np.float32), out, h_target


@torch.no_grad()
def rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, *, free_running):
    """Iterated temporal rollout: predict slot k from the context of slots {0..k-1},
    for k=1..Tp-1; return per-clip L1 at each horizon → (B, Tp-1).

    free_running=True  : context bank starts as encoder(slot0); each predicted slot's OUTPUT
                         features are APPENDED to the bank (errors compound → drift).
    free_running=False : teacher-forced — at each k re-encode the REAL visible slots {0..k-1}.

    Targets are the full-forward encoder features per slot (same target space as future_mse's
    h_target). Both branches are the SAME predictor call; only the context differs.

    GPU-VALIDATION REQUIRED post-eval (§3.1): the free-running bank injects predicted (6656-dim)
    features as predictor context — the exact rollout semantics need a GPU smoke + sanity that
    L1 grows monotonically with horizon. Design per plan §2.1.
    """
    Tp, _, _, S = token_grid(num_frames)
    if Tp < 2:
        raise ValueError(f"rollout needs Tp>=2 temporal slots; got {Tp}")
    b = pixel.shape[0]
    h = encoder(pixel)                                   # full forward → per-slot targets
    if isinstance(h, (list, tuple)):
        h = torch.cat(list(h), dim=-1)                   # (B, N, 6656)

    l1s = []
    if free_running:
        bank = encoder(pixel, masks=[expand_mask(temporal_token_idx(num_frames, [0]), b)])
        if isinstance(bank, (list, tuple)):
            bank = torch.cat(list(bank), dim=-1)         # (B, S, 6656) slot-0 context
        bank_idx = temporal_token_idx(num_frames, [0])
        for k in range(1, Tp):
            m_enc = expand_mask(bank_idx, b)
            m_pred = expand_mask(temporal_token_idx(num_frames, [k]), b)
            out = predictor(bank, [m_enc], [m_pred], mask_index=0)
            if isinstance(out, tuple) and len(out) == 2:
                out = out[0]
            tgt = h[:, k * S:(k + 1) * S, :]
            l1s.append((out.float() - tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
            bank = torch.cat([bank, out], dim=1)         # APPEND predicted features (drift)
            bank_idx = torch.cat([bank_idx, temporal_token_idx(num_frames, [k])])
    else:
        for k in range(1, Tp):
            m_enc = expand_mask(temporal_token_idx(num_frames, range(0, k)), b)
            z = encoder(pixel, masks=[m_enc])            # re-encode REAL visible slots
            if isinstance(z, (list, tuple)):
                z = torch.cat(list(z), dim=-1)
            m_pred = expand_mask(temporal_token_idx(num_frames, [k]), b)
            out = predictor(z, [m_enc], [m_pred], mask_index=0)
            if isinstance(out, tuple) and len(out) == 2:
                out = out[0]
            tgt = h[:, k * S:(k + 1) * S, :]
            l1s.append((out.float() - tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.stack(l1s, axis=1).astype(np.float32)      # (B, Tp-1)


def perclip_slope(L: np.ndarray, x) -> np.ndarray:
    """Per-clip OLS slope of L (B, k) against x (k,). Vectorised cov(x,L)/var(x).
    Used by the slope metrics (#1 rollout, #3 tdist, #5 maskratio)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        raise ValueError("perclip_slope: x has zero variance (need ≥2 distinct sweep points)")
    return ((L.astype(np.float64) - L.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / denom


def expand_mask(idx: torch.Tensor, B: int) -> torch.Tensor:
    """(n,) token-index LongTensor → (B, n) on cuda (same mask broadcast across the batch)."""
    return idx.unsqueeze(0).expand(B, -1).contiguous().to("cuda")


__all__ = [
    "load_encoder_predictor", "build_mask_gen", "token_grid", "temporal_token_idx",
    "to_pixel", "masked_predict_l1", "perclip_slope", "expand_mask", "bootstrap_ci",
    "NUM_FRAMES_DEFAULT", "CROP", "TUBELET_SIZE", "PATCH_SIZE",
]
