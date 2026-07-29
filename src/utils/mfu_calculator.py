"""Model FLOPs Utilization (MFU) — training + inference — for m09* / m12*.

Gold-standard analytic FLOPs-per-token (PaLM App. B / nanoGPT ``estimate_mfu`` /
stas00 ml-engineering):

    training flops/token = 6N + 12*L*H*Q*T        (fwd 2N + act-grad 2N + wt-grad 2N)
    forward  flops/token = 2N + 4*L*H*Q*T         (= 1/3 of training; bwd ~= 2x fwd)
    MFU = flops_per_token * tokens_per_sec / peak_flops

Frozen-aware (surgery / LoRA freeze most params, so the backward only runs on the
TRAINABLE params). We therefore log BOTH, per user decision 2026-07-28:

    honest   flops/token = 2*N_fwd + 4*N_train + 12*L*H*Q*T   (== 6N when N_train==N_fwd)
    dense-6N flops/token = 6*N_fwd            + 12*L*H*Q*T     (all-params-trainable convention)

Inference MFU uses the forward-only count:

    inference flops/token = 2*N_fwd + 4*L*H*Q*T

The peak-FLOPs denominator is MEASURED at runtime by a dense BF16 GEMM micro-benchmark
on the actual device (``measure_peak_flops``) — hardware-agnostic, no hardcoded per-GPU
number, and free of the 2:4-sparse-vs-dense ambiguity (a dense matmul measures dense
peak by construction). It reports the card's *achievable* dense-GEMM ceiling
(typically ~85-95% of the marketing spec), which is the denominator practitioners prefer.

Attention term note: L/H/Q are the ENCODER's (it dominates); the predictor's params are
counted in N but its (smaller, shorter-T) attention quadratic is not added separately.

Refs (gold standard):
  PaLM, Chowdhery et al. 2022, Appendix B.
  https://github.com/karpathy/nanoGPT/blob/master/model.py  (estimate_mfu)
  https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:                     # type-only: torch stays out of the runtime import path
    import torch


def measure_peak_flops(device: torch.device, dtype: torch.dtype,
                       size: int, iters: int, warmup: int) -> float:
    """Achieved dense matmul FLOPS on ``device`` at ``dtype`` — the MFU denominator.

    Times ``iters`` square matmuls of shape (size, size); each is 2*size**3 FLOPs.
    All parameters are supplied by the caller (yaml-resolved) — no hardcoded magic
    numbers per the project no-defaults rule.
    """
    import torch  # deferred: the analytic MFU math stays importable without torch (Mac/CPU)
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    c = None
    for _ in range(warmup):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    if dt <= 0:
        raise RuntimeError(f"GEMM peak benchmark measured non-positive time dt={dt}")
    _ = float(c.reshape(-1)[0].item())  # force materialization; block dead-code elim
    return (2.0 * size ** 3 * iters) / dt


def measured_peak_flops(pipeline_cfg: dict, device: torch.device,
                        dtype=None) -> float:
    """Read the GEMM-benchmark parameters from pipeline.yaml ``mfu`` and measure peak.

    FAIL LOUD: missing yaml keys raise KeyError (no silent defaults).
    ``dtype`` defaults to bf16, resolved lazily so this module imports without torch.
    """
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    m = pipeline_cfg["mfu"]
    return measure_peak_flops(
        device=device, dtype=dtype,
        size=m["gemm_bench_size"], iters=m["gemm_bench_iters"],
        warmup=m["gemm_bench_warmup"])


def tokens_per_clip(crop_size: int, patch_size: int, tubelet_size: int,
                    num_frames: int) -> int:
    """Spatiotemporal token count for one V-JEPA clip: (T/tubelet) * (H/patch)^2."""
    if num_frames % tubelet_size != 0:
        raise ValueError(f"num_frames {num_frames} not divisible by tubelet_size {tubelet_size}")
    if crop_size % patch_size != 0:
        raise ValueError(f"crop_size {crop_size} not divisible by patch_size {patch_size}")
    return (num_frames // tubelet_size) * (crop_size // patch_size) ** 2


class MFUCalculator:
    """Analytic MFU for a transformer stack. See module docstring for the formulas."""

    def __init__(self, n_fwd: int, n_train: int, n_layers: int, embed_dim: int,
                 seq_len: int, peak_flops: float):
        if peak_flops <= 0:
            raise ValueError(f"peak_flops must be > 0, got {peak_flops}")
        if n_train > n_fwd:
            raise ValueError(f"n_train ({n_train}) exceeds n_fwd ({n_fwd}) — impossible")
        if min(n_fwd, n_layers, embed_dim, seq_len) <= 0:
            raise ValueError("n_fwd/n_layers/embed_dim/seq_len must all be > 0")
        self.n_fwd = n_fwd
        self.n_train = n_train
        self.peak = float(peak_flops)
        self.seq_len = seq_len
        # attn flops/token = n_layers*(n_heads*head_dim)*seq_len = n_layers*embed_dim*seq_len
        attn = n_layers * embed_dim * seq_len
        self.f_fwd = 2 * n_fwd + 4 * attn                       # forward / inference
        self.f_train = 2 * n_fwd + 4 * n_train + 12 * attn      # honest fwd+bwd (frozen-aware)
        self.f_train_6n = 6 * n_fwd + 12 * attn                 # dense 6N convention

    def training_mfu(self, tokens_per_sec: float) -> float:
        return tokens_per_sec * self.f_train / self.peak

    def training_mfu_6n(self, tokens_per_sec: float) -> float:
        return tokens_per_sec * self.f_train_6n / self.peak

    def inference_mfu(self, tokens_per_sec: float) -> float:
        return tokens_per_sec * self.f_fwd / self.peak

    def tokens_per_sec_from_steps(self, steps_per_sec: float,
                                  batch_size_clips: int) -> float:
        """steps/sec -> tokens/sec = steps/sec * clips/step * tokens/clip (self.seq_len)."""
        return steps_per_sec * batch_size_clips * self.seq_len

    def report(self, tokens_per_sec: float, mode: str) -> dict:
        """Small dict for real-time step logging (JSONL / pbar / wandb)."""
        if mode == "train":
            return {
                "mfu": round(self.training_mfu(tokens_per_sec), 6),
                "mfu_6n": round(self.training_mfu_6n(tokens_per_sec), 6),
                "tokens_per_sec": round(tokens_per_sec, 1),
            }
        if mode == "inference":
            return {
                "inference_mfu": round(self.inference_mfu(tokens_per_sec), 6),
                "tokens_per_sec": round(tokens_per_sec, 1),
            }
        raise ValueError(f"mode must be 'train' or 'inference', got {mode!r}")


def build_calculator(*, forward_modules, trainable_modules, model_cfg: dict,
                     num_frames: int, peak_flops: float) -> MFUCalculator:
    """Construct an MFUCalculator with N counted at runtime and arch from model_cfg.

    forward_modules   : iterable of nn.Modules whose weights do FLOPs in the forward
                        pass (e.g. [encoder, predictor]) -> N_fwd.
    trainable_modules : iterable of nn.Modules holding the params that receive grads
                        -> N_train (only p.requires_grad params are counted).
    """
    n_fwd = sum(p.numel() for mod in forward_modules for p in mod.parameters())
    n_train = sum(p.numel() for mod in trainable_modules for p in mod.parameters()
                  if p.requires_grad)
    seq_len = tokens_per_clip(
        crop_size=model_cfg["crop_size"], patch_size=model_cfg["patch_size"],
        tubelet_size=model_cfg["tubelet_size"], num_frames=num_frames)
    return MFUCalculator(
        n_fwd=n_fwd, n_train=n_train, n_layers=model_cfg["depth"],
        embed_dim=model_cfg["embed_dim"], seq_len=seq_len, peak_flops=peak_flops)


def _introspect_depth(encoder) -> int:
    """Transformer-block count for a ViT encoder (.blocks; HF-style .encoder.layer)."""
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None and hasattr(blocks, "__len__"):
        return len(blocks)
    enc = getattr(encoder, "encoder", None)
    layer = getattr(enc, "layer", None) if enc is not None else None
    if layer is not None and hasattr(layer, "__len__"):
        return len(layer)
    raise AttributeError(
        f"cannot introspect transformer depth from {type(encoder).__name__}: "
        "expected .blocks or .encoder.layer")


def build_inference_calculator(*, encoder, embed_dim: int, num_frames: int,
                               crop_size: int, patch_size: int, tubelet_size: int,
                               peak_flops: float) -> MFUCalculator:
    """Inference-only MFU for a frozen/adapted encoder — N and depth introspected at runtime."""
    n_fwd = sum(p.numel() for p in encoder.parameters())
    seq_len = tokens_per_clip(crop_size=crop_size, patch_size=patch_size,
                              tubelet_size=tubelet_size, num_frames=num_frames)
    return MFUCalculator(n_fwd=n_fwd, n_train=0, n_layers=_introspect_depth(encoder),
                         embed_dim=embed_dim, seq_len=seq_len, peak_flops=peak_flops)


def _selftest():
    """CPU self-test of the analytic math (no GPU needed).

    USAGE: source venv_walkindia/bin/activate && python3 -m utils.mfu_calculator --selftest
    """
    # V-JEPA 2.1 ViT-g (1B) at 16 training frames, 384 crop, patch 16, tubelet 2.
    cfg = {"embed_dim": 1408, "num_heads": 22, "depth": 40,
           "crop_size": 384, "patch_size": 16, "tubelet_size": 2}
    T = tokens_per_clip(cfg["crop_size"], cfg["patch_size"], cfg["tubelet_size"], 16)
    assert T == 8 * 24 * 24 == 4608, T
    N = 1_010_000_000  # ~1.01B params (encoder + predictor, illustrative)
    peak = 5.0e14      # 500 TFLOPS dense bf16 (illustrative; real run measures it)
    # Full fine-tune: honest == 6N.
    full = MFUCalculator(N, N, cfg["depth"], cfg["embed_dim"], T, peak)
    assert full.f_train == full.f_train_6n, (full.f_train, full.f_train_6n)
    # Surgery-like: 10% trainable -> honest < 6N.
    surg = MFUCalculator(N, N // 10, cfg["depth"], cfg["embed_dim"], T, peak)
    assert surg.f_train < surg.f_train_6n
    assert surg.f_fwd < surg.f_train
    # Realistic full-1B throughput: ~23 s/step at batch 112 (matches the measured
    # continual-SSL full run, 17.7 h). MFU > 1 would signal an unrealistic step time
    # or a wrong peak FLOPs — a useful diagnostic — so we assert sanity here.
    tok_s = 112 * T / 23.0
    rep = surg.report(tok_s, "train")
    inf = surg.report(tok_s, "inference")
    assert 0 < rep["mfu"] < 1, rep
    assert 0 < rep["mfu_6n"] < 1, rep
    assert 0 < inf["inference_mfu"] < 1, inf
    assert rep["mfu"] < rep["mfu_6n"], (rep, "honest must be < 6N for frozen-heavy")
    print(f"T(tokens/clip)={T}  N={N/1e9:.2f}B  peak={peak/1e12:.0f} TFLOPS  (~23 s/step, batch 112)")
    print(f"tokens/sec={tok_s:,.0f}")
    print(f"train:     mfu(honest)={rep['mfu']:.4f}  mfu_6n={rep['mfu_6n']:.4f}")
    print(f"inference: mfu={inf['inference_mfu']:.4f}")
    print("OK: analytic MFU self-test passed.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MFU calculator self-test / peak benchmark")
    ap.add_argument("--selftest", action="store_true",
                    help="run the CPU analytic self-test (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.error("nothing to do; pass --selftest (GPU peak benchmark runs inside m09/m12)")
