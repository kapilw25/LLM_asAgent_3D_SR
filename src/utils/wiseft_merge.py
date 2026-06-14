"""
WiSE-FT weight-space merge — lever #1 of plan_outperform_FT.md (iter18 2026-06-13).

Produces a NEW encoder by linearly interpolating a fine-tuned (surgery) encoder with the
FROZEN base encoder, WITHOUT any retraining:

        θ*(α) = (1 − α) · θ_frozen  +  α · θ_surgery          α ∈ [0, 1]

This recovers Frozen-level temporal structure (TCC) that fine-tuning erodes, while keeping
surgery's prediction gains — the interpolation curve is typically convex, so a mid α can
Pareto-dominate either endpoint. Pick α on the VAL split, then that ckpt IS the new arm.

Gold standard: Wortsman et al., "Robust fine-tuning of zero-shot models" (WiSE-FT), CVPR 2022
(arXiv:2109.01903); generalised by "Model soups" (ICML 2022, arXiv:2203.05482).

ADDITIVE: writes ONLY new files under a NEW arm dir — the surgery + frozen checkpoints are
read-only, so previously-trained endpoints are untouched and remain comparable in eval.

USAGE:
  # inspect both checkpoints' state-dict structure first (no write):
  python -u src/utils/wiseft_merge.py --inspect \
      --frozen-ckpt checkpoints/vjepa2_1_vitG_384.pt \
      --surgery-ckpt outputs/poc/vjepa_2_1_vitG/m09c_surgery_3stage_DI_encoder/student_encoder.pt

  # sweep α and write one candidate per α (for val-selection):
  python -u src/utils/wiseft_merge.py \
      --frozen-ckpt  checkpoints/vjepa2_1_vitG_384.pt \
      --surgery-ckpt outputs/poc/vjepa_2_1_vitG/m09c_surgery_3stage_DI_encoder/student_encoder.pt \
      --out-dir      outputs/poc/vjepa_2_1_vitG/m09c_surgical_3stage_DI_wiseft_encoder \
      --alphas 0.5 0.6 0.7 0.8

  # write the single chosen α as the arm's exported encoder:
  python -u src/utils/wiseft_merge.py --alpha 0.7 ... --out-dir <arm dir>   # → <arm dir>/student_encoder.pt
"""
import argparse
import sys
from pathlib import Path

import torch

# Common containers a checkpoint's parameter dict may hide under.
_STATE_KEYS = ("state_dict", "model", "encoder", "module", "model_state_dict")


def _find_state_dict(obj, label: str) -> dict:
    """Return the {name: Tensor} mapping from a loaded checkpoint, FAIL LOUD if not found."""
    if isinstance(obj, dict):
        # raw state_dict: dict whose values are mostly tensors
        tensor_vals = sum(1 for v in obj.values() if torch.is_tensor(v))
        if tensor_vals >= max(1, len(obj) // 2):
            return obj
        for k in _STATE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    sys.exit(f"FATAL: could not locate a parameter state_dict inside {label} "
             f"(top-level type={type(obj).__name__}, keys={list(obj)[:8] if isinstance(obj, dict) else 'n/a'}). "
             f"Add the container key to _STATE_KEYS.")


def _strip(prefixes, sd: dict) -> dict:
    """Drop a leading module prefix (e.g. 'encoder.', 'backbone.') so the two dicts align."""
    for p in prefixes:
        if all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        sys.exit(f"FATAL: {label} not found: {path}")
    sd = _find_state_dict(torch.load(str(path), map_location="cpu", weights_only=False), label)
    return _strip(("module.", "encoder.", "backbone."), sd)


def _common_float_keys(a: dict, b: dict):
    """Keys present + same-shape + floating in BOTH — only these get interpolated."""
    inter, skipped = [], []
    for k in a:
        if k not in b:
            skipped.append((k, "missing in surgery"))
        elif a[k].shape != b[k].shape:
            skipped.append((k, f"shape {tuple(a[k].shape)} vs {tuple(b[k].shape)}"))
        elif not torch.is_floating_point(a[k]):
            skipped.append((k, "non-float (copied from surgery)"))
        else:
            inter.append(k)
    return inter, skipped


def main():
    ap = argparse.ArgumentParser(description="WiSE-FT weight-space merge (frozen × surgery).")
    ap.add_argument("--frozen-ckpt",  type=Path, required=True, help="base FROZEN encoder ckpt")
    ap.add_argument("--surgery-ckpt", type=Path, required=True, help="fine-tuned surgery student_encoder.pt")
    ap.add_argument("--out-dir",      type=Path, default=None, help="arm output dir (required unless --inspect)")
    ap.add_argument("--alpha",  type=float, default=None, help="single α → writes <out-dir>/student_encoder.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="α sweep → writes <out-dir>/alpha<α>/student_encoder.pt per α (for val-selection)")
    ap.add_argument("--inspect", action="store_true", help="report state-dict alignment, write nothing")
    a = ap.parse_args()

    fro = _load(a.frozen_ckpt, "frozen-ckpt")
    sur = _load(a.surgery_ckpt, "surgery-ckpt")
    inter, skipped = _common_float_keys(fro, sur)
    print(f"  [wiseft] frozen={len(fro)} keys · surgery={len(sur)} keys · "
          f"interpolated={len(inter)} · skipped={len(skipped)}")
    for k, why in skipped[:8]:
        print(f"    skip {k}: {why}")
    if not inter:
        sys.exit("FATAL: 0 interpolable keys — the two checkpoints don't align (see skips above).")

    if a.inspect:
        print("  [wiseft] --inspect only; nothing written.")
        return

    if a.out_dir is None:
        sys.exit("FATAL: --out-dir required unless --inspect")
    alphas = a.alphas if a.alphas is not None else [a.alpha if a.alpha is not None else 0.7]

    for al in alphas:
        if not 0.0 <= al <= 1.0:
            sys.exit(f"FATAL: alpha {al} out of [0,1]")
        merged = dict(sur)                                  # start from surgery (keeps non-float buffers)
        for k in inter:
            merged[k] = (1.0 - al) * fro[k].float() + al * sur[k].float()
        sub = a.out_dir if (a.alphas is None) else (a.out_dir / f"alpha{al:g}")
        sub.mkdir(parents=True, exist_ok=True)
        out = sub / "student_encoder.pt"
        torch.save(merged, str(out))
        print(f"  [wiseft] α={al:g} → {out}  ({len(inter)} keys blended)")


if __name__ == "__main__":
    main()
