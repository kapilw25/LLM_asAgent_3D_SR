#!/usr/bin/env python3
"""Clear regenerable resume anchors (`*ckpt_latest*.pt` + `*ckpt_stage*.pt`) for arms whose
training is FINISHED — reclaiming disk before an HF upload — WITHOUT touching the final
deliverables (`student_encoder.pt`, `*ckpt_best*.pt`, `motion_aux_head.pt`, eval `.npy/.json`,
`probe.pt`, plots) or the resume state of PARKED arms.

WHY a dedicated guarded helper (not `find -delete`):
  · delete-protection (src/CLAUDE.md): destructive deletes live in .py behind `--cache-policy`,
    never as shell `rm`/`find -delete` (the checkpoint-protection hook blocks those anyway).
  · the parked-arm carve-out is the whole point: an arm with NO `student_encoder.pt` (training
    interrupted / parked — e.g. iter18's cassle_encoder, ewc_encoder per runbook §0.C C4) STILL
    needs its `*ckpt_latest*.pt` to resume. A blanket anchor-wipe would force it to retrain from
    scratch. This helper PRESERVES anchors for any arm missing its final exported encoder.

RULE: clear an arm's resume anchors IFF `<arm>/student_encoder.pt` exists (training done →
      `ckpt_best.pt` + `student_encoder.pt` are exported → the latest/stage anchors are pure
      regenerable scaffolding). Otherwise preserve them and say so.

USAGE:
  # DRY-RUN (policy=1=keep, default): list every anchor that WOULD be cleared + what's preserved.
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --cache-policy 1

  # DELETE (policy=2=recompute): actually remove the DONE-arm anchors via guarded_delete.
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --cache-policy 2 \
      2>&1 | tee logs/clear_resume_anchors_$(date +%Y%m%d_%H%M%S).log

  # also drop the predictor-bearing best ckpts (only if you will NEVER re-run eval):
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --also-best --cache-policy 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cache_policy import (  # noqa: E402
    add_cache_policy_arg,
    guarded_delete,
    is_recompute,
    resolve_cache_policy_interactive,
)

# Regenerable resume scaffolding — safe to drop once an arm has exported its final encoder.
_ANCHOR_SUBSTRINGS = ("ckpt_latest", "ckpt_stage")
# The predictor-bearing best ckpt — needed ONLY to re-run eval (stage 8); opt-in via --also-best.
_BEST_SUBSTRING = "ckpt_best"
# Final deliverables that are NEVER touched (here for the docstring/self-check, not matched).
_KEEP_ALWAYS = ("student_encoder.pt", "motion_aux_head.pt", "probe.pt")


def _arm_anchors(arm_dir: Path, include_best: bool):
    """(anchor .pt paths, total_bytes) for one arm dir — latest/stage always, best if requested."""
    subs = _ANCHOR_SUBSTRINGS + ((_BEST_SUBSTRING,) if include_best else ())
    hits = [p for p in arm_dir.rglob("*.pt") if any(s in p.name for s in subs)]
    return hits, sum(p.stat().st_size for p in hits)


def _gb(n):
    return n / 1024 ** 3


def clear_resume_anchors(root: Path, policy: str, include_best: bool) -> dict:
    """Walk `root`'s arm dirs; clear (policy=2) or preview (policy=1) anchors for DONE arms only.
    Returns {"cleared_bytes", "preserved_bytes", "done": [...], "parked": [...]}."""
    if not root.is_dir():
        sys.exit(f"FATAL: --root not a directory: {root}")
    arms = sorted(d for d in root.iterdir() if d.is_dir())
    if not arms:
        sys.exit(f"FATAL: no arm subdirectories under {root}")
    target_bytes = preserved_bytes = 0   # target = anchors on DONE arms (cleared@2 / would-clear@1)
    done, parked = [], []
    recompute = is_recompute(policy)
    print(f"  root: {root}  ·  arms: {len(arms)}  ·  policy={policy} "
          f"({'DELETE' if recompute else 'dry-run/keep'})  ·  also-best={include_best}\n")
    for arm in arms:
        hits, nbytes = _arm_anchors(arm, include_best)   # measured BEFORE any delete
        if not (arm / "student_encoder.pt").exists():
            # PARKED / interrupted — its anchors are the only way to resume → preserve, loudly.
            preserved_bytes += nbytes
            parked.append(arm.name)
            print(f"  ⏸️  PRESERVE  {arm.name:34s} no student_encoder.pt (parked/resumable) "
                  f"— kept {_gb(nbytes):.1f} G of anchors")
            continue
        done.append(arm.name)
        target_bytes += nbytes
        if not hits:
            print(f"  ✅ clean    {arm.name:34s} DONE · no anchors left")
            continue
        print(f"  🧹 {'deleting' if recompute else 'WOULD clear':12s} {arm.name:34s} DONE · "
              f"{len(hits)} anchor(s) · {_gb(nbytes):.1f} G")
        for p in hits:                                   # guarded_delete: no-op unless policy=2
            guarded_delete(p, policy, label=f"resume-anchor {arm.name}/{p.name}")
    return {"target_bytes": target_bytes, "preserved_bytes": preserved_bytes,
            "done": done, "parked": parked}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="training output root for ONE backbone, e.g. outputs/poc/vjepa_2_1_vitG")
    ap.add_argument("--also-best", action="store_true",
                    help="ALSO clear *ckpt_best*.pt (the predictor-bearing best ckpt) — only safe if "
                         "you will never re-run eval (stage 8 needs it). OFF by default.")
    add_cache_policy_arg(ap)
    args = ap.parse_args()
    policy = resolve_cache_policy_interactive(args.cache_policy)

    res = clear_resume_anchors(args.root, policy, args.also_best)
    print()
    if is_recompute(policy):
        print(f"  ✅ CLEARED ~{_gb(res['target_bytes']):.1f} G of resume anchors "
              f"from {len(res['done'])} DONE arms.")
    else:
        print(f"  🔎 DRY-RUN: would clear ~{_gb(res['target_bytes']):.1f} G from "
              f"{len(res['done'])} DONE arms — rerun with --cache-policy 2 to delete.")
    print(f"  ⏸️  preserved ~{_gb(res['preserved_bytes']):.1f} G of anchors for "
          f"{len(res['parked'])} PARKED arm(s): {res['parked'] or '—'}")


if __name__ == "__main__":
    main()
