"""m12e — Predictor-only temporal metric suite (EVALUATION band). GPU-only.

Six mutually-orthogonal predictor latent-L1 metrics, each sweeping ONE axis future_mse holds
fixed (none redundant with future_mse or each other) — see iter/iter16_ablations/
plan_metrics_temporal.md §2.1:
  #1 rollout       multi-step free-running drift slope          ★ (pt_rollout)
  #2 causal        causal future-block L1                          (pt_causal)
  #3 tdist         single-shot L1-vs-Δt slope                      (pt_tdist)
  #4 teacher_free  free−teacher exposure-bias gap               ★ (pt_teacher_free)
  #5 maskratio     L1-vs-mask-ratio slope                          (pt_maskratio)
  #6 order         shuffled−ordered ΔL1                            (pt_order)

ONE standalone entry: loads the ViT-G encoder+predictor ONCE per variant (rule-32 + load cost)
and dispatches --metric {one|all} to the utils/pt_*.py compute fns (which share
utils/predictor_eval.py). Mirrors probe_future_mse.py's forward + paired_per_variant structure.

Gold standard: facebookresearch/vjepa2/app/vjepa_2_1/train.py
GPU-VALIDATION REQUIRED post-eval (§3.1) — written CPU-checked; not yet GPU-smoked (eval holds GPU).

USAGE:
    python -u src/m12e_predictor_temporal.py --POC --stage forward --metric all \\
        --variant vjepa_2_1_frozen --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --action-probe-root outputs/poc/probe_action --local-data data/eval_10k_local \\
        --output-root outputs/poc/predictor_temporal --cache-policy 1 --no-wandb

    python -u src/m12e_predictor_temporal.py --POC --stage paired_per_variant \\
        --metric all --output-root outputs/poc/predictor_temporal --cache-policy 1 --no-wandb
"""
import argparse
import json
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import pt_causal, pt_maskratio, pt_order, pt_rollout, pt_tdist, pt_teacher_free

from utils.action_labels import load_action_labels
from utils.bootstrap import paired_bca
from utils.cache_policy import (
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.frozen_features import ENCODERS, decode_to_tensor
from utils.gpu_batch import cleanup_temp, cuda_cleanup
from utils.predictor_eval import CROP, NUM_FRAMES_DEFAULT, bootstrap_ci, load_encoder_predictor
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics

_PCFG = get_pipeline_config()
CHECKPOINT_EVERY = _PCFG["streaming"]["checkpoint_every"]

# metric name → (compute fn, lower_is_better, one-line interpretation)
METRICS = {
    "rollout":      (pt_rollout.compute,      True,  "free-run drift slope; lower = stabler dynamics"),
    "causal":       (pt_causal.compute,       True,  "causal future-block L1; lower = better"),
    "tdist":        (pt_tdist.compute,        True,  "L1-vs-Δt slope; lower = slower decay w/ distance"),
    "teacher_free": (pt_teacher_free.compute, True,  "free−teacher gap; lower = less exposure bias"),
    "maskratio":    (pt_maskratio.compute,    True,  "L1-vs-maskratio slope; lower = graceful degrade"),
    "order":        (pt_order.compute,        None,  "shuffled−ordered ΔL1; sign = order reliance"),
}

# V-JEPA-only (predictor required) — runtime-discovered from the encoder registry.
KNOWN_VARIANTS = tuple(n for n, s in ENCODERS.items() if s.get("kind") == "vjepa")


def _safe_metric(fn, encoder, predictor, batch, num_frames, min_bs=1):
    """Run a metric on `batch`, sub-batching with OOM backoff (mirrors AdaptiveBatchSizer
    intent). Returns concatenated per-clip array."""
    try:
        return fn(encoder, predictor, batch, num_frames)
    except torch.cuda.OutOfMemoryError:
        cuda_cleanup()
        b = batch.shape[0]
        if b <= min_bs:
            raise
        mid = b // 2
        lo = _safe_metric(fn, encoder, predictor, batch[:mid], num_frames, min_bs)
        hi = _safe_metric(fn, encoder, predictor, batch[mid:], num_frames, min_bs)
        return np.concatenate([lo, hi], axis=0)


def run_forward_stage(args, wb) -> None:
    for need, msg in [(args.variant, "--variant"), (args.encoder_ckpt, "--encoder-ckpt"),
                      (args.local_data, "--local-data"), (args.action_probe_root, "--action-probe-root")]:
        if need is None:
            sys.exit(f"FATAL: --stage forward requires {msg}")
    if not args.variant.startswith("vjepa"):
        sys.exit(f"FATAL: forward only supports vjepa_* variants (predictor required); got {args.variant!r}")

    metrics = list(METRICS) if args.metric == "all" else [args.metric]
    check_gpu()
    print_cgroup_header(prefix="[m12e_predictor_temporal]")
    start_oom_watchdog(prefix="[m12e_predictor_temporal]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for m in metrics:
        agg = out_dir / f"aggregate_{m}.json"
        pcl = out_dir / f"per_clip_{m}.npy"
        if agg.exists() and pcl.exists() and args.cache_policy == "1":
            print(f"  [keep] {m}: {agg.name} present — skipping (--cache-policy 2 to redo)")
            continue
        guarded_delete(pcl, args.cache_policy, f"per_clip_{m}")
        guarded_delete(agg, args.cache_policy, f"aggregate_{m}")
        todo.append(m)
    if not todo:
        print("  all requested metrics cached — nothing to do")
        return

    labels = load_action_labels(args.action_probe_root / "action_labels.json")
    test_keys = [k for k, info in labels.items() if info["split"] == "test"]
    if not test_keys:
        sys.exit("FATAL: 0 test clips in action_labels.json — re-run the labels stage")
    print(f"Forward {todo} on {len(test_keys)} test clips, variant={args.variant}")

    encoder, predictor, embed_concat = load_encoder_predictor(args.encoder_ckpt, args.num_frames)
    print(f"  encoder+predictor loaded (concat dim={embed_concat})")

    # Resume: one ckpt holds the shared keys + each metric's per-clip accumulator in lockstep.
    ckpt = out_dir / ".m12e_ckpt.npz"
    acc = {m: [] for m in todo}
    keys_acc, processed = [], set()
    if ckpt.exists():
        try:
            d = np.load(ckpt, allow_pickle=True)
            keys_acc = list(d["keys"])
            processed = set(keys_acc)
            for m in todo:
                if m in d.files:
                    acc[m] = list(d[m].tolist())
                if len(acc[m]) != len(keys_acc):
                    raise ValueError(f"resume ckpt {m} len {len(acc[m])} != keys {len(keys_acc)}")
            print(f"  Resume: {len(processed)} clips cached")
        except (ValueError, KeyError) as e:
            print(f"  WARN: resume ckpt unusable ({e}); starting fresh")
            acc = {m: [] for m in todo}
            keys_acc, processed = [], set()

    target = set(test_keys) - processed
    if target:
        bs = args.batch_size
        pbar = make_pbar(total=len(test_keys), desc="m12e", unit="clip", initial=len(processed))
        tmp_root = out_dir / "tmp_decode"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=tmp_root)
        clip_q, tar_stop, _r = iter_clips_parallel(
            local_data=args.local_data, subset_keys=target, processed_keys=processed)
        pend_t, pend_k, n_since = [], [], 0

        def _flush():
            nonlocal pend_t, pend_k, n_since
            if not pend_t:
                return
            batch = torch.stack(pend_t)
            for m in todo:
                vals = _safe_metric(METRICS[m][0], encoder, predictor, batch, args.num_frames)
                if len(vals) != len(pend_k):
                    sys.exit(f"FATAL: metric {m} returned {len(vals)} vals for {len(pend_k)} clips")
                acc[m].extend(float(v) for v in vals)
            keys_acc.extend(pend_k)
            pbar.update(len(pend_k))
            n_since += len(pend_k)
            pend_t, pend_k = [], []

        try:
            while True:
                try:
                    item = clip_q.get(timeout=300)
                except queue.Empty:
                    print("  WARN: clip queue timeout — flushing pending")
                    break
                if item is None:
                    break
                clip_key, mp4 = item
                t = decode_to_tensor(mp4, tmp_dir, clip_key, args.num_frames, CROP)
                if t is None:
                    continue
                pend_t.append(t)
                pend_k.append(clip_key)
                if len(pend_t) >= bs:
                    _flush()
                    if n_since >= CHECKPOINT_EVERY:
                        np.savez(ckpt.with_suffix(".tmp.npz"),
                                 keys=np.array(keys_acc, dtype=object),
                                 **{m: np.array(acc[m], dtype=np.float32) for m in todo})
                        ckpt.with_suffix(".tmp.npz").replace(ckpt)
                        n_since = 0
            _flush()
        finally:
            tar_stop.set()
            pbar.close()

    for m in todo:
        arr = np.asarray(acc[m], dtype=np.float32)
        if arr.size == 0:
            sys.exit(f"FATAL: metric {m} produced 0 clips — pipeline failure")
        save_array_checkpoint(arr, out_dir / f"per_clip_{m}.npy")
        np.save(out_dir / f"clip_keys_{m}.npy", np.array(keys_acc, dtype=object))
        ci = bootstrap_ci(arr.astype(np.float64))
        save_json_checkpoint({
            "variant": args.variant, "metric": m, "n_test": int(arr.size),
            "mean": round(float(arr.mean()), 6), "std": round(float(arr.std()), 6),
            "ci": ci, "lower_is_better": METRICS[m][1], "interpretation": METRICS[m][2],
            "num_frames": args.num_frames, "encoder_ckpt": str(args.encoder_ckpt),
        }, out_dir / f"aggregate_{m}.json")
        print(f"  {m}: mean={arr.mean():.5f} std={arr.std():.5f} N={arr.size}")
        log_metrics(wb, {f"{args.variant}_{m}_mean": float(arr.mean())})


def run_paired_per_variant_stage(args, wb) -> None:
    """Pairwise BCa Δ across discovered vjepa variants, per metric. Mirrors
    probe_future_mse.run_paired_per_variant_stage."""
    metrics = list(METRICS) if args.metric == "all" else [args.metric]
    out = {"metrics": {}}
    for m in metrics:
        by_variant = {}
        for v in KNOWN_VARIANTS:
            pcl = args.output_root / v / f"per_clip_{m}.npy"
            keys = args.output_root / v / f"clip_keys_{m}.npy"
            agg = args.output_root / v / f"aggregate_{m}.json"
            if not (pcl.exists() and keys.exists() and agg.exists()):
                continue
            by_variant[v] = {
                "agg": json.loads(agg.read_text()),
                "vals": np.load(pcl).astype(np.float64),
                "keys": [str(k) for k in np.load(keys, allow_pickle=True)],
            }
        avail = sorted(by_variant)
        deltas = {}
        for i, a in enumerate(avail):
            for b in avail[i + 1:]:
                ka, kb = by_variant[a]["keys"], by_variant[b]["keys"]
                shared = sorted(set(ka) & set(kb))
                if not shared:
                    sys.exit(f"FATAL [{m}]: {a} vs {b} share 0 clips — re-extract on same subset")
                ai = {k: j for j, k in enumerate(ka)}
                bi = {k: j for j, k in enumerate(kb)}
                d = (np.array([by_variant[a]["vals"][ai[k]] for k in shared])
                     - np.array([by_variant[b]["vals"][bi[k]] for k in shared]))
                bca = paired_bca(d)
                deltas[f"{a}_minus_{b}"] = {
                    "n": len(shared), "delta_mean": round(float(d.mean()), 6),
                    "delta_ci_lo": round(float(bca["ci_lo"]), 6),
                    "delta_ci_hi": round(float(bca["ci_hi"]), 6),
                    "p_value": float(bca["p_value_vs_zero"]),
                }
        out["metrics"][m] = {
            "by_variant": {v: by_variant[v]["agg"] for v in avail},
            "pairwise_deltas": deltas,
            "lower_is_better": METRICS[m][1],
        }
    save_json_checkpoint(out, args.output_root / "predictor_temporal_per_variant.json")
    log_metrics(wb, {"n_metrics": len(out["metrics"])})
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m12e predictor-only temporal metric suite")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC", action="store_true")
    p.add_argument("--FULL", action="store_true")
    p.add_argument("--stage", required=True, choices=["forward", "paired_per_variant"])
    p.add_argument("--metric", required=True, choices=list(METRICS) + ["all"])
    p.add_argument("--variant", choices=list(KNOWN_VARIANTS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--motion-aux-head", type=Path, default=None,
                   help="head-cell symmetry (reserved; not yet consumed by these metrics)")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="probe_action output dir (action_labels.json test split)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--batch-size", type=int, default=4,
                   help="clips per forward (predictor fwd is heavy; OOM-subbatched downward)")
    p.add_argument("--seed", type=int, default=_PCFG["probe"]["seed"])
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"m12e_predictor_temporal_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        torch.manual_seed(args.seed)
        if args.stage == "forward":
            t0 = time.time()
            run_forward_stage(args, wb)
            print(f"forward stage: {time.time() - t0:.0f}s")
        else:
            run_paired_per_variant_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
