"""m12f — Encoder-temporal metric suite (EVALUATION band). GPU-only.

Four encoder-feature-based temporal metrics (plan_metrics_temporal.md §6) — none read the
predictor; all read frozen-or-surgical ENCODER features. Trainable heads for AoT/TOV/Pace
(tiny linear, train on train-split features, eval per-clip on test); TCC is training-free
(soft-NN cycle-back alignment on per-frame features).

  --metric aot   Arrow-of-Time  (Wei CVPR18)            ★ direction
  --metric tov   TOV / VCOP     (Misra ECCV16 / Xu CVPR19) ★ order
  --metric pace  Pace           (Wang ECCV20)             ★ rate
  --metric tcc   TCC (training-free) (Dwibedi CVPR19)    ★ correspondence

ONE standalone entry: loads the encoder ONCE per variant (rule-32 + load cost) via
utils.predictor_eval.load_encoder_only (no predictor build — iter16 §3.3 R2).
Dispatches --metric {one|all} to the et_*.py util fns. Mirrors m12e_predictor_temporal.py's
forward + paired_per_variant structure. Pace requires OVERSAMPLE decode (--pace-source-frames
= num_frames × max(strides)); the orchestrator switches the decode frame-count for the pace
pass only.

ALL sweep hyperparameters (n_permutations, strides, head LR/epochs/WD/BS, TCC temperature)
arrive via argparse — NO module-level constants per CLAUDE.md "no hardcoded values in Python".
The thin shell (run_eval.sh) will source them from `configs/pipeline.yaml probe.encoder_temporal`
at §3.3 integration.

Gold standard: docstring of each utils/et_*.py — Wei CVPR18, Misra ECCV16, Xu CVPR19, Wang
ECCV20, Dwibedi CVPR19.
GPU-VALIDATION REQUIRED post-eval (§3.1) — written CPU-checked; eval holds GPU now.

USAGE:
    python -u src/m12f_encoder_temporal.py --POC --stage forward --metric all \\
        --variant vjepa_2_1_frozen --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --action-probe-root outputs/poc/probe_action --local-data data/eval_10k_local \\
        --output-root outputs/poc/encoder_temporal \\
        --tubelet-size 2 --tov-n-permutations 4 --pace-strides 1,2,4 \\
        --pace-source-frames 64 --tcc-temperature 0.1 \\
        --head-lr 1e-3 --head-epochs 20 --head-weight-decay 1e-4 --head-batch-size 64 \\
        --cache-policy 1 --no-wandb
"""
import argparse
import json
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import et_aot, et_pace, et_tcc, et_tov  # noqa: E402

from utils.action_labels import load_action_labels  # noqa: E402
from utils.bootstrap import paired_bca  # noqa: E402
from utils.cache_policy import (  # noqa: E402
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog  # noqa: E402
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint  # noqa: E402
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config  # noqa: E402
from utils.data_download import ensure_local_data, iter_clips_parallel  # noqa: E402
from utils.frozen_features import ENCODERS, decode_to_tensor  # noqa: E402
from utils.gpu_batch import cleanup_temp, cuda_cleanup  # noqa: E402
from utils.predictor_eval import CROP, NUM_FRAMES_DEFAULT, bootstrap_ci, load_encoder_only  # noqa: E402
from utils.progress import make_pbar  # noqa: E402
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics  # noqa: E402

_PCFG = get_pipeline_config()
CHECKPOINT_EVERY = _PCFG["streaming"]["checkpoint_every"]

# metric → (lower_is_better, interpretation). TCC writes 2 per-clip arrays.
METRICS = {
    "aot":  (False, "Arrow-of-Time binary acc; higher = encoder preserves temporal direction"),
    "tov":  (False, "TOV/VCOP top-1 over n_permutations; higher = encoder retains frame order"),
    "pace": (False, "Pace top-1 over playback rates; higher = encoder is rate-sensitive"),
    "tcc":  (None,  "TCC cycle-back error (lower better) + Kendall's τ (higher better)"),
}


def _resolve_metrics(metric_arg):
    """Expand --metric all → the 4 metrics, then DROP any listed in ET_SKIP_METRICS (comma list). Lets
    the orchestrator skip an expensive metric without a code change — e.g. ET_SKIP_METRICS=pace, since
    pace needs an uncacheable nf64 oversample-decode (~28 s/clip cold) that dwarfs aot/tov/tcc. A single
    --metric is returned verbatim (the skip-list is ignored — you asked for exactly that one)."""
    if metric_arg != "all":
        return [metric_arg]                              # explicit single metric → run exactly it (skip-list ignored)
    skip = {m.strip() for m in os.environ.get("ET_SKIP_METRICS", "").split(",") if m.strip()}
    return [m for m in METRICS if m not in skip]

# V-JEPA-only registry mirror (image-JEPAs skipped — they have no native temporal axis here).
KNOWN_VARIANTS = tuple(n for n, s in ENCODERS.items() if s.get("kind") == "vjepa")


def _gen_permutations(n_permutations, T, seed):
    """Identity + (n_permutations - 1) deterministic non-identity permutations of length T.
    FAIL LOUD if T is too small to provide enough distinct non-identity permutations."""
    if n_permutations < 2:
        raise SystemExit(f"FATAL: --tov-n-permutations must be >= 2; got {n_permutations}")
    if T < 3:
        raise SystemExit(f"FATAL: T={T} too small for TOV (need >=3 frames)")
    perms = [torch.arange(T)]
    seen = {tuple(perms[0].tolist())}
    g = torch.Generator(device="cpu").manual_seed(seed)
    attempts = 0
    while len(perms) < n_permutations:
        cand = torch.randperm(T, generator=g)
        key = tuple(cand.tolist())
        if key not in seen:
            perms.append(cand)
            seen.add(key)
        attempts += 1
        if attempts > 10_000:
            raise SystemExit(
                f"FATAL: could not draw {n_permutations} distinct permutations for T={T}")
    return perms


def _run_metric_forward(metric, encoder, batch, num_frames, args):
    """Per-metric dispatch — returns (per_example_features, per_example_labels) on CPU, where
    each clip contributes K variant examples (K=2 aot, K=n_permutations tov, K=len(strides)
    pace, K=1 tcc-per-frame stack)."""
    if metric == "aot":
        f_fwd, f_rev = et_aot.compute_features(encoder, batch, num_frames, args.tubelet_size)
        feats = torch.cat([f_fwd, f_rev], dim=0)
        labels = torch.cat([torch.zeros(f_fwd.shape[0], dtype=torch.long),
                            torch.ones(f_rev.shape[0], dtype=torch.long)], dim=0)
        return feats, labels
    if metric == "tov":
        T = batch.shape[1]
        perms = _gen_permutations(args.tov_n_permutations, T, seed=args.seed)
        stack = et_tov.compute_features(encoder, batch, num_frames, args.tubelet_size, perms)  # (n_perm,B,D)
        n_perm, B, D = stack.shape
        # clip-major flatten: (clip0_p0, clip0_p1, ..., clip0_p{n-1}, clip1_p0, ...)
        feats = stack.permute(1, 0, 2).reshape(B * n_perm, D)
        labels = torch.arange(n_perm).repeat(B)
        return feats, labels
    if metric == "pace":
        strides = [int(s) for s in args.pace_strides.split(",")]
        if sorted(strides) != strides or len(set(strides)) != len(strides):
            raise SystemExit(f"FATAL: --pace-strides must be sorted + distinct; got {strides}")
        stack = et_pace.compute_features(encoder, batch, num_frames, args.tubelet_size, strides)
        n_str, B, D = stack.shape
        feats = stack.permute(1, 0, 2).reshape(B * n_str, D)
        labels = torch.arange(n_str).repeat(B)
        return feats, labels
    if metric == "tcc":
        per_frame = et_tcc.compute_per_frame(encoder, batch, num_frames, args.tubelet_size)
        # TCC stores per-frame features (B, T_eff, D) ; "labels" carry no class meaning here
        # and are unused by compute_pair downstream. We use a sentinel -1 array.
        labels = -torch.ones(per_frame.shape[0], dtype=torch.long)
        return per_frame, labels  # NOTE: 3-D feats for tcc — handled by stage 2 dispatch
    raise SystemExit(f"FATAL: unknown metric {metric!r}")


def _extract_split(args, encoder, metric, split_keys, split_name, out_dir):
    """Stream the requested split through the encoder once for `metric`, dumping (feats, labels,
    clip_ids) to disk. FAIL LOUD if the resulting feature N is 0."""
    if not split_keys:
        raise SystemExit(f"FATAL: split={split_name} has 0 keys — pipeline failure")
    tmp_root = out_dir / f"tmp_decode_{metric}_{split_name}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    # Pace decodes oversample frames; all other metrics use args.num_frames.
    decode_T = args.pace_source_frames if metric == "pace" else args.num_frames
    if metric == "pace":
        max_stride = max(int(s) for s in args.pace_strides.split(","))
        if decode_T < args.num_frames * max_stride:
            print(f"  WARN: --pace-source-frames {decode_T} < num_frames * max_stride "
                  f"({args.num_frames * max_stride}); stride_indices will cyclic-wrap")
    clip_q, tar_stop, _r = iter_clips_parallel(
        local_data=args.local_data, subset_keys=set(split_keys), processed_keys=set())
    pend_t, pend_k, feats_all, labels_all, clip_ids_all = [], [], [], [], []
    pbar = make_pbar(total=len(split_keys), desc=f"m12f[{metric}/{split_name}]", unit="clip")

    def _flush():
        nonlocal pend_t, pend_k
        if not pend_t:
            return
        batch = torch.stack(pend_t)
        feats, labels = _run_metric_forward(metric, encoder, batch, args.num_frames, args)
        k_per_clip = feats.shape[0] // batch.shape[0] if feats.ndim >= 2 else 1
        feats_all.append(feats)
        labels_all.append(labels)
        clip_ids_all.extend([ck for ck in pend_k for _ in range(k_per_clip)])
        pbar.update(len(pend_k))
        pend_t, pend_k = [], []

    try:
        while True:
            try:
                item = clip_q.get(timeout=300)
            except queue.Empty:
                print(f"  WARN: clip queue timeout on {metric}/{split_name} — flushing")
                break
            if item is None:
                break
            clip_key, mp4 = item
            t = decode_to_tensor(mp4, tmp_dir, clip_key, decode_T, CROP)
            if t is None:
                continue
            pend_t.append(t)
            pend_k.append(clip_key)
            if len(pend_t) >= args.batch_size:
                try:
                    _flush()
                except torch.cuda.OutOfMemoryError:
                    cuda_cleanup()
                    raise SystemExit(
                        f"FATAL: OOM at batch_size={args.batch_size} for metric={metric}; "
                        "lower --batch-size and re-run")
        _flush()
    finally:
        tar_stop.set()
        pbar.close()
    if not feats_all:
        raise SystemExit(f"FATAL: {metric}/{split_name} produced 0 features — pipeline failure")
    feats_cat = torch.cat(feats_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    np.save(out_dir / f"{metric}_{split_name}_feats.npy", feats_cat.cpu().numpy())
    np.save(out_dir / f"{metric}_{split_name}_labels.npy", labels_cat.cpu().numpy())
    np.save(out_dir / f"{metric}_{split_name}_clip_ids.npy",
            np.array(clip_ids_all, dtype=object))
    return feats_cat, labels_cat, clip_ids_all


def _train_eval_classifier(metric, train_feats, train_labels, test_feats, test_labels, args, embed_dim):
    """Trainable-head metrics (aot/tov/pace): train head on train split, evaluate per-example
    on test split, return per-test-example correctness."""
    device = "cuda"
    # per-metric, NOT a dict literal — a dict eagerly evaluates pace_strides.split() even for
    # aot/tov (where --pace-strides is legitimately None) → AttributeError.
    if metric == "aot":
        n_classes = 2
    elif metric == "tov":
        n_classes = args.tov_n_permutations
    elif metric == "pace":
        n_classes = len(args.pace_strides.split(","))
    else:
        raise SystemExit(f"FATAL: _train_eval_classifier got non-head metric {metric!r}")
    mod = {"aot": et_aot, "tov": et_tov, "pace": et_pace}[metric]
    kwargs = dict(embed_dim=embed_dim, lr=args.head_lr, epochs=args.head_epochs,
                  weight_decay=args.head_weight_decay, batch_size=args.head_batch_size,
                  device=device, seed=args.seed)
    if metric == "aot":
        head = mod.train_head(train_feats, train_labels, **kwargs)
    else:
        head = mod.train_head(train_feats, train_labels, n_classes=n_classes, **kwargs)
    return mod.eval_head(head, test_feats, test_labels, device=device, batch_size=args.head_batch_size)


def _tcc_scores(test_feats_per_frame, test_clip_ids, args):
    """Training-free TCC scoring: pair test clips by action label, average cycle-back error
    and Kendall's τ per clip across its pairs."""
    labels_map = load_action_labels(args.action_probe_root / "action_labels.json")
    by_action = {}
    for i, ck in enumerate(test_clip_ids):
        a = labels_map[ck]["class_id"]  # action_labels schema: {class, class_id, split}
        by_action.setdefault(a, []).append(i)
    per_clip_cycle = {i: [] for i in range(test_feats_per_frame.shape[0])}
    per_clip_tau = {i: [] for i in range(test_feats_per_frame.shape[0])}
    n_pairs = n_degenerate = 0
    for a, idxs in by_action.items():
        if len(idxs) < 2:
            continue
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                a_i, b_i = idxs[ii], idxs[jj]
                cyc, tau = et_tcc.compute_pair(
                    test_feats_per_frame[a_i], test_feats_per_frame[b_i],
                    temperature=args.tcc_temperature)
                per_clip_cycle[a_i].append(cyc); per_clip_cycle[b_i].append(cyc)
                per_clip_tau[a_i].append(tau);   per_clip_tau[b_i].append(tau)
                n_pairs += 1
                if np.isnan(tau):                       # degenerate (collinear) → τ undefined
                    n_degenerate += 1
    if n_pairs == 0:
        raise SystemExit("FATAL: TCC produced 0 pairs — every action class has <2 test clips")
    # Kendall's τ is NaN for a degenerate pair (collinear per-frame features — expected for static
    # clips, esp. frozen encoders). nanmean excludes those τ so one bad pair does not abort the eval;
    # report the count so the degeneracy stays VISIBLE (FAIL-LOUD) rather than silently swallowed.
    if n_degenerate:
        print(f"  [tcc] {n_degenerate}/{n_pairs} pairs degenerate (collinear features) → τ=NaN, "
              f"excluded from per-clip τ mean (cycle-back kept).", flush=True)
    cycle_arr = np.array([np.mean(v) if v else np.nan for v in per_clip_cycle.values()], dtype=np.float32)
    tau_arr = np.array([float(np.mean([x for x in v if not np.isnan(x)]))
                        if any(not np.isnan(x) for x in v) else np.nan
                        for v in per_clip_tau.values()], dtype=np.float32)
    keep = ~(np.isnan(cycle_arr) | np.isnan(tau_arr))
    if not keep.any():
        raise SystemExit("FATAL: TCC every test clip lacked a same-action pair")
    return cycle_arr[keep], tau_arr[keep], int(n_pairs), [test_clip_ids[i] for i in range(len(keep)) if keep[i]]


def run_forward_stage(args, wb):
    for need, msg in [(args.variant, "--variant"), (args.encoder_ckpt, "--encoder-ckpt"),
                      (args.local_data, "--local-data"),
                      (args.action_probe_root, "--action-probe-root")]:
        if need is None:
            raise SystemExit(f"FATAL: --stage forward requires {msg}")
    if not args.variant.startswith("vjepa"):
        raise SystemExit(
            f"FATAL: m12f forward only supports vjepa_* variants (encoder-temporal "
            f"needs a video encoder); got {args.variant!r}")
    metrics = _resolve_metrics(args.metric)

    check_gpu()
    print_cgroup_header(prefix="[m12f_encoder_temporal]")
    start_oom_watchdog(prefix="[m12f_encoder_temporal]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for m in metrics:
        agg = out_dir / f"aggregate_{m}.json"
        if agg.exists() and args.cache_policy == "1":
            print(f"  [keep] {m}: {agg.name} present — skipping (--cache-policy 2 to redo)")
            continue
        for p in out_dir.glob(f"per_clip_{m}*.npy"):
            guarded_delete(p, args.cache_policy, p.name)
        guarded_delete(agg, args.cache_policy, agg.name)
        todo.append(m)
    if not todo:
        print("  all requested metrics cached — nothing to do")
        return

    labels_map = load_action_labels(args.action_probe_root / "action_labels.json")
    train_keys = sorted(k for k, info in labels_map.items() if info["split"] == "train")
    test_keys = sorted(k for k, info in labels_map.items() if info["split"] == "test")
    # §opt: cap head-fit TRAIN clips (linear head saturates well before full N); TEST + BCa CI untouched.
    if args.head_train_cap and len(train_keys) > args.head_train_cap:
        print(f"  [head-train-cap] train {len(train_keys)} → {args.head_train_cap} (head-fit only; test/CI unaffected)")
        train_keys = train_keys[:args.head_train_cap]
    print(f"  splits: train={len(train_keys)}  test={len(test_keys)}  metrics={todo}")

    encoder, _ckpt, embed_concat = load_encoder_only(args.encoder_ckpt, args.num_frames, args.model_config)
    del _ckpt  # encoder-temporal needs no predictor (iter16 §3.3 R2 — no predictor build)
    print(f"  encoder loaded (concat dim={embed_concat}; encoder-only — no predictor build)")

    for m in todo:
        t0 = time.time()
        # 1) extract test features (always)
        test_feats, test_labels, test_ids = _extract_split(args, encoder, m, test_keys, "test", out_dir)
        # 2) trainable-head metrics also need train split
        if m in ("aot", "tov", "pace"):
            tr_feats, tr_labels, _tr_ids = _extract_split(args, encoder, m, train_keys, "train", out_dir)
            per_example = _train_eval_classifier(m, tr_feats, tr_labels, test_feats, test_labels,
                                                 args, embed_dim=embed_concat)
            # collapse per-example (K per clip) to per-clip mean
            k = per_example.size // len(test_keys)
            per_clip = per_example.reshape(-1, k).mean(axis=1) if k > 1 else per_example
            save_array_checkpoint(per_clip, out_dir / f"per_clip_{m}.npy")
            ci = bootstrap_ci(per_clip.astype(np.float64))
            save_json_checkpoint({
                "variant": args.variant, "metric": m, "n_test": int(per_clip.size),
                "k_per_clip": int(k),
                "mean": round(float(per_clip.mean()), 6),
                "std": round(float(per_clip.std()), 6),
                "ci": ci, "lower_is_better": METRICS[m][0], "interpretation": METRICS[m][1],
                "num_frames": args.num_frames, "tubelet_size": args.tubelet_size,
                "encoder_ckpt": str(args.encoder_ckpt),
                "head_lr": args.head_lr, "head_epochs": args.head_epochs,
                "head_weight_decay": args.head_weight_decay,
                "head_batch_size": args.head_batch_size,
                **({"n_permutations": args.tov_n_permutations} if m == "tov" else {}),
                **({"strides": args.pace_strides, "source_frames": args.pace_source_frames}
                   if m == "pace" else {}),
            }, out_dir / f"aggregate_{m}.json")
            print(f"  {m}: mean={per_clip.mean():.4f} std={per_clip.std():.4f} "
                  f"N={per_clip.size} ({time.time()-t0:.0f}s)")
            log_metrics(wb, {f"{args.variant}_{m}_mean": float(per_clip.mean())})
        elif m == "tcc":
            cyc, tau, n_pairs, kept_ids = _tcc_scores(test_feats, test_ids, args)
            save_array_checkpoint(cyc, out_dir / "per_clip_tcc_cycle.npy")
            save_array_checkpoint(tau, out_dir / "per_clip_tcc_tau.npy")
            np.save(out_dir / "per_clip_tcc_clip_ids.npy", np.array(kept_ids, dtype=object))
            ci_cyc = bootstrap_ci(cyc.astype(np.float64))
            ci_tau = bootstrap_ci(tau.astype(np.float64))
            save_json_checkpoint({
                "variant": args.variant, "metric": "tcc",
                "n_test": int(cyc.size), "n_pairs": int(n_pairs),
                "cycle_back": {"mean": round(float(cyc.mean()), 6),
                               "std": round(float(cyc.std()), 6),
                               "ci": ci_cyc, "lower_is_better": True},
                "kendalls_tau": {"mean": round(float(tau.mean()), 6),
                                 "std": round(float(tau.std()), 6),
                                 "ci": ci_tau, "lower_is_better": False},
                "temperature": args.tcc_temperature,
                "num_frames": args.num_frames, "tubelet_size": args.tubelet_size,
                "encoder_ckpt": str(args.encoder_ckpt),
            }, out_dir / "aggregate_tcc.json")
            print(f"  tcc: cycle={cyc.mean():.4f}±{cyc.std():.4f}  τ={tau.mean():.4f}±{tau.std():.4f} "
                  f"N={cyc.size} pairs={n_pairs} ({time.time()-t0:.0f}s)")
            log_metrics(wb, {f"{args.variant}_tcc_cycle_mean": float(cyc.mean()),
                             f"{args.variant}_tcc_tau_mean": float(tau.mean())})


def run_paired_per_variant_stage(args, wb):
    """Pairwise BCa Δ across discovered vjepa variants, per metric."""
    metrics = _resolve_metrics(args.metric)
    out = {"metrics": {}}
    for m in metrics:
        by_variant = {}
        if m == "tcc":
            for v in KNOWN_VARIANTS:
                cyc = args.output_root / v / "per_clip_tcc_cycle.npy"
                tau = args.output_root / v / "per_clip_tcc_tau.npy"
                ids = args.output_root / v / "per_clip_tcc_clip_ids.npy"
                agg = args.output_root / v / "aggregate_tcc.json"
                if not (cyc.exists() and tau.exists() and ids.exists() and agg.exists()):
                    continue
                by_variant[v] = {
                    "agg": json.loads(agg.read_text()),
                    "cycle": np.load(cyc).astype(np.float64),
                    "tau": np.load(tau).astype(np.float64),
                    "keys": [str(k) for k in np.load(ids, allow_pickle=True)],
                }
            avail = sorted(by_variant)
            deltas = {"cycle_back": {}, "kendalls_tau": {}}
            for i, a in enumerate(avail):
                for b in avail[i + 1:]:
                    ka, kb = by_variant[a]["keys"], by_variant[b]["keys"]
                    shared = sorted(set(ka) & set(kb))
                    if not shared:
                        raise SystemExit(f"FATAL [tcc]: {a} vs {b} share 0 clips")
                    ai = {k: j for j, k in enumerate(ka)}
                    bi = {k: j for j, k in enumerate(kb)}
                    for fld in ("cycle", "tau"):
                        d = (np.array([by_variant[a][fld][ai[k]] for k in shared])
                             - np.array([by_variant[b][fld][bi[k]] for k in shared]))
                        bca = paired_bca(d)
                        outfld = "cycle_back" if fld == "cycle" else "kendalls_tau"
                        deltas[outfld][f"{a}_minus_{b}"] = {
                            "n": len(shared), "delta_mean": round(float(d.mean()), 6),
                            "delta_ci_lo": round(float(bca["ci_lo"]), 6),
                            "delta_ci_hi": round(float(bca["ci_hi"]), 6),
                            "p_value": float(bca["p_value_vs_zero"]),
                        }
            out["metrics"]["tcc"] = {
                "by_variant": {v: by_variant[v]["agg"] for v in avail},
                "pairwise_deltas": deltas,
            }
            continue
        # trainable-head metrics
        for v in KNOWN_VARIANTS:
            pcl = args.output_root / v / f"per_clip_{m}.npy"
            agg = args.output_root / v / f"aggregate_{m}.json"
            ids = args.output_root / v / f"{m}_test_clip_ids.npy"
            if not (pcl.exists() and agg.exists() and ids.exists()):
                continue
            raw_ids = [str(k) for k in np.load(ids, allow_pickle=True)]
            # ids file contains K-per-clip duplicates ; collapse to unique in order.
            seen, dedup = set(), []
            for k in raw_ids:
                if k not in seen:
                    seen.add(k); dedup.append(k)
            by_variant[v] = {
                "agg": json.loads(agg.read_text()),
                "vals": np.load(pcl).astype(np.float64),
                "keys": dedup,
            }
        avail = sorted(by_variant)
        deltas = {}
        for i, a in enumerate(avail):
            for b in avail[i + 1:]:
                ka, kb = by_variant[a]["keys"], by_variant[b]["keys"]
                shared = sorted(set(ka) & set(kb))
                if not shared:
                    raise SystemExit(f"FATAL [{m}]: {a} vs {b} share 0 clips")
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
            "lower_is_better": METRICS[m][0],
        }
    save_json_checkpoint(out, args.output_root / "encoder_temporal_per_variant.json")
    log_metrics(wb, {"n_metrics": len(out["metrics"])})
    print(json.dumps(out, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="m12f encoder-temporal metric suite (TOV/AoT/Pace/TCC)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC", action="store_true")
    p.add_argument("--FULL", action="store_true")
    p.add_argument("--stage", required=True, choices=["forward", "paired_per_variant"])
    p.add_argument("--metric", required=True, choices=list(METRICS) + ["all"])
    p.add_argument("--variant", choices=list(KNOWN_VARIANTS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--model-config", type=str, default=None,
                   help="configs/model/<backbone>.yaml — encoder arch/dims (WS-B3 arch-aware). "
                        "Required for --stage forward; unused by --stage paired_per_variant (None → ViT-G).")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="probe_action output dir (action_labels.json train/test splits)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--tubelet-size", type=int, required=True,
                   help="encoder tubelet size (V-JEPA 2.1 ViT-G = 2); per model-config")
    p.add_argument("--batch-size", type=int, default=4,
                   help="clips per encoder forward (OOM-protected; lower if Pace OOMs)")
    p.add_argument("--seed", type=int, default=_PCFG["probe"]["seed"])
    # — per-metric sweep params (NO defaults — must come from pipeline.yaml at §3.3) —
    p.add_argument("--tov-n-permutations", type=int, required=False,
                   help="REQUIRED if --metric tov|all : number of TOV permutation classes (incl identity)")
    p.add_argument("--pace-strides", type=str, required=False,
                   help="REQUIRED if --metric pace|all : comma-separated sorted strides, e.g. 1,2,4")
    p.add_argument("--pace-source-frames", type=int, required=False,
                   help="REQUIRED if --metric pace|all : decode T_src; should be num_frames × max(strides)")
    p.add_argument("--tcc-temperature", type=float, required=False,
                   help="REQUIRED if --metric tcc|all : soft-NN temperature (Dwibedi eq.1)")
    # — head training (trainable metrics) —
    p.add_argument("--head-lr", type=float, required=False,
                   help="REQUIRED for aot|tov|pace|all : tiny linear head AdamW LR")
    p.add_argument("--head-epochs", type=int, required=False,
                   help="REQUIRED for aot|tov|pace|all : head training epochs")
    p.add_argument("--head-weight-decay", type=float, required=False,
                   help="REQUIRED for aot|tov|pace|all : head AdamW weight decay")
    p.add_argument("--head-batch-size", type=int, required=False,
                   help="REQUIRED for aot|tov|pace|all : head training batch size")
    p.add_argument("--head-train-cap", type=int, default=None,
                   help="Cap head-fit TRAIN clips to this many (linear head saturates ~1.5-2k). "
                        "TEST split + BCa CI are UNTOUCHED. None = full train. yaml encoder_temporal.head_train_cap.")
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def _check_required(args):
    """Stage-aware required-arg gate — argparse's required= can't condition on --metric, so we
    enforce here. FAIL LOUD on any missing knob for the requested metric."""
    if args.stage != "forward":
        return
    needs = {"tov": ["tov_n_permutations"],
             "pace": ["pace_strides", "pace_source_frames"],
             "tcc": ["tcc_temperature"]}
    head_needs = ["head_lr", "head_epochs", "head_weight_decay", "head_batch_size"]
    metrics = _resolve_metrics(args.metric)
    missing = []
    for m in metrics:
        for k in needs.get(m, []):
            if getattr(args, k) is None:
                missing.append(f"--{k.replace('_', '-')} (metric={m})")
        if m in ("aot", "tov", "pace"):
            for k in head_needs:
                if getattr(args, k) is None:
                    missing.append(f"--{k.replace('_', '-')} (metric={m})")
    if missing:
        raise SystemExit("FATAL: missing required args for --metric=" + args.metric
                         + ":\n  " + "\n  ".join(missing))


def main():
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        raise SystemExit("ERROR: specify --SANITY, --POC, or --FULL")
    _check_required(args)
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"m12f_encoder_temporal_{args.stage}", mode,
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
