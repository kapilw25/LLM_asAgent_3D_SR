"""Live post-warmup rate + ETA monitor for the m10 multi-worker run.

Reads current `.npz` mask counts under `data/<MODE>_local/m10_sam_segment_w*/masks/`,
persists a sliding-window of (timestamp, counts) snapshots to a state JSONL file
in /tmp, and on every invocation computes:

  • per-worker s/clip over the trailing N-second window (default 180s = 3 min)
  • per-worker remaining clips (from /tmp/m10_parallel_subsets_*/subset_w{i}.json)
  • per-worker ETA hours = remaining × s/clip / 3600
  • worst-case wall ETA = max(per_worker_eta) — workers run in parallel, slowest
    finisher gates the whole run

This complements tqdm's whole-run-AVG rate (which is dragged DOWN by warmup) by
showing the *current* post-warmup steady-state rate. tqdm shows the conservative
upper-bound ETA; this script shows the optimistic-if-current-rate-sustains ETA.

USAGE — recommended invocation as `watch`:
    watch -n 30 'python /workspace/factorjepa/scripts/m10_live_rate.py \\
        --workers-dir /workspace/factorjepa/data/full_local \\
        --window-sec 180'

  # First 3 min: prints "⏳ Collecting baseline..." (insufficient history).
  # After 3 min: ASCII table with real per-worker rates + ETA.

  # To start a fresh measurement window (e.g., after restart or rate-regime
  # change), clear the state and re-launch watch:
  python /workspace/factorjepa/scripts/m10_live_rate.py --reset

USAGE — one-shot (single sample, no live refresh):
    python /workspace/factorjepa/scripts/m10_live_rate.py \\
        --workers-dir /workspace/factorjepa/data/full_local
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path


def detect_worker_dirs(workers_dir: Path) -> list[Path]:
    """Return sorted list of m10_sam_segment_w*/ dirs under workers_dir."""
    return sorted(workers_dir.glob("m10_sam_segment_w*"))


def load_total_per_worker(n_workers: int, subset_dir_glob: str) -> list[int]:
    """Return total clip count per worker by reading subset_w{i}.json files.

    Looks under /tmp/m10_parallel_subsets_*/subset_w{i}.json (auto-detects the
    most-recent orchestrator's subset dir). FAIL LOUD if not found — totals
    are needed for ETA, and a stale 0 would silently produce ETA=0.
    """
    candidates = sorted(glob.glob(subset_dir_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No subset directories matching '{subset_dir_glob}' — "
            "orchestrator's Step 1 output is missing. "
            "Is run_factor_prep_parallel.sh actually running? "
            "(/tmp/m10_parallel_subsets_*/ is auto-cleaned at orchestrator exit.)"
        )
    latest = Path(candidates[-1])
    totals = []
    for i in range(n_workers):
        f = latest / f"subset_w{i}.json"
        if not f.exists():
            raise FileNotFoundError(
                f"Missing subset file: {f} — found {len(candidates)} candidate "
                "subset dirs but the one selected does not have all N worker files. "
                "Stale leftover from a prior run? Try --reset, then verify the "
                "running orchestrator's SUBSET_DIR matches this glob."
            )
        data = json.loads(f.read_text())
        if "clip_keys" not in data:
            raise KeyError(
                f"{f} has no 'clip_keys' field — subset schema changed?"
            )
        totals.append(len(data["clip_keys"]))
    return totals


def count_masks(worker_dir: Path) -> int:
    """Count *.npz mask files in worker_dir/masks/."""
    masks = worker_dir / "masks"
    if not masks.exists():
        return 0
    return sum(1 for _ in masks.glob("*.npz"))


def load_history(state_file: Path) -> list[dict]:
    """Load prior snapshots from state file. FAIL LOUD on corruption — caller
    can recover by running --reset to clear the file."""
    if not state_file.exists():
        return []
    history = []
    for lineno, raw in enumerate(state_file.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Corrupt JSONL in state file {state_file} at line {lineno}: {e}. "
                f"Recover with: python {Path(__file__).name} --reset"
            ) from e
    return history


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workers-dir", type=Path,
                   help="Parent of m10_sam_segment_w*/ dirs "
                        "(e.g. data/full_local). Required unless --reset.")
    p.add_argument("--window-sec", type=int, default=180,
                   help="Sliding-window size in seconds (default: 180 = 3 min)")
    p.add_argument("--state-file", type=Path,
                   default=Path("/tmp/m10_live_rate_state.jsonl"),
                   help="Where to persist (timestamp, counts) snapshots")
    p.add_argument("--subset-dir-glob", type=str,
                   default="/tmp/m10_parallel_subsets_*",
                   help="Glob to find the orchestrator's subset_w*.json dir")
    p.add_argument("--reset", action="store_true",
                   help="Clear --state-file and exit (start fresh window)")
    args = p.parse_args()

    if args.reset:
        if args.state_file.exists():
            args.state_file.unlink()
            print(f"State cleared: {args.state_file}")
        else:
            print(f"No state file to clear: {args.state_file}")
        return

    if args.workers_dir is None:
        print("ERROR: --workers-dir required (unless --reset)", file=sys.stderr)
        sys.exit(2)
    if not args.workers_dir.exists():
        print(f"ERROR: workers-dir does not exist: {args.workers_dir}",
              file=sys.stderr)
        sys.exit(2)

    worker_dirs = detect_worker_dirs(args.workers_dir)
    n_workers = len(worker_dirs)
    if n_workers == 0:
        print(f"ERROR: no m10_sam_segment_w*/ dirs found under "
              f"{args.workers_dir}", file=sys.stderr)
        sys.exit(2)

    now = time.time()
    cur_counts = [count_masks(d) for d in worker_dirs]
    totals = load_total_per_worker(n_workers, args.subset_dir_glob)

    history = load_history(args.state_file)
    history.append({"ts": now, "counts": cur_counts})

    # Prune entries older than 2× window (keep buffer for re-derivation)
    prune_cutoff = now - 2 * args.window_sec
    history = [h for h in history if h["ts"] >= prune_cutoff]
    args.state_file.write_text(
        "\n".join(json.dumps(h) for h in history) + "\n"
    )

    target = now - args.window_sec
    older = [h for h in history if h["ts"] <= target]

    hdr_time = time.strftime("%H:%M:%S")
    print(f"=== {hdr_time} · {args.window_sec}s-sliding-window rate "
          f"({n_workers} workers) ===")

    if not older:
        oldest = history[0]["ts"]
        elapsed = now - oldest
        need = args.window_sec - elapsed
        ready_at = time.strftime("%H:%M:%S", time.localtime(now + need))
        print(f"⏳ Collecting baseline: {elapsed:.0f}s gathered "
              f"({len(history)} snapshots), need {need:.0f}s more.")
        print(f"   First useful reading at {ready_at}.")
        return

    snap = older[-1]
    dt = now - snap["ts"]

    rows = []
    max_eta_h = 0.0
    sum_remaining = 0
    sum_rate = 0.0
    for i in range(n_workers):
        before = snap["counts"][i] if i < len(snap["counts"]) else 0
        nowcnt = cur_counts[i]
        delta = nowcnt - before
        tot = totals[i]
        rem = max(0, tot - nowcnt)
        sum_remaining += rem
        if delta > 0 and dt > 0:
            s_per = dt / delta
            eta_h = rem * s_per / 3600.0
            rate_clips_per_sec = delta / dt
            sum_rate += rate_clips_per_sec
            rows.append((f"w{i}", before, nowcnt, delta, s_per, rem, eta_h))
            max_eta_h = max(max_eta_h, eta_h)
        else:
            rows.append((f"w{i}", before, nowcnt, delta, None, rem, None))

    sep = "┌─────┬──────────┬──────────┬────────┬──────────┬───────────┬──────────┐"
    mid = "├─────┼──────────┼──────────┼────────┼──────────┼───────────┼──────────┤"
    bot = "└─────┴──────────┴──────────┴────────┴──────────┴───────────┴──────────┘"
    print(sep)
    print(f"│ wkr │{dt:6.0f}s ago│  current │ delta  │  s/clip  │ remaining │  ETA(h)  │")
    print(mid)
    for w, b, c, d, s, rem, e in rows:
        s_str = f"{s:>6.2f}  " if s is not None else "   —    "
        e_str = f"{e:>6.2f}  " if e is not None else "   —    "
        print(f"│ {w:<3} │ {b:>8} │ {c:>8} │ {d:>+5}  │ {s_str} │ {rem:>9,} │ {e_str} │")
    print(bot)

    if max_eta_h > 0:
        eta_wall = time.localtime(now + max_eta_h * 3600)
        eta_str = time.strftime("%H:%M %a", eta_wall)
        print(f"🎯 Worst-case wall ETA (slowest worker): "
              f"{max_eta_h:.1f}h → finishes ~{eta_str}")
        if sum_rate > 0:
            agg_per_min = sum_rate * 60
            print(f"📊 Aggregate throughput: {agg_per_min:.1f} clips/min "
                  f"({sum_remaining:,} clips remaining across {n_workers} workers)")


if __name__ == "__main__":
    main()
