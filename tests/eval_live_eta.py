"""Live aggregate-ETA monitor for the sequential run_eval.sh pipeline.

run_eval.sh chains ~40 independent `python probe_*.py` processes (8 encoders ×
{features,train} × {action,taxonomy,motion_cos,future_mse}); each owns its OWN
tqdm and only knows its own loop, so there is no whole-pipeline ETA. This script
reconstructs one by parsing the single eval log:

  total work  = n_encoders × per-encoder feature-extraction clips (test+train+val)
  work done   = Σ "Saved …features_*.npy (N,…)"  +  the current in-flight tqdm pos
  rate        = Δ(work done) over a trailing window (default 180s), from a /tmp state JSONL
  aggregate ETA = (total − done) / rate

This is the GPU feature-extraction BULK (the dominant cost). future_mse/motion_cos
add a tail beyond this estimate — labelled as such. Mirrors tests/m10_live_rate.py
(sliding window + FAIL-LOUD), but for one sequential log instead of N parallel dirs.

USAGE (watch):
    watch -n 30 'python tests/eval_live_eta.py --log $(ls -t logs/iter15_v2_*_eval_*.log | head -1)'
ONE-SHOT:
    python tests/eval_live_eta.py --log logs/iter15_v2_poc_eval_20260527_163314.log
RESET the window:
    python tests/eval_live_eta.py --reset
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

# "Saved …/features_train.npy (6854, 16, 1664)"  → ('train', 6854)
_SAVED = re.compile(r"features_(\w+)\.npy\s*\((\d+),")
# "probe_features_train:  8%|…| 545/6854 ["  → ('train', 545, 6854)
_TQDM = re.compile(r"probe_features_(\w+):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
# "→ final ENCODERS after head preflight: a b c …"  /  "Stage 8 ENCODERS: a b c …"
_ENCS = re.compile(r"(?:final ENCODERS after head preflight|Stage 8 ENCODERS):\s*(.+)")


def parse_log(log: Path):
    """Return (n_encoders, per_encoder_clips, clips_done) from the eval log.

    FAIL LOUD if the structural anchors (encoder list, any feature tqdm/save) are
    missing — a silent 0 would print a meaningless ETA.
    """
    text = log.read_text(errors="replace")

    enc_lines = _ENCS.findall(text)
    if not enc_lines:
        raise RuntimeError(
            f"{log}: no 'ENCODERS:' line yet — eval still in STAGE-1 label prep, "
            "or log path wrong. Re-run once STAGE 2 (features) starts.")
    n_encoders = len(enc_lines[-1].split())

    split_total = {}                      # split -> clips/encoder (the tqdm denominator)
    for split, _pos, tot in _TQDM.findall(text):
        split_total[split] = max(split_total.get(split, 0), int(tot))
    saved = _SAVED.findall(text)          # [(split, N), …] across all encoders so far
    for split, n in saved:
        split_total.setdefault(split, int(n))
    if not split_total:
        raise RuntimeError(
            f"{log}: no 'probe_features_*' tqdm or 'features_*.npy' save yet — "
            "no feature-extraction work to measure. Re-run after STAGE 2 begins.")
    per_encoder_clips = sum(split_total.values())

    clips_done = sum(int(n) for _split, n in saved)     # completed extractions
    tqdm_hits = _TQDM.findall(text)
    if tqdm_hits:                                       # + the in-flight (not-yet-saved) one
        clips_done += int(tqdm_hits[-1][1])
    return n_encoders, per_encoder_clips, clips_done, split_total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, help="eval log (required unless --reset)")
    p.add_argument("--window-sec", type=int, default=180)
    p.add_argument("--state-file", type=Path, default=Path("/tmp/eval_live_eta_state.jsonl"))
    p.add_argument("--reset", action="store_true", help="clear --state-file and exit")
    args = p.parse_args()

    if args.reset:
        args.state_file.unlink(missing_ok=True)
        print(f"State cleared: {args.state_file}")
        return
    if args.log is None or not args.log.exists():
        print(f"ERROR: --log missing or not found: {args.log}", file=sys.stderr)
        sys.exit(2)

    now = time.time()
    n_enc, per_enc, done, split_total = parse_log(args.log)
    total = n_enc * per_enc

    # persist (ts, done) snapshot; prune to 2× window
    hist = []
    if args.state_file.exists():
        for ln in args.state_file.read_text().splitlines():
            if ln.strip():
                hist.append(json.loads(ln))
    hist.append({"ts": now, "done": done})
    hist = [h for h in hist if h["ts"] >= now - 2 * args.window_sec]
    args.state_file.write_text("\n".join(json.dumps(h) for h in hist) + "\n")

    pct = 100.0 * done / total if total else 0.0

    def _bar(p, w=28):
        f = int(round(w * p / 100.0))
        return "█" * f + "░" * (w - f)

    def _hhmm(hours):                                    # 7.7h -> "7:42"  (ETA in hh:mm)
        m = int(round(hours * 60))
        return f"{m // 60:d}:{m % 60:02d}"

    splits_str = " + ".join(f"{s}={n:,}" for s, n in sorted(split_total.items()))
    print(f"=== {time.strftime('%H:%M:%S')} · run_eval aggregate ETA "
          f"({n_enc} encoders × ({splits_str}) = {total:,} feat-clips) ===")

    older = [h for h in hist if h["ts"] <= now - args.window_sec]
    snap = older[-1] if older else None
    dt = now - snap["ts"] if snap else 0.0
    dd = done - snap["done"] if snap else 0
    if snap and dd > 0 and dt > 0:
        rate = dd / dt                                   # clips/sec
        eta_h = (total - done) / rate / 3600.0
        eta_wall = time.strftime("%H:%M %a", time.localtime(now + eta_h * 3600))
        # tqdm-style bar + ETA in hh:mm duration
        print(f"  feat |{_bar(pct)}| {pct:5.1f}%  {done:,}/{total:,}  "
              f"[ETA {_hhmm(eta_h)}<{eta_wall}, {rate*60:.0f} clip/min]")
        print("       (feature-extraction bulk; future_mse + motion_cos add a tail)")
    else:
        # no window yet, or no forward progress — still show the bar + %
        gathered = now - hist[0]["ts"]
        note = (f"baseline {gathered:.0f}/{args.window_sec}s — ETA next tick"
                if not snap else "stalled / between stages")
        print(f"  feat |{_bar(pct)}| {pct:5.1f}%  {done:,}/{total:,}  [ETA --:-- · {note}]")


if __name__ == "__main__":
    main()
