"""WalkIndia DRIVE-tier turn hunt — CPU-only honest visibility probe (iter20 last untested cell).

WHY this exists (and why it is NOT just `ood_turn_probe --stage scan`):
  The plan was to download ~10 FULL YouTube drive-tour videos and scan each with the committed
  `ood_turn_probe.py`. On 2026-07-19 that path is HARD-BLOCKED: this datacenter IP is gated by
  YouTube to logged-out `LOGIN_REQUIRED` on every player client, and the only cookies on the box
  (`cookies_youtube.txt`) are a CONSENT-ONLY set (SID / SAPISID / __Secure-1PSID / LOGIN_INFO all
  missing) so they do not authenticate. A bgutil PO-token provider was stood up and DID mint tokens
  (BotGuard solved), but PO tokens cannot fix LOGIN_REQUIRED — that needs a live login session we do
  not have and must never request. yt-dlp last succeeded here 2026-07-15.

  Fallback (MORE in-domain, not less): the actual WalkIndia drive-tier training clips are already on
  disk in `data/demo_src/data/train-00025.tar` — 284 delhi drive clips from source videos
  `yMfgkU9YRms` + `xwCVvSZU_i4`, BOTH of which are in `YT_videos_raw.json > drive_tours.delhi`. These
  are literally "the source videos OURS was surgery-tuned on", just cut into 8-10s @ 480p/30fps clips.

  Because they are 8-10s clips (~2 windows each), per-clip percentile thresholding is meaningless.
  This wrapper reuses `ood_turn_probe._gray` (the EXACT coarse global-translation SAD flow) and its
  window / cum_dx logic, but pools every window across all clips and applies ONE GLOBAL percentile
  cut — the honest way to threshold a many-short-clip corpus. Sign convention is kept identical to
  `ood_turn_probe.stage_scan` (cum_dx > 0 -> TURN_LEFT, < 0 -> TURN_RIGHT) so a human can run
  `--stage feats/probe` next (feats must read each segment's `clip` path — see turn_labels.json).

stages (both CPU): scan (flow -> segments.json + turn_labels.json) · sheet (two contact-sheet PNGs)

USAGE (run from repo root, venv_walkindia):
    PYTHONPATH=src python -m utils.tmp.walkindia_drive_turn_hunt --stage scan \
        --clips-dir data/youtube_demo/walkindia_drive \
        --out outputs/demo/walkindia_drive_turns --seg-s 6 --step-s 2
    PYTHONPATH=src python -m utils.tmp.walkindia_drive_turn_hunt --stage sheet \
        --clips-dir data/youtube_demo/walkindia_drive \
        --out outputs/demo/walkindia_drive_turns
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from utils.tmp.ood_turn_probe import _gray  # EXACT same SAD flow as the committed scanner


def _clip_dx(gray, m=8):
    """Per-frame best horizontal shift (coarse yaw proxy) — identical inner loop to
    ood_turn_probe.stage_scan, factored out so it runs per short clip."""
    H, W = gray.shape[1], gray.shape[2]
    dxs = []
    for t in range(len(gray) - 1):
        aa, bb = gray[t, m:H - m, m:W - m], gray[t + 1]
        best, bd = 1e18, 0
        for dx in range(-m, m + 1):
            s = float(np.abs(aa - bb[m:H - m, m + dx:W - m + dx]).mean())
            if s < best:
                best, bd = s, dx
        dxs.append(bd)
    return np.asarray(dxs, float)


def _video_id(clip_path: Path) -> str:
    name = clip_path.stem
    return name.rsplit("_", 1)[0] if "_" in name else name


def stage_scan(a):
    clips = sorted(a.clips_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"FATAL: no .mp4 under {a.clips_dir}")
    win, step = int(a.seg_s * a.scan_fps), int(a.step_s * a.scan_fps)
    windows = []
    too_short = 0
    for i, clip in enumerate(clips):
        g = _gray(clip, a.scan_fps, 160, 90)
        if len(g) < 2:
            too_short += 1
            continue
        dxs = _clip_dx(g)
        if len(dxs) <= win:
            too_short += 1
            continue
        for s in range(0, len(dxs) - win, step):
            windows.append({
                "clip": str(clip),
                "video_id": _video_id(clip),
                "t0": round(s / a.scan_fps, 3),
                "dur": float(a.seg_s),
                "cum_dx": float(dxs[s:s + win].sum()),
            })
        if (i + 1) % 50 == 0:
            print(f"[scan] {i + 1}/{len(clips)} clips -> {len(windows)} windows so far")

    cums = np.array([w["cum_dx"] for w in windows], float)
    acum = np.abs(cums)
    hi = float(np.percentile(acum, a.turn_pct))
    lo = float(np.percentile(acum, a.straight_pct))
    pctls = {str(p): float(np.percentile(acum, p)) for p in (5, 10, 25, 30, 50, 70, 75, 90, 95, 99)}
    print(f"\n[scan] {len(clips)} clips ({too_short} too short for a {a.seg_s}s window) "
          f"-> {len(windows)} windows")
    print(f"[scan] |cum_dx| percentiles: {json.dumps({k: round(v, 1) for k, v in pctls.items()})}"
          f"  max={acum.max():.1f}")
    print(f"[scan] cut: STRAIGHT if |cum_dx|<=p{a.straight_pct}={lo:.1f} · "
          f"TURN if |cum_dx|>=p{a.turn_pct}={hi:.1f}")

    for w in windows:
        c = w["cum_dx"]
        w["cls"] = (("TURN_LEFT" if c > 0 else "TURN_RIGHT") if abs(c) >= hi
                    else ("STRAIGHT" if abs(c) <= lo else "AMB"))
    labelled = [w for w in windows if w["cls"] != "AMB"]
    print(f"[scan] labelled (non-AMB): {len(labelled)} · {dict(Counter(w['cls'] for w in labelled))}")

    a.out.mkdir(parents=True, exist_ok=True)
    # segments.json — ood_turn_probe-compatible (labelled only), for the human's feats/probe stage
    json.dump(labelled, open(a.out / "segments.json", "w"), indent=1)
    # turn_labels.json — richer, ALL windows incl AMB + the threshold/percentile provenance
    json.dump({
        "source": "IN-DOMAIN WalkIndia drive-tier clips (data/demo_src train-00025), "
                  "YouTube download IP-blocked 2026-07-19",
        "n_clips": len(clips), "n_windows": len(windows), "too_short": too_short,
        "seg_s": a.seg_s, "step_s": a.step_s, "scan_fps": a.scan_fps,
        "turn_pct": a.turn_pct, "straight_pct": a.straight_pct, "hi": hi, "lo": lo,
        "abs_cum_dx_percentiles": pctls,
        "class_counts": dict(Counter(w["cls"] for w in windows)),
        "segments": windows,
    }, open(a.out / "turn_labels.json", "w"), indent=1)
    print(f"[scan] wrote {a.out/'segments.json'} and {a.out/'turn_labels.json'}")


# ---------- contact sheets ----------
def _font(sz):
    from matplotlib import font_manager
    from PIL import ImageFont
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    return ImageFont.truetype(path, sz)


def _decode_span(clip: Path, t0: float, dur: float, n: int, tw: int, th: int):
    """n frames evenly spanning [t0, t0+dur] of clip, each resized to tw x th uint8."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.2f}", "-t", f"{dur:.2f}", "-i", str(clip),
         "-vf", f"fps={n/dur:.4f},scale={tw}:{th}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    if len(fr) == 0:
        raise SystemExit(f"FATAL: 0 frames from {clip} @ {t0}")
    if len(fr) < n:
        fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)], 0)
    return fr[:n]


def _build_sheet(segs, clips_dir, out_png, title, n_frames=8, tw=192, th=108):
    from PIL import Image, ImageDraw
    pad, cap_w, hdr = 3, 300, 46
    rows = len(segs)
    row_h = th + pad
    W = n_frames * (tw + pad) + pad + cap_w
    Hh = hdr + rows * row_h + pad
    canvas = Image.new("RGB", (W, Hh), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 12), title, font=_font(22), fill=(255, 255, 255))
    col = {"TURN_LEFT": (90, 200, 255), "TURN_RIGHT": (255, 170, 80)}
    for r, s in enumerate(segs):
        y = hdr + r * row_h
        frames = _decode_span(Path(s["clip"]), s["t0"], s["dur"], n_frames, tw, th)
        for c in range(n_frames):
            canvas.paste(Image.fromarray(frames[c]), (pad + c * (tw + pad), y))
        cx = pad + n_frames * (tw + pad)
        cls = s["cls"]
        d.rectangle([cx, y, cx + cap_w, y + th], fill=(32, 32, 32))
        arrow = "<-- LEFT" if cls == "TURN_LEFT" else "RIGHT -->"
        d.text((cx + 8, y + 6), f"{cls}", font=_font(20), fill=col.get(cls, (230, 230, 230)))
        d.text((cx + 8, y + 32), arrow, font=_font(18), fill=col.get(cls, (230, 230, 230)))
        d.text((cx + 8, y + 58), f"cum_dx={s['cum_dx']:+.0f}", font=_font(16), fill=(210, 210, 210))
        d.text((cx + 8, y + 80), f"{s['video_id']}  t0={s['t0']:.0f}s",
               font=_font(14), fill=(150, 150, 150))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print(f"[sheet] {out_png}  ({rows} rows)")


def stage_sheet(a):
    segs = json.load(open(a.out / "segments.json"))
    turns = [s for s in segs if s["cls"] in ("TURN_LEFT", "TURN_RIGHT")]
    L = sorted([s for s in turns if s["cls"] == "TURN_LEFT"], key=lambda s: -abs(s["cum_dx"]))
    R = sorted([s for s in turns if s["cls"] == "TURN_RIGHT"], key=lambda s: -abs(s["cum_dx"]))
    # TOP sheet: 8 strongest LEFT + 8 strongest RIGHT (cherry-picked — NOT the verdict basis)
    top = (L[:8] + R[:8])
    top = sorted(top, key=lambda s: (s["cls"], -abs(s["cum_dx"])))
    _build_sheet(top, a.clips_dir, a.out / "contact_sheet.png",
                 "TOP-16 strongest-yaw TURN segments (cherry-picked) — in-domain WalkIndia delhi drive")
    # RANDOM sheet: 12 unbiased strict turns (THE verdict basis)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(turns), size=min(12, len(turns)), replace=False)
    rand = [turns[i] for i in sorted(idx)]
    _build_sheet(rand, a.clips_dir, a.out / "contact_sheet_random12.png",
                 f"RANDOM-12 strict turns (unbiased, seed={a.seed}) — VERDICT basis — WalkIndia delhi drive")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["scan", "sheet"])
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scan-fps", type=int, default=6)
    p.add_argument("--seg-s", type=float, default=6.0)
    p.add_argument("--step-s", type=float, default=2.0)
    p.add_argument("--turn-pct", type=int, default=70)
    p.add_argument("--straight-pct", type=int, default=30)
    p.add_argument("--seed", type=int, default=20)
    a = p.parse_args()
    {"scan": stage_scan, "sheet": stage_sheet}[a.stage](a)
