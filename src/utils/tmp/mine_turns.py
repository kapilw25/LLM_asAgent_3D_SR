"""mine_turns — mine LEFT/RIGHT turn candidates from m04d motion features (IN-DOMAIN WalkIndia).

m04d never writes a "LEFT"/"RIGHT" label, but it writes the raw ego-yaw signal:
    camera_motion_x  = FEATURE_NAMES[11]  = median horizontal optical flow over the clip
Sign = pan direction, |value| = how hard the camera swung. So turns are DERIVABLE for every
clip already processed by m04d — no YouTube download, no hand-spotting.

SIGN CONVENTION IS NOT ASSUMED. For a forward-facing camera the scene slides OPPOSITE the turn
(turn right -> content flows left -> dx<0), but that depends on m04d's flow orientation, so this
tool reports both poles as POLE_NEG / POLE_POS and the mapping to LEFT/RIGHT must be fixed by
EYEBALLING extracted clips (stage `clips`). Never label a demo off the unverified convention.

WHY THIS MATTERS (honesty): surgery's aux supervision bins on vec[13]=fg_mean_mag, which is
CAMERA-SUBTRACTED (utils/action_labels.py:99-141) — i.e. trained with the turn signal removed.
So the expectation is OURS <= FROZEN on turn direction. This tool exists to TEST that, not to
manufacture a win.

stages:
    pull  -> download the 3 m04d artifacts for a dataset dir from the HF outputs repo
    rank  -> rank clips by camera_motion_x, report distribution + top-N per pole (multi-video)
    clips -> extract the top candidates to mp4 + a contact sheet for the human/Claude eyeball

USAGE (run from repo root):
    PYTHONPATH=src python -m utils.tmp.mine_turns --stage pull --hf-dir data/full_local \
        --repo anonymousML123/factorjepa-outputs --out <scratch>/turn_mine
    PYTHONPATH=src python -m utils.tmp.mine_turns --stage rank --out <scratch>/turn_mine \
        --top-n 40 --per-video-cap 3
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

MEAN_MAG = 0        # FEATURE_NAMES[0]  == "mean_magnitude" (total motion — the normaliser)
CAM_X = 11          # FEATURE_NAMES[11] == "camera_motion_x" (the ego-yaw / turn signal)
FG_MEAN_MAG = 13    # FEATURE_NAMES[13] == "fg_mean_mag" (what surgery was actually trained on)
_ARTIFACTS = ("motion_features.npy", "motion_features.paths.npy", "motion_features.meta.json")


def stage_pull(a):
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("FATAL: HF_TOKEN not in env — `set -a; . ./.env; set +a` first")
    a.out.mkdir(parents=True, exist_ok=True)
    for fn in _ARTIFACTS:
        rp = f"{a.hf_dir}/m04d_motion_features/{fn}"
        p = hf_hub_download(a.repo, rp, repo_type="dataset", token=tok)
        dst = a.out / fn
        dst.write_bytes(Path(p).read_bytes())
        print(f"  pulled {rp}  ->  {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
    meta = json.load(open(a.out / "motion_features.meta.json"))
    print(f"[pull] n_clips={meta['n_clips']:,} feature_dim={meta['feature_dim']} "
          f"n_frame_pairs={meta['n_frame_pairs']}")
    names = meta["feature_names"]
    print(f"[pull] confirming index map: [{CAM_X}]={names[CAM_X]!r}  [{FG_MEAN_MAG}]={names[FG_MEAN_MAG]!r}")
    if names[CAM_X] != "camera_motion_x":
        raise RuntimeError(f"FATAL: feature index {CAM_X} is {names[CAM_X]!r}, expected camera_motion_x")


def _video_of(key):
    """The video is the leakage unit = everything above the clip file.

    Keys are '<tier>/<city>/<section>/<video_id>/<video_id>-NNN.mp4' (5 parts), NOT the
    3-part shape the docstring of m04d implies. An earlier parts[:2] grouping silently
    collapsed 1,559 videos into 25 CITIES, which would have made a "leave-one-video-out"
    split actually leave-one-city-out — and passed every per-video cap unnoticed.
    """
    return "/".join(str(key).split("/")[:-1])


def stage_rank(a):
    X = np.load(a.out / "motion_features.npy")
    keys = np.load(a.out / "motion_features.paths.npy", allow_pickle=True)
    if len(X) != len(keys):
        raise RuntimeError(f"FATAL: features {len(X)} != paths {len(keys)}")
    cam_x = X[:, CAM_X].astype(np.float64)
    mean_mag = X[:, MEAN_MAG].astype(np.float64)
    fg = X[:, FG_MEAN_MAG].astype(np.float64)
    vids = np.array([_video_of(k) for k in keys])

    # MAGNITUDE-NORMALISED signal. Raw camera_motion_x correlates +0.849 with
    # mean_magnitude, so ranking on it returns "clips with the most motion" (drone
    # shots, rain static), not turns. Dividing by total motion drops that to +0.380.
    sig = cam_x / (mean_mag + 1e-6)

    print(f"\n[rank] n={len(X):,} clips · {len(set(vids)):,} source videos")
    print(f"[rank] corr(|raw cam_x|, mean_magnitude) = "
          f"{np.corrcoef(np.abs(cam_x), mean_mag)[0, 1]:+.3f}  -> normalised: "
          f"{np.corrcoef(np.abs(sig), mean_mag)[0, 1]:+.3f}")

    # ── VALIDITY GATE: is the SIGN a turn direction, or a per-video constant? ──
    # A real drive turns both ways, so its clips must carry BOTH signs. If a video's
    # clips are ~all one sign, the sign is encoding the CAMERA (mount angle, lens
    # offset, scene asymmetry), not the manoeuvre — and any L/R label mined from it
    # is fabricated. Measured on full_local: 60.3% of drive videos are single-signed,
    # median |frac-0.5| = 0.425. Hence this gate, not a warning line.
    fracs, big_vids = {}, []
    for v in set(vids):
        s = vids == v
        if s.sum() >= a.min_clips_per_video:
            fracs[v] = float(np.mean(sig[s] > 0))
            big_vids.append(v)
    if not fracs:
        raise RuntimeError(f"FATAL: no video has >={a.min_clips_per_video} clips")
    fr = np.array(list(fracs.values()))
    mixed = {v for v, f in fracs.items() if a.mixed_lo <= f <= a.mixed_hi}
    print(f"[rank] SIGN-VALIDITY over {len(fr)} videos (>= {a.min_clips_per_video} clips): "
          f"single-signed={100*((fr < 0.1) | (fr > 0.9)).mean():.1f}%  "
          f"mixed={100*len(mixed)/len(fr):.1f}%  median|frac-0.5|={np.median(np.abs(fr-0.5)):.3f}")
    if len(mixed) / len(fr) < a.min_mixed_frac:
        raise RuntimeError(
            f"FATAL: only {100*len(mixed)/len(fr):.1f}% of videos show BOTH signs "
            f"(need >={100*a.min_mixed_frac:.0f}%). The sign of camera_motion_x is a PER-VIDEO "
            f"CONSTANT (camera mount/lens/scene bias), NOT turn direction — L/R labels mined "
            f"from it would be fabricated. Use a real ego-yaw estimator (frame-shift matching, "
            f"see utils/tmp/ood_turn_probe.py) on decoded clips instead.")
    # only clips from MIXED videos can carry a trustworthy sign
    ok = np.isin(vids, list(mixed))
    print(f"[rank] restricting to {len(mixed)} MIXED videos -> {int(ok.sum()):,} eligible clips")
    sig = np.where(ok, sig, 0.0)

    # top-N per pole, capped per video so a probe can do leave-one-VIDEO-out without leakage
    def top(sign):
        order = np.argsort(sig * sign)            # most extreme for this pole first
        picked, per_vid = [], Counter()
        for i in order:
            v = vids[i]
            if per_vid[v] >= a.per_video_cap:
                continue
            per_vid[v] += 1
            picked.append(int(i))
            if len(picked) >= a.top_n:
                break
        return picked

    out = {}
    for pole, sign in (("POLE_NEG", 1), ("POLE_POS", -1)):
        idx = top(sign)
        out[pole] = [{"clip": str(keys[i]), "video": vids[i],
                      "camera_motion_x": float(cam_x[i]), "fg_mean_mag": float(fg[i])}
                     for i in idx]
        vals = [cam_x[i] for i in idx]
        print(f"\n[rank] {pole}: {len(idx)} clips from {len({vids[i] for i in idx})} videos · "
              f"camera_motion_x range {min(vals):+.2f} .. {max(vals):+.2f}")
        for r in out[pole][:8]:
            print(f"        {r['camera_motion_x']:+7.2f}  {r['clip']}")

    dst = a.out / "turn_candidates.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n[rank] wrote {dst}")
    print("[rank] ⚠️  POLE_NEG/POLE_POS -> LEFT/RIGHT is NOT yet established — "
          "run --stage clips and EYEBALL before labelling anything.")


def stage_clips(a):
    """Extract candidate clips from local TAR shards + build a contact sheet to eyeball."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    from utils.data_download import iter_clips_parallel
    cands = json.load(open(a.out / "turn_candidates.json"))
    want = {}
    for pole in ("POLE_NEG", "POLE_POS"):
        for r in cands[pole][:a.n_eyeball]:
            want[r["clip"]] = (pole, r["camera_motion_x"])
    print(f"[clips] fetching {len(want)} clips from {a.local_data} ...")
    cdir = a.out / "clips"
    cdir.mkdir(parents=True, exist_ok=True)
    got = {}
    q, stop, _r = iter_clips_parallel(str(a.local_data), subset_keys=set(want), processed_keys=set())
    while len(got) < len(want):
        item = q.get(timeout=180)
        if item is None:
            break
        k, b = item
        if k in want and b:
            p = cdir / (k.replace("/", "__"))
            p = p.with_suffix(".mp4")
            p.write_bytes(b)
            got[k] = p
    stop.set()
    print(f"[clips] got {len(got)}/{len(want)}")
    if not got:
        raise RuntimeError("FATAL: no clips extracted — is --local-data pointing at the TAR shards?")

    FB = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    F = lambda s: ImageFont.truetype(FB, s)
    N, CW, CH = 8, 150, 105
    rows = sorted(got.items(), key=lambda kv: want[kv[0]][0])
    sheet = Image.new("RGB", (N * CW + 190, len(rows) * (CH + 6) + 22), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), "EYEBALL: which POLE turns which way? (this fixes the sign convention)",
           font=F(13), fill=(255, 255, 255))
    for r, (k, p) in enumerate(rows):
        y = 22 + r * (CH + 6)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", str(p)], capture_output=True).stdout or 6)
        for i in range(N):
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur*i/N:.2f}", "-i", str(p),
                                  "-frames:v", "1", "-vf", f"scale={CW}:{CH}", "-pix_fmt", "rgb24",
                                  "-f", "rawvideo", "-"], capture_output=True).stdout
            if len(raw) >= CW * CH * 3:
                sheet.paste(Image.frombytes("RGB", (CW, CH), raw[:CW * CH * 3]), (i * CW, y))
        pole, cx = want[k]
        col = (120, 220, 120) if pole == "POLE_NEG" else (255, 150, 90)
        d.text((N * CW + 6, y + CH // 2 - 12), f"{pole}\nx={cx:+.1f}", font=F(14), fill=col)
    dst = a.out / "turn_candidates_sheet.png"
    sheet.save(dst)
    print(f"[clips] {len(rows)} clips -> {dst}  — EYEBALL to fix POLE->LEFT/RIGHT")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["pull", "rank", "clips"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=str, default=None)
    p.add_argument("--hf-dir", type=str, default=None)
    p.add_argument("--local-data", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--per-video-cap", type=int, default=3)
    p.add_argument("--n-eyeball", type=int, default=6)
    # sign-validity gate (see stage_rank): a video whose clips are ~all one sign is
    # reporting its CAMERA, not its turns, so it cannot supply L/R labels.
    p.add_argument("--min-clips-per-video", type=int, default=20)
    p.add_argument("--mixed-lo", type=float, default=0.3)
    p.add_argument("--mixed-hi", type=float, default=0.7)
    p.add_argument("--min-mixed-frac", type=float, default=0.5,
                   help="fail if fewer than this fraction of videos show BOTH signs")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "pull" and not (a.repo and a.hf_dir):
        raise SystemExit("FATAL: --stage pull needs --repo and --hf-dir")
    if a.stage == "clips" and not a.local_data:
        raise SystemExit("FATAL: --stage clips needs --local-data")
    {"pull": stage_pull, "rank": stage_rank, "clips": stage_clips}[a.stage](a)
