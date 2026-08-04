#!/usr/bin/env python3
"""
Reconstruct DENSEWORLD-115k clips from YouTube.

DENSEWORLD-115k is a METADATA-ONLY dataset. We do NOT redistribute the source videos:
they are downloaded from YouTube and remain under their creators' copyright. This script
re-downloads the 714 source videos listed in `sources.json` and regenerates the 115,687
clips deterministically, reproducing the exact pipeline used to build the dataset.
`data/data_prep/clip_durations.json` is the ground-truth manifest used for verification.

This file is dual-use: it is version-controlled in the FactorJEPA repo (src/hf_denseworld_115k/)
AND shipped verbatim as the root of the Hugging Face `denseworld-115k` dataset repo, so it is
intentionally self-contained (no repo imports; the pipeline constants below are inlined rather
than read from configs/pipeline.yaml, and mirror it exactly).

PIPELINE (identical to the DENSEWORLD-115k build; constants mirror configs/pipeline.yaml):
  1. Download each source video at 480p                 (yt-dlp; pipeline.yaml data.download_resolution)
  2. Detect scene boundaries                            (PySceneDetect ContentDetector, threshold=15.0)
  3. Greedy-split into 4-10 s clips at scene boundaries (scene_detection.clip_{min,max}_duration)
  4. Encode each clip                                   (ffmpeg libx264 CRF 28, aac 128k, +faststart)
Output: <out>/<section>/<video_id>-<NNN>.mp4  (e.g. goa/walking/04YKvC8kAgI-000.mp4)

NOTE: 3 very long videos (see sources.json "chunked_videos") were originally processed in
fixed windows rather than whole-video scene detection; this script scene-detects them like the
rest, so their clip boundaries may differ slightly (<1% of the dataset). The verify step reports
any per-section clip-count mismatch against the manifest.

REQUIREMENTS:
  pip install -r requirements.txt          # yt-dlp, scenedetect[opencv]
  ffmpeg + ffprobe on PATH                 # brew install ffmpeg  /  apt-get install ffmpeg

USAGE:
  python reconstruct.py --out ./denseworld_clips                 # rebuild all 714 videos
  python reconstruct.py --out ./denseworld_clips --limit 1       # smoke test: first video only
  python reconstruct.py --out ./denseworld_clips --only 04YKvC8kAgI   # one specific YouTube ID
  python reconstruct.py --out ./denseworld_clips --verify-only   # check existing clips vs manifest
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---- pipeline constants (inlined; mirror configs/pipeline.yaml so this file runs standalone) ----
CLIP_MIN_DURATION = 4.0     # pipeline.yaml scene_detection.clip_min_duration
CLIP_MAX_DURATION = 10.0    # pipeline.yaml scene_detection.clip_max_duration
REENCODE_CRF = 28           # pipeline.yaml reencode_crf (libx264)
SCENE_THRESHOLD = 15.0      # PySceneDetect ContentDetector threshold
DOWNLOAD_RES = 480          # pipeline.yaml data.download_resolution (yt-dlp max height)

HERE = Path(__file__).resolve().parent
SOURCES_JSON = HERE / "sources.json"
MANIFEST_JSON = HERE / "data" / "data_prep" / "clip_durations.json"


def download_video(video_id: str, url: str, dst: Path, res: int = DOWNLOAD_RES) -> bool:
    """Download one YouTube video at <=res p (mp4). Returns True on success, False if unavailable."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    fmt = (f"bestvideo[height<={res}][ext=mp4]+bestaudio[ext=m4a]/"
           f"bestvideo[height<={res}]+bestaudio/best[height<={res}]/best")
    cmd = ["yt-dlp", "-f", fmt, "-o", str(dst), "--merge-output-format", "mp4",
           "--no-warnings", "--no-progress", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
            print(f"    yt-dlp failed for {video_id}: {last}")
        return dst.exists() and dst.stat().st_size > 0
    except FileNotFoundError:
        sys.exit("ERROR: yt-dlp not found. pip install yt-dlp")
    except subprocess.TimeoutExpired:
        print(f"    yt-dlp timeout for {video_id}")
        return False


def get_duration(path: Path) -> float:
    """Duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out) if out else 0.0
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def detect_boundaries(path: Path, video_dur: float) -> list:
    """PySceneDetect ContentDetector(threshold=15) -> sorted interior boundary seconds."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    video = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
    sm.detect_scenes(video, show_progress=False)
    boundaries = set()
    for start, end in sm.get_scene_list():
        for t in (start.get_seconds(), end.get_seconds()):
            if 0 < t < video_dur:
                boundaries.add(t)
    return sorted(boundaries)


def greedy_split_plan(video_duration: float, boundaries: list,
                      min_dur: float, max_dur: float) -> list:
    """Verbatim from the build pipeline: contiguous [4,10]s greedy cut at scene boundaries."""
    if video_duration <= 0:
        return []
    if video_duration <= max_dur:
        return [(0.0, video_duration)]
    clips, pos = [], 0.0
    while pos < video_duration:
        remaining = video_duration - pos
        if remaining < min_dur:
            if clips:
                prev_start, _ = clips[-1]
                clips[-1] = (prev_start, video_duration)
            else:
                clips.append((0.0, video_duration))
            break
        window_start, window_end = pos + min_dur, pos + max_dur
        candidates = [x for x in boundaries if window_start <= x <= min(window_end, video_duration)]
        cut = max(candidates) if candidates else min(window_end, video_duration)
        if 0 < (video_duration - cut) < min_dur:
            cut = video_duration
        clips.append((pos, cut))
        pos = cut
    return clips


def encode_clip(src: Path, start: float, duration: float, dst: Path) -> bool:
    """Encode one clip: libx264 CRF 28, aac 128k, +faststart. -ss before -i for fast seek."""
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(src), "-t", str(duration),
           "-c:v", "libx264", "-crf", str(REENCODE_CRF), "-preset", "medium",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
           "-loglevel", "error", str(dst)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0 and dst.exists()
    except subprocess.SubprocessError:
        return False


def process_video(v: dict, out_root: Path, work: Path) -> int:
    """Download + scene-detect + split + encode one source video. Returns clip count."""
    vid, url = v["id"], v["url"]
    section = v["sections"][0] if v.get("sections") else "unsorted"
    clip_dir = out_root / section
    clip_dir.mkdir(parents=True, exist_ok=True)

    src = work / f"{vid}.mp4"
    print(f"  [{vid}] downloading ({section}) ...")
    if not download_video(vid, url, src):
        print(f"  [{vid}] SKIP (download failed / video unavailable)")
        return 0
    dur = get_duration(src)
    boundaries = detect_boundaries(src, dur)
    plan = greedy_split_plan(dur, boundaries, CLIP_MIN_DURATION, CLIP_MAX_DURATION)
    n = 0
    for i, (s, e) in enumerate(plan):
        dst = clip_dir / f"{vid}-{i:03d}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            n += 1
            continue
        if encode_clip(src, s, e - s, dst):
            n += 1
    print(f"  [{vid}] {n} clips  (manifest expects {v['n_clips']})")
    return n


def verify(out_root: Path, manifest: dict) -> None:
    """Compare on-disk clip counts to clip_durations.json per section."""
    print("\n=== VERIFY vs clip_durations.json ===")
    exp_total = manifest["summary"]["total_clips"]
    got_total, mism = 0, 0
    for section, sd in manifest["sections"].items():
        expected = sum(len(c) for c in sd.get("videos", {}).values())
        got = len(list((out_root / section).glob("*.mp4"))) if (out_root / section).exists() else 0
        got_total += got
        if got != expected:
            mism += 1
            if mism <= 20:
                print(f"  MISMATCH {section}: got {got}, expected {expected}")
    print(f"\n  total clips: got {got_total:,} / expected {exp_total:,}  | section mismatches: {mism}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct DENSEWORLD-115k clips from YouTube.")
    ap.add_argument("--out", required=True, help="output root for reconstructed clips")
    ap.add_argument("--work", default=None, help="scratch dir for source videos (default: <out>/_sources)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N videos (smoke test)")
    ap.add_argument("--only", default=None, help="process only this base YouTube ID")
    ap.add_argument("--verify-only", action="store_true", help="only verify existing clips vs the manifest")
    args = ap.parse_args()

    out_root = Path(args.out)
    manifest = json.load(open(MANIFEST_JSON))
    if args.verify_only:
        verify(out_root, manifest)
        return

    out_root.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out_root / "_sources"
    work.mkdir(parents=True, exist_ok=True)

    videos = json.load(open(SOURCES_JSON))["videos"]
    if args.only:
        videos = [v for v in videos if v["id"] == args.only]
    if args.limit:
        videos = videos[:args.limit]

    print(f"Reconstructing {len(videos)} source video(s) -> {out_root}")
    total = 0
    for j, v in enumerate(videos, 1):
        print(f"[{j}/{len(videos)}]")
        total += process_video(v, out_root, work)
    print(f"\nDONE: {total:,} clips reconstructed.")
    verify(out_root, manifest)


if __name__ == "__main__":
    main()
