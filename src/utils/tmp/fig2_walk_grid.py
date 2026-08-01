"""Fig 2 (scene-type coverage) regenerator: replace the aerial/drone panels with
GROUND-LEVEL WALKING frames that show crowd. Pulls one high-crowd walking clip
per scene type from the downloaded subset TAR shards (member name == clip_key),
grabs a representative mid-clip frame, and composes the same 4x3 navy-labelled
grid as the original denseworld_scene_types_overview.png.

Usage:
    python fig2_walk_grid.py <shards_dir> <keys_json> <out_png>
"""
import io
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

SHARDS_DIR, KEYS_JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

# 4x3 layout order + display labels (matches the original figure exactly)
ORDER = [
    ("market", "Market"), ("temple", "Temple"), ("commercial", "Commercial"),
    ("transit", "Transit"), ("residential_lane", "Residential"), ("promenade", "Promenade"),
    ("ghat", "Ghat"), ("heritage_tourist", "Heritage"), ("highway", "Highway"),
    ("junction", "Junction"), ("flyover_underpass", "Flyover"), ("beach_coastal", "Beach"),
]
# these source videos carry a "YouTube.com/AnujN..." channel watermark as a thin
# band along the very top edge; crop that fraction off the raw frame to drop it.
WATERMARK_CROP_TOP = {
    "residential_lane": 0.075,
    "flyover_underpass": 0.075,
}
# top-right corner logos over sky (e.g. "STEP OUT") — inpaint rather than crop so
# the subject (temple gopuram reaching the top edge) is preserved. Fractional boxes.
WATERMARK_INPAINT = {
    "temple": [(0.885, 0.0, 1.0, 0.15)],
}
NAVY = (27, 42, 74)
W, H = 1612, 1404
GAP = 8
COLS, ROWS = 3, 4
CELL_W = (W - (COLS + 1) * GAP) // COLS          # 526
CELL_H = (H - (ROWS + 1) * GAP) // ROWS          # 341
BAR_H = 44
IMG_H = CELL_H - BAR_H                            # 297

import json  # noqa: E402
keys = json.load(open(KEYS_JSON))
INDEX = {}  # populated after build_index() is defined


def build_index():
    """Members are numeric (011916.mp4/.json); reconstruct clip_key from each
    sidecar (section/video_id/source_file) -> {clip_key: (tar_path, member)}."""
    idx = {}
    for tar_path in sorted(Path(SHARDS_DIR).rglob("subset-*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            for m in tar.getmembers():
                if not m.name.endswith(".json"):
                    continue
                d = json.load(tar.extractfile(m))
                key = f"{d['section']}/{d['video_id']}/{d['source_file']}"
                idx[key] = (str(tar_path), m.name[:-5] + ".mp4")
    return idx


def find_member_bytes(clip_key):
    tar_path, mp4 = INDEX[clip_key]
    with tarfile.open(tar_path, "r") as tar:
        return tar.extractfile(tar.getmember(mp4)).read()


def mid_frame(mp4_bytes):
    """Decode the clip, return the middle frame as an RGB numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
        tf.write(mp4_bytes)
        tf.flush()
        cap = cv2.VideoCapture(tf.name)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("could not decode frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def inpaint_boxes(arr, boxes):
    """Inpaint fractional boxes (corner logos over sky) with cv2 TELEA."""
    h, w, _ = arr.shape
    mask = np.zeros((h, w), np.uint8)
    for fx0, fy0, fx1, fy1 in boxes:
        mask[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = 255
    return cv2.inpaint(arr, mask, 4, cv2.INPAINT_TELEA)


def center_crop_resize(arr, tw, th):
    """Center-crop to target aspect then resize to (tw, th)."""
    h, w, _ = arr.shape
    target_ar = tw / th
    ar = w / h
    if ar > target_ar:                      # too wide -> crop width
        nw = int(h * target_ar)
        x0 = (w - nw) // 2
        arr = arr[:, x0:x0 + nw]
    else:                                   # too tall -> crop height
        nh = int(w / target_ar)
        y0 = (h - nh) // 2
        arr = arr[y0:y0 + nh, :]
    return np.array(Image.fromarray(arr).resize((tw, th), Image.LANCZOS))


# bold TrueType font for the labels
bold_path = fm.findfont(fm.FontProperties(weight="bold"))
font = ImageFont.truetype(bold_path, 26)

INDEX = build_index()
print(f"indexed {len(INDEX)} clips across shards", flush=True)

canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for idx, (scene, label) in enumerate(ORDER):
    r, c = divmod(idx, COLS)
    x0 = GAP + c * (CELL_W + GAP)
    y0 = GAP + r * (CELL_H + GAP)
    frame = mid_frame(find_member_bytes(keys[scene]))
    if scene in WATERMARK_INPAINT:
        frame = inpaint_boxes(frame, WATERMARK_INPAINT[scene])
    if scene in WATERMARK_CROP_TOP:
        frame = frame[int(frame.shape[0] * WATERMARK_CROP_TOP[scene]):]
    img = center_crop_resize(frame, CELL_W, IMG_H)
    canvas.paste(Image.fromarray(img), (x0, y0))
    # navy label bar
    draw.rectangle([x0, y0 + IMG_H, x0 + CELL_W, y0 + CELL_H], fill=NAVY)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x0 + (CELL_W - tw) // 2 - tb[0], y0 + IMG_H + (BAR_H - th) // 2 - tb[1]),
              label, fill=(255, 255, 255), font=font)
    print(f"  {scene:20s} {label:12s} <- {keys[scene]}", flush=True)

canvas.save(OUT)
print("wrote", OUT, canvas.size)
