"""Contact-sheet browser: for a given scene type, tile the mid-frames of every
high-crowd walking clip present in the downloaded shards, numbered, so a human
can pick the best (most crowd, no overlay). Usage:
    python fig2_candidates.py <shards_dir> <scene> <out_png> [max]
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

SHARDS_DIR, SCENE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX = int(sys.argv[4]) if len(sys.argv) > 4 else 12
CROWD = sys.argv[5] if len(sys.argv) > 5 else "high"  # "any" to disable

lab = json.load(open("outputs/full/probe_taxonomy/taxonomy_labels.json"))
dims, labels = lab["dims"], lab["labels"]
sc, cd = dims["scene_type"]["values"], dims["crowd_density"]["values"]


def dec(lb, dim, vals):
    v = lb.get(dim)
    return vals[v] if isinstance(v, int) else v


# index shards: reconstructed clip_key -> (tar, mp4 member)
INDEX = {}
for tp in sorted(Path(SHARDS_DIR).rglob("subset-*.tar")):
    with tarfile.open(tp) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".json"):
                d = json.load(tar.extractfile(m))
                INDEX[f"{d['section']}/{d['video_id']}/{d['source_file']}"] = (str(tp), m.name[:-5] + ".mp4")

# candidates: high-crowd walking, this scene, present in shards
cands = []
for k, lb in labels.items():
    if "walking" not in k.lower() or k not in INDEX:
        continue
    if dec(lb, "scene_type", sc) == SCENE and dec(lb, "crowd_density", cd) == "high":
        cands.append(k)
cands = cands[:MAX]
print(f"{SCENE}: {len(cands)} candidates in shards")


def mid_frame(key):
    tp, mp4 = INDEX[key]
    with tarfile.open(tp) as tar:
        b = tar.extractfile(tar.getmember(mp4)).read()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        tf.write(b); tf.flush()
        cap = cv2.VideoCapture(tf.name)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read(); cap.release()
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


CW, CH = 400, 225
cols = 3
rows = (len(cands) + cols - 1) // cols
font = ImageFont.truetype(fm.findfont(fm.FontProperties(weight="bold")), 20)
sheet = Image.new("RGB", (cols * CW, rows * (CH + 26)), (30, 30, 30))
draw = ImageDraw.Draw(sheet)
for i, k in enumerate(cands):
    fr = mid_frame(k)
    if fr is None:
        continue
    im = Image.fromarray(fr).resize((CW, CH), Image.LANCZOS)
    r, c = divmod(i, cols)
    x, y = c * CW, r * (CH + 26)
    sheet.paste(im, (x, y))
    draw.text((x + 4, y + CH + 2), f"[{i}] {k.split('/')[-1]}", fill=(255, 255, 0), font=font)
    print(f"  [{i}] {k}")
sheet.save(OUT)
print("wrote", OUT)
