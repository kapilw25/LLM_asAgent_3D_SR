"""Rank walking clips per coverage type by a metadata-derived CROWD score
(from taxonomy_labels.json — no pixels, no eyeballing), then render the top-K
mid-frames per scene as a scored contact sheet so the final pick is grounded.

crowd score = 3*crowd_density + 3*pedestrian_present + 2*pedestrian_dominant
              + 1*street_vendor + 1*shared_space + 1*traffic_density
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

SHARDS_DIR, OUT_SHEET, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 3

lab = json.load(open("outputs/full/probe_taxonomy/taxonomy_labels.json"))
dims, labels = lab["dims"], lab["labels"]
SC = dims["scene_type"]["values"]
CD = dims["crowd_density"]["values"]
TD = dims["traffic_density"]["values"]
TM = dims["traffic_mix"]["values"]
PVS = dims["pedestrian_vehicle_separation"]["values"]
NO = dims["notable_objects"]["values"]
PED_I, VEND_I = NO.index("pedestrian"), NO.index("street_vendor")

ORDER = ["market", "bazaar", "commercial", "transit", "residential_lane", "promenade",
         "ghat", "heritage_tourist", "highway", "junction", "flyover_underpass", "beach_coastal"]


def as_idx(v):
    return v if isinstance(v, int) else None


def crowd_score(lb):
    cd = as_idx(lb.get("crowd_density"))          # 0..2
    td = as_idx(lb.get("traffic_density"))
    tm = lb.get("traffic_mix")
    pvs = lb.get("pedestrian_vehicle_separation")
    no = lb.get("notable_objects") or []
    ped = no[PED_I] if len(no) > PED_I else 0
    vend = no[VEND_I] if len(no) > VEND_I else 0
    pd = 1 if (isinstance(tm, int) and TM[tm] == "pedestrian_dominant") else 0
    shared = 1 if (isinstance(pvs, int) and PVS[pvs] == "shared_space") else 0
    return 3 * (cd or 0) + 3 * ped + 2 * pd + 1 * vend + 1 * shared + 1 * (td or 0), \
        dict(crowd=CD[cd] if cd is not None else "?", ped=ped, vend=vend, pd=pd, td=TD[td] if td is not None else "?")


# index shards
INDEX = {}
for tp in sorted(Path(SHARDS_DIR).rglob("subset-*.tar")):
    with tarfile.open(tp) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".json"):
                d = json.load(tar.extractfile(m))
                INDEX[f"{d['section']}/{d['video_id']}/{d['source_file']}"] = (str(tp), m.name[:-5] + ".mp4")

# rank
ranked = {s: [] for s in ORDER}
for k, lb in labels.items():
    if "walking" not in k.lower() or k not in INDEX:
        continue
    st = as_idx(lb.get("scene_type"))
    if st is None or SC[st] not in ranked:
        continue
    sco, parts = crowd_score(lb)
    ranked[SC[st]].append((sco, k, parts))
for s in ORDER:
    ranked[s].sort(key=lambda x: -x[0])

json.dump({s: [(sc, k, p) for sc, k, p in ranked[s][:TOPK]] for s in ORDER},
          open(OUT_JSON, "w"), indent=1)
for s in ORDER:
    top = ranked[s][:TOPK]
    print(f"{s:20s} n={len(ranked[s]):3d}  top:", [(sc, p['crowd'], f"ped{p['ped']}") for sc, k, p in top])


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
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else np.zeros((225, 400, 3), np.uint8)


CW, CH = 380, 214
font = ImageFont.truetype(fm.findfont(fm.FontProperties(weight="bold")), 17)
sheet = Image.new("RGB", (TOPK * CW, len(ORDER) * (CH + 24)), (25, 25, 25))
draw = ImageDraw.Draw(sheet)
for r, s in enumerate(ORDER):
    for c, (sc, k, p) in enumerate(ranked[s][:TOPK]):
        im = Image.fromarray(mid_frame(k)).resize((CW, CH), Image.LANCZOS)
        x, y = c * CW, r * (CH + 24)
        sheet.paste(im, (x, y))
        tag = f"{s}[{c}] sc={sc} cr={p['crowd']} ped={p['ped']} pd={p['pd']}"
        draw.text((x + 3, y + CH + 2), tag, fill=(255, 255, 0), font=font)
sheet.save(OUT_SHEET)
print("wrote", OUT_SHEET)
