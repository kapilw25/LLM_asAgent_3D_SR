"""OOD turn probe — does OURS's surgery edge survive OUT-OF-DOMAIN on VISIBLE ego-yaw?

The load-bearing experiment behind iter20's closing finding (plan_v2 §F): FROZEN beats OURS by
−6.0 (MLP) / −10.8pp (LINEAR) on ghat-road POV driving turns (n=167, majority baseline 0.503, so the
test had real power), replicating Diving48. See memory project_iter20_ood_edge_indomain_only.

Encoder-level (NO LLM → immune to the projector-starvation that made the VLM early gate inconclusive).
Ghat-road POV driving video → sliding segments → flow-derived {STRAIGHT, TURN_LEFT, TURN_RIGHT} →
V-JEPA features (FROZEN vs OURS) → small head → accuracy.

VALIDITY: segments from ONE continuous drive are strongly autocorrelated. A random split would leak
(neighbouring windows share frames/scenery) and fabricate a win. We use LEAVE-ONE-BLOCK-OUT over
contiguous temporal blocks, so each test block is temporally distant from its training data.

stages: scan (CPU) → feats (GPU) → probe (CPU / GPU-head)

USAGE (run from repo root):
    PYTHONPATH=src python -m utils.tmp.ood_turn_probe --stage scan  --video <mp4> --out <dir>
    PYTHONPATH=src python -m utils.tmp.ood_turn_probe --stage feats --video <mp4> --out <dir> \
        --frozen-ckpt checkpoints/vjepa2_1_vitg_384.pt \
        --ours-ckpt outputs/full/.../m09c_ckpt_best.pt --model-config configs/model/vjepa2_1_vitg.yaml
    PYTHONPATH=src python -m utils.tmp.ood_turn_probe --stage probe --video <mp4> --out <dir> --blocks 5
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _gray(video, fps, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},scale={w}:{h}",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def stage_scan(a):
    g = _gray(a.video, a.scan_fps, 160, 90)
    H, W, m = 90, 160, 8
    dxs = []
    for t in range(len(g) - 1):
        aa, bb = g[t, m:H - m, m:W - m], g[t + 1]
        best, bd = 1e18, 0
        for dx in range(-m, m + 1):
            s = float(np.abs(aa - bb[m:H - m, m + dx:W - m + dx]).mean())
            if s < best:
                best, bd = s, dx
        dxs.append(bd)
    dxs = np.array(dxs, float)
    print(f"[scan] {len(g)} frames @ {a.scan_fps}fps ({len(g)/a.scan_fps:.0f}s)")

    win, step = int(a.seg_s * a.scan_fps), int(a.step_s * a.scan_fps)
    segs = []
    for s in range(0, len(dxs) - win, step):
        cum = float(dxs[s:s + win].sum())
        segs.append({"t0": s / a.scan_fps, "dur": a.seg_s, "cum_dx": cum})
    cums = np.array([x["cum_dx"] for x in segs])
    hi = np.percentile(np.abs(cums), a.turn_pct)
    lo = np.percentile(np.abs(cums), a.straight_pct)
    print(f"[scan] |cum_dx| percentiles: p{a.straight_pct}={lo:.1f}  p{a.turn_pct}={hi:.1f}  max={np.abs(cums).max():.1f}")
    for x in segs:
        c = x["cum_dx"]
        x["cls"] = ("TURN_LEFT" if c > 0 else "TURN_RIGHT") if abs(c) >= hi else ("STRAIGHT" if abs(c) <= lo else "AMB")
    keep = [x for x in segs if x["cls"] != "AMB"]
    from collections import Counter
    print(f"[scan] {len(segs)} windows -> {len(keep)} labelled · {dict(Counter(x['cls'] for x in keep))}")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(a.out / "segments.json", "w"), indent=1)


def stage_feats(a):
    import torch
    from tqdm import tqdm
    from utils.predictor_eval import load_encoder_only
    from utils.frozen_features import resize_and_normalize
    segs = json.load(open(a.out / "segments.json"))
    arms = {"frozen": a.frozen_ckpt, "ours": a.ours_ckpt}
    # decode each segment ONCE (16 frames @ 384), reuse for both arms
    px = []
    for s in tqdm(segs, desc="decode"):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{s['t0']:.2f}", "-t", f"{s['dur']:.2f}",
                              "-i", str(a.video), "-vf", f"fps={16/s['dur']:.4f},scale=384:384",
                              "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, 384, 384, 3)
        if len(fr) < 16:
            fr = np.concatenate([fr, np.repeat(fr[-1:], 16 - len(fr), 0)], 0)
        px.append(fr[:16])
    for arm, ck in arms.items():
        enc, _, _dim = load_encoder_only(ck, 16, a.model_config)
        feats = []
        for fr in tqdm(px, desc=f"feat[{arm}]"):
            n = resize_and_normalize(fr, 384).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to("cuda", torch.bfloat16)
            with torch.no_grad():
                h = enc(n)
                if isinstance(h, (list, tuple)):
                    h = torch.cat(list(h), -1)
            feats.append(h[0].float().mean(0).cpu().numpy())
        np.save(a.out / f"feats_{arm}.npy", np.stack(feats))
        del enc
        torch.cuda.empty_cache()
        print(f"[feats] {arm}: {np.stack(feats).shape}")


def stage_probe(a):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    segs = json.load(open(a.out / "segments.json"))
    y = np.array([{"STRAIGHT": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2}[s["cls"]] for s in segs])
    t0 = np.array([s["t0"] for s in segs])
    block = np.floor(t0 / (t0.max() + 1e-9) * a.blocks).astype(int)   # contiguous TEMPORAL blocks
    block[block == a.blocks] = a.blocks - 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def run(X, mlp):
        pred = np.empty(len(y), int)
        for b in range(a.blocks):
            tr, te = block != b, block == b
            if te.sum() == 0 or len(set(y[tr])) < 2:
                pred[te] = 0
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=dev)
            ytr = torch.tensor(y[tr], device=dev)
            D = Xtr.shape[1]
            head = (nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 3))
                    if mlp else nn.Linear(D, 3)).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
            lf = nn.CrossEntropyLoss()
            for _ in range(300):
                opt.zero_grad()
                lf(head(Xtr), ytr).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                pred[te] = head(Xte).argmax(1).cpu().numpy()
        return (pred == y).mean()

    from collections import Counter
    chance = max(Counter(y).values()) / len(y)
    print(f"\n[probe] n={len(y)} · classes {dict(Counter(y))} · majority-class baseline={chance:.3f} "
          f"· leave-one-of-{a.blocks}-temporal-blocks-out")
    print(f"{'head':7s} {'FROZEN':>8s} {'OURS':>8s} {'d(O-F)':>8s}")
    res = {}
    for mlp, nm in [(False, "LINEAR"), (True, "MLP")]:
        f = run(np.load(a.out / "feats_frozen.npy"), mlp)
        o = run(np.load(a.out / "feats_ours.npy"), mlp)
        res[nm] = (f, o)
        print(f"{nm:7s} {f:8.3f} {o:8.3f} {o-f:+8.3f}")
    print("\n=== VERDICT (OOD visible ego-yaw, encoder level) ===")
    for nm, (f, o) in res.items():
        d = (o - f) * 100
        v = "OURS wins" if d >= 3 else ("OURS does NOT beat FROZEN" if d <= 0 else "marginal (<3pp)")
        print(f"  {nm}: {d:+.1f}pp -> {v}")
    json.dump({k: {"frozen": v[0], "ours": v[1]} for k, v in res.items()},
              open(a.out / "probe_result.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["scan", "feats", "probe"])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frozen-ckpt", type=Path, default=None)
    p.add_argument("--ours-ckpt", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--scan-fps", type=int, default=6)
    p.add_argument("--seg-s", type=float, default=8.0)
    p.add_argument("--step-s", type=float, default=4.0)
    p.add_argument("--turn-pct", type=int, default=70)
    p.add_argument("--straight-pct", type=int, default=30)
    p.add_argument("--blocks", type=int, default=5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"scan": stage_scan, "feats": stage_feats, "probe": stage_probe}[a.stage](a)
