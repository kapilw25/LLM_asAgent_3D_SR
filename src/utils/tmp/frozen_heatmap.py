"""Compact single-column heatmap of the frozen-encoder scorecard (ablation evidence).

Reads m13_frozen_scorecard.csv (10 frozen encoders x 3 metrics) and renders a small,
paper-ready heatmap: cells coloured per-column (green = better) with value +/- 95% CI,
title emphasising the collapse band. Vector PDF, TrueType (AAAI-safe).
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC, OUT = sys.argv[1], sys.argv[2]

NAME = {
    "vjepa_2_1_frozen": "V-JEPA 2.1", "vjepa_2_1_vitL_frozen": "V-JEPA 2.1 ViT-L",
    "vjepa_1_vitH_frozen": "V-JEPA 1 ViT-H", "lejepa_vitL_frozen": "LeJEPA ViT-L",
    "vjepa_1_vitL_frozen": "V-JEPA 1 ViT-L", "ijepa_vitH14": "I-JEPA ViT-H",
    "vjepa_2_0_vitg_ssv2": "V-JEPA 2.0 (SSv2)", "dinov2": "DINOv2",
    "vjepa_2_vitL_256_frozen": "V-JEPA 2 ViT-L", "ijepa_vitG16": "I-JEPA ViT-G",
}
COLS = [("action_top1", "Action\ntop-1 (%)"), ("motion_cos", "motion-cos\nsep."),
        ("taxonomy_f1", "taxonomy\nF1")]

rows, vals, cis = [], [], []
with open(SRC) as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append(NAME.get(d["encoder"], d["encoder"]))
        v, c = [], []
        for key, _ in COLS:
            mean, ci = d[key].split("\n")
            v.append(float(mean)); c.append(ci.strip())
        vals.append(v); cis.append(c)
vals = np.array(vals)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.2})
fig, ax = plt.subplots(figsize=(3.34, 3.15))
norm = (vals - vals.min(0)) / (vals.max(0) - vals.min(0) + 1e-9)   # per-column 0..1
ax.imshow(norm, cmap="YlGn", aspect="auto", vmin=-0.15, vmax=1.05)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], fontsize=7.2, fontweight="bold")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
ax.tick_params(length=0)
for i in range(len(rows)):
    for j in range(len(COLS)):
        txt = f"{vals[i, j]:.1f}" if j == 0 else f"{vals[i, j]:.3f}"
        ax.text(j, i, txt + "\n" + cis[i][j], ha="center", va="center", fontsize=5.7,
                color="black")
ax.set_title("Frozen encoders collapse:\n37.5-44.4% Action top-1 band",
             fontsize=8.2, fontweight="bold", pad=5)
for s in ax.spines.values():
    s.set_visible(False)
fig.text(0.5, 0.005, "19.5% majority baseline; DENSEWORLD-adapted reach 50.3-53.2%.",
         ha="center", fontsize=5.8, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT)
print("wrote", OUT)
