"""Emit a LaTeX-native heatmap table (\\cellcolor) for the frozen-encoder scorecard.
No image — pure LaTeX (needs xcolor[table] + makecell, both already in the preamble).
Cell shade is per-column min-max -> green!20..green!68."""
import csv
import sys

SRC = sys.argv[1]
NAME = {
    "vjepa_2_1_frozen": "V-JEPA 2.1", "vjepa_2_1_vitL_frozen": "V-JEPA 2.1 ViT-L",
    "vjepa_1_vitH_frozen": "V-JEPA 1 ViT-H", "lejepa_vitL_frozen": "LeJEPA ViT-L",
    "vjepa_1_vitL_frozen": "V-JEPA 1 ViT-L", "ijepa_vitH14": "I-JEPA ViT-H",
    "vjepa_2_0_vitg_ssv2": "V-JEPA 2.0 (SSv2)", "dinov2": "DINOv2",
    "vjepa_2_vitL_256_frozen": "V-JEPA 2 ViT-L", "ijepa_vitG16": "I-JEPA ViT-G",
}
COLS = ["action_top1", "motion_cos", "taxonomy_f1"]
rows = []
with open(SRC) as f:
    for d in csv.DictReader(f):
        m = {c: float(d[c].split("\n")[0]) for c in COLS}
        rows.append((NAME.get(d["encoder"], d["encoder"]), m))
lo = {c: min(r[1][c] for r in rows) for c in COLS}
hi = {c: max(r[1][c] for r in rows) for c in COLS}


def shade(c, v):
    n = (v - lo[c]) / (hi[c] - lo[c] + 1e-9)
    return int(round(20 + 48 * n))


def fmt(c, v):
    return f"{v:.1f}" if c == "action_top1" else f"{v:.3f}"


print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{\textbf{\emph{Frozen encoders collapse.}} Ten off-the-shelf frozen encoders "
      r"on the DENSEWORLD motion-class probe ($n_{\mathrm{test}}=1{,}825$; $\pm$95\% BCa CI "
      r"$\approx\!2.2$--$2.3$ on Action top-1). Action top-1 stays in a narrow "
      r"$37.5$--$44.4\%$ band (19.5\% majority baseline), while DENSEWORLD-adapted encoders reach "
      r"$50.3$--$53.2\%$. Cell shade $\propto$ within-column rank ($\uparrow$ better).}")
print(r"\label{tab:frozen_scorecard}")
print(r"\scriptsize")
print(r"\setlength{\tabcolsep}{3.5pt}")
print(r"\renewcommand{\arraystretch}{1.18}")
print(r"\begin{tabular}{@{}l ccc@{}}")
print(r"\toprule")
print(r"Frozen encoder & \makecell{Action\\top-1\,(\%)} & \makecell{motion-cos\\sep.} "
      r"& \makecell{taxonomy\\F1} \\")
print(r"\midrule")
for name, m in rows:
    cells = " & ".join(rf"\cellcolor{{green!{shade(c, m[c])}}}{fmt(c, m[c])}" for c in COLS)
    print(rf"{name} & {cells} \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
