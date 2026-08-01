"""Fig 5 space-saver: merge the two stacked bar plots (count density + agent
occupancy) into ONE compact dual-y-axis line plot with a shared x-axis.
Left axis = count density (solid), right axis = occupancy (dashed);
blue = Representative West, orange = DENSEWORLD. Values read off the
professor-provided figure (same as src/utils/fig_denseworld_vs_west_density.py).
Vector PDF, TrueType (AAAI-safe).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = sys.argv[1]
SCENES = ["Residential lane", "Promenade", "Market",
          "Heritage / Tourist", "Flyover / Underpass", "Commercial"]
COUNT_WEST = [1.8, 1.4, 4.6, 1.7, 2.6, 4.1]
COUNT_DW = [3.2, 2.5, 12.5, 3.0, 5.0, 11.0]
OCC_WEST = [1.9, 1.5, 4.2, 0.8, 8.7, 3.8]
OCC_DW = [3.4, 2.7, 11.1, 1.4, 20.9, 9.0]
BLUE, ORANGE = "#1f77b4", "#ff7f0e"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5,
                     "font.weight": "bold", "axes.labelweight": "bold"})
x = np.arange(len(SCENES))
fig, ax1 = plt.subplots(figsize=(3.36, 1.98))
ax2 = ax1.twinx()

ax1.plot(x, COUNT_WEST, "o-", color=BLUE, lw=1.6, ms=4)
ax1.plot(x, COUNT_DW, "o-", color=ORANGE, lw=1.6, ms=4)
ax2.plot(x, OCC_WEST, "s--", color=BLUE, lw=1.4, ms=3.5, alpha=0.9)
ax2.plot(x, OCC_DW, "s--", color=ORANGE, lw=1.4, ms=3.5, alpha=0.9)

ax1.set_ylabel("Mean agents / clip", fontsize=8)
ax2.set_ylabel("Agent pixel ratio (%)", fontsize=8)
ax1.set_ylim(0, 13.5)
ax2.set_ylim(0, 22.5)
ax1.set_xticks(x)
ax1.set_xticklabels(SCENES, fontsize=6.6, rotation=22, ha="right", rotation_mode="anchor")
ax1.tick_params(labelsize=7)
ax2.tick_params(labelsize=7)
ax1.grid(axis="y", alpha=0.25, lw=0.6)
ax1.set_axisbelow(True)

handles = [Line2D([], [], color=BLUE, marker="o", lw=1.6, label="Representative West"),
           Line2D([], [], color=ORANGE, marker="o", lw=1.6, label="DENSEWORLD"),
           Line2D([], [], color="0.35", ls="-", marker="o", label="count (left)"),
           Line2D([], [], color="0.35", ls="--", marker="s", label="occupancy (right)")]
ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
           fontsize=6.3, ncol=4, handlelength=1.5, columnspacing=1.1, framealpha=0.92)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
