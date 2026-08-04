---
id: VM1
title: winner-arm annotations struck through the metric labels (2026-07-12)
category: latex-figure-placement
keywords: [column, y-tick]
---

## VM1 — winner-arm annotations struck through the metric labels (2026-07-12)

**Symptom**: per-row winner names rendered in the LEFT margin under each metric tick label overlapped/struck
through the labels in `forest_plot_*` and `scale_poc_vs_full_*`.
**Root cause**: side text placed in the margin that ALREADY belongs to y-tick labels; collision worsens at
compressed row pitch.
**Fix**: dedicated RIGHT-hand column — `annotate(xy=(1.05, y), xycoords=("axes fraction","data"))`, right
margin reserved in inches (`m13_eval_plot.py::plot_forest`).
**Prevention**: side text columns go in RESERVED margin space on the RHS, never the LHS.
