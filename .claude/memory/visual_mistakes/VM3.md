---
id: VM3
title: multi-panel comparison rendered side-by-side landscape (2026-07-12)
category: latex-figure-placement
keywords: [column]
---

## VM3 — multi-panel comparison rendered side-by-side landscape (2026-07-12)

**Symptom**: `forest_plot_best_ci.png` (2 backbones) was 2 panels side-by-side — cannot fit one column of a
2-column AAAI paper.
**Root cause**: `subplots(1, n)` default with a `vertical=` opt-in switch.
**Fix**: `plot_forest` always `subplots(n, 1)` portrait; the `vertical` parameter was deleted.
**Prevention**: comparison panels stack VERTICALLY, always — homogeneous across every figure destined for
the paper.
