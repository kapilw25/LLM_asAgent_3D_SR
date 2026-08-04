---
id: VM8
title: chart y-axis label drawn over the neighbouring image panel (2026-07-12)
category: plot-authoring
keywords: []
---

## VM8 — chart y-axis label drawn over the neighbouring image panel (2026-07-12)

**Symptom**: Scene C ylabel "prediction error (L1)" rendered on top of the predicted-frame heatmap
(white glyphs on light-yellow pixels — invisible).
**Root cause**: matplotlib places the ylabel OUTSIDE the axes bbox; the compositor packed the image
panel flush against the axes' left spine, so the label landed on the image.
**Fix**: reserve a gutter (~40 px) between an image panel and any chart whose ylabel extends left;
or use labelpad/inset label.
**Prevention**: when compositing images next to matplotlib axes, budget for text that extends beyond
the axes bbox, not just the bbox itself.
