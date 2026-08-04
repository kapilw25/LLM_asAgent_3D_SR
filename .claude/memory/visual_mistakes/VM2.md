---
id: VM2
title: suptitle wider than the plot → wasted flank space (2026-07-12)
category: plot-authoring
keywords: [suptitle]
---

## VM2 — suptitle wider than the plot → wasted flank space (2026-07-12)

**Symptom**: one/two-line suptitle spanned the full image width while the axes used ~50%, leaving dead
space beside every other element.
**Root cause**: raw f-string suptitle, no wrapping; figure width driven by the title.
**Fix**: `textwrap.fill(seg, width=68)` per newline-segment; top margin sized from wrapped line count
(0.26 in/line).
**Prevention**: wrap EVERY suptitle; title must be narrower than the figure.
