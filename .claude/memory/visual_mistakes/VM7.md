---
id: VM7
title: metric name in scene title contradicts the on-frame unit label (2026-07-12)
category: demo-readability
keywords: []
---

## VM7 — metric name in scene title contradicts the on-frame unit label (2026-07-12)

**Symptom**: demo Scene B title said "future-frame MSE" while the value panel on the same frame said
"mean latent L1 (lower = better)" (JSON key B_l1); the verdict card repeated "MSE".
**Root cause**: title text written from the storyboard, unit label written from the metric code — two
sources for one fact.
**Fix**: derive the displayed metric name from ONE constant shared by title, unit label, JSON key, and
verdict row.
**Prevention**: every number shown must carry exactly one name; grep the render script for the metric
string appearing more than once.
