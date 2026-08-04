---
id: VM5
title: compressed vertical panels made per-row annotations collide (2026-07-12)
category: plot-authoring
keywords: []
---

## VM5 — compressed vertical panels made per-row annotations collide (2026-07-12)

**Symptom**: at 5.6 in/panel × 13 rows, winner names struck through metric labels (worked at horizontal
6.8 in panels).
**Root cause**: per-row pitch dropped to ~0.30 in — below what two stacked 9–11pt text lines need.
**Fix**: 6.8 in/panel everywhere → pitch ≈ 0.4 in.
**Prevention**: keep per-row pitch ≥ ~0.4 in whenever a row carries annotations; recompute pitch whenever
figure height or row count changes.
