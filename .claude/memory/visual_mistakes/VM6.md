---
id: VM6
title: hardcoded arm name in a figure title (2026-07-12)
category: plot-authoring
keywords: [suptitle]
---

## VM6 — hardcoded arm name in a figure title (2026-07-12)

**Symptom**: every forest suptitle said "surgery_raw ∈ OURs" even in plots whose roster contains NO
surgery_raw (FULL = frozen/lora/diheavy only).
**Root cause**: blanket static title text describing a per-row fact.
**Fix**: per-row winner declaration (» exact arm name, registry colour); title only explains the notation.
**Prevention**: figure text must never hardcode arm names — derive per-row/per-panel facts from the data
being plotted.
