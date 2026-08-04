---
id: VM4
title: value label at an edge-hugging marker ran into the RHS column (2026-07-12)
category: latex-figure-placement
keywords: [column]
---

## VM4 — value label at an edge-hugging marker ran into the RHS column (2026-07-12)

**Symptom**: "33.2×" / "4.8×" value labels of the right-most markers touched the winner-arm column text.
**Root cause**: symlog xlim headroom (×3) sized for the marker, not for the marker + its offset text.
**Fix**: positive-side xlim headroom ×6 in `plot_forest`.
**Prevention**: reserve headroom on the side where offset text extends, sized for the TEXT end, not the marker.
