---
id: VM9
title: sequential dark cmap + black bold cell text = illegible endpoint cells (2026-07-12)
category: plot-authoring
keywords: [bold]
---

## VM9 — sequential dark cmap + black bold cell text = illegible endpoint cells (2026-07-12)

**Symptom**: Scene E cosine matrix used a Reds-style cmap; the diagonal "1.00" cells were near-black
maroon with black bold text — unreadable; and an all-red map reads "everything bad" for a metric where
high = good.
**Root cause**: plotting.md rule 1 (always black text) presumes a white-through diverging cmap; a
sequential dark-endpoint cmap silently breaks it.
**Fix**: desirability cmap red→white→blue with moderate endpoints (rule 4); drop leading zeros and size
values to fill ≥70% cell width (rule 2).
**Prevention**: black-bold-text rule and moderate-endpoint-cmap rule are a PAIR — never apply one
without the other.
