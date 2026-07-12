---
name: visual-mistakes-kb
description: Append-only KB of every figure/demo layout mistake ever made in this repo — the visual-audit agent re-reads this file on EVERY audit, so a logged mistake can never be repeated silently
type: project
---

# Visual mistakes KB (append-only)

> Format per entry: **Symptom** → **Root cause** → **Fix** → **Prevention**. The visual-audit agent
> (`.claude/agents/visual-audit.md`) turns every entry into a checklist item. NEVER delete or renumber.

## VM1 — winner-arm annotations struck through the metric labels (2026-07-12)

**Symptom**: per-row winner names rendered in the LEFT margin under each metric tick label overlapped/struck
through the labels in `forest_plot_*` and `scale_poc_vs_full_*`.
**Root cause**: side text placed in the margin that ALREADY belongs to y-tick labels; collision worsens at
compressed row pitch.
**Fix**: dedicated RIGHT-hand column — `annotate(xy=(1.05, y), xycoords=("axes fraction","data"))`, right
margin reserved in inches (`m13_eval_plot.py::plot_forest`).
**Prevention**: side text columns go in RESERVED margin space on the RHS, never the LHS.

## VM2 — suptitle wider than the plot → wasted flank space (2026-07-12)

**Symptom**: one/two-line suptitle spanned the full image width while the axes used ~50%, leaving dead
space beside every other element.
**Root cause**: raw f-string suptitle, no wrapping; figure width driven by the title.
**Fix**: `textwrap.fill(seg, width=68)` per newline-segment; top margin sized from wrapped line count
(0.26 in/line).
**Prevention**: wrap EVERY suptitle; title must be narrower than the figure.

## VM3 — multi-panel comparison rendered side-by-side landscape (2026-07-12)

**Symptom**: `forest_plot_best_ci.png` (2 backbones) was 2 panels side-by-side — cannot fit one column of a
2-column AAAI paper.
**Root cause**: `subplots(1, n)` default with a `vertical=` opt-in switch.
**Fix**: `plot_forest` always `subplots(n, 1)` portrait; the `vertical` parameter was deleted.
**Prevention**: comparison panels stack VERTICALLY, always — homogeneous across every figure destined for
the paper.

## VM4 — value label at an edge-hugging marker ran into the RHS column (2026-07-12)

**Symptom**: "33.2×" / "4.8×" value labels of the right-most markers touched the winner-arm column text.
**Root cause**: symlog xlim headroom (×3) sized for the marker, not for the marker + its offset text.
**Fix**: positive-side xlim headroom ×6 in `plot_forest`.
**Prevention**: reserve headroom on the side where offset text extends, sized for the TEXT end, not the marker.

## VM5 — compressed vertical panels made per-row annotations collide (2026-07-12)

**Symptom**: at 5.6 in/panel × 13 rows, winner names struck through metric labels (worked at horizontal
6.8 in panels).
**Root cause**: per-row pitch dropped to ~0.30 in — below what two stacked 9–11pt text lines need.
**Fix**: 6.8 in/panel everywhere → pitch ≈ 0.4 in.
**Prevention**: keep per-row pitch ≥ ~0.4 in whenever a row carries annotations; recompute pitch whenever
figure height or row count changes.

## VM7 — metric name in scene title contradicts the on-frame unit label (2026-07-12)

**Symptom**: demo Scene B title said "future-frame MSE" while the value panel on the same frame said
"mean latent L1 (lower = better)" (JSON key B_l1); the verdict card repeated "MSE".
**Root cause**: title text written from the storyboard, unit label written from the metric code — two
sources for one fact.
**Fix**: derive the displayed metric name from ONE constant shared by title, unit label, JSON key, and
verdict row.
**Prevention**: every number shown must carry exactly one name; grep the render script for the metric
string appearing more than once.

## VM8 — chart y-axis label drawn over the neighbouring image panel (2026-07-12)

**Symptom**: Scene C ylabel "prediction error (L1)" rendered on top of the predicted-frame heatmap
(white glyphs on light-yellow pixels — invisible).
**Root cause**: matplotlib places the ylabel OUTSIDE the axes bbox; the compositor packed the image
panel flush against the axes' left spine, so the label landed on the image.
**Fix**: reserve a gutter (~40 px) between an image panel and any chart whose ylabel extends left;
or use labelpad/inset label.
**Prevention**: when compositing images next to matplotlib axes, budget for text that extends beyond
the axes bbox, not just the bbox itself.

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

## VM10 — identical duplicate row/col labels for distinct entities (2026-07-12)

**Symptom**: Scene E matrix had two rows both labelled "drive delhi" and two both "walking goa" — a
stranger cannot tell the 4 clips apart, defeating the "same-motion PAIRS" premise.
**Root cause**: labels derived from the motion category only, dropping the clip ID.
**Fix**: append a distinguishing suffix (clip ID) to every entity label.
**Prevention**: axis labels must be unique per row/col; assert uniqueness in the plotting helper.

## VM11 — plotted element with no legend entry (dashed fit line) (2026-07-12)

**Symptom**: Scene D drew a grey dashed linear-fit line across the chart; the legend explained only the
solid data line, leaving the dashed line's meaning to guesswork.
**Root cause**: fit line added with label=None.
**Fix**: legend entry "linear fit (slope = the metric)" or inline end-of-line label.
**Prevention**: C6 applies to EVERY artist — count artists vs legend entries before saving.

## VM12 — outro/verdict references entities never introduced in the demo (2026-07-12)

**Symptom**: verdict footer said "surgery separates from the best competitor by 43.3x / 33.2x / 20.0x /
13.9x CI" — "surgery" and "CI" undefined in a single-model (FROZEN) demo, and the 4 ratios were not
attributed to the 4 metrics.
**Root cause**: paper-context sentence pasted into a self-contained artifact.
**Fix**: name the method explicitly and attribute each ratio to its metric on the card.
**Prevention**: C8 — every proper noun and number on a demo frame must be defined on-screen.

## VM13 — single-model rendering of a multi-model layout leaves a dead half-frame (2026-07-12)

**Symptom**: Scene E matrix sat in the left ~32% of the 1280 px frame with the right ~55% empty; the
verdict card left the right 43% and bottom 40% black.
**Root cause**: layout grid sized for N model columns, rendered with N=1 without recentering.
**Fix**: compute panel placement from the actual model count; center when N=1.
**Prevention**: after any roster change, re-check frames for dead flanks, not just overlap.

## VM6 — hardcoded arm name in a figure title (2026-07-12)

**Symptom**: every forest suptitle said "surgery_raw ∈ OURs" even in plots whose roster contains NO
surgery_raw (FULL = frozen/lora/diheavy only).
**Root cause**: blanket static title text describing a per-row fact.
**Fix**: per-row winner declaration (» exact arm name, registry colour); title only explains the notation.
**Prevention**: figure text must never hardcode arm names — derive per-row/per-panel facts from the data
being plotted.
