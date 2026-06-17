# Plot guidance (analysis-phase figures)

Rules for **analysis-phase** figures (eyeballed in the terminal / a research meeting / a projector).
Paper-phase figures may revert to subtler styling — see `feedback_analysis_phase_vs_paper_phase_styling`
in memory. When in doubt for an analysis figure, optimise for **"readable across the room"**, not pretty.

## Cell-annotated figures (heatmaps, scorecards, confusion/correlation matrices, ax.table)

1. **ALL text BLACK + BOLD. Never white-on-dark, never grey.**
   - White text on a saturated cell vanishes when the figure is printed B&W or thrown on a projector.
   - Set `color="black", fontweight="bold"` on every `ax.text`, tick label, axis label, title, colorbar label.
   - Do **not** branch the text colour on the cell value (`color="white" if abs(v)>0.6 else "black"` is BANNED).
2. **Numbers must fill the cell — target ≥ ~70% of cell width.**
   - Size the font from the cell size, not a tiny fixed point size: `cell_pt = cell_inches*72`; a value
     font of `≈0.32*cell_pt` makes a 4-char value (`-.91`) fill ~70% of the cell width.
   - **Drop the leading zero** so glyphs are bigger: `0.91 → .91`, `-0.91 → -.91`. Fewer chars → larger font.
   - Make cells big enough to begin with (`≥ ~0.8 in/cell`); a 14×14 grid → a ~16 in figure. Disk is cheap.
   - Width is the binding constraint for multi-char numbers (you can't also fill 70% of *height* with
     4 chars without overflow) — fill the width; height follows.
3. **Tick + family/group labels:** black, bold, sized from the cells (`≈0.22*cell_pt`), not 7 pt.
4. **Diverging quantity** symmetric about 0 (`vmin=-1, vmax=1`); outline logical blocks with a thick **black**
   `Rectangle`. Pick the cmap by what the COLOUR should mean:
   - **Sign convention** (raw correlation, no good/bad direction): `RdBu_r` — red=+, blue=−. Colour ≠ desirability.
   - **Desirability convention** (high IS the goal, e.g. a validity matrix that SHOULD be high): **blue = +1
     (best), white = 0, red = −1 (worst)** via `LinearSegmentedColormap.from_list([red, white, blue])` with
     MODERATE endpoints (so the bold-black numbers stay legible; builtin `RdBu` ends too dark). You're
     overriding the textbook sign-convention → **say so in the colorbar label.**
   - ♿ **Use blue↔red (or blue↔orange), NEVER green↔red** — green/red is the worst pair for colour-blind
     viewers (~8% of men); blue/red via white is colour-blind-safe.
   - A convergent/discriminant validity matrix is a **Campbell–Fiske MTMM** figure (within-family ρ ≫ between);
     cite it. Ref impl: `plot_metric_validity` in `src/m13_eval_plot.py`.
5. **No missing glyphs.** DejaVu Serif lacks `≫` (U+226B), `≤`, etc. → use ASCII (`>>`, `<=`) in figure text.
   (`ρ`, `↑`, `Δ`, `–` are fine.) A missing glyph renders as a tofu box.
6. Save **both** `.png` and `.pdf` (`save_fig` does both); raster dpi from `pipeline.yaml plots.dpi`.

## General

- Tunables (permutation counts, seeds, dpi) live in `configs/pipeline.yaml`, never hardcoded — read via
  `get_pipeline_config()` (CLAUDE.md "No hardcoded values in Python").
- Reference implementation that follows this: `src/m13_eval_plot.py::plot_metric_validity`.
