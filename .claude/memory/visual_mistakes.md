---
name: visual-mistakes-kb
description: Append-only KB of every figure/demo layout mistake ever made in this repo — the visual-audit agent re-reads this file on EVERY audit, so a logged mistake can never be repeated silently
type: project
---

# Visual mistakes KB (append-only)

## VM30 — a probe win on a TECHNICAL label ≠ a layman-verifiable demo (GT anti-correlates with perception) (2026-07-14)

**Symptom**: the m17 VQA demo asked "how much motion is in this clip?" and marked OURS right / FROZEN
wrong per the action-label magnitude (still/slow/medium/fast). On real clips it looked BACKWARDS: a calm
man fishing was labeled "fast" (OURS agreed → looked wrong to the eye); a busy market was labeled "still"
(OURS agreed → looked wrong). A layman would side with FROZEN's "wrong" answers.
**Root cause**: the magnitude label is a camera-subtracted RAFT foreground-flow quartile, which
ANTI-correlates with naive visible motion (Spearman −0.363, p=2.5e-6, confirms VM29). OURS predicting the
LABEL better (+8.2pp, real, leakage-safe) does NOT make the GROUND TRUTH eye-verifiable — a viewer cannot
confirm "fast" on a calm scene, so "OURS right" is unreadable/misleading.
**Fix**: before rendering ANY OURS-vs-FROZEN VQA/quiz demo, GATE the GROUND TRUTH itself — verify the
label correlates with what a layman can SEE (model-free perceptual check), not merely that OURS predicts
it. Only ship questions whose answer is OBSERVABLE in the clip (semantic future events: turn L/R, open
fridge, crash y/n).
**Prevention**: visual-audit C10 gains a GT-PERCEPTIBILITY sub-check — a demo's GROUND TRUTH must be
verifiable from the pixels by a non-expert; a latent/flow-derived label that anti-correlates with
perception FAILS even when the probe delta is real and significant. See [[feedback_no_hallucinated_victory]].

## VM29 — the eval metric wins are REAL but SUB-PERCEPTUAL: a naked-eye layman demo is not honestly achievable (2026-07-14)

**Symptom**: after ~7 render iterations + 3 demo modalities, no honest video lets a layman SEE OURS beat
FROZEN with labels removed. Evidence chain, each with a FAIL-LOUD gate:
  1. decoder pixels (causal-L1): fog + tiny +8.9% mean effect → invisible.
  2. retrieval by home-grown motion proxies (crude translation, dense flow-histogram): FROZEN won both
     → the proxies didn't match the metric's ground truth.
  3. retrieval by the eval's TRUE motion classes: OURS wins 0.387 vs 0.208 (2x) — REAL — BUT the classes
     are camera-subtracted RAFT foreground-flow signatures; my flow-arrow can't reproduce them.
  4. SPEED sub-axis: OURS wins 0.699 vs 0.361 — BUT model-free motion energy ANTI-correlates with the
     eval "speed" label (Spearman -0.374: "fast"-labelled clips have LOWER visible frame-change than
     "still"), because "speed" = camera-subtracted foreground flow, not visible motion.
**Root cause**: the metrics OURS wins (motion-cosine on RAFT-flow classes, latent-prediction L1) measure
TECHNICAL representation quantities that do not map onto untrained human perception. A large, overwhelmingly
significant feature-space win can be entirely sub-perceptual.
**Fix**: do NOT fake perceptual visibility. Present these wins as what they honestly are — rigorous forest
plots + real retrieval precision numbers (0.39 vs 0.21). A retrieval demo with OFFICIAL-label green/red is
honest as "OURS matches the ground-truth motion fingerprint 2x more often", but a layman reads the GREEN
COUNT, not the motion (strict blind test with colour masked would fail — that's the honest truth, not a
bug to paper over).
**Prevention**: before promising a "layman sees it" demo for a metric, first test whether the metric's
quantity even correlates with a model-free perceptual proxy. If it anti-/non-correlates, the win is
sub-perceptual and the honest deliverable is numbers/plots, not a naked-eye video.

**Update (downstream capabilities also tested, 2026-07-14)** — websearch surfaced two canonical
eye-visible V-JEPA apps; both fail for OURS:
  5. TRACKING / mask propagation (DAVIS-style feature correspondence): OURS is architecturally WEAKEST
     here (TCC cycle-back −31.6%, TCC τ −5.1% — OURS LOSES correspondence). Would favour FROZEN.
  6. ANOMALY / surprise via prediction error: per-slot pre-check OURS 0.041 vs FROZEN −0.110 (both ≈
     noise); proper SLIDING-WINDOW test on 10s clips FROZEN 0.283 > OURS 0.141 — FROZEN localizes
     genuine motion-surprises BETTER. Not an OURS win.
FINAL: across 6 modalities (decoder pixels, 3 retrieval variants, 2 anomaly variants) + the full metric
profile, there is NO honest layman-eye-visible task where OURS beats FROZEN. OURS' wins are real,
statistically overwhelming, and either sub-perceptual (motion-cosine fingerprint) or moderate probe
gains (6–27%) that don't transfer to perception; its only architectural strengths are motion/prediction,
and the eye-visible versions of those (perceptual retrieval, anomaly) are won by FROZEN. Honest
deliverable = the rigorous forest plots + retrieval-precision numbers, NOT a naked-eye demo.

## VM27 — audit agent FALSE-APPROVED a layman claim it pattern-matched from the captions (2026-07-14)

**Symptom**: the visual-audit agent returned "LAYMAN: APPROVED — a layman can spot OURS beats FROZEN"
on demo_W; the real human (user) saw NO difference. The agent's evidence was "OURS-squint pairs with
REAL-squint" — i.e. it read the demo's own captions/framing and confirmed them, and it upgraded a
"consistent lean" into an approval.
**Root cause**: the layman check let the agent (a) use the demo's captions as evidence, (b) look at
flattering frames, (c) approve a subtle/leaning difference, (d) not test whether the visible difference
was a DECODER COLOUR ARTIFACT (measured after: OURS-decoder R−B +51 vs FROZEN +43 vs real +55 — the
"warm vs blue" was ~8 pts, partly separate-decoder calibration, not a clean model-quality signal).
**Fix**: C8-LAYMAN blind pre-attentive protocol in `.claude/agents/visual-audit.md` — mask captions,
guess from raw pixels blind, adversarial frame pick, artifact-falsification, POP-OUT bar (any "lean" =
NOT APPROVED), and a humility clause ("you are not a human eye; a false APPROVE is worse than a FAIL").
**Prevention**: an automated audit is a PRE-filter, never a substitute for the human's eye on a
"can-a-layman-see-it" claim; when the signal is subtle enough to argue about, it fails by definition.

## VM28 — forced a LATENT-space metric into a PIXEL demo it cannot show (2026-07-14)

**Symptom**: 5 render iterations to make causal-future-block-L1 (a latent-L1 number) layman-visible via
a trained pixel decoder — all fog; the honest gap lives in abstract features, and a small decoder
smears it away. Also chose LOW-MOTION walking clips where future ≈ past, so there was barely any
prediction signal to begin with.
**Root cause**: matching the wrong DEMO MODALITY to the metric. Pixel-decode of predictor latents is
V-JEPA's OFF-LABEL visualization (Meta: "decoder for visualization purposes, not used in training");
the model's eye-visible strength is MOTION UNDERSTANDING (SSv2 77.3%), shown by a probe readout on the
REAL video, no decoder.
**Fix**: pick the demo modality from where the signal is legible — motion/action probe overlay on the
real clip for motion metrics; forest plots / numbers for latent-only metrics; and HIGH-MOTION clips
(future ≠ past) whenever "predict the future" is the framing.
**Prevention**: before building a demo, ask "is this metric's signal legible in the medium I'm about to
render?" — if it's a latent number smeared by a lossy decoder, the medium is wrong, not the polish.

## VM26 — title-card legend line orphaned by a design swap of the element it describes (2026-07-14)

**Symptom**: demo_W v4 title card still said "red box = where the models disagree most (zoomed at the
bottom)" after the ROI-zoom row was replaced by the full-frame SQUINT row; the legend taught a FALSE
reading of the strongest evidence row.
**Root cause**: the title card's legend is hand-written prose, not derived from the renderer's current
element roster; swapping a visual component (VM25 fix) did not touch the intro text that defined it.
**Fix**: rewrite the legend from the CURRENT elements ("red-bordered tiles at the bottom = the same
heavy blur of all three videos — match FROZEN / OURS to the REAL one").
**Prevention**: after ANY change to a demo's visual element roster, re-audit the title/intro card line
by line against the elements that actually render (VM19 extended to legend/intro cards).

## VM25 — crop-scale evidence is unusable when generative output is foggy; compare at GESTALT scale (2026-07-14)

**Symptom**: two ROI strategies in a row (max pixel-gap VM24, then structure-weighted) produced zoom
insets where BOTH models show featureless waffle — at 112px crop scale every predictor-latent decode
is fog, so the inset row kept undermining a demo whose WIDE panels were decisive (red canopy vs blue).
**Root cause**: predictor uncertainty destroys local detail; the model gap lives in the large-scale
colour/layout gestalt, not in any crop.
**Fix**: replace the zoom-inset row with a SQUINT TEST — identical aggressive downsample (NxN blocks)
of FROZEN-imagined / OURS-imagined / REAL: hue-layout survives squinting, fog cannot fake it; the two
"sibling" mosaics are obvious.
**Prevention**: match the comparator's SCALE to where the signal lives — crops for sharp outputs,
gestalt/squint for blurry ones; never ship a crop comparator without eyeballing that the winner wins
IN THE CROP.

## VM24 — numerically-decisive ROI is visually MISLEADING when fog hue coincides with the object (2026-07-14)

**Symptom**: the zoom inset picked a blue dustbin region where OURS was numerically closer, but
FROZEN's blue fog LOOKED like the blue bin — a layman reading the inset would pick FROZEN. The inset
inverted the demo's verdict while the wide panels (red canopy vs blue fog) were decisive.
**Root cause**: ROI scored by per-tile mean pixel gap only; flat single-hue objects can be "matched"
by a same-hue fog with zero structure.
**Fix**: weight the tile gap by the REAL tile's structure (per-tile std) so the ROI lands on a salient
patterned object (e.g. the red canopy), where fog cannot fake it; enlarge the ROI for context.
**Prevention**: any auto-selected "evidence crop" must be eyeballed for the layman reading; a crop
where the LOSER's rendering resembles the target by coincidence is an automatic re-select.

## VM21 — verdict/summary rows ship placeholder glyphs instead of values (2026-07-14)

**Symptom**: demo_W verdict card rendered its 4 result rows as "walking / varanasi · FROZEN 2.1  —"
(em-dash where the number belongs); demo_metrics.json carried the real C_l1 values and a NaN
motion_cos_margin that the formatter silently swallowed.
**Root cause**: the row formatter fell back to a placeholder on missing scene keys instead of raising;
the render "succeeded" so nothing flagged the empty verdict.
**Fix**: raise on NaN/missing verdict values; render the values that exist plus a per-row winner mark.
**Prevention**: any "—", "N/A", "?" or blank in a value slot of a sign-off artifact is an automatic FAIL.

## VM22 — hero clip chosen without checking the visual evidence is decisive (2026-07-14)

**Symptom**: demo_W hero clip 2's entire hidden half showed featureless fog in BOTH imagined panels;
a layman could not pick a winner, and the clip suggested OURS also produces fog.
**Root cause**: hero clips/ROIs selected by a LATENT-space numeric criterion with no check that the
winning model's RENDERING contains recognizable structure.
**Fix**: select heroes + the zoom ROI by PIXEL-space decisiveness — |imagined_FROZEN − real| −
|imagined_OURS − real| — and FAIL LOUD if no clip has a positive gap.
**Prevention**: every hero segment must independently pass the layman test before locking the roster.

## VM23 — fog-vs-fog: a colour-temperature difference is NOT a layman-visible win (2026-07-14)

**Symptom**: user (demo4.png): "I don't see any layman-spottable difference" — FROZEN imagined a BLUE
fog, OURS a WARM fog; the palette difference reads as a colour shift, not a better prediction.
**Root cause**: decoded PREDICTOR latents average away detail; side-by-side fog panels only differ in
hue, and hue is dismissible.
**Fix (layman comparators, honest)**: (1) CONTINUITY CUT — the imagined panels PLAY THE REAL past at
full brightness, then cut into their own imagination: FROZEN's cut visibly "breaks" the video (blue
jump), OURS' continues it; (2) pixel-decisive ROI zoom (VM22) so the inset shows FROZEN dissolving an
object OURS keeps; (3) identical mild contrast stretch on BOTH imagined streams (presentation-only,
same op both sides).
**Prevention**: for generative-ish comparisons, design for PRE-ATTENTIVE judgments (continuity breaks,
object persistence) — never rely on hue or texture quality of blurry outputs.

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

## VM19 — mask caption describes a render style the renderer does not produce (2026-07-14)

**Symptom**: demo Scene C panel 2 captioned "model input (black = hidden)" while the hidden half renders
as clearly legible DIMMED footage (~15-30% brightness); a stranger concludes the model still sees a
darker copy of the future.
**Root cause**: caption written from the mask spec, pixels written by a ghost-dim visualizer — two
sources for one fact (the VM7 pattern applied to a render style).
**Fix**: caption describes the actual render ("dim = hidden from the model") or the render matches the
caption (true black).
**Prevention**: every caption naming a visual encoding (black/dimmed/hatched/blurred) must be checked
against the rendered pixels of the frame that carries it.

## VM20 — video frame layout leaves a dead horizontal band between content strip and pinned footer (2026-07-14)

**Symptom**: demo Scene C frames: panel strip ended y≈400, footer pinned at y≈698 of 720 px — ~37% of
every frame empty black while the evidence panels were only ~255 px; verdict card had a ~45% dead band.
**Root cause**: panels sized to a fixed edge, footer anchored to the bottom; nothing sizes content to the
frame height (VM13's dead-flank failure, vertical axis).
**Fix**: move the tickers into the dead band under the panels (bigger panels + no side column), center
text blocks vertically.
**Prevention**: after composing any video frame, check all four flanks AND interior bands for dead space;
content should claim ≥ ~80% of frame height.

## VM15 — CI whisker strikes the value label; a struck minus reads as PLUS (2026-07-13)

**Symptom**: forest value labels placed at a fixed dx right of the marker are struck through by the CI
whisker line whenever the CI is wide; worst case the whisker right-cap crosses the minus sign so
"-2.6x" READS "+2.6x" — a visually flipped sign.
**Root cause**: label y = marker y exactly, on the same horizontal line the whisker occupies; caps are
vertical strokes that compose with "-" into "+".
**Fix**: offset value labels vertically above the whisker (va="bottom", +6pt).
**Prevention**: never place text at the exact y of an errorbar; audit every negative label for a
readable minus sign.

## VM16 — in-axes annotation overflows and is struck by the axes spine (2026-07-13)

**Symptom**: the "1xCI - separated ->" annotation anchored at the dashed x=1 line extends PAST the axes
right spine; the "139.1x" value label of an edge-hugging marker crosses the spine into the winner column.
**Root cause**: left-anchored in-axes text sized without checking its rendered end against the axes bbox.
**Fix**: right-align in-axes annotations (ha="right" at axes fraction ~0.99); flip edge-hugging value
labels to the LEFT of the marker.
**Prevention**: any text anchored inside the axes must have its far end verified inside the axes bbox.

## VM17 — same-titled sibling panels computed on DIFFERENT arm rosters (2026-07-13)

**Symptom**: two figures titled identically "ViT-g . 1B . POC 10k (n_test = 1,825)" disagree (4.8x winner
surgery_raw vs 3.3x winner surgical_3stage_DI_diheavy) because one silently restricts to the shared-arm
roster; the restriction lived only in suptitle small print.
**Root cause**: the entity-title builder carries backbone/size/scale/n_test but NOT the arm roster, which
also changes the best-competitor baseline.
**Fix**: builder gains a roster component ("· shared-arm roster (3)") emitted whenever restricted.
**Prevention**: an entity title must pin EVERY fact that changes the numbers; C9 compares values, not
just title strings, for identically titled panels.

## VM18 — blanket suptitle claim false in one instantiation (2026-07-13)

**Symptom**: scale_poc_vs_full_vjepa_2_1_vitG suptitle said "FULL panel matches forest_plot_best_ci"
while its FULL panel is an "N/A — no eval yet" placeholder (claim true only for the vitg sibling).
**Root cause**: static suptitle template shared across figures; the claim is a per-figure fact.
**Fix**: emit the claim segment only when the FULL-side data exists.
**Prevention**: every factual clause in a templated title must be gated on the data that makes it true.

## VM14 — same entity titled DIFFERENTLY across sibling figures (2026-07-13)

**Symptom**: `forest_plot_best_ci.png` panel said "ViT-g · 1B" while `scale_poc_vs_full_vjepa_2_1_vitg.png`
said "ViT-g · POC 10k (n_test = 1,825)" — the same backbone/corpus, two different titles; the reader can't
tell they are the same thing. Correct: "ViT-g · 1B · POC 10k (n_test = 1,825)" everywhere.
**Root cause**: two label builders — `cross_backbone_report` composed `tag · size`, `scale_forest_report`
composed `tag · scale (n_test)` — each carrying HALF the facts. The visual-audit agent missed it because
(a) it was only handed one artifact set, (b) checklist C1-C8 were all SINGLE-image checks with no
cross-figure consistency item.
**Fix**: ONE `_xb_panel_label(tag, size, mtag, n_test)` builder used by every forest; agent gained C9
(cross-artifact consistency: Glob sibling figures, compare entity titles/units).
**Prevention**: any string identifying an entity (backbone, arm, corpus, n) is built by ONE function;
the audit always samples SIBLING figures from the same output dir, not just the artifact under review.

## VM6 — hardcoded arm name in a figure title (2026-07-12)

**Symptom**: every forest suptitle said "surgery_raw ∈ OURs" even in plots whose roster contains NO
surgery_raw (FULL = frozen/lora/diheavy only).
**Root cause**: blanket static title text describing a per-row fact.
**Fix**: per-row winner declaration (» exact arm name, registry colour); title only explains the notation.
**Prevention**: figure text must never hardcode arm names — derive per-row/per-panel facts from the data
being plotted.

## VM31 — color-emoji tofu in PIL/DejaVu video renders
**Symptom**: demo_mcq answer panel showed empty □ boxes before "FROZEN"/after "LOOKS" — `🧊 🥇 ❌ ✅`
rendered as missing-glyph tofu in the PIL `ImageDraw.text` frames.
**Root cause**: DejaVu Sans (the render font) has NO color-emoji glyphs (U+274C ❌, U+2705 ✅, 🧊, 🥇).
GFM/terminal emojis do NOT transfer to a rasterized PIL video.
**Fix**: use plain dingbat glyphs that ARE in DejaVu — ✓ U+2713, ✗ U+2717, → U+2192, · U+00B7, ● U+25CF —
plus colour (red/green) as the primary right/wrong signal.
**Prevention**: before putting any glyph in a PIL frame, check the font cmap
(`fontTools.ttLib.TTFont(fp)["cmap"].getBestCmap()` → `codepoint in cmap`). Emojis are for chat/markdown, NOT
DejaVu video frames. This is the render-side twin of [[feedback_ascii_table_emojis]] (variation-selector width).

## VM32 — overlay/text detector misses text baked over MOVING footage → human-eyeball gate is mandatory
**Symptom**: a temporal-static-edge overlay detector caught graphic overlays (floating-dollars, captions,
watermarks, YouTube icons) but MISSED "8.55% Muslims" burned over a moving crowd (scored 0.000, same as clean)
and false-positived on a static night monument (0.203, no text).
**Root cause**: the "static edges = overlay" heuristic only fires when the overlay is static AND the scene
moves; text over a moving crowd isn't distinguishable, and a static scene makes everything look like overlay.
**Fix**: use the detector as a PRE-FILTER only; the real gate is a human contact-sheet review of the FINAL
selected clips (mid-frame, large) before rendering — I eyeballed all 20 clips and hand-picked the clean 4.
**Prevention**: for "no baked-in text/political content" guarantees, never trust an automatic score alone —
always view the final shipped clips at size. Political/sensitive frames must never sit in an output file.

## VM33 — full-width figure* caterpillar rendered near-SQUARE → y-tick arm NAMES fall below the 9pt print floor (2026-07-26)

**Symptom**: `eval_scorecard_winners` (Fig 7, `figure*` at `width=\textwidth`) placed its 2x2 per-arm
caterpillar in a near-square canvas (content 6.65in x 6.44in, aspect 0.97). With ~18 arms per panel the
y-axis encoder names (`surgical_3stage_DI_intervene`, `peft_lora`, `frozen`, ...) rendered at only
~7.2-7.6pt effective in the compiled PDF with a ~1px (0.007in) inter-line gap — the SMALLEST text on the
page (below the tick values ~12pt, panel titles ~12pt, suptitle ~9.3pt) and below the plotting.md C5 >=9pt
floor. Readable, but exactly the element the rebuild existed to make comfortable, left sub-contract.
**Root cause**: font size scales with `width=\textwidth`, but a many-row caterpillar's name legibility is
set by the figure's HEIGHT (row pitch), not its width. A near-square aspect under-provisions vertical space,
so the per-row pitch (hence the y-tick font) collapses even though the width is fine. VM5 covers annotation
collision at low pitch; this is the print-SIZE floor of the y-tick names themselves.
**Fix**: for a full-width figure*, size the HEIGHT from n_rows x target_pitch where target_pitch admits a
>=9pt bold line box plus a real gap (~0.18in/row); a 2x18-row scorecard wants ~8.3in tall (aspect ~1.25),
i.e. a genuinely tall full-page figure*. Bump the y-tick label fontsize ~x1.25 AND grow figsize height the
same factor so lines don't touch.
**Prevention**: audit any \textwidth caterpillar/forest by measuring the y-tick name effective pt at the
PLACED width (png_content_px / placed_inches -> pt); if <9pt or inter-line gap <~2px @150dpi, FAIL and give
the height bump. Near-square aspect + many rows is the tell.

## VM34 — a "readability" fix that DROPS / REFORMATS / GROUPS-AWAY spec'd content is a spec VIOLATION, not a fix (2026-07-27)

**Symptom**: Fig 7 was specified as a 2×2 of the top-4 metric **BAR** panels showing **ALL ~20 FT rivals by
name** (from `outputs/poc/probe_plot/metrics_watch/eval_scorecard_combined.png`), readable. To satisfy
"readable", the main agent silently (a) dropped 4 metrics → 2, (b) changed vertical bars → a per-arm dot±CI
"caterpillar", and (c) [Fig 8] collapsed every named rival into "FactorJEPA (green) vs competitors (grey)",
deleting the arm names/legend that `scale_replication.png` carries. The visual-audit PASSED all of it — C1-C10
only check RENDER quality of the artifact in isolation, never whether it is the figure the user asked for.
User (furious): *"how did you dare to remove it without my permission... do you want me rejected at AAAI?"*.
**Root cause**: treating "make it readable" as license to change WHAT is shown, not just HOW. Readability is a
HOW-constraint; the metrics / named entities / plot-type / panel-count are the WHAT-spec and are inviolable.
Compounded by the audit having no spec-fidelity gate, so a well-rendered WRONG figure passed clean.
**Fix**: restore the exact spec — 4-metric BAR 2×2 with every rival NAMED (Fig 7); causal scatter with every
rival's own colour + naming legend (Fig 8). When spec+readability truly conflict, STOP and ask a
spec-PRESERVING question (horizontal bars? full-page figure? smaller font?), NEVER a menu that drops content.
Added audit check **C0 SPEC-FIDELITY** (outranks C1-C10): the auditor must be given the user's spec + cited
reference and FAIL any dropped/reformatted/substituted/grouped-away content, no matter how clean the render.
**Prevention**: before "optimizing" any figure, LIST what the user explicitly named (metrics, rivals, plot
type, reference image); if the fix removes ANY of it, it is wrong — solve the harder problem or ask. A
visual-audit PASS on a figure that isn't the spec'd one is a false PASS. See [[feedback_never_override_user_spec]].

## VM35 — non-BOLD value/tick numbers + wasteful WHITE space slipped through because the audit had no bold-check or space-check (2026-07-27)

**Symptom**: `eval_scorecard_winbars` (Fig 7) shipped with the per-bar VALUE labels (`0.496`…) and the
x-axis TICK NUMBERS rendered at REGULAR weight while the titles / y-codes / key were bold — the figure
"looked" bold but every number on it was thin. Separately it carried ~86% white with a 5.3% dead band
between the stacked panel rows and `xlim = hix + pad×8` right-margins (bars sat left, right third empty).
The user caught both; the visual-audit had NOT been run on it AND — worse — its checklist had no explicit
"every text bold" check and no white-space measurement, so it would have PASSED both defects even if run.
**Root cause**: (a) matplotlib text calls default to regular weight; `ax.annotate(...)`, and tick labels
styled only via `tick_params`, need an explicit `fontweight="bold"` / `set_fontweight("bold")` — bolding the
title alone leaves the numbers thin. (b) an over-tall figure + large `hspace` + large `xlim` right padding
create dead white that a glance rationalizes as "clean". (c) the audit checked legibility PT-size but never
weight, and never quantified blank space.
**Fix**: set `fontweight="bold"` on EVERY text emitter — value labels (`annotate`), x/y tick labels (loop
`set_fontweight("bold")`), legend (`prop={"weight":"bold"}` + `leg.get_title().set_fontweight("bold")`),
titles/suptitle. Tighten white: cut `xlim` right pad (×8→×4), reduce `hspace`, and size the figure to the
content (measure white% + empty bands with PIL, target tight-not-touching). Added audit checks **C11
BOLD-TEXT** (grep the generator: every text call must carry a bold weight; a single regular-weight text =
FAIL) and **C12 SPACE-BALANCE** (PIL-measure blank %, FAIL any >3% empty band OR cramped gaps).
**Prevention**: before shipping any figure, (1) `grep` the generator for `annotate|set_title|suptitle|
set_xticklabels|set_yticklabels|legend|bar_label|ax.text` and confirm each has a bold weight; (2) PIL-measure
the white fraction + biggest empty bands and tighten. "Bold title" ≠ "bold figure". See [[feedback_never_override_user_spec]].

## VM36 — figsize HEIGHT shrunk but suptitle y / top margin not recomputed -> suptitle 2nd line overlaps the panel titles (2026-07-27)

**Symptom**: `eval_scorecard_winbars` (Fig 7) height was cut 9.6->8.6->7.8in to save paper space; the suptitle kept
`y=0.984` fontsize 11.5 (2 lines) and `subplots_adjust(top=0.905)` with 2-line panel titles (`pad=5`). At 7.8in the
header stack (2 suptitle lines + 2 panel-title lines) no longer fits the ~0.62in above the axes, so the suptitle's 2nd
line "(POC 10k . n_test = 1,825 . 95% BCa CI)" OVERLAPPED both top panel titles by ~16px across ~97 columns
("n_test = 1,825" struck "future-frame L1"; "95% BCa CI)" sat on "causal future-block L1"). Measured column-wise:
min clearance = -16px, 97 columns with <=1px gap.
**Root cause**: suptitle `y` and the `top` subplots margin are absolute FIGURE-FRACTION constants; when figure HEIGHT
drops, the same fraction is fewer inches, so a header stack that fit at 8.6in collides at 7.8in. The height edit
recomputed nothing above the axes (same class of bug as VM33 but on the vertical header budget, not the y-tick pt).
**Fix**: after ANY figsize height change, re-derive the top headroom in INCHES from the header line count (plotting.md
rule 3 / VM2: ~0.26in/line) -> here grow height back to ~8.3in and/or lower `top` to ~0.87 so the 2-line suptitle
clears the 2-line panel titles; or collapse suptitle/titles to fewer lines. The paired shrink also left the bottom
rotated x-ticks only 19px above the legend box (tight-but-clear this round -> next shrink will collide there too).
**Prevention**: audit BOTH the suptitle<->panel-title band AND the bottom-tick<->legend band after every figsize edit
by measuring per-column clearance (negative = overlap). Fixed figure-fraction margins do NOT survive a height edit;
the header/footer budget must be recomputed from the line counts each time.

## VM37 — a C5 font-bump that ignores the AUTHORING-width vs PLACEMENT-width downscale under-delivers (still sub-9pt) (2026-07-27)

**Symptom**: `eval_scorecard_winbars` (Fig 7, `figure*` at `width=\textwidth` ~= 7.0in) was authored at `figsize=(7.8, 8.2)`.
A prior C5 fix "bumped fonts" (y-codes 8.5->9, value labels 7->8.5, x-ticks 7.5->8.5, legend 7.5->8, panel titles 9.5->10)
believing that cleared the >=9pt floor. But LaTeX scales the 7.8in-wide artifact DOWN to the 7.0in column, a
`7.0/7.84 = 0.893x` shrink that drops EVERY glyph ~11% below its authored pt. Net effective sizes at placement:
value labels ~7.6pt, x-tick numbers ~7.6pt, y-codes ~8.0pt, legend entries+title ~7.1pt, panel titles ~8.9pt -- ALL
below the ~9pt floor; only the suptitle (~9.8pt) cleared. The bump raised the authored numbers but the downscale ate
the gain, so the SECOND C5 pass still failed. (Validated: measured suptitle cap-span 9.99pt vs 9.82 predicted,
panel-title 9.11 vs 8.93, legend-title 7.35 vs 7.14 -- the `effective = authored x 0.893` model is exact.)
**Root cause**: effective print pt is `authored_pt x (saved_content_inches / placed_inches)`, NOT `authored_pt`. When the
matplotlib `figsize` width (7.8) is WIDER than the `\includegraphics` width (`\textwidth` ~= 7.0), there is a hidden
sub-1.0 scale factor that no amount of in-figure font tuning reveals unless you convert to the PLACED width. Bumping
fonts while measuring at the authoring width (or eyeballing the PNG) hides it.
**Fix**: author the figure AT its placement width -- set `figsize` width == the `\includegraphics` width (7.8->7.0, keep
aspect so height 8.2->~7.6) so 1px maps 1:1 and `effective_pt == authored_pt`; THEN raise the sub-floor classes to >=9pt
(value labels 8.5->9, x-ticks 8.5->9, legend size 8->9 + title 8->9; y-codes/panel-titles already >=9 at 1:1). Re-audit
the bottom-tick<->legend band (was only 20px) and the 2-col legend full-names after the change -- narrower width + bigger
fonts tighten both; grow height (7.6->~8.0) if either collides.
**Prevention**: C5 must ALWAYS convert to the intended PLACED width, never the authoring width: read the `.tex`
`\includegraphics[width=...]` + the figure env (`figure` vs `figure*`) to get the real placed inches, compute
`effective_pt = authored_pt x placed_in / saved_content_in`, and FAIL if the smallest class < ~9pt EVEN IF a prior
"font bump" was applied. The tell: matplotlib `figsize` width != the LaTeX include width. Sibling of VM33 (there the
height under-provisioned the y-tick pt; here the width-mismatch downscale does).
