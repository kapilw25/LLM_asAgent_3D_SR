#!/usr/bin/env python3
"""Compile ppt_july05.md into ppt_july05.pdf (16:9 slides) — VECTOR edition.

Unlike make_ppt_june{07,14}_pdf.py (matplotlib + mpimg.imread, which RASTERIZES → blurs on
zoom), this embeds the source .pdf plots as TRUE VECTOR via PyMuPDF `show_pdf_page()`, so every
figure stays razor-sharp at any zoom. Text/titles/bullets are drawn with the base-14 PDF fonts.

The 5 figures are the iter18 v5_1B CROSS-BACKBONE report (m13 --cross-plots): three forest plots
+ scale-replication + the combined 2B/1B scorecard — all from the two per-backbone eval_metrics
(Task 1 data sources on slide 0b). Content matches ppt_july05.md verbatim.

USAGE:  cd iter/utils/ppt && python make_ppt_july05_pdf.py
"""
from pathlib import Path

import fitz  # PyMuPDF — the ONLY engine that embeds vector PDF pages into a new PDF

HERE = Path(__file__).resolve().parent
MW = (HERE.parent.parent / "iter18_ablations_FTtechniues" / "result_outputs"
      / "v5_1B" / "poc" / "probe_plot" / "metrics_watch")
OUT = HERE / "ppt_july05.pdf"

PW, PH = 960.0, 540.0        # 16:9 slide @ 72 pt/in  (= 13.33 x 7.5 in)
INK, SLATE, GREEN = (0.12, 0.12, 0.12), (0.22, 0.28, 0.31), (0.18, 0.49, 0.20)
FONT, FONTB, MONO = "helv", "hebo", "cour"

# ── Task 1: the data source behind every figure (m13 cross_backbone_report reads these) ──
_SOURCES = [  # figure | what it compares | source
    ("forest_plot_best_ci / _mean", "OURS vs BEST competitor (stat + magnitude)", "both eval_metrics.json  (plot_forest, vs=best)"),
    ("forest_plot_frozen_ci", "OURS vs FROZEN  (the paper claim)", "both eval_metrics.json  (plot_forest, vs=frozen)"),
    ("scale_replication", "1B rank vs 2B rank (Spearman rho)", "both eval_metrics.json  (plot_scale_replication)"),
    ("eval_scorecard_combined", "2B champion stacked over 1B", "the two per-backbone eval_scorecard.pdf"),
]
_SRC_PATHS = [
    "2B  v5_1B/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.{csv,json}",
    "1B  v5_1B/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.{csv,json}",
]

# ── image slides: (title, source .pdf, [bullet, bullet]) — SAME ORDER the user listed them ──
IMG_SLIDES = [
    ("1 · Surgery vs the STRONGEST competitor - statistical (CI-widths)",
     MW / "forest_plot_best_ci.pdf",
     ["Hardest bar (vs the best NON-ours arm, not frozen): future-L1 6.3x/4.8x and causal-L1 2.3x/2.7x CI-widths clear significance at 2B/1B.",
      "Motion-cosine sits far left (-14.5x / -9.2x) -- surgery cedes semantic separation to full-FT; reported, not hidden."]),
    ("2 · Surgery vs the strongest competitor - effect size (raw %)",
     MW / "forest_plot_best_mean.pdf",
     ["Magnitude view: the predictive lead is small-but-consistent -- future-L1 +1.7%/+1.4%, mask-ratio +1.1%/+2.9%, teacher-free +3.8% (1B).",
      "The motion-cosine deficit is large (-44.9% / -33.4%) -- a deliberate trade: surgery optimizes prediction, full-FT optimizes semantics."]),
    ("3 · THE PAPER CLAIM - surgery vs FROZEN (CI-widths)",
     MW / "forest_plot_frozen_ci.pdf",
     ["Decisive: surgery separates from frozen by 5-47 CI-widths on EVERY predictive + motion metric (future-L1 47x/27x, motion-cos 27x/26x) at BOTH scales.",
      "The cost is coherence -- surgery regresses on TCC-tau / TCC-cycle (grey/negative), the frame-timing metrics frozen still owns; a stated trade."]),
    ("4 · Scale transfer - does the 1B rank like the 2B? (Spearman rho)",
     MW / "scale_replication.pdf",
     ["Core metrics replicate 2B->1B: causal-L1 rho=0.978, motion-cos 0.952, future-L1 0.938, mask-ratio 0.895 -> the cheap 1B is a faithful proxy for the 2B.",
      "12/15 metrics replicate (rho>0.2); 3 secondary ones fail (rollout -0.60, teacher-free -0.25, temporal-order -0.11) -- flagged, not over-claimed."]),
    ("5 · Full appendix scorecard - 2B champion + 1B, all 15 metrics",
     MW / "eval_scorecard_combined.pdf",
     ["Every arm x 15 metrics at both scales (n=1,825, 95% BCa CI): the same three-behaviour split -- surgery wins prediction, full-FT semantics, frozen coherence.",
      "The 1B (bottom) is a scaled-down twin of the 2B champion (top) -- the bar orderings mirror, which is exactly what scale_replication quantifies."]),
]


def _center(page, y, text, size, font=FONTB, color=INK):
    w = fitz.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text(((PW - w) / 2, y), text, fontname=font, fontsize=size, color=color)


def _fit(src_rect, box):
    """Aspect-preserving fit of src_rect into box (fitz.Rect), centered — no distortion."""
    sa, ba = src_rect.width / src_rect.height, box.width / box.height
    if sa >= ba:
        w, h = box.width, box.width / sa
    else:
        h, w = box.height, box.height * sa
    x0 = box.x0 + (box.width - w) / 2
    y0 = box.y0 + (box.height - h) / 2
    return fitz.Rect(x0, y0, x0 + w, y0 + h)


doc = fitz.open()

# ── slide 0: title ──
pg = doc.new_page(width=PW, height=PH)
_center(pg, 250, "iter18 -> iter19  ·  July-05 - 1B replication + scale transfer", 22)
_center(pg, 288, "Surgery >> frozen HOLDS on the 1B (ViT-g)   AND   replicates 2B -> 1B", 14, FONT, SLATE)
_center(pg, 322, "=> greenlights the iter19 full-scale (116k) run on the cheaper 1B backbone   ·   n=1,825   ·   95% BCa CI",
        11, FONT, GREEN)

# ── slide 0b: Task 1 — data sources ──
pg = doc.new_page(width=PW, height=PH)
_center(pg, 48, "0 · Data sources (Task 1) - every figure -> its eval_metrics.{csv,json}", 15)
cx = (44, 320, 560)
for j, h in enumerate(("figure", "what it compares", "source (m13 --cross-plots)")):
    pg.insert_text((cx[j], 100), h, fontname=FONTB, fontsize=10.5, color=INK)
pg.draw_line((40, 108), (PW - 40, 108), color=(0.6, 0.6, 0.6), width=0.8)
for i, row in enumerate(_SOURCES):
    y = 132 + i * 30
    for j, val in enumerate(row):
        pg.insert_text((cx[j], y), val, fontname=FONT, fontsize=9.0, color=INK)
pg.insert_text((44, 300), "canonical per-backbone metrics (under iter18_.../result_outputs/):",
               fontname=FONTB, fontsize=10, color=SLATE)
for i, s in enumerate(_SRC_PATHS):
    pg.insert_text((52, 326 + i * 22), s, fontname=MONO, fontsize=7.6, color=INK)

# ── image slides (the 5 plots, in order) — VECTOR via show_pdf_page ──
for title, pdfp, bullets in IMG_SLIDES:
    pg = doc.new_page(width=PW, height=PH)
    _center(pg, 34, title, 15)
    src = fitz.open(str(pdfp))
    pg.show_pdf_page(_fit(src[0].rect, fitz.Rect(22, 50, PW - 22, 448)), src, 0)
    src.close()
    for i, b in enumerate(bullets):
        by = 460 + i * 32
        pg.draw_circle(fitz.Point(44, by + 6), 2.2, color=INK, fill=INK)   # base-14 has no U+2022 → draw it
        pg.insert_textbox(fitz.Rect(54, by, PW - 36, by + 30),
                          b, fontname=FONT, fontsize=9.0, color=INK, align=0)

doc.save(str(OUT), deflate=True, garbage=3)
doc.close()
print(f"PDF compiled (VECTOR) -> {OUT}  ({2 + len(IMG_SLIDES)} slides: title + sources + {len(IMG_SLIDES)} figures)")
