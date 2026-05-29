#!/usr/bin/env python3
"""Generate iter16 progress.pdf — 3 pages (1 per metric), iter15 v1 vs v2 side-by-side.

Audience: research lead / professor. Snapshot of the data-leakage fix:
  v1 = trainHead/freezeEncoder, test-clip leakage in surgery SSL pool, n=220 test clips
  v2 = leakage fixed (universe - val - test), n=1825 test clips

Embeds the source *.pdf (VECTOR) via fitz.show_pdf_page so text stays sharp on zoom,
and auto-crops each source page just below the x-axis labels (drops the bottom
explanation box + whitespace). All takeaway numbers verified against the committed
probe_*.json under each result_outputs dir (2026-05-28). Run:
  python iter/iter15_v2_data_leakage/result_outputs/make_progress_pdf.py
"""
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path("/workspace/factorjepa")
V1 = ROOT / "iter/iter15_v1_trainHead_freezeEncoder/result_outputs/poc/probe_plot/eval"
V2 = ROOT / "iter/iter15_v2_data_leakage/result_outputs/poc/probe_plot/eval"
OUT = ROOT / "iter/iter15_v2_data_leakage/result_outputs/progress.pdf"

# A4 landscape (points)
W, H = 841.89, 595.28
# colors (0-1 rgb)
NAVY = (0.086, 0.192, 0.31); GREY = (0.27, 0.27, 0.27)
RED = (0.69, 0.0, 0.125); GREEN = (0.106, 0.478, 0.239)
INK = (0.1, 0.1, 0.1); BOXFILL = (0.957, 0.965, 0.973); BOXEDGE = (0.78, 0.8, 0.82)
FAINT = (0.53, 0.53, 0.53)

FOOTER = ("FactorJEPA  |  V-JEPA 2.1 ViT-G (2B)  |  POC eval  |  "
          "iter15 v1 (head-probe, test-clip leakage) vs v2 (leakage-fixed, 8x eval clips)  |  "
          "8 encoders  |  95% BCa CI, 10K bootstrap  |  2026-05-28")

PAGES = [
    dict(
        png="probe_action_acc_compare.pdf",
        title="Action probe top-1 accuracy",
        sub="Linear separability of motion-action classes on encoder features  (higher is better)",
        lines=[
            "v2 removes test-clip leakage and grows the eval set 220 -> 1825 clips; 95% CI half-width: 6.1 -> 2.3 pp  (~2.7x, the 1/sqrt(N) law).",
            "Encoder adaptation lifts top-1 to 50-53% - CI-separated +6 to +9 pp above frozen (44.4%) in v2.",
            "Surgery vs continual-pretrain: top-six CIs OVERLAP in BOTH versions - no separation, no leakage-driven reversal.",
        ],
    ),
    dict(
        png="probe_future_mse_compare.pdf",
        title="Future-frame prediction error (L1)",
        sub="Predictor rollout error on held-out future frames  (lower is better)",
        lines=[
            "Same leakage fix + 8x eval clips; 95% CI half-width: 0.0024 -> 0.0008  (~2.9x).",
            "Cleanest win: surgery-on-encoder gives the LOWEST error (0.515), CI-separated below all pretrain (>=0.540) and frozen (0.557) - both versions.",
            "The gain is encoder-specific: surgery-on-head ~ continual-pretrain (~0.540).",
        ],
    ),
    dict(
        png="probe_motion_cos_compare.pdf",
        title="Motion cosine: intra-inter class separation",
        sub="Same-class minus different-class cosine of motion features  (higher is better)",
        lines=[
            "Same fix + 8x clips; with the clean larger set the class separation roughly DOUBLES (~0.08 -> 0.17).",
            "Encoder adaptation creates motion structure: 6 adapted variants at 0.16-0.18; frozen & head-only pretrain stay near 0 (~0.01).",
            "Best surgery (noDI-head 0.179) > best pretrain (2X 0.169): CI-separated in v2, overlapping CIs in v1.",
        ],
    ),
]

CAP_KEYS = ("whiskers", "TEST split", "AttentiveClassifier", "encoder clusters", "predictor")


def crop_rect(pg):
    """Rect of the source page to keep: top down to just below x-axis labels."""
    tops = []
    for b in pg.get_text("dict")["blocks"]:
        txt = " ".join(s["text"] for ln in b.get("lines", []) for s in ln["spans"])
        if any(k in txt for k in CAP_KEYS):
            tops.append(b["bbox"][1])
    cap_top = min(tops) if tops else None
    content_bottom = max((w[3] for w in pg.get_text("words")
                          if cap_top is None or w[3] < cap_top - 2), default=pg.rect.height)
    return fitz.Rect(0, 0, pg.rect.width, min(content_bottom + 8, pg.rect.height))


def fit(clip, box):
    """Aspect-preserving target rect for `clip` centered inside `box`."""
    a = clip.width / clip.height
    if a > box.width / box.height:
        tw, th = box.width, box.width / a
    else:
        th, tw = box.height, box.height * a
    x = box.x0 + (box.width - tw) / 2
    y = box.y0 + (box.height - th) / 2
    return fitz.Rect(x, y, x + tw, y + th)


def src(path):
    if not path.exists():
        sys.exit(f"FATAL: missing source plot {path}")
    return fitz.open(str(path))


out = fitz.open()
open_docs = []
for i, pg in enumerate(PAGES, 1):
    d1, d2 = src(V1 / pg["png"]), src(V2 / pg["png"])
    open_docs += [d1, d2]
    page = out.new_page(width=W, height=H)

    t1 = page.insert_textbox(fitz.Rect(0, 12, W, 50), f'Metric {i} / 3      {pg["title"]}',
                             fontsize=16, fontname="hebo", color=NAVY, align=fitz.TEXT_ALIGN_CENTER)
    t2 = page.insert_textbox(fitz.Rect(0, 52, W, 76), pg["sub"],
                             fontsize=10.5, fontname="heit", color=GREY, align=fitz.TEXT_ALIGN_CENTER)
    t3 = page.insert_textbox(fitz.Rect(0, 80, W / 2, 106), "v1    |    n = 220    |    test-clip leakage",
                             fontsize=11, fontname="hebo", color=RED, align=fitz.TEXT_ALIGN_CENTER)
    t4 = page.insert_textbox(fitz.Rect(W / 2, 80, W, 106), "v2    |    n = 1825    |    leakage fixed",
                             fontsize=11, fontname="hebo", color=GREEN, align=fitz.TEXT_ALIGN_CENTER)

    boxL = fitz.Rect(18, 112, W / 2 - 8, 432)
    boxR = fitz.Rect(W / 2 + 8, 112, W - 18, 432)
    c1, c2 = crop_rect(d1[0]), crop_rect(d2[0])
    page.show_pdf_page(fit(c1, boxL), d1, 0, clip=c1)
    page.show_pdf_page(fit(c2, boxR), d2, 0, clip=c2)

    page.insert_textbox(fitz.Rect(40, 436, 300, 458), "TAKEAWAY",
                        fontsize=10, fontname="hebo", color=NAVY)
    page.draw_rect(fitz.Rect(40, 460, W - 40, 553), color=BOXEDGE, fill=BOXFILL, width=0.8)
    body = "\n".join(f"{j}.   {ln}" for j, ln in enumerate(pg["lines"], 1))
    t5 = page.insert_textbox(fitz.Rect(54, 467, W - 54, 549), body,
                             fontsize=9.5, fontname="helv", color=INK,
                             align=fitz.TEXT_ALIGN_LEFT, lineheight=1.55)
    page.insert_textbox(fitz.Rect(0, 567, W, 588), FOOTER,
                        fontsize=7, fontname="helv", color=FAINT, align=fitz.TEXT_ALIGN_CENTER)
    for nm, lo in [("title", t1), ("subtitle", t2), ("v1", t3), ("v2", t4), ("takeaway", t5)]:
        if lo < 0:
            sys.exit(f"FATAL: '{nm}' text overflow on page {i} (leftover={lo:.1f})")

out.save(str(OUT), deflate=True)
out.close()
for d in open_docs:
    d.close()
print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB, {len(PAGES)} pages, vector-embedded)")
