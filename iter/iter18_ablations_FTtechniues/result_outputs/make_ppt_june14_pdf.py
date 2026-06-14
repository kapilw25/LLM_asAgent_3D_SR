#!/usr/bin/env python3
"""Compile ppt_june14.md into ppt_june14.pdf (16:9 slides) with matplotlib — the only
PDF engine on this box (no pandoc/LaTeX). Mirrors make_ppt_pdf.py: image/table per slide.
Content (paths, bullets, tables) is hardcoded to MATCH ppt_june14.md verbatim — this is a
doc-gen utility under result_outputs/, not a src/ or scripts/ pipeline file.

USAGE:  cd iter/iter18_ablations_FTtechniues/result_outputs && python make_ppt_june14_pdf.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
MW = HERE / "v2/poc/probe_plot/metrics_watch/vjepa_2_1_vitG"
V3 = HERE / "v3/plan_tweak_surgery"
OUT = HERE / "ppt_june14.pdf"

# ── the 5 NEW arms (plain text — matplotlib's default font has no emoji glyphs) ──
_NEW_ARMS = [
    ["Replay-25", "raw replay 50% -> 25%", "beat raw on prediction; cut semantic forgetting"],
    ["DI-heavy", "D_I stage budget 30% -> 45%", "causal-L1 / future-MSE via interaction factor"],
    ["TCC-aux", "+ gamma*TCC cycle loss", "recover the TCC-tau coherence regression"],
    ["Intervene", "+ object-tube mask (Causal-JEPA)", "push causal-L1 / future-MSE further"],
    ["WiSE-FT", "0.7*ours + 0.3*frozen (post-hoc, no train)", "buy back coherence, keep the lead"],
]

# ── §3 three-behaviour verdict (mirrors ppt_june14.md table; means rounded) ──
_VERDICT = [
    ["Semantics", "action top-1  ↑", "pretrain .528", ".492", "trails, but within CI ±.023 (noise)"],
    ["Semantics", "motion-cos  ↑", "full-FT .244", ".119", "full-FT sig> OURS;  OURS sig> frozen .009"],
    ["Semantics", "taxonomy-F1  ↑", "frozen .797", ".789", "tied — not a differentiator"],
    ["Prediction", "future-MSE  ↓", "raw .497 ≈ OURS", ".498", "surgery sig< full-FT .541 / frozen .557"],
    ["Prediction", "causal-L1  ↓", "raw .528 ≈ OURS", ".530", "surgery sig< full-FT .566 / frozen .583"],
    ["Coherence", "TCC-tau  ↑", "frozen .293", ".269", "OURS sig regresses; full-FT .284 keeps more"],
    ["Coherence", "TCC-cycle  ↓", "frozen 2.110", "2.259", "training hurts all; OURS best-trained"],
    ["Coherence", "AoT / pace  ↑", "frozen .963/.741", ".932/.731", "training costs ~3pp; rest within noise"],
]

# ── image slides: (title, png, [bullet, bullet]) ──
IMG_SLIDES = [
    ("1 · Full 14-panel TEST scorecard (appendix figure)",
     MW / "eval_scorecard.png",
     ["All 14 metrics, 11 trained arms + frozen, n=1,825, 95% BCa CI; common-exponent axes keep clustered bars legible.",
      "Three behaviours separate: prediction (surgery wins), semantics (full-FT/pretrain win), coherence (frozen wins)."]),
    ("2 · HERO paper figure — Causal-L1 & Future-MSE, full baseline set",
     MW / "eval_scorecard_paper.png",
     ["Green surgery family tops both predictive panels — 95% CIs clear of full-FT, LoRA/DoRA and frozen.",
      "Honest read: OURS matches RAW surgery within CI (gain is surgery itself); trades a small motion/TCC margin, reported not hidden."]),
    ("4a · Arm tree — where the 5 new arms attach",
     V3 / "architechture.png",
     ["13 arms done on one backbone (ViT-G 2B); the 5 new arms branch off SURGERY, each = OURS + ONE change.",
      "WiSE-FT (amber) is post-hoc weight-merge — no training, depends only on the trained OURS + frozen ckpts."]),
    ("4b · Model-data pipeline — factor clips vs raw clips -> one scorecard",
     V3 / "model_data_pipeline.png",
     ["Factor clips (m10 SAM masks -> m11 factor sets) feed SURGERY + the 5 arms; raw clips feed every baseline.",
      "All 18 endpoints land in the same m12a-m12f eval (14 metrics, n=1,825, 95% BCa CI)."]),
]


def _table_slide(pdf, title, col_labels, rows, col_widths, fontsize=9.0, scale_y=1.5):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
                 loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, scale_y)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor("#999")
        if r == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


with PdfPages(OUT) as pdf:
    # ── slide 0: title ──
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.5, 0.62, "iter18 · June-14 — weekly progress",
             ha="center", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.50, "Surgery vs FT-techniques · POC = paper run · ViT-G 2B",
             ha="center", fontsize=16, color="#37474f")
    fig.text(0.5, 0.36,
             "13 arms trained + 14-metric TEST eval (n=1,825, 95% BCa CI) DONE  ·  "
             "5 improvement arms built, in SANITY",
             ha="center", fontsize=12, color="#2e7d32")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    # ── slide 0b: the 5 new arms glossary ──
    _table_slide(pdf, "0 · The 5 NEW arms — each = SURGERY (ours) + ONE change",
                 ["new arm", "the one change", "what it targets"],
                 _NEW_ARMS, [0.16, 0.40, 0.44], fontsize=11, scale_y=2.4)

    # ── image slide 1 + 2 (scorecards) ──
    for title, img, bullets in IMG_SLIDES[:2]:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.965)
        ax = fig.add_axes([0.02, 0.16, 0.96, 0.76])
        ax.imshow(mpimg.imread(str(img)))
        ax.axis("off")
        for i, b in enumerate(bullets):
            fig.text(0.05, 0.105 - i * 0.05, f"•  {b}", fontsize=11.5, va="top")
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

    # ── slide 3: §3 three-behaviour verdict table ──
    _table_slide(pdf, "3 · Conclusions from eval_metrics.csv — 3 behaviours, 3 winners",
                 ["behaviour", "metric", "winner (mean)", "OURS", "verdict (95% BCa CI)"],
                 _VERDICT, [0.12, 0.16, 0.18, 0.10, 0.44], fontsize=9.5, scale_y=1.9)

    # ── image slides 4a + 4b (architecture + pipeline) ──
    for title, img, bullets in IMG_SLIDES[2:]:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.965)
        ax = fig.add_axes([0.02, 0.16, 0.96, 0.76])
        ax.imshow(mpimg.imread(str(img)))
        ax.axis("off")
        for i, b in enumerate(bullets):
            fig.text(0.05, 0.105 - i * 0.05, f"•  {b}", fontsize=11.5, va="top")
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

print(f"PDF compiled -> {OUT}  ({3 + len(IMG_SLIDES)} slides: title + glossary + verdict + {len(IMG_SLIDES)} figures)")
