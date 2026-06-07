#!/usr/bin/env python3
"""Compile ppt.md into ppt.pdf (16:9 slides) with matplotlib — the only PDF engine
on this box (no pandoc/LaTeX/weasyprint). One slide per plot: image + the 2 bullets.

USAGE:  cd iter/iter18_ablations_FTtechniues/result_outputs && python make_ppt_pdf.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
I17 = HERE.parent.parent / "iter17_ablations_model" / "result_outputs"
OUT = HERE / "ppt.pdf"

_TECHS = [  # technique → (FULL FORM per 📖 glossary, 5-year-old line ≤10 words) — mirrors ppt.md §0
    ("frozen", "frozen backbone (no adaptation)",
     "Don't touch the brain; just test what it knows."),
    ("pretrain (vCSSL)", "vanilla continual SSL (Self-Supervised Learning) — OURS anchor",
     "Keep practicing videos the same old way."),
    ("surgery 3stage-DI (OURS)", "staged factor-curriculum continual-FT, with D_I tubes",
     "Special lesson clips; unlock slowly; includes things-touching clips."),
    ("surgery noDI (OURS)", "staged factor-curriculum continual-FT, without D_I",
     "Same slow unlocking; skip the things-touching clips."),
    ("surgery heads (OURS)", "staged factor-curriculum continual-FT, head-only",
     "Brain stays locked; only a tiny helper learns."),
    ("surgery RAW (control)", "staged factor-curriculum continual-FT on raw clips",
     "Surgery's slow unlocking, but with plain videos."),
    ("Auto-RGN", "Automatic Relative Gradient Norm (Lee et al. ICLR'23)",
     "Layers that pull harder get bigger learning steps."),
    ("Full-FT", "Full Fine-Tuning",
     "Unlock the whole brain; change everything at once."),
    ("LP-FT", "Linear-Probing then Fine-Tuning (Kumar et al. ICLR'22)",
     "First teach the helper, then unlock everything."),
    ("LoRA", "Low-Rank Adaptation (Hu et al. 2021)",
     "Stick tiny add-on notes; never rewrite the book."),
    ("DoRA", "Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024)",
     "Sticky notes plus a volume knob per page."),
    ("CaSSLe", "continual self-supervised distillation (Fini et al. CVPR'22)",
     "New learning must still match the old teacher."),
    ("EWC", "Elastic Weight Consolidation (Kirkpatrick et al. PNAS'17)",
     "Important old memories get glued — hard to change."),
]

SLIDES = [
    ("1 · HERO — kept-checkpoint scorecard",
     HERE / "v2/poc/probe_plot/metrics_watch/kept_scorecard.png",
     ["Surgery leads future-L1 (0.4995) and causal (0.530); LoRA/DoRA trail ≤0.005; full-FT owns semantics.",
      "Semantic gaps sit inside N=451 noise; surgery's ~0.035 future-L1 lead over pretrain/full-FT holds."]),
    ("2 · Training — every probe checkpoint, every arm",
     HERE / "v2/poc/probe_plot/metrics_watch/train_trajectories.png",
     ["Surgery's future-L1 falls every stage (0.523→0.4995); pretrain stays flat (~0.54) — progressive unfreeze works.",
      "Selector keeps mid-training minima — LP-FT probe 1/5, full-FT 2/4; last checkpoints regress."]),
    ("3 · Upcoming — per-encoder TEST eval scorecard",
     HERE / "v2/poc/probe_plot/metrics_watch/eval_scorecard.png",
     ["TEST verdicts (n=1,825, 95% BCa CIs) auto-fill as evals finish; val trends need confirmation.",
      "Paired-delta finale decides significance — val-probe leads are directional only."]),
    ("4a · iter17 — frozen backbone selection",
     I17 / "v17a_frozen_eval/poc/probe_plot/eval/m13_frozen_scorecard.png",
     ["V-JEPA 2.1 ViT-G is the strongest frozen backbone — justified as iter18's sole backbone.",
      "Every frozen model is motion-blind (motion-cos ~ 0) — adaptation is necessary, not optional."]),
    ("4b · iter17 — ViT-G family raw values after training",
     I17 / "v17b_train_eval/poc/probe_plot/eval/m13_hero_raw_values.png",
     ["Any continual training beats frozen by +6-7pp top-1; gains are not surgery-specific.",
      "Only surgery moves future-MSE and causal — encoder updates drive temporal gains, heads don't."]),
    ("4c · iter17 — paired deltas with 95% CI",
     I17 / "v17b_train_eval/poc/probe_plot/eval/m13_paired_diff_heatmap.png",
     ["Surgery beats frozen significantly on action and motion-cos — CIs exclude zero.",
      "Versus pretrain: action ties; motion/prediction edge persists — surgery's value is temporal, not semantic."]),
]

with PdfPages(OUT) as pdf:
    # ── slide 0a: experiment map image ──
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("0 · Experiment map — 13 arms, one backbone (ViT-G 2B)",
                 fontsize=18, fontweight="bold", y=0.965)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
    ax.imshow(mpimg.imread(str(HERE.parent / "iter18_ft_baselines_pipeline.png")))
    ax.axis("off")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    # ── slide 0b: technique glossary (FULL FORM per 📖 glossary + 5-year-old line) ──
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("0 · Techniques — FULL FORM + like you're 5",
                 fontsize=18, fontweight="bold", y=0.965)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis("off")
    t = ax.table(cellText=[[n, f, w] for n, f, w in _TECHS],
                 colLabels=["technique", "FULL FORM (glossary)", "like you're 5"],
                 colWidths=[0.18, 0.42, 0.40], loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.55)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor("#999")
        if r == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    for title, img, bullets in SLIDES:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
        ax = fig.add_axes([0.02, 0.16, 0.96, 0.76])
        ax.imshow(mpimg.imread(str(img)))
        ax.axis("off")
        for i, b in enumerate(bullets):
            fig.text(0.06, 0.095 - i * 0.05, f"•  {b}", fontsize=13, va="top")
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

print(f"PDF compiled → {OUT} ({len(SLIDES) + 2} slides incl. map + glossary)")
