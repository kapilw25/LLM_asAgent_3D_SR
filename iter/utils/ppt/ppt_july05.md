# iter18 → iter19 · July-05 — 1B replication + scale transfer

> **Headline:** Surgery ≫ frozen **holds on the 1B (ViT-g)** *and* **replicates 2B → 1B** →
> greenlights the iter19 full-scale (116k) run on the cheaper 1B backbone.
> POC eval · n = 1,825 · 95% BCa CIs · `m13 --cross-plots` over the two per-backbone `eval_metrics`.

Build the PDF: `cd iter/utils/ppt && python make_ppt_july05_pdf.py` → `ppt_july05.pdf` (7 slides).
The plots are embedded as **true vector** (PyMuPDF `show_pdf_page` on the source `.pdf`, not the `.png`) — sharp at any zoom.

---

## 0 · Data sources (Task 1) — every figure → its `eval_metrics.{csv,json}`

All 5 figures are the **cross-backbone report** (`m13_eval_plot.py --cross-plots`, `cross_backbone_report()`),
which reads the **canonical per-backbone `eval_metrics.json`** for each backbone and writes the forest /
scale / combined figures beside them.

| figure | what it compares | source |
|---|---|---|
| `forest_plot_best_{ci,mean}` | OURS vs **best competitor** (stat + magnitude) | both `eval_metrics.json` · `plot_forest(vs="best")` |
| `forest_plot_frozen_ci` | OURS vs **frozen** (the paper claim) | both `eval_metrics.json` · `plot_forest(vs="frozen")` |
| `scale_replication` | 1B rank vs 2B rank (Spearman ρ) | both `eval_metrics.json` · `plot_scale_replication()` |
| `eval_scorecard_combined` | 2B champion stacked over 1B | the two per-backbone `eval_scorecard.pdf` |

**The two canonical metric files** (under `iter18_ablations_FTtechniues/result_outputs/`):
```text
2B  v5_1B/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.{csv,json}
1B  v5_1B/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.{csv,json}
```

---

## 1 · `forest_plot_best_ci` — Surgery vs the STRONGEST competitor, statistical (CI-widths)

- Hardest bar (vs the best **non**-ours arm, not frozen): future-L1 **6.3×/4.8×** and causal-L1 **2.3×/2.7×** CI-widths clear significance at 2B/1B.
- Motion-cosine sits far left (**−14.5× / −9.2×**) — surgery cedes semantic separation to full-FT; reported, not hidden.

## 2 · `forest_plot_best_mean` — Surgery vs the strongest competitor, effect size (raw %)

- Magnitude view: the predictive lead is small-but-consistent — future-L1 **+1.7%/+1.4%**, mask-ratio **+1.1%/+2.9%**, teacher-free **+3.8%** (1B).
- The motion-cosine deficit is large (**−44.9% / −33.4%**) — a deliberate trade: surgery optimizes prediction, full-FT optimizes semantics.

## 3 · `forest_plot_frozen_ci` — THE PAPER CLAIM: surgery vs FROZEN (CI-widths)

- Decisive: surgery separates from frozen by **5–47 CI-widths** on **every** predictive + motion metric (future-L1 47×/27×, motion-cos 27×/26×) at **both** scales.
- The cost is coherence — surgery regresses on TCC-τ / TCC-cycle (grey/negative), the frame-timing metrics frozen still owns; a stated trade.

## 4 · `scale_replication` — does the 1B rank like the 2B? (Spearman ρ)

- Core metrics replicate 2B→1B: causal-L1 **ρ=0.978**, motion-cos 0.952, future-L1 0.938, mask-ratio 0.895 → the cheap 1B is a faithful proxy for the 2B.
- **12/15** metrics replicate (ρ>0.2); 3 secondary ones fail (rollout −0.60, teacher-free −0.25, temporal-order −0.11) — flagged, not over-claimed.

## 5 · `eval_scorecard_combined` — full appendix: 2B champion + 1B, all 15 metrics

- Every arm × 15 metrics at both scales (n=1,825, 95% BCa CI): the same **three-behaviour split** — surgery wins prediction, full-FT semantics, frozen coherence.
- The 1B (bottom) is a scaled-down **twin** of the 2B champion (top) — the bar orderings mirror, exactly what `scale_replication` quantifies.

---

### One-line takeaway
The 1B reproduces the 2B's surgery-wins-prediction / concedes-semantics-and-coherence story with ρ≈0.9–0.98
on the headline metrics → the iter19 full-scale run on the 1B is a sound, ~½-cost stand-in for the 2B champion.
