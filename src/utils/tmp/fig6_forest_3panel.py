"""Fig 6 (paper): 3-panel horizontal forest — FactorJEPA's advantage over the best competitor at
POC ViT-G 2B, POC ViT-g 1B, and FULL ViT-g 1B. Metric short-names shown ONCE (shared y-axis), sorted
BEST->WORST by the FULL ViT-g 1B panel. Colours: green = separated win (>=1x), yellow = ahead but within
CI (0..1x), red = competitor ahead (<0). x-axis is symlog with decade ticks; xlim is tight to the data so
bars fill the panel width. n_total in each title (paper scale label). Authored AT \\textwidth (7.0in).
Run: venv_walkindia/bin/python src/utils/tmp/fig6_forest_3panel.py
"""
import sys
import math
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "src")
import matplotlib.pyplot as plt  # noqa: E402
import m13_eval_plot as m13  # noqa: E402

P2 = "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.json"
P1 = "outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.json"
PF = "outputs/full/vjepa_2_1_vitg_1B/eval/full/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.json"
FIGS = "overleaf/2026___FactorJEPA_AAAI/figures"

PANELS = [
    ("POC · ViT-G 2B\nn$_{\\mathrm{total}}$ = 10k", m13._xb_load_metrics(P2)),
    ("POC · ViT-g 1B\nn$_{\\mathrm{total}}$ = 10k", m13._xb_load_metrics(P1)),
    ("FULL · ViT-g 1B\nn$_{\\mathrm{total}}$ = 116k", m13._xb_load_metrics(PF)),
]
SHORT = {
    "fut": "future-frame L1", "causal": "causal-block L1", "maskratio": "mask-ratio slope",
    "teacher_free": "exposure-bias", "rollout": "rollout drift", "tdist": "L1-vs-Δt",
    "act": "action top-1", "mcos": "motion-cos", "aot": "arrow-of-time",
    "tov": "temporal-order", "pace": "pace", "tcc_cycle": "TCC cycle", "tcc_tau": "TCC τ",
}
DIRN = {k: d for k, d, _l in m13._XB_METRICS}
GREEN, YELLOW, RED, GREY = "#2E7D32", "#F9A825", "#B02C2C", "#9E9E9E"


def adv_ci(data, key):
    hi = DIRN[key] == "hi"
    bo = m13._xb_best(data, m13._xb_is_ours, key, hi)
    bc = m13._xb_best(data, lambda s: not m13._xb_is_ours(s), key, hi)
    if not bo or not bc:
        return None
    adv = (bo[1] - bc[1]) if hi else (bc[1] - bo[1])
    ci = ((bo[2] or 0) ** 2 + (bc[2] or 0) ** 2) ** 0.5 or 1e-9
    return adv / ci


def barcol(v):
    if v is None:
        return GREY
    if v >= 1.0:
        return GREEN            # separated win
    if v >= 0.0:
        return YELLOW           # ahead but within CI
    return RED                  # competitor ahead


# y-axis order = BEST -> WORST by the FULL ViT-g 1B panel (mask-ratio at top, TCC-cycle at bottom)
_full = PANELS[2][1]
ORDER = sorted(SHORT.keys(), key=lambda k: (adv_ci(_full, k) if adv_ci(_full, k) is not None else -1e9), reverse=True)

m13.init_style()
ys = list(range(len(ORDER)))[::-1]          # first (best) metric at the TOP
fig, axes = plt.subplots(1, 3, figsize=(7.0, 5.0), sharey=True)
for ax, (label, data) in zip(axes, PANELS):
    vals = [adv_ci(data, k) for k in ORDER]
    plot_v = [(v if v is not None else 0.0) for v in vals]
    ax.barh(ys, plot_v, color=[barcol(v) for v in vals], height=0.68, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(1.0, ls="--", color="#37474F", lw=1.1, zorder=2)      # separation threshold (1xCI)
    ax.axvline(0.0, ls="-", color="#546E7A", lw=0.8, zorder=1)
    ax.set_xscale("symlog", linthresh=1.0)
    valid = [v for v in vals if v is not None]
    lo, hi = min(valid + [0.0]), max(valid + [1.0])
    xlo, xhi = min(lo * 1.10, -0.5), max(hi * 1.10, 1.3)             # TIGHT to data -> bars fill the width
    ax.set_xlim(xlo, xhi)
    maxmag = max(abs(xlo), abs(xhi), 1.0)
    hd = int(math.floor(math.log10(maxmag)))
    ticks = [t for t in sorted({0} | {s * 10 ** e for e in range(hd + 1) for s in (1, -1)}) if xlo <= t <= xhi]
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_title(label, fontsize=10, fontweight="bold", pad=5, color="#1B2A4A", linespacing=1.3)
    for y, v in zip(ys, vals):
        if v is None:
            continue
        if abs(v) >= 2.0:                        # long bar: label INSIDE the tip (toward 0), white
            ax.annotate(f"{v:.1f}×", (v, y), xytext=(5 if v < 0 else -5, 0), textcoords="offset points",
                        va="center", ha="left" if v < 0 else "right", fontsize=7.5, fontweight="bold", color="white")
        else:                                    # short bar: label OUTSIDE the tip, dark
            ax.annotate(f"{v:.1f}×", (v, y), xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7.5, fontweight="bold", color="#1A1A1A")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == 0 else f"{t:g}" for t in ticks], fontsize=7.5, fontweight="bold")
    ax.tick_params(axis="x", length=2)
    ax.grid(axis="x", alpha=0.18, zorder=0)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([SHORT[k] for k in ORDER], fontsize=9, fontweight="bold")
fig.suptitle("FactorJEPA vs. the best competitor, across scale: separation in 95%-CI units\n"
             "(rows sorted best-to-worst by the FULL 1B panel; dashed line = the 1× separation threshold)",
             fontsize=10.3, fontweight="bold", y=0.99)
fig.text(0.5, 0.055, "green: separated win (≥1×)      yellow: ahead but within CI (0–1×)      red: competitor ahead (<0)",
         ha="center", fontsize=8.5, fontweight="bold", color="#333")
fig.text(0.5, 0.017, "n$_{\\mathrm{total}}$ = train + val + test.   FULL ViT-G 2B omitted: not run under the compute "
                     "budget (≈ 2× the 1B cost).", ha="center", fontsize=8, style="italic", color="#555")
fig.subplots_adjust(left=0.175, right=0.985, top=0.80, bottom=0.135, wspace=0.12)
m13.save_fig(fig, FIGS + "/forest_plot_best_ci_paper")
print("wrote forest_plot_best_ci_paper (3-panel) · order:", [SHORT[k] for k in ORDER])
