"""Fig 8 (paper): 2B→1B rank-preservation scatter for the top-replicating metric (causal future-block L1,
ρ=0.978), with EACH encoder in its own colour + name (legend in the empty lower-right triangle). OURS family
gets distinct green shades; rivals use their registry colours. Uses the FULL 14-arm roster so the reported
ρ matches the body's cross-scale range (0.895 to 0.978); dropping arms would shift the statistic.
Run: venv_walkindia/bin/python src/utils/tmp/fig8_scale_causal.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "src")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import m13_eval_plot as m13  # noqa: E402

P2 = "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.json"
P1 = "outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.json"
FIGS = "overleaf/2026___FactorJEPA_AAAI/figures"
KEY = "causal"
DROP = {"surgical_intervene_wiseft_f50", "surgical_intervene_wiseft_f70"}   # match Fig 7's 12 encoders
SHORT = {
    "surgical_3stage_DI": "3sDI", "surgical_3stage_DI_intervene": "3sDI·int",
    "surgical_3stage_DI_diheavy": "3sDI·diH", "surgical_intervene_wiseft_f30": "wSE·f30",
    "surgery_raw": "raw", "peft_lora": "LoRA", "peft_dora": "DoRA", "full_ft": "full-FT",
    "lpft": "LP-FT", "frozen": "frozen", "vanilla_continual_SSL": "cont-SSL", "surgical_autorgn": "Auto-RGN",
}
# OURS family all map to one registry green — give each a DISTINCT green shade so every encoder has its own colour
OURS_SHADES = {"surgical_3stage_DI": "#1B5E20", "surgical_3stage_DI_diheavy": "#388E3C",
               "surgical_3stage_DI_intervene": "#43A047", "surgical_intervene_wiseft_f30": "#66BB6A",
               "surgery_raw": "#81C784"}
d2 = m13._xb_load_metrics(P2)
d1 = m13._xb_load_metrics(P1)


def full(s):
    return m13._xb_arm_short(s)


def sh(s):
    return SHORT.get(full(s), full(s))


def col(s):
    return OURS_SHADES.get(full(s), m13._color_for(s, 0))   # OURS = green shades; rivals = registry colour


shared = [s for s in d2 if s in d1 and d2[s][KEY][0] is not None and d1[s][KEY][0] is not None and full(s) not in DROP]
xs = [d2[s][KEY][0] for s in shared]
ys = [d1[s][KEY][0] for s in shared]
rho = spearmanr(xs, ys).correlation

m13.init_style()
fig, ax = plt.subplots(figsize=(3.5, 3.7))
lo, hi = min(xs + ys), max(xs + ys)
pad = (hi - lo) * 0.16 or 0.01
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#90A4AE", lw=1.3, zorder=1)
handles = []
for s in sorted(shared, key=lambda z: (not m13._xb_is_ours(z), sh(z))):
    x, y = d2[s][KEY][0], d1[s][KEY][0]
    c = col(s)
    ax.scatter(x, y, c=[c], s=95, edgecolors="white", linewidths=0.9, zorder=4)
    handles.append(Line2D([], [], marker="o", ls="none", mfc=c, mec="white", mew=0.6, ms=6.5, label=sh(s)))
handles.append(Line2D([], [], ls="--", color="#90A4AE", lw=1.3, label="identity  y=x"))
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(lo - pad, hi + pad)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"causal future-block L1  ↓      ρ = {rho:.3f}", fontsize=9.5, fontweight="bold", pad=6)
ax.set_xlabel("ViT-G · 2B · per-method score", fontsize=8.5, fontweight="bold")
ax.set_ylabel("ViT-g · 1B · per-method score", fontsize=8.5, fontweight="bold")
ax.tick_params(labelsize=8)
ax.grid(alpha=0.25)
leg = ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=6.6, frameon=True, framealpha=0.93,
                handletextpad=0.3, borderpad=0.4, columnspacing=0.9, labelspacing=0.28,
                title="encoders (green = FactorJEPA)", title_fontsize=6.8)
leg.get_title().set_fontweight("bold")
fig.subplots_adjust(top=0.905, bottom=0.135, left=0.165, right=0.965)
m13.save_fig(fig, FIGS + "/scale_replication_single")
print(f"wrote scale_replication_single (per-encoder colours + legend) · {len(shared)} arms · ρ={rho:.3f}")
