import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "src")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import m13_eval_plot as m13  # noqa: E402

P2 = "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.json"
P1 = "outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_plot/metrics_watch/vjepa_2_1_vitg/eval_metrics.json"
FIGS = "overleaf/2026___FactorJEPA_AAAI/figures"
BB = [("ViT-G · 2B", m13._xb_load_metrics(P2)), ("ViT-g · 1B", m13._xb_load_metrics(P1))]
SEL = [("fut", 0), ("causal", 1), ("maskratio", 1), ("teacher_free", 1)]
DIRNAME = {k: (d, nm) for k, d, nm in m13._XB_ALL15}
# Show the SAME encoder set in all four panels. Drop f50/f70 (keep the winning f30 WiSE-FT merge), and drop
# the five ablation arms that have NO 1B eval (present only at 2B) so the 2B panel matches the three 1B panels.
DROP = {"surgical_intervene_wiseft_f50", "surgical_intervene_wiseft_f70",
        "surgical_3stage_DI_head", "surgical_3stage_DI_replay25", "surgical_3stage_DI_tccaux",
        "surgical_noDI", "surgical_noDI_head"}

SHORT = {
    "surgical_3stage_DI_intervene": "3sDI·int", "surgical_3stage_DI_tccaux": "3sDI·tcc",
    "surgical_3stage_DI_diheavy": "3sDI·diH", "surgical_3stage_DI_replay25": "3sDI·rp25",
    "surgical_3stage_DI": "3sDI", "surgical_3stage_DI_head": "3sDI·head",
    "surgical_3stage_DI_wiseft": "3sDI·wSE", "surgical_noDI": "noDI", "surgical_noDI_head": "noDI·head",
    "surgical_intervene_wiseft_f30": "wSE·f30", "surgical_intervene_wiseft_f50": "wSE·f50",
    "surgical_intervene_wiseft_f70": "wSE·f70", "surgery_raw": "raw",
    "peft_lora": "LoRA", "peft_dora": "DoRA", "full_ft": "full-FT", "lpft": "LP-FT",
    "frozen": "frozen", "vanilla_continual_SSL": "cont-SSL", "surgical_autorgn": "Auto-RGN",
}
# proper, human-readable full names for the KEY (no internal snake_case codes)
FULLNAME = {
    "surgery_raw": "FactorJEPA-RAW (no factor targets)",
    "surgical_3stage_DI": "FactorJEPA · 3-stage (L→A→interaction)",
    "surgical_3stage_DI_intervene": "FactorJEPA · 3-stage + intervention",
    "surgical_3stage_DI_diheavy": "FactorJEPA · 3-stage (interaction-heavy)",
    "surgical_3stage_DI_tccaux": "FactorJEPA · 3-stage + TCC aux-loss",
    "surgical_3stage_DI_replay25": "FactorJEPA · 3-stage + 25% replay",
    "surgical_3stage_DI_head": "FactorJEPA · 3-stage (head-only)",
    "surgical_noDI": "FactorJEPA · 2-stage (no interaction)",
    "surgical_noDI_head": "FactorJEPA · 2-stage (head-only)",
    "surgical_intervene_wiseft_f30": "FactorJEPA + WiSE-FT merge 30%",
    "surgical_intervene_wiseft_f50": "FactorJEPA + WiSE-FT merge 50%",
    "surgical_intervene_wiseft_f70": "FactorJEPA + WiSE-FT merge 70%",
    "peft_lora": "LoRA (low-rank adaptation)",
    "peft_dora": "DoRA (weight-decomposed LoRA)",
    "lpft": "LP-FT (linear-probe then fine-tune)",
    "full_ft": "full fine-tuning",
    "vanilla_continual_SSL": "continual SSL (V-JEPA pretrain)",
    "surgical_autorgn": "Auto-RGN gradient-norm surgery",
    "frozen": "frozen V-JEPA 2.1 encoder",
}


def col(s):
    return m13._color_for("vjepa_2_1_vitG_" + s, 0)


def full(s):
    return m13._xb_arm_short(s)


def sh(s):
    return SHORT.get(full(s), full(s))


def fn(s):
    return FULLNAME.get(full(s), full(s))


m13.init_style()
fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))   # author AT \textwidth (7.0in) so fonts render 1:1 (VM37)
axf = axes.flatten()
seen = {}
for idx, (key, bi) in enumerate(SEL):
    ax = axf[idx]
    blabel, data = BB[bi]
    _dir, name = DIRNAME[key]
    hi = _dir == "hi"
    items = [(s, data[s][key][0], data[s][key][1] or 0.0) for s in data if data[s][key][0] is not None and full(s) not in DROP]
    items.sort(key=lambda t: (t[1] if hi else -t[1]))
    ys = list(range(len(items)))
    vals = [m for _s, m, _c in items]
    errs = [c for _s, _m, c in items]
    cols = [col(s) for s, _m, _c in items]
    for s, _m, _c in items:
        seen[s] = col(s)
    lo = min(v - e for v, e in zip(vals, errs))
    hix = max(v + e for v, e in zip(vals, errs))
    pad = (hix - lo) * 0.05 or 0.01
    base = lo - pad
    ax.barh(ys, [v - base for v in vals], left=base, xerr=errs, color=cols, height=0.78,
            edgecolor="white", linewidth=0.4, error_kw=dict(ecolor="#37474F", elinewidth=0.8, capsize=1.6), zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([sh(s) for s, _m, _c in items], fontsize=9)
    for t, (s, _m, _c) in zip(ax.get_yticklabels(), items):
        t.set_color(col(s) if col(s) != "#616161" else "#000000")
        t.set_fontweight("bold")
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlim(base, hix + pad * 4.0)                     # just enough right room for the value label (less blank)
    for y, v, e in zip(ys, vals, errs):
        ax.annotate(f"{v:.3f}", (v + e, y), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, fontweight="bold", color="#1A1A1A")
    bo = m13._xb_best(data, m13._xb_is_ours, key, hi)
    bc = m13._xb_best(data, lambda s: not m13._xb_is_ours(s), key, hi)
    adv = (bo[1] - bc[1]) if hi else (bc[1] - bo[1])
    ci = ((bo[2] or 0) ** 2 + (bc[2] or 0) ** 2) ** 0.5 or 1e-9
    ax.set_title(f"{name}  {'↑' if hi else '↓'}\n{blabel}: FactorJEPA wins +{adv / ci:.1f}×CI",
                 fontsize=10, fontweight="bold", color="#1B5E20", pad=3, linespacing=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))       # (1.1) fewer x-ticks so they don't crowd
    ax.tick_params(axis="x", labelsize=9)
    for t in ax.get_xticklabels():                         # (1.1) rotate x numbers 45° + BOLD
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
        t.set_fontweight("bold")
        t.set_color("#1A1A1A")
    ax.grid(axis="x", alpha=0.25, zorder=0)

ours = [s for s in seen if m13._xb_is_ours(s)]
riv = [s for s in seen if not m13._xb_is_ours(s)]
order = sorted(ours, key=lambda s: sh(s)) + sorted(riv, key=lambda s: sh(s))
handles = [Patch(facecolor=seen[s], edgecolor="white", label=f"{sh(s)}  =  {fn(s)}") for s in order]
_leg = fig.legend(handles=handles, loc="lower center", ncol=2, prop={"weight": "bold", "size": 9},
                  frameon=True, bbox_to_anchor=(0.5, 0.006),
                  title="ENCODER KEY   ·   short code = full name   ·   ↓ = lower is better (all four are error metrics)\n"
                        "green shades = FactorJEPA / OURs   ·   other colours = fine-tuning rivals\n"
                        "DI = interaction factor D_I   ·   3-stage = Layout → Agent → Interaction surgery",
                  title_fontsize=9, handlelength=1.1, columnspacing=2.2, labelspacing=0.38, borderpad=0.6)
_leg.get_title().set_fontweight("bold")                    # every text BOLD (title + entries)
fig.suptitle("FactorJEPA vs. every fine-tuning rival: the four predictive metrics it wins\n"
             "(POC 10k · n_test = 1,825 · 95% BCa CI)", fontsize=11, fontweight="bold", y=0.986)
fig.subplots_adjust(left=0.115, right=0.985, top=0.87, bottom=0.30, hspace=0.52, wspace=0.42)
m13.save_fig(fig, FIGS + "/eval_scorecard_winbars")
print("wrote eval_scorecard_winbars.{png,pdf} · panels:", SEL, "· key arms:", len(order))
