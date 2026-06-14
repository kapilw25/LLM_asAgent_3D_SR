"""
iter18 — single BIG, eyeball-able TCC comparison chart (user request 2026-06-13).

Two horizontal-bar panels (the two TCC sub-metrics that live in eval_scorecard.png):
  · TCC Kendall τ   (↑ higher better) — frame-order rank correlation
  · TCC cycle-back  (↓ lower  better) — cycle-consistency error
Every encoder with TCC data is shown (honest context: ALL fine-tuning drops below
Frozen). The 4 named players (Frozen / Pretrain / OURS-3stage-DI / raw) are boldly
coloured; the rest are grey. Values are PLAIN DECIMALS, identical to eval_metrics.csv
(NO ×10⁻ⁿ rescale) so they can be verified by eye against the csv. A dashed line marks
Frozen's value = "the bar to beat". x-axis is zoomed so the small gaps are visible;
the exact number is printed on every bar so the zoom can't mislead.

USAGE:
  python scripts/iter18_tcc_chart.py \
      --metrics-json outputs/poc/probe_plot/metrics_watch/vjepa_2_1_vitG/eval_metrics.json \
      --out-dir      outputs/poc/probe_plot/metrics_watch/vjepa_2_1_vitG \
      --out-stem     tcc_comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# display name + colour + bold? — keyed by encoder. The 4 named players pop; rest grey.
ARMS = {
    "vjepa_2_1_frozen":                     ("Frozen  (reference)",   "#E53935", True),
    "vjepa_2_1_pretrain_encoder":           ("Pretrain  (start pt)",  "#1565C0", True),
    "vjepa_2_1_surgical_3stage_DI_encoder": ("OURS  (3-stage DI)",    "#1B5E20", True),
    "vjepa_2_1_surgery_raw_encoder":        ("raw  (surgery control)", "#C0CA33", True),
    "vjepa_2_1_surgical_noDI_encoder":      ("Surgery no-DI",         "#9E9E9E", False),
    "vjepa_2_1_surgical_autorgn_encoder":   ("Surgery auto-RGN",      "#9E9E9E", False),
    "vjepa_2_1_full_ft_encoder":            ("Full fine-tune",        "#9E9E9E", False),
    "vjepa_2_1_lpft_encoder":               ("LP-FT",                 "#9E9E9E", False),
    "vjepa_2_1_peft_lora_encoder":          ("LoRA",                  "#9E9E9E", False),
    "vjepa_2_1_peft_dora_encoder":          ("DoRA",                  "#9E9E9E", False),
}
# iter18 (2026-06-14): any scheduler encoder NOT explicitly styled above → grey default (same treatment as
# the other fine-tuners), so a NEW arm auto-appears in this chart with no edit here. Roster comes from the
# single source (configs/arm_registry.yaml); the 4 named players above keep their bold custom colours.
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.arm_registry import display_arms as _display_arms  # noqa: E402
for _a, _enc, _grp, _kind in _display_arms():
    _full = f"vjepa_2_1_{_enc}"
    if _full not in ARMS:
        ARMS[_full] = (_enc.replace("_encoder", "").replace("_", " "), "#9E9E9E", False)
PANELS = [
    ("tcc_tau",   "TCC Kendall τ   (↑ higher = better)", "higher",
     "frame-order rank correlation"),
    ("tcc_cycle", "TCC cycle-back   (↓ lower = better)",      "lower",
     "cycle-consistency error"),
]


def _panel(ax, rows, key, title, direction, subtitle, frozen_val):
    hi = direction == "higher"
    items = [(ARMS[e][0], ARMS[e][1], ARMS[e][2], d[key]["mean"], d[key]["ci_half"] or 0.0)
             for e, d in rows.items() if e in ARMS and d[key]["mean"] is not None]
    items.sort(key=lambda t: -t[3] if hi else t[3])          # best at TOP
    ys = np.arange(len(items))[::-1]
    for y, (name, col, bold, m, ci) in zip(ys, items):
        ax.barh(y, m, color=col, alpha=0.95 if bold else 0.5, height=0.66,
                edgecolor="black" if bold else col, linewidth=1.8 if bold else 0.6,
                xerr=ci, capsize=4, error_kw=dict(lw=1.2, ecolor="#222"))
        ax.text(m + ci, y, f"  {m:.3f}", va="center", fontsize=12.5,
                fontweight="bold" if bold else "normal",
                color="#000" if bold else "#666")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, *_ in items], fontsize=12.5)
    for tick, (_, col, bold, *_2) in zip(ax.get_yticklabels(), items):
        tick.set_color(col if bold else "#777")
        tick.set_fontweight("bold" if bold else "normal")
    # "the bar to beat" — Frozen reference line
    ax.axvline(frozen_val, color="#E53935", ls="--", lw=1.6, alpha=0.7, zorder=0)
    ax.text(frozen_val, len(items) - 0.3, " Frozen = the bar to beat",
            color="#E53935", fontsize=10.5, fontweight="bold", va="bottom")
    lo = min(m - ci for *_3, m, ci in items)
    hivv = max(m + ci for *_3, m, ci in items)
    pad = (hivv - lo) * 0.18
    ax.set_xlim(lo - pad, hivv + pad * 2.6)                   # zoom so gaps are visible
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(f"score  ({subtitle}) — exact decimals, = eval_metrics.csv", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def build(metrics_json: Path, out_dir: Path, stem: str):
    rows = {x["encoder"]: x for x in json.load(open(metrics_json))}
    frozen = rows["vjepa_2_1_frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    for ax, (key, title, direction, sub) in zip(axes, PANELS):
        _panel(ax, rows, key, title, direction, sub, frozen[key]["mean"])
    o = rows["vjepa_2_1_surgical_3stage_DI_encoder"]
    take = (f"Frozen wins BOTH · every fine-tuner drops below it · "
            f"OURS (τ {o['tcc_tau']['mean']:.3f}) is the GENTLEST surgery — "
            f"beats raw ({rows['vjepa_2_1_surgery_raw_encoder']['tcc_tau']['mean']:.3f}) "
            f"but still below Frozen ({frozen['tcc_tau']['mean']:.3f}).")
    fig.suptitle("TCC — can the encoder keep video frames in time-order?   "
                 "(bold = the 4 players · grey = other fine-tuners · whiskers = 95% CI)",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, take, ha="center", fontsize=12.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#F9A825", lw=1.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.13, wspace=0.45)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved → {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Big eyeball-able TCC-only comparison chart.")
    ap.add_argument("--metrics-json", type=Path, required=True)
    ap.add_argument("--out-dir",      type=Path, required=True)
    ap.add_argument("--out-stem",     type=str, default="tcc_comparison")
    a = ap.parse_args()
    build(a.metrics_json, a.out_dir, a.out_stem)


if __name__ == "__main__":
    main()
