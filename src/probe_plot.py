"""m08d — N-encoder visualization for the probe trio (action probe + motion cos + future MSE).

CPU-only. Pure-visualization function — like m08b, ALWAYS recomputes; no cache_policy.

Reads (auto-discovers ALL encoders present in the JSON `by_encoder` dict — no
hardcoded encoder list, so the same plotter works for {frozen, pretrain, surgical}
or any future fine-tuning technique that lands a row in by_encoder):
  <action-probe-root>/probe_paired_delta.json         {by_encoder, pairwise_deltas}
  <action-probe-root>/<enc>[/lr_*]/train_log.jsonl   probe train history (per-LR)
  <motion-cos-root>/probe_motion_cos_paired.json      {by_encoder, pairwise_deltas}
  <future-mse-root>/probe_future_mse_per_variant.json {by_variant: {<enc>: ... or "n/a"}}

Writes (under --output-dir):
  probe_action_loss.{png,pdf}       train loss per encoder/LR
  probe_action_acc.{png,pdf}        val acc per encoder/LR (train dropped — see loss plot for overfit signal)
  probe_action_acc_compare.{png,pdf}      bars sorted ↓ highest-first (higher=better)
  probe_motion_cos_compare.{png,pdf}      bars sorted ↓ highest-first (higher=better)
  probe_future_mse_compare.{png,pdf}      bars sorted ↑ lowest-first  (lower=better)

USAGE:
  python -u src/probe_plot.py --SANITY \\
    --action-probe-root outputs/sanity/probe_action \\
    --motion-cos-root   outputs/sanity/probe_motion_cos \\
    --future-mse-root   outputs/sanity/probe_future_mse \\
    --output-dir        outputs/sanity/probe_plot \\
    2>&1 | tee logs/probe_plot_sanity.log
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.plots import COLORS, ENCODER_COLORS, init_style, save_fig
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics
# iter13 v13 (2026-05-07): pull bootstrap iter count from the single source of
# truth (utils.bootstrap.N_BOOTSTRAP) instead of hardcoding "10 K" in caption.
from utils.bootstrap import N_BOOTSTRAP


# ── Display helpers (no hardcoded encoder list — derived per-call) ───

def _display_label(enc: str) -> str:
    """Human-readable encoder name for plot legends. Falls back to enc verbatim.

    iter15 (2026-05-16): added the 6 head/encoder POC variants from
    run_eval.sh:158 ENCODERS list. Caller at line ~229 applies
    `.replace(" ", "\\n")` to stack labels vertically on x-ticks; design
    each label so the newline-splits land on natural word boundaries.
    Dropped the universal `vjepa_2_1_` prefix since every label would
    otherwise repeat it (waste vertical space). The head/encoder suffix
    is preserved verbatim so the paired-Δ visual mapping is unambiguous.
    """
    return {
        # iter14 anchors
        "vjepa_2_1_frozen":                       "V-JEPA 2.1 frozen",
        "dinov2":                                 "DINOv2 frozen",
        # iter15 ENCODER-update variants (full ViT-G backward).
        # Spaces between [variant] and [encoder/head] suffix → caller's
        # .replace(" ", "\\n") splits them onto separate tick-label lines.
        "vjepa_2_1_pretrain_encoder":             "pretrain encoder",
        "vjepa_2_1_pretrain_2X_encoder":          "pretrain_2X encoder",
        "vjepa_2_1_surgical_3stage_DI_encoder":   "surgery 3stage_DI encoder",
        "vjepa_2_1_surgical_noDI_encoder":        "surgery noDI encoder",
        # iter15 HEAD-only variants (frozen encoder, motion_aux head trains)
        "vjepa_2_1_pretrain_head":                "pretrain head",
        "vjepa_2_1_surgical_3stage_DI_head":      "surgery 3stage_DI head",
        "vjepa_2_1_surgical_noDI_head":           "surgery noDI head",
    }.get(enc, enc.replace("_", " "))


# Color rotation for unrecognized encoder names (future fine-tuning techniques).
_FALLBACK_COLOR_CYCLE = ("blue", "green", "orange", "purple", "red", "cyan", "gold")


def _color_for(enc: str, idx: int) -> str:
    """Pick a color for an encoder bar. Tries the canonical map first
    (utils.plots.ENCODER_COLORS understands vjepa/vjepa_*/dinov2/clip/...),
    falls back to a deterministic rotation indexed by display-order position.
    """
    # Direct hit (e.g. "dinov2") or vjepa-family hit (anything starting with "vjepa")
    if enc in ENCODER_COLORS:
        return ENCODER_COLORS[enc]
    if enc.startswith("vjepa"):
        return ENCODER_COLORS["vjepa"]
    fallback_key = _FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)]
    return COLORS.get(fallback_key, COLORS["gray"])


# ── Loaders ──────────────────────────────────────────────────────────

def _load_train_logs(action_probe_root: Path, encoder: str):
    """Return [{label, records[]}] — one entry per LR (or one if no sweep).

    Layout #1 (no sweep):  <action_probe_root>/<encoder>/train_log.jsonl
    Layout #2 (LR sweep):  <action_probe_root>/<encoder>/lr_*/train_log.jsonl
    """
    enc_dir = action_probe_root / encoder
    runs = []
    flat = enc_dir / "train_log.jsonl"
    if flat.exists():
        runs.append({"label": "single LR", "path": flat})
    for lr_dir in sorted(enc_dir.glob("lr_*")):
        p = lr_dir / "train_log.jsonl"
        if p.exists():
            runs.append({"label": lr_dir.name.replace("lr_", "lr="), "path": p})
    out = []
    for r in runs:
        records = []
        for line in r["path"].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        if records:
            out.append({"label": r["label"], "records": records, "path": str(r["path"])})
    return out


def _load_json(path: Path, stage_hint: str) -> dict:
    if not path.exists():
        sys.exit(f"FATAL: {path} not found — run {stage_hint} first")
    return json.loads(path.read_text())


# ── Plot 1: probe loss curves (auto-discovers encoders by directory presence) ─

def plot_loss_curves(action_probe_root: Path, encoders: list, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew = False
    for idx, enc in enumerate(encoders):
        color = _color_for(enc, idx)
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            print(f"  [loss] no train_log.jsonl for {enc} — skipping")
            continue
        for i, run in enumerate(runs):
            recs = run["records"]
            x = [r["step"] for r in recs]
            y = [r["train_loss"] for r in recs]
            linestyle = "-" if i == 0 else "--"
            label = _display_label(enc) if len(runs) == 1 \
                else f"{_display_label(enc)} ({run['label']})"
            ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2.5, label=label)
            drew = True
    if not drew:
        plt.close(fig)
        print("  [loss] no curves to plot — skipping figure")
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Train cross-entropy loss")
    ax.set_title("probe Action Probe — Training Loss")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    save_fig(fig, str(output_dir / "probe_action_loss"))


# ── Plot 2: probe accuracy curves ────────────────────────────────────

def plot_acc_curves(action_probe_root: Path, encoders: list, n_classes: int,
                    output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew = False
    for idx, enc in enumerate(encoders):
        color = _color_for(enc, idx)
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            continue
        for run in runs:
            recs = run["records"]
            x = [r["step"] for r in recs]
            va = [r["val_acc"]   for r in recs]
            sweep = "" if len(runs) == 1 else f" ({run['label']})"
            ax.plot(x, va, linestyle="-", color=color, linewidth=2.6,
                    label=f"{_display_label(enc)} val{sweep}")
            drew = True
    if not drew:
        plt.close(fig)
        print("  [acc] no curves to plot — skipping figure")
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0 / n_classes, color="black", linestyle=":", linewidth=1.0,
               label=f"chance ({n_classes}-class)")
    ax.set_title("probe Action Probe — Validation Top-1 Accuracy")
    ax.legend(loc="lower right", fontsize=10)
    save_fig(fig, str(output_dir / "probe_action_acc"))


# ── Plot 3: 3-panel encoder comparison (N-bar generic) ───────────────

def _bar_with_ci(ax, encoders: list, vals: list, errs: list,
                 ylabel: str, title: str, na_set: set = None,
                 direction: str = ""):
    """Render N bars with 95% CI error caps + value labels above each bar.

    Args:
      direction: "higher" → annotate "↑ higher = better" badge (green).
                 "lower"  → annotate "↓ lower = better" badge (orange).
                 ""       → omit (default).
    `na_set` = encoder names with no measurement → render hatched 'N/A'.
    """
    na_set = na_set or set()
    x = np.arange(len(encoders))
    colors = [_color_for(e, i) for i, e in enumerate(encoders)]
    plot_vals = [0.0 if e in na_set else v for e, v in zip(encoders, vals)]
    plot_errs = [0.0 if e in na_set else er for e, er in zip(encoders, errs)]
    bars = ax.bar(x, plot_vals, 0.6, color=colors, alpha=0.85,
                  yerr=plot_errs, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})
    for i, e in enumerate(encoders):
        if e in na_set:
            bars[i].set_hatch("//")
            bars[i].set_alpha(0.25)
    real_v = np.array([v for e, v in zip(encoders, plot_vals) if e not in na_set])
    real_e = np.array([er for e, er in zip(encoders, plot_errs) if e not in na_set])
    if real_v.size:
        # NaN-safe ylim: a degenerate BCa CI (perfect predictions on tiny test
        # sets — e.g. SANITY's N=22 with V-JEPA hitting 22/22) returns ci_half
        # = NaN, which propagates through .min()/.max() and makes
        # ax.set_ylim(NaN, NaN) raise ValueError. Substitute NaN errors with
        # 0 (renders as "no error bar" rather than crashing) BEFORE computing
        # ylim. matplotlib handles NaN in yerr itself fine — the bars draw
        # without error caps. Only the explicit set_ylim call needs sanitizing.
        real_e_safe = np.nan_to_num(real_e, nan=0.0)
        lo = max(0.0, float((real_v - real_e_safe).min()))
        hi = float((real_v + real_e_safe).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1.0
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    else:
        pad = 0.05
    y_lo, y_hi = ax.get_ylim()
    for i, (xi, e, v, er) in enumerate(zip(x, encoders, plot_vals, plot_errs)):
        if e in na_set:
            ax.text(xi, y_lo + (y_hi - y_lo) * 0.5, "N/A", ha="center", va="center",
                    fontsize=12, color="#555", fontweight="bold")
        else:
            # NaN-safe value-label placement — degenerate-CI bars (NaN er) would
            # otherwise place text at v+NaN+... = NaN → matplotlib silently drops
            # the label. Substitute 0 so the label sits just above the bar top.
            er_safe = 0.0 if (isinstance(er, float) and np.isnan(er)) else er
            ax.text(xi, v + er_safe + pad * 0.1, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_xticks(x)
    # iter15 Phase 8 (2026-05-17): rotated 25° instead of multi-line wrap — keeps
    # vertical footprint to ~1 line height so the per-axes caption below doesn't
    # collide with the tick labels. ha="right" anchors the label end at the tick
    # for visual alignment with the bar.
    ax.set_xticklabels([_display_label(e) for e in encoders],
                       fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    # Direction badge (top-left of axes) — at-a-glance "tall bar = good or bad".
    if direction == "higher":
        ax.text(0.02, 0.97, "↑ higher = better",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="#2E7D32", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                          edgecolor="#2E7D32", linewidth=1.0, alpha=0.85))
    elif direction == "lower":
        ax.text(0.02, 0.97, "↓ lower = better",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="#E65100", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                          edgecolor="#E65100", linewidth=1.0, alpha=0.85))


def _sort_by_metric(encoders: list, vals: list, errs: list,
                    na_set: set, direction: str):
    """Sort (encoder, val, err) triples by val; N/A entries always pushed to the tail.

    direction = "desc" → descending (higher=better metrics: top1, motion_cos)
    direction = "asc"  → ascending  (lower=better metrics: future_mse)
    NaN-safe via np.isnan check; NaN values are treated as N/A (sent to tail).
    """
    triples = list(zip(encoders, vals, errs))
    real = [(e, v, er) for e, v, er in triples
            if e not in na_set and not (isinstance(v, float) and np.isnan(v))]
    na = [(e, v, er) for e, v, er in triples
          if e in na_set or (isinstance(v, float) and np.isnan(v))]
    real.sort(key=lambda t: t[1], reverse=(direction == "desc"))
    out = real + na  # N/A always last regardless of direction
    return [t[0] for t in out], [t[1] for t in out], [t[2] for t in out]


def plot_encoder_comparison(paired_delta: dict, motion_cos: dict,
                            future_mse: dict, output_dir: Path):
    """Emit 3 SEPARATE comparison figures (one per metric), bars sorted in-panel.

    iter15 Phase 8 (2026-05-17): split the prior 3-subplot stacked figure into
    3 individual files so each can be embedded standalone in slides/paper.
    Bars within each figure are sorted by the metric value:
      probe_action_acc_compare.png   — DESCENDING by top1 (higher=better)
      probe_motion_cos_compare.png   — DESCENDING by score_mean (higher=better)
      probe_future_mse_compare.png   — ASCENDING  by mse_mean  (lower=better)
    N/A bars always tail-position regardless of direction.
    """
    # Encoder set = union of all encoders that appear in any of the 3 JSONs'
    # by_encoder/by_variant blocks. The same encoder name may appear with
    # `null` in future_mse.by_variant if its trainer didn't run — we treat
    # that as N/A in the MSE panel.
    ap_enc = set(paired_delta.get("by_encoder", {}).keys())
    mc_enc = set(motion_cos.get("by_encoder", {}).keys())
    bv = future_mse.get("by_variant", {})
    fm_enc = {k for k, v in bv.items() if isinstance(v, dict)}    # only real entries
    encoders = sorted(ap_enc | mc_enc | fm_enc)
    if not encoders:
        sys.exit("FATAL: no encoders found in any paired-Δ JSON")

    if N_BOOTSTRAP >= 1000:
        boot_str = f"{N_BOOTSTRAP // 1000} K bootstrap"
    else:
        boot_str = f"{N_BOOTSTRAP} bootstrap"

    # Shared figure dimensions per metric — width scales with encoder count.
    # iter15 Phase 8 (2026-05-17): bumped height 6.5 → 8.5 + caption-y −0.27
    # → −0.55 + tight_layout bottom rect 0.18 → 0.32 so the rotated xtick
    # labels (25° angle, full encoder names) clear the caption box without
    # overlap. Reduced top rect 0.94 → 0.93 to compensate.
    def _fig_size(n_enc: int):
        return (max(8, 1.4 * n_enc + 2), 8.5)

    def _emit_one(out_name: str, sorted_enc, sorted_vals, sorted_errs,
                  na_set: set, ylabel: str, title: str, direction_badge: str,
                  caption: str):
        fig, ax = plt.subplots(figsize=_fig_size(len(encoders)))
        _bar_with_ci(ax, sorted_enc, sorted_vals, sorted_errs,
                     ylabel=ylabel, title=title,
                     na_set=na_set, direction=direction_badge)
        ax.text(0.5, -0.55, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#000", fontweight="medium",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA",
                          edgecolor="#888", linewidth=0.8))
        fig.suptitle(f"{title} · {len(encoders)} encoders · 95 % BCa CI",
                     fontsize=14, fontweight="bold", y=0.97)
        fig.tight_layout(rect=[0, 0.32, 1, 0.93])
        save_fig(fig, str(output_dir / out_name))

    # ─── Figure 1: probe_action top-1 (DESCENDING — higher = better) ───
    ap_be = paired_delta.get("by_encoder", {})
    acc_vals = [ap_be.get(e, {}).get("acc_pct", 0.0) for e in encoders]
    acc_errs = [ap_be.get(e, {}).get("top1_ci", {}).get("ci_half", 0.0) * 100.0
                for e in encoders]
    ap_na = {e for e in encoders if e not in ap_be}
    enc_acc, acc_vals, acc_errs = _sort_by_metric(
        encoders, acc_vals, acc_errs, ap_na, direction="desc"
    )
    _n_test = next(iter(ap_be.values()), {}).get("n", "?")
    _emit_one(
        "probe_action_acc_compare",
        enc_acc, acc_vals, acc_errs, ap_na,
        ylabel="Top-1 accuracy (%) — higher is better",
        title=f"Action probe top-1 (n={_n_test}) — sorted ↓ highest first",
        direction_badge="higher",
        caption=("AttentiveClassifier on encoder features →\n"
                 "K-class motion-flow accuracy (magnitude × direction).\n"
                 "Tests linear separability — are motion classes encoded?\n"
                 f"TEST split;  whiskers = BCa 95 % CI from {boot_str}."),
    )

    # ─── Figure 2: motion cosine intra−inter (DESCENDING — higher = better) ───
    mc_be = motion_cos.get("by_encoder", {})
    cos_vals = [mc_be.get(e, {}).get("score_mean", 0.0) for e in encoders]
    cos_errs = [mc_be.get(e, {}).get("score_ci", {}).get("ci_half", 0.0)
                for e in encoders]
    mc_na = {e for e in encoders if e not in mc_be}
    enc_cos, cos_vals, cos_errs = _sort_by_metric(
        encoders, cos_vals, cos_errs, mc_na, direction="desc"
    )
    _n_cos = next(iter(mc_be.values()), {}).get("n", "?")
    _emit_one(
        "probe_motion_cos_compare",
        enc_cos, cos_vals, cos_errs, mc_na,
        ylabel="Intra − Inter cosine — higher is better",
        title=f"Motion cosine (n={_n_cos}) — sorted ↓ highest first",
        direction_badge="higher",
        caption=("Same-class minus different-class cosine separation.\n"
                 "> 0 ⇒ encoder clusters clips semantically by motion;\n"
                 "near 0 ⇒ no clustering.\n"
                 f"TEST split;  whiskers = BCa 95 % CI from {boot_str}."),
    )

    # ─── Figure 3: future-frame L1 (ASCENDING — lower = better) ───
    mse_vals, mse_errs = [], []
    fm_na = set()
    n_test_mse = "?"
    for e in encoders:
        entry = bv.get(e)
        if isinstance(entry, dict):
            mse_vals.append(entry["mse_mean"])
            mse_errs.append(entry["mse_ci"]["ci_half"])
            n_test_mse = entry["n"]
        else:
            mse_vals.append(0.0)
            mse_errs.append(0.0)
            fm_na.add(e)
    enc_mse, mse_vals, mse_errs = _sort_by_metric(
        encoders, mse_vals, mse_errs, fm_na, direction="asc"
    )
    _emit_one(
        "probe_future_mse_compare",
        enc_mse, mse_vals, mse_errs, fm_na,
        ylabel="Future L1 — lower is better",
        title=f"Future-frame MSE (n={n_test_mse}) — sorted ↑ lowest first",
        direction_badge="lower",
        caption=("V-JEPA predictor L1 on masked next-frame tokens —\n"
                 "the JEPA objective itself.\n"
                 "Tests temporal predictive capability.\n"
                 "V-JEPA-only (no predictor for DINOv2 / CLIP → N/A)."),
    )


# ── Training-side N-run comparison (iter15 Phase 8, 2026-05-17) ──────
#
# Reads <training_root>/m09*/{loss_log.csv, probe_history.jsonl,
# *_block_drift_history.json, training_summary.json} and emits 1 PNG per
# metric × N lines (one per training run). Style: 7 distinct colors + family
# linestyle (encoder=solid, head=dashed). X-axis: dual (bottom = % of training
# 0-100%, top = raw step) per user preference. PNG-only (no PDF/SVG).

TRAINING_RUNS_CANONICAL = [
    # (subdir_name,                       display_label,             family,    color)
    ("m09a_pretrain_encoder",              "pretrain_encoder (2ep)",    "encoder", "tab:blue"),
    ("m09a_pretrain_2X_encoder",           "pretrain_2X_encoder (4ep)", "encoder", "tab:cyan"),
    ("m09c_surgery_3stage_DI_encoder",     "surgery_3stage_DI_enc",     "encoder", "tab:red"),
    ("m09c_surgery_noDI_encoder",          "surgery_noDI_encoder",      "encoder", "tab:orange"),
    ("m09a_pretrain_head",                 "pretrain_head",             "head",    "tab:purple"),
    ("m09c_surgery_3stage_DI_head",        "surgery_3stage_DI_head",    "head",    "tab:green"),
    ("m09c_surgery_noDI_head",             "surgery_noDI_head",         "head",    "tab:brown"),
]
TRAINING_LINESTYLE_BY_FAMILY = {"encoder": "-", "head": "--"}


def _read_csv_typed(path: Path):
    """csv.DictReader → list-of-dicts; numeric strings cast to float; empty → NaN."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            out = {}
            for k, v in row.items():
                if v is None or v == "":
                    out[k] = float("nan")
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def _col(rows: list, col: str):
    """Pair (step, col-value) from list-of-dicts as (steps, values) np arrays; NaN-filtered."""
    steps, vals = [], []
    for r in rows:
        v = r.get(col)
        if not isinstance(v, (int, float)) or (isinstance(v, float) and np.isnan(v)):
            continue
        s = r.get("step")
        if isinstance(s, (int, float)) and not np.isnan(s):
            steps.append(float(s))
            vals.append(float(v))
    return np.array(steps), np.array(vals)


def _probe_col(probe_rows: list, col: str):
    """Probe-history column — prefers global_step over step (m09c1 stage-keyed)."""
    steps, vals = [], []
    for r in probe_rows:
        v = r.get(col)
        if not isinstance(v, (int, float)) or (isinstance(v, float) and np.isnan(v)):
            continue
        s = r.get("global_step", r.get("step"))
        if isinstance(s, (int, float)) and not np.isnan(s):
            steps.append(float(s))
            vals.append(float(v))
    return np.array(steps), np.array(vals)


def _derive_loss_total_m09c1(loss_rows: list):
    """m09c1 loss_log lacks loss_total → Σ(jepa+masked+context+infonce+tcc) per row."""
    cols = ("loss_jepa", "loss_masked", "loss_context", "loss_infonce", "loss_tcc")
    steps, vals = [], []
    for r in loss_rows:
        comps = [r.get(c) for c in cols]
        comps = [c for c in comps
                 if isinstance(c, (int, float)) and not (isinstance(c, float) and np.isnan(c))]
        if not comps:
            continue
        s = r.get("step")
        if isinstance(s, (int, float)) and not np.isnan(s):
            steps.append(float(s))
            vals.append(float(sum(comps)))
    return np.array(steps), np.array(vals)


def _drift_mean_series(drift_list: list):
    """list of {step, rel_l2_per_block} → (steps, mean_rel_l2_across_blocks)."""
    if not drift_list:
        return np.array([]), np.array([])
    steps = np.array([d["step"] for d in drift_list], dtype=np.float64)
    means = np.array([float(np.mean(d["rel_l2_per_block"])) for d in drift_list],
                     dtype=np.float64)
    return steps, means


def _load_training_run(subdir: Path) -> dict:
    """Load loss_log.csv + probe_history.jsonl + *block_drift_history.json + training_summary.json
    from a single training output subdir. FAIL LOUD if any required file is missing.
    """
    if not subdir.is_dir():
        sys.exit(f"FATAL: training subdir missing: {subdir}")
    loss_csv = subdir / "loss_log.csv"
    if not loss_csv.is_file():
        sys.exit(f"FATAL: {loss_csv} missing")
    loss_rows = _read_csv_typed(loss_csv)

    probe_jsonl = subdir / "probe_history.jsonl"
    if not probe_jsonl.is_file():
        sys.exit(f"FATAL: {probe_jsonl} missing")
    probe_rows = [json.loads(line) for line in probe_jsonl.read_text().splitlines() if line.strip()]

    drift_files = list(subdir.glob("*block_drift_history.json"))
    if not drift_files:
        sys.exit(f"FATAL: no *block_drift_history.json in {subdir}")
    drift_list = json.loads(drift_files[0].read_text())

    summary_path = subdir / "training_summary.json"
    if not summary_path.is_file():
        sys.exit(f"FATAL: {summary_path} missing")
    summary = json.loads(summary_path.read_text())

    if "steps" in summary:
        total_steps = int(summary["steps"])
    elif "total_steps" in summary:
        total_steps = int(summary["total_steps"])
    else:
        max_step = 0
        for r in loss_rows:
            s = r.get("step")
            if isinstance(s, (int, float)) and not np.isnan(s):
                max_step = max(max_step, int(s))
        for r in probe_rows:
            s = r.get("global_step", r.get("step", 0))
            if isinstance(s, (int, float)):
                max_step = max(max_step, int(s))
        total_steps = max_step

    return {"loss_rows": loss_rows, "probe_rows": probe_rows,
            "drift_list": drift_list, "summary": summary, "total_steps": total_steps}


def _plot_training_metric(runs: dict, extractor, ylabel: str,
                          title: str, out_name: str, output_dir: Path,
                          y_log: bool = False):
    """Emit 1 figure with N lines (1 per canonical run) + dual x-axis (% bottom, step top)."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = 0
    max_total_steps = 0
    for subdir_name, display, family, color in TRAINING_RUNS_CANONICAL:
        rd = runs.get(subdir_name)
        if rd is None:
            continue
        steps_arr, values_arr, total_steps = extractor(rd)
        if steps_arr is None or len(steps_arr) == 0:
            continue
        max_total_steps = max(max_total_steps, total_steps)
        pct = 100.0 * steps_arr.astype(np.float64) / max(total_steps, 1)
        ax.plot(pct, values_arr,
                color=color,
                linestyle=TRAINING_LINESTYLE_BY_FAMILY[family],
                linewidth=1.8,
                marker="o", markersize=4,
                label=f"{display}  [{family}, n_steps={total_steps}]")
        plotted += 1
    if plotted == 0:
        print(f"  [SKIP] {out_name}: no run carries this metric")
        plt.close(fig)
        return
    ax.set_xlabel("% of training (0 → 100)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.25, linestyle=":")
    if y_log:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
    if max_total_steps > 0:
        ax_top = ax.twiny()
        ax_top.set_xlim(0, max_total_steps)
        ax_top.set_xlabel(f"raw step (max across runs = {max_total_steps})",
                          fontsize=10, color="#555")
        ax_top.tick_params(axis="x", labelsize=8, colors="#555")
    fig.tight_layout()
    out_path = output_dir / f"{out_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} ({plotted}/{len(TRAINING_RUNS_CANONICAL)} lines)")


def run_training_side_compare(training_root: Path, output_dir: Path):
    """Generate 12 N-run training-side comparison PNGs in output_dir.
    Splits the per-run m09{a,c}_probe_trajectory_trio.png into 3 separate plots
    (probe_top1 / motion_cos / future_l1) per user request.
    """
    runs = {}
    for subdir_name, _, _, _ in TRAINING_RUNS_CANONICAL:
        sub = training_root / subdir_name
        if sub.is_dir():
            runs[subdir_name] = _load_training_run(sub)
        else:
            print(f"  [missing] {sub} — skipping this run")
    if not runs:
        sys.exit(f"FATAL: no training subdirs found under {training_root}/m09*/")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _loss_col(col):
        def _x(rd):
            s, v = _col(rd["loss_rows"], col)
            return s, v, rd["total_steps"]
        return _x

    def _loss_total(rd):
        s, v = _col(rd["loss_rows"], "loss_total")
        if len(s) > 0:
            return s, v, rd["total_steps"]
        s, v = _derive_loss_total_m09c1(rd["loss_rows"])
        return s, v, rd["total_steps"]

    def _probe(col):
        def _x(rd):
            s, v = _probe_col(rd["probe_rows"], col)
            return s, v, rd["total_steps"]
        return _x

    def _drift(rd):
        s, v = _drift_mean_series(rd["drift_list"])
        return s, v, rd["total_steps"]

    plots = [
        ("train_loss_jepa_compare",     _loss_col("loss_jepa"),
         "loss_jepa",                    "Training — JEPA loss",                          False),
        ("train_loss_total_compare",    _loss_total,
         "loss_total (m09c1: Σ jepa+masked+context+infonce+tcc)",
                                         "Training — total loss",                         False),
        ("train_grad_norm_compare",     _loss_col("grad_norm"),
         "grad_norm",                    "Training — gradient norm",                      False),
        ("train_lr_compare",            _loss_col("lr"),
         "learning rate (log y)",        "Training — LR schedule",                        True),
        ("val_loss_jepa_compare",       _probe("val_jepa_loss"),
         "val_jepa_loss",                "Validation — JEPA loss",                        False),
        ("val_motion_aux_loss_compare", _probe("val_motion_aux_loss"),
         "val_motion_aux_loss",          "Validation — motion_aux (CE + MSE)",            False),
        ("val_total_loss_compare",      _probe("val_total_loss"),
         "val_total_loss",               "Validation — total loss",                       False),
        ("probe_top1_compare",          _probe("probe_top1"),
         "probe_top1 (kNN-LOOCV)",       "Probe-trio split 1/3 — probe_top1 (HIGHER better)",  False),
        ("motion_cos_compare",          _probe("motion_cos"),
         "motion_cos (intra − inter)",   "Probe-trio split 2/3 — motion_cos (HIGHER better)",  False),
        ("future_l1_compare",           _probe("future_l1"),
         "future_l1",                    "Probe-trio split 3/3 — future_l1 (LOWER better)",    False),
        ("block_drift_mean_compare",    _drift,
         "mean(rel_l2 across blocks)",   "Encoder weight drift — mean over blocks",       False),
        ("val_drift_loss_compare",      _probe("val_drift_loss"),
         "val_drift_loss",               "Validation — drift penalty (SPD)",              False),
    ]
    pbar = make_pbar(total=len(plots), desc="train_compare", unit="plot")
    for out_name, extractor, ylabel, title, y_log in plots:
        _plot_training_metric(runs, extractor, ylabel, title, out_name, output_dir, y_log=y_log)
        pbar.update(1)
    pbar.close()
    print(f"  Training-side compare plots written to: {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="m08d — visualize probe action_probe + motion_cos + future_mse outputs (N-encoder generic).")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    # iter15 Phase 8 (2026-05-17): --training-side splits the output_dir into
    # /eval and /train subdirs. Eval-side path (default) requires the 3 probe
    # roots; training-side requires --training-root. Body validation below
    # raises a clear FATAL if the required flags for the chosen mode are absent.
    p.add_argument("--training-side", action="store_true",
                   help="Generate N-run training-side comparison plots (12 PNGs) from "
                        "<training-root>/m09*/. Output goes to <output-dir>/train/.")
    p.add_argument("--training-root", type=Path, default=None,
                   help="Required when --training-side: dir containing m09*/ subdirs "
                        "(e.g. outputs/poc).")
    p.add_argument("--action-probe-root", type=Path, default=None)
    p.add_argument("--motion-cos-root",   type=Path, default=None)
    p.add_argument("--future-mse-root",   type=Path, default=None)
    p.add_argument("--output-dir",        type=Path, required=True)
    add_wandb_args(p)
    args = p.parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # Output-dir auto-suffix: --training-side → <output_dir>/train/, else /eval/.
    # Only wipe the active subdir, leaving the sibling (other-side) plots intact.
    sub_name = "train" if args.training_side else "eval"
    sub_dir = args.output_dir / sub_name
    if sub_dir.exists():
        n = sum(1 for _ in sub_dir.rglob("*") if _.is_file())
        print(f"  [probe_plot] wiping {sub_name}/ — {n} stale file(s)")
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = sub_dir

    # Mode-specific arg validation (FAIL LOUD per CLAUDE.md — no silent defaults).
    if args.training_side:
        if args.training_root is None:
            sys.exit("ERROR: --training-side requires --training-root (e.g. outputs/poc)")
    else:
        missing = [name for name, val in [
            ("--action-probe-root", args.action_probe_root),
            ("--motion-cos-root",   args.motion_cos_root),
            ("--future-mse-root",   args.future_mse_root),
        ] if val is None]
        if missing:
            sys.exit(f"ERROR: eval-side mode requires: {', '.join(missing)}")

    wb = init_wandb("probe_plot", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()

        # iter15 Phase 8 (2026-05-17): branch on --training-side. The training
        # path reads outputs/poc/m09*/{loss_log.csv, probe_history.jsonl, ...}
        # and emits 12 N-run comparison PNGs. The eval path below stays
        # unchanged — same 5 plots (loss, acc, 3 sorted-bar compares).
        if args.training_side:
            run_training_side_compare(args.training_root, args.output_dir)
            return

        paired_delta = _load_json(args.action_probe_root / "probe_paired_delta.json", "Stage 4")
        motion_cos   = _load_json(args.motion_cos_root   / "probe_motion_cos_paired.json", "Stage 7")
        future_mse   = _load_json(args.future_mse_root   / "probe_future_mse_per_variant.json", "Stage 9")

        # Encoder list for loss/acc curves: discover from action_probe by_encoder.
        ap_be = paired_delta.get("by_encoder", {})
        if not ap_be:
            sys.exit("FATAL: paired_delta has empty by_encoder — re-run Stage 4 with N-way schema")
        encoders = sorted(ap_be.keys())
        n_classes = len(next(iter(ap_be.values())).get("top1_ci", {})) and 3
        # n_classes is best-effort; chance line on acc plot. Default to 3 if unknown.
        if not isinstance(n_classes, int) or n_classes < 2:
            n_classes = 3

        pbar = make_pbar(total=3, desc="probe_plot", unit="plot")
        plot_loss_curves(args.action_probe_root, encoders, args.output_dir);              pbar.update(1)
        plot_acc_curves(args.action_probe_root, encoders, n_classes, args.output_dir);    pbar.update(1)
        plot_encoder_comparison(paired_delta, motion_cos, future_mse, args.output_dir);   pbar.update(1)
        pbar.close()

        # wandb metric upload — generic prefix + encoder name (NO hardcoded vjepa/dinov2 keys).
        wb_metrics = {"n_encoders": len(encoders)}
        for e, v in ap_be.items():
            wb_metrics[f"acc_pct__{e}"] = float(v.get("acc_pct", 0.0))
        for e, v in motion_cos.get("by_encoder", {}).items():
            wb_metrics[f"motion_cos__{e}"] = float(v.get("score_mean", 0.0))
        for e, v in future_mse.get("by_variant", {}).items():
            if isinstance(v, dict):
                wb_metrics[f"future_mse_mean__{e}"] = float(v.get("mse_mean", 0.0))
        log_metrics(wb, wb_metrics)
        print(f"  Plots written to: {args.output_dir}")
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
