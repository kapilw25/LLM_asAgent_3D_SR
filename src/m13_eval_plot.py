"""m13 — N-encoder visualization for the FULL probe eval suite (VISUALIZATION band, LAST).
Renamed from probe_plot.py (was m08d) — strict pipeline direction: visualization is the LAST band.

CPU-only. Pure-visualization — ALWAYS recomputes; no cache_policy.

iter16 §3.2/§3.3/§7 (2026-05-28): ALL LINE PLOTS REMOVED (§3.3a). Emits ONLY bar-with-CI panels
for ALL 14 metrics (§3.3c) + (pending) the HERO (table-with-CI + plot-with-CI). Every number
carries its BCa 95% CI. The 14 metrics (plan §7.1):
  HEAD (4): action_top1 · motion_cos · taxonomy_f1 · future_mse
  PRED (6): rollout · causal · tdist · teacher_free · maskratio · order        (m12e)
  ENC  (4): aot · tov · pace · tcc(τ)   [+ tcc_cycle appendix]                 (m12f)

Reads (auto-discovers encoders from each JSON; graceful — absent source → that metric skipped):
  <action-probe-root>/probe_paired_delta.json          {by_encoder}            → action_top1
  <motion-cos-root>/probe_motion_cos_paired.json        {by_encoder}            → motion_cos
  <future-mse-root>/probe_future_mse_per_variant.json   {by_variant}            → future_mse
  <taxonomy-root>/per_dim_acc.json                      {dims[*].by_encoder}    → taxonomy_f1
  <predictor-temporal-root>/predictor_temporal_per_variant.json {metrics}       → 6 PRED metrics
  <encoder-temporal-root>/encoder_temporal_per_variant.json     {metrics}       → 4 ENC metrics

Writes (under <output-dir>/eval/{head,predictor,encoder}/):
  one bar-with-CI panel per available metric (legacy names kept for the 3 v15a headline panels:
  probe_action_acc_compare / probe_motion_cos_compare / probe_future_mse_compare; the rest m13_*).
  (§3.3c-hero will add m13_hero_table + m13_hero_raw_values.)

USAGE:
  python -u src/m13_eval_plot.py --SANITY \\
    --action-probe-root outputs/sanity/probe_action \\
    --motion-cos-root   outputs/sanity/probe_motion_cos \\
    --future-mse-root   outputs/sanity/probe_future_mse \\
    --taxonomy-root     outputs/sanity/probe_taxonomy \\
    --predictor-temporal-root outputs/sanity/predictor_temporal \\
    --encoder-temporal-root   outputs/sanity/encoder_temporal \\
    --output-dir        outputs/sanity/probe_plot \\
    2>&1 | tee logs/m13_eval_plot_sanity.log
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from utils.plots import COLORS, ENCODER_COLORS, init_style, save_fig
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics
from utils.bootstrap import N_BOOTSTRAP
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants.
_AXIS_DECIMAL_MIN = 0.01   # tick-format band edges (display only)
_AXIS_INT_MIN = 10
_FONT_MIN_PT = 28          # hero-figure auto-shrink floor
_K_FMT_MIN = 1000          # render bootstrap count as 'N K' above this
_MIN_COMPARABLE = 2


# ── Display helpers (no hardcoded encoder list — derived per-call) ───

def _canon(enc: str) -> str:
    """The iter16 champion (V-JEPA 2.1 ViT-G) was evaluated under the LEGACY name vjepa_2_1_<arm>;
    rewrite it to vjepa_2_1_vitG_<arm> so §G labels it consistently with this run's vjepa_2_1_vitg /
    vjepa_2_0_vitg. Anything else (vjepa_2_1_vitg/_vitL, vjepa_2_0_*, frozen baselines) falls straight
    through unchanged — DISPLAY ONLY, never used to key into metric data."""
    if enc.startswith("vjepa_2_1_") and not enc.startswith("vjepa_2_1_vit"):
        return "vjepa_2_1_vitG_" + enc[len("vjepa_2_1_"):]
    return enc


# (backbone prefix → long tag, short tag) for the 3 trained backbones; everything else (frozen-9) falls through.
_BB_TAG = (("vjepa_2_1_vitG_", "ViT-G", "G"), ("vjepa_2_1_vitg_", "ViT-g", "g"), ("vjepa_2_0_vitg_", "ViT-g·2.0", "g0"))
_ARM_LONG = {
    "frozen": "frozen", "pretrain_encoder": "vanilla continual-SSL", "pretrain_2X_encoder": "vanilla continual-SSL 2x",
    "pretrain_head": "vanilla continual-SSL (head)", "surgical_3stage_DI_encoder": "surgery 3stage_DI encoder",
    "surgical_noDI_encoder": "surgery noDI encoder", "surgical_3stage_DI_head": "surgery 3stage_DI head",
    "surgical_noDI_head": "surgery noDI head",
}
_ARM_SHORT = {
    "frozen": "frozen", "pretrain_encoder": "vCSSL-enc", "pretrain_2X_encoder": "vCSSL-2X", "pretrain_head": "vCSSL-hd",
    "surgical_3stage_DI_encoder": "s3DI-enc", "surgical_noDI_encoder": "sNoDI-enc",
    "surgical_3stage_DI_head": "s3DI-hd", "surgical_noDI_head": "sNoDI-hd",
    # iter18 FT baselines (encoder-only; no head twin → no -enc suffix needed) — clean distinct WINNER tags
    "surgery_raw_encoder": "raw", "surgical_autorgn_encoder": "argn",
    "full_ft_encoder": "fullFT", "lpft_encoder": "LPFT",
    "peft_lora_encoder": "LoRA", "peft_dora_encoder": "DoRA",
    "cassle_encoder": "CaSSLe", "ewc_encoder": "EWC",
}
# Per-backbone header label (name + model size) for the stacked §G overview. Architecture facts are
# pinned in configs/model/*.yaml (embed_dim/depth VERIFIED there); mirrored here as the plot caption.
_BB_LABEL = {
    "vjepa_2_1_vitG": "V-JEPA 2.1   ·   ViT-G   ·   ~2B params   ·   1664-dim, 48 blocks   (CHAMPION)",
    "vjepa_2_1_vitg": "V-JEPA 2.1   ·   ViT-g   ·   ~1B params   ·   1408-dim, 40 blocks",
    "vjepa_2_0_vitg": "V-JEPA 2.0   ·   ViT-g   ·   ~1B params   ·   1408-dim, 40 blocks",
}


def _display_label(enc: str) -> str:
    """Human-readable encoder name for plot legends. All 3 trained backbones → '<ViT-tag> <arm>';
    frozen-9 baselines fall through verbatim."""
    enc = _canon(enc)
    for pre, long_tag, _ in _BB_TAG:
        if enc.startswith(pre):
            arm = enc[len(pre):]
            return f"{long_tag} {_ARM_LONG.get(arm, arm.replace('_', ' '))}"
    return {"dinov2": "DINOv2 frozen"}.get(enc, enc.replace("_", " "))


_FALLBACK_COLOR_CYCLE = ("blue", "green", "orange", "purple", "red", "cyan", "gold")


# iter18 2026-06-08: the 4 OURS (surgery novelty) share GREEN so they read as ONE group in every
# bar — but stay 4 separately-labelled bars (NOT merged) so you can see WHICH ours wins by height.
_OURS_GREEN = {"surgical_3stage_DI_encoder", "surgical_noDI_encoder",
               "surgical_3stage_DI_head", "surgical_noDI_head"}
# every OTHER arm gets its OWN distinct colour, keyed by NAME (stable across all metric panels —
# the old code keyed off enc.startswith("vjepa") so ALL 14 encoders rendered in one identical blue).
_ITER18_ENC_COLOR = {
    "frozen":                   "#616161",  # gray   — frozen baseline
    "pretrain_encoder":         "#1565C0",  # blue   — vanilla cont-SSL anchor
    "surgical_autorgn_encoder": "#E65100",  # orange — Auto-RGN (Surgical-FT baseline)
    "surgery_raw_encoder":      "#8D6E63",  # brown  — surgery-on-raw control
    "full_ft_encoder":          "#C62828",  # red    — Full-FT
    "lpft_encoder":             "#D81B60",  # magenta— LP-FT
    "peft_lora_encoder":        "#6A1B9A",  # purple — LoRA
    "peft_dora_encoder":        "#00838F",  # cyan   — DoRA
    "cassle_encoder":           "#F9A825",  # gold   — CaSSLe
    "ewc_encoder":              "#827717",  # olive  — EWC
}
_SURG_GREEN = "#2E7D32"   # the OURS-group green, reused to flag surgery in the WINNER row
# iter18 2026-06-08: WINNER-row render style for the hero heatmap. At POC the per-metric 95% CIs are
# wide enough that a surgery arm is a CO-LEADER (CI overlaps the point-best) on EVERY metric, so a
# single point-best name under-sells it. Three honest styles, selected by env so all three can be
# rendered side-by-side for the user to pick (none invents significance — every style keeps the "~"/
# tie marker and boxes ALL CI-tied co-leaders):
#   coleader_set     — list ALL arms tied for #1 (surgery bold-green), competitors shown too
#   surgery_coleader — name the best SURGERY arm (green); "~" prefix when it's a CI-tie
#   tie_badge        — "~ tie (n)" / "<arm> WIN (sole)"; names nobody, boxes carry the co-leaders
_WINNER_MODE = os.environ.get("HERO_WINNER_MODE", "coleader_set")


def _color_for(enc: str, idx: int) -> str:
    """Per-encoder colour: the 4 OURS → one GREEN; every other arm → its own distinct colour keyed
    by name (stable across panels). Legacy/cross-arch encoders fall back to the canonical map."""
    short = enc.replace("vjepa_2_1_", "")
    if short in _OURS_GREEN:
        return "#2E7D32"                                  # GREEN — the OURS group
    if short in _ITER18_ENC_COLOR:
        return _ITER18_ENC_COLOR[short]
    if enc in ENCODER_COLORS:
        return ENCODER_COLORS[enc]
    return COLORS.get(_FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)], COLORS["gray"])


# ── Loaders ──────────────────────────────────────────────────────────

def _load_json(path: Path, stage_hint: str) -> dict:
    """REQUIRED JSON — FAIL LOUD if missing (CLAUDE.md: no silent defaults)."""
    if not path.exists():
        sys.exit(f"FATAL: {path} not found — run {stage_hint} first")
    return json.loads(path.read_text())


def _opt_json(path: Path):
    """OPTIONAL JSON — returns None (graceful) if absent. For the newer temporal/taxonomy
    sources that a SKIP_STAGES run may not have produced; m13 degrades to fewer metrics."""
    return json.loads(path.read_text()) if path.exists() else None


# ── Bar-with-CI primitive (N-bar generic) ───────────────────────────

def _bar_with_ci(ax, encoders: list, vals: list, errs: list,
                 ylabel: str, title: str, na_set: set = None, direction: str = ""):
    """Render N bars with 95% CI error caps + value labels. `direction`: higher/lower → badge;
    "" → none. `na_set` = encoders with no measurement → hatched 'N/A'."""
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
        real_e_safe = np.nan_to_num(real_e, nan=0.0)
        lo = float((real_v - real_e_safe).min())
        hi = float((real_v + real_e_safe).max())
        span = hi - lo
        pad = max(0.15 * span, 0.02 * abs(hi)) if span > 0 else (abs(hi) * 0.1 or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        pad = 0.05
    y_lo, y_hi = ax.get_ylim()
    for xi, e, v, er in zip(x, encoders, plot_vals, plot_errs):
        if e in na_set:
            ax.text(xi, y_lo + (y_hi - y_lo) * 0.5, "N/A", ha="center", va="center",
                    fontsize=12, color="#555", fontweight="bold")
        else:
            er_safe = 0.0 if (isinstance(er, float) and np.isnan(er)) else er
            ax.text(xi, v + er_safe + (y_hi - y_lo) * 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_label(e) for e in encoders], fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if direction == "higher":
        ax.text(0.02, 0.97, "↑ higher = better", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color="#2E7D32", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                          edgecolor="#2E7D32", linewidth=1.0, alpha=0.85))
    elif direction == "lower":
        ax.text(0.02, 0.97, "↓ lower = better", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color="#E65100", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                          edgecolor="#E65100", linewidth=1.0, alpha=0.85))


def _sort_by_metric(encoders: list, vals: list, errs: list, na_set: set, direction: str):
    """Sort (encoder, val, err) by val; N/A always tail. direction 'desc' (higher=better) /
    'asc' (lower=better). NaN treated as N/A."""
    triples = list(zip(encoders, vals, errs))
    real = [(e, v, er) for e, v, er in triples
            if e not in na_set and not (isinstance(v, float) and np.isnan(v))]
    na = [(e, v, er) for e, v, er in triples
          if e in na_set or (isinstance(v, float) and np.isnan(v))]
    real.sort(key=lambda t: t[1], reverse=(direction == "desc"))
    out = real + na
    return [t[0] for t in out], [t[1] for t in out], [t[2] for t in out]


def _emit_bar(out_path: Path, sorted_enc, sorted_vals, sorted_errs, na_set,
              ylabel, title, badge, caption, layman, n_enc, boot_str):
    """One bar-with-CI figure (png+pdf). Module-level (was a closure) so the 14-metric loop
    and the legacy 3 headline panels share one code path.

    iter16 fix: the caption lives in FIGURE coords at the bottom (not ax.text at a large
    negative axes-fraction, which fed back into tight_layout and collapsed the axes into a
    thin strip). Fixed margins via subplots_adjust give a TALL plot area (~1.5:1), matching
    the iter15 reference. Caption now carries a plain-language `Layman:` example."""
    fig, ax = plt.subplots(figsize=(max(8.5, 0.95 * n_enc + 3.0), 9.5))
    _bar_with_ci(ax, sorted_enc, sorted_vals, sorted_errs,
                 ylabel=ylabel, title=title, na_set=na_set, direction=badge)
    cap = (f"{caption}\nLayman: e.g. {layman}\n"
           f"TEST split · whiskers = BCa 95 % CI from {boot_str}.")
    fig.text(0.5, 0.03, cap, ha="center", va="bottom", fontsize=10, color="#000",
             linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#888", linewidth=0.8))
    fig.suptitle(f"{title} · {n_enc} encoders · 95 % BCa CI", fontsize=14, fontweight="bold", y=0.985)
    # Fixed margins: TALL axes (top-bottom = 0.59 of a 9.5in fig ≈ 5.6in) + room for rotated
    # xticklabels (bottom 0.33) and the caption box (figure y≈0.03).
    fig.subplots_adjust(left=0.11, right=0.96, top=0.92, bottom=0.33)
    save_fig(fig, str(out_path))


# ── §3.3c: 14-metric catalog + unified loader + per-metric bars (plan §7) ──
# (key, family, out_name, direction, ylabel, caption, layman). direction: higher|lower|signed.
# `layman` = one-line plain-language example shown in every panel's caption box (iter16 fix).
# Legacy names kept for the 3 v15a headline panels; the rest take m13_ names. Mirrors §7.1.
_CATALOG = [
    ("action_top1",  "HEAD", "probe_action_acc_compare", "higher", "Top-1 accuracy (%)",
     "AttentiveClassifier → K-class motion-flow accuracy (magnitude × direction).",
     "from the clip's features alone, can a simple classifier tell 'fast leftward motion' from 'slow upward motion'?"),
    ("motion_cos",   "HEAD", "probe_motion_cos_compare", "higher", "Intra − Inter cosine",
     "Same-class minus different-class cosine separation (>0 ⇒ motion-semantic clustering).",
     "do two clips of the SAME motion sit closer in feature space than two clips of DIFFERENT motions?"),
    ("taxonomy_f1",  "HEAD", "m13_taxonomy_f1_compare",  "higher", "Taxonomy mean (top1+F1)",
     "Mean over 15 VLM-tag dims of per-dim top-1 (single) / sample-F1 (multi).",
     "can a linear read-off name scene attributes — crowd density, lighting, camera motion — from the features?"),
    ("future_mse",   "HEAD", "probe_future_mse_compare", "lower",  "Future L1",
     "V-JEPA predictor L1 on masked next-frame tokens (the JEPA objective; V-JEPA-only).",
     "given the first frames, how well does the model's predictor guess the next frame's content?"),
    ("rollout",      "PRED", "m13_rollout_compare",      "lower",  "Free-run drift slope",
     "Iterated multi-step rollout L1 slope — lower = stabler free-running dynamics.",
     "if the model predicts step-by-step into the future, how fast does its error snowball?"),
    ("causal",       "PRED", "m13_causal_compare",       "lower",  "Causal future-block L1",
     "Predict future temporal half from past half — lower = better causal prediction.",
     "shown only the first half of a clip, how well does it predict the second half?"),
    ("tdist",        "PRED", "m13_tdist_compare",        "lower",  "L1-vs-Δt slope",
     "Single-shot L1 vs target offset Δt — lower = slower decay with temporal distance.",
     "does prediction stay accurate as you ask it to look further ahead in time?"),
    ("teacher_free", "PRED", "m13_teacher_free_compare", "lower",  "Free − teacher gap",
     "Free-running minus teacher-forced L1 — lower = less exposure bias.",
     "how much worse is the model when it must rely on its OWN predictions vs being fed the truth each step?"),
    ("maskratio",    "PRED", "m13_maskratio_compare",    "lower",  "L1-vs-maskratio slope",
     "L1 vs mask-ratio sweep — lower = graceful degradation under heavier masking.",
     "how gracefully does prediction degrade as more of the video is hidden?"),
    ("order",        "PRED", "m13_order_compare",        "signed", "Shuffled − ordered ΔL1",
     "Shuffled-context minus ordered-context L1 — sign = predictor's reliance on frame order.",
     "does scrambling the frame order hurt the model — i.e. does it actually use time?"),
    ("aot",          "ENC",  "m13_aot_compare",          "higher", "Arrow-of-Time acc",
     "Forward-vs-reversed binary head accuracy — higher = encoder preserves time's arrow.",
     "can the model tell a video played forwards from the same video played backwards?"),
    ("tov",          "ENC",  "m13_tov_compare",          "higher", "TOV / VCOP top-1",
     "N-way frame-permutation classification top-1 — higher = retains temporal order.",
     "can it put a handful of shuffled frames back into the right order?"),
    ("pace",         "ENC",  "m13_pace_compare",         "higher", "Pace top-1",
     "Playback-rate (1×/2×/4×) classification top-1 — higher = temporal-scale sensitive.",
     "can it tell normal-speed video from 2× / 4× fast-forwarded video?"),
    ("tcc_tau",      "ENC",  "m13_tcc_tau_compare",      "higher", "TCC Kendall's τ",
     "Per-frame soft-NN alignment Kendall's τ across same-action pairs (training-free).",
     "do matching moments in two clips of the same action line up in feature space?"),
    ("tcc_cycle",    "ENC",  "m13_tcc_cycle_compare",    "lower",  "TCC cycle-back error",
     "TCC A→B→A cycle-back frame-index error (appendix companion to τ) — lower = better.",
     "follow a frame to its match in clip B and back — how far from where it started does it land?"),
]
_FAMILY_DIR = {"HEAD": "head", "PRED": "predictor", "ENC": "encoder"}
_DIR_TAG = {"higher": "↑ better", "lower": "↓ better", "signed": "± signed"}   # column direction badge


def _short_label(enc: str) -> str:
    """Compact encoder tag for the hero WINNER row (cells are narrow). '<g-tag>-<arm>' for the 3 backbones."""
    enc = _canon(enc)
    for pre, _, short_tag in _BB_TAG:
        if enc.startswith(pre):
            return f"{short_tag}-{_ARM_SHORT.get(enc[len(pre):], enc[len(pre):][:8])}"
    return {"dinov2": "dinov2"}.get(enc, enc.replace("vjepa_", "")[:10])


def _fmt_compact(x: float) -> str:
    """Sign-prefixed number ROUNDED TO 2 decimals (0.01) for heatmap cells. Anything that rounds to
    zero prints '0' (so −0.0003 → 0, −0.07 stays −0.07, +5.81 stays +5.81)."""
    r = round(float(x), 2)
    if r == 0:                                # also catches -0.0 (e.g. round(-0.0003, 2))
        return "0"
    return f"{r:+.2f}"


def _fmt_fine(x: float) -> str:
    """Adaptive-precision sign-prefixed number for the PAIRED-diff heatmap, where the values are tiny
    (predictor Δs ~1e-3) — 0.01 rounding would collapse a significant Δ to a misleading '0'. Keeps 2–4
    sig decimals so a coloured 'SURGERY' cell never reads as 0."""
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    ax = abs(x)
    if ax >= 1:
        return f"{x:+.2f}"
    if ax >= _AXIS_DECIMAL_MIN:
        return f"{x:+.3f}"
    if ax > 0:
        return f"{x:+.4f}"
    return "0"


def _fmt_val(x: float) -> str:
    """Adaptive-precision RAW metric value (NO forced sign — these are absolute values, not deltas).
    Raw metrics span ~0.009 (predictor) → ~53 (action top1), so a fixed 0.01 rounding would collapse the
    tiny ones; big numbers stay terse (1 decimal), small ones get the precision they need."""
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    ax = abs(x)
    if ax >= _AXIS_INT_MIN:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= _AXIS_DECIMAL_MIN:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _taxonomy_f1_by_encoder(taxonomy_json: dict, encoders: list) -> dict:
    """taxonomy_f1 per encoder = mean over the 15 dims of per-dim test_mean; ci = mean of the
    per-dim ci_half (a conservative aggregate of the per-dim BCa half-widths)."""
    dims = taxonomy_json.get("dims", {})
    out = {}
    for v in encoders:
        vals, cis = [], []
        for spec in dims.values():
            be = spec.get("by_encoder", {}).get(v)
            if be is not None:
                vals.append(be["test_mean"])
                cis.append(be["test_ci"]["ci_half"])
        if vals:
            # nan-filter the per-dim CI halves (a degenerate per-dim BCa → nan; don't poison the
            # aggregate — tiny SANITY splits trip this, real POC/FULL won't).
            clean = [c for c in cis if not (isinstance(c, float) and np.isnan(c))]
            out[v] = (float(np.mean(vals)), float(np.mean(clean)) if clean else 0.0)
    return out


_DM = ("delta_mean", "delta_ci_lo", "delta_ci_hi")   # motion/future/pred/enc/tcc pairwise schema


def _norm_deltas(pairwise: dict, fields: tuple) -> dict:
    """Normalize a pairwise_deltas block → {pair_key: (delta, lo, hi)} using this schema's
    (delta, lo, hi) field names. pair_key is the JSON's '{a}_minus_{b}' (= a − b)."""
    fd, fl, fh = fields
    return {k: (v[fd], v[fl], v[fh]) for k, v in pairwise.items()}


def _agg_taxonomy_deltas(taxonomy_json: dict) -> dict:
    """taxonomy aggregate Δ per pair = mean over the 15 dims of per-dim (delta, ci_lo, ci_hi)."""
    acc = {}
    for spec in taxonomy_json.get("dims", {}).values():
        for pk, v in spec.get("pairwise_deltas", {}).items():
            acc.setdefault(pk, []).append((v["delta"], v["ci_lo"], v["ci_hi"]))
    return {pk: tuple(float(np.mean(col)) for col in zip(*rows)) for pk, rows in acc.items()}


def _load_all_metrics(srcs: dict, encoders: list) -> dict:
    """Return {metric_key: {"by_encoder": {enc: (val, ci_half)}, "na": set, "deltas": {pair_key:
    (Δ, lo, hi)}}} for every catalog metric whose source JSON is present (graceful: absent → omitted).
    `deltas` (the paired Δ across variants) feeds the hero's Δ-vs-frozen + CI-exclusion ★."""
    out = {}

    def _pack(by_enc, deltas):
        return {"by_encoder": by_enc, "na": {e for e in encoders if e not in by_enc}, "deltas": deltas}

    if srcs.get("action"):
        be = srcs["action"].get("by_encoder", {})
        out["action_top1"] = _pack(
            {v: (be[v]["acc_pct"], be[v]["top1_ci"]["ci_half"] * 100.0) for v in be},
            _norm_deltas(srcs["action"].get("pairwise_deltas", {}), ("delta_pp", "ci_lo_pp", "ci_hi_pp")))
    if srcs.get("motion"):
        be = srcs["motion"].get("by_encoder", {})
        out["motion_cos"] = _pack(
            {v: (be[v]["score_mean"], be[v]["score_ci"]["ci_half"]) for v in be},
            _norm_deltas(srcs["motion"].get("pairwise_deltas", {}), _DM))
    if srcs.get("future"):
        bv = srcs["future"].get("by_variant", {})
        real = {v: e for v, e in bv.items() if isinstance(e, dict)}
        out["future_mse"] = _pack(
            {v: (real[v]["mse_mean"], real[v]["mse_ci"]["ci_half"]) for v in real},
            _norm_deltas(srcs["future"].get("pairwise_deltas", {}), _DM))
    if srcs.get("taxonomy"):
        out["taxonomy_f1"] = _pack(_taxonomy_f1_by_encoder(srcs["taxonomy"], encoders),
                                   _agg_taxonomy_deltas(srcs["taxonomy"]))
    if srcs.get("pred"):
        m = srcs["pred"].get("metrics", {})
        for key in ("rollout", "causal", "tdist", "teacher_free", "maskratio", "order"):
            blk = m.get(key, {})
            bv = blk.get("by_variant", {})
            if bv:
                out[key] = _pack({v: (bv[v]["mean"], bv[v]["ci"]["ci_half"]) for v in bv},
                                 _norm_deltas(blk.get("pairwise_deltas", {}), _DM))
    if srcs.get("enc"):
        m = srcs["enc"].get("metrics", {})
        for key in ("aot", "tov", "pace"):
            blk = m.get(key, {})
            bv = blk.get("by_variant", {})
            if bv:
                out[key] = _pack({v: (bv[v]["mean"], bv[v]["ci"]["ci_half"]) for v in bv},
                                 _norm_deltas(blk.get("pairwise_deltas", {}), _DM))
        tcc = m.get("tcc", {})
        tcc_bv = tcc.get("by_variant", {})
        if tcc_bv:
            tpd = tcc.get("pairwise_deltas", {})
            out["tcc_tau"] = _pack(
                {v: (tcc_bv[v]["kendalls_tau"]["mean"], tcc_bv[v]["kendalls_tau"]["ci"]["ci_half"]) for v in tcc_bv},
                _norm_deltas(tpd.get("kendalls_tau", {}), _DM))
            out["tcc_cycle"] = _pack(
                {v: (tcc_bv[v]["cycle_back"]["mean"], tcc_bv[v]["cycle_back"]["ci"]["ci_half"]) for v in tcc_bv},
                _norm_deltas(tpd.get("cycle_back", {}), _DM))
    return out


# ── §3.3c HERO (B1 table-with-CI + B2 Δ-vs-frozen heatmap) — plan §7.3 ──

def _delta_v_vs_frozen(deltas: dict, v: str, frozen: str):
    """Δ(v − frozen), (lo, hi) from a normalized deltas dict keyed '{a}_minus_{b}' (=a−b).
    Orients + sign-flips (and swaps CI bounds) when stored as frozen−v. None if absent."""
    kvf, kfv = f"{v}_minus_{frozen}", f"{frozen}_minus_{v}"
    if kvf in deltas:
        d, lo, hi = deltas[kvf]
        return d, lo, hi
    if kfv in deltas:
        d, lo, hi = deltas[kfv]
        return -d, -hi, -lo          # v−f = −(f−v); CI [lo,hi] → [−hi,−lo]
    return None


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _is_good_win(delta: float, lo: float, hi: float, direction: str) -> bool:
    """Significant win vs frozen in the GOOD direction. higher→lo>0 ; lower→hi<0 ; signed→never."""
    if direction == "higher":
        return lo > 0
    if direction == "lower":
        return hi < 0
    return False


def _hero_catalog(metrics: dict):
    """The 14 hero metrics present (tcc_cycle is the appendix companion — excluded from the grid)."""
    return [c for c in _CATALOG if c[0] in metrics and c[0] != "tcc_cycle"]


def plot_hero_table(metrics: dict, encoders: list, frozen: str, output_dir: Path, boot_str: str):
    """B1: value±CI scorecard PNG + CSV, rendered as a COLORED heatmap-table.
    rows = encoders (frozen pinned top) + a WINNER row ; cols = hero metrics + WINS.
      • cell colour  = per-column min-max normalized, direction-aware (green=best in that metric);
                       'order' (signed) + N/A → neutral.
      • '*'          = Δ-vs-frozen 95% BCa CI excludes 0 (significance vs baseline).
      • winner       = single best encoder per scorable metric (argmax/argmin); its cell is BOLD,
                       named in the WINNER row.
      • WINS         = #metrics this encoder is THE winner, of n_scorable → the column PARTITIONS
                       the metrics (Σ over all rows = n_scorable; iter16 fix — was an overlapping
                       'beats-frozen' count that summed to >1)."""
    cat = _hero_catalog(metrics)
    if not cat:
        print("  [hero-table] no metrics present — skip")
        return
    scorable = [c for c in cat if c[3] in ("higher", "lower")]
    ordered = [frozen] + [e for e in encoders if e != frozen]
    metric_keys = [c[0] for c in cat]
    cmap = plt.get_cmap("RdYlGn")

    # raw point values for colouring + winner determination
    valmat = {(e, k): (metrics[k]["by_encoder"].get(e, (None, None))[0]) for e in ordered for k in metric_keys}
    colstat = {}                                   # key -> (vmin, vmax) over present encoders
    for k in metric_keys:
        present = [valmat[(e, k)] for e in ordered if valmat[(e, k)] is not None]
        colstat[k] = (min(present), max(present)) if present else (0.0, 0.0)
    # WINNER = canonical surgery-vs-pretrain CHAMPION DUEL (same _family_verdict the heatmap WINNER
    # row / scoreboard / grouped use) — NOT a raw best-encoder argmax (which never ties and could
    # even crown the frozen baseline). Decisive metric → the winning arm; paired CI overlaps 0 → tie
    # (no winner, no blue box). Keeps all four §G views telling ONE story.
    _surg = [e for e in ordered if _arm_family(e) == "surgery"]
    _pre = [e for e in ordered if _arm_family(e) == "pretrain"]
    _per = _family_verdict(metrics, encoders, frozen)[3]
    winner = {}
    for key, _f, _o, direction, _y, _c, _l in scorable:
        v = _per.get(key)
        if v == "surgery":
            winner[key] = _family_champion(metrics, key, frozen, _surg)
        elif v == "pretrain":
            winner[key] = _family_champion(metrics, key, frozen, _pre)
        # v == "tie" (or absent) → no winner entry → WINNER col shows "tie", no blue box

    def _cell_colour(val, key, direction):
        if val is None:
            return (0.90, 0.90, 0.90, 1.0)                       # N/A grey
        if direction not in ("higher", "lower"):
            return (1.0, 1.0, 1.0, 1.0)                          # signed → neutral
        vmin, vmax = colstat[key]
        t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
        if direction == "lower":
            t = 1.0 - t                                          # low value = good = green
        r, g, b, _ = cmap(t)
        return (r, g, b, 0.55)                                   # alpha keeps black text legible

    # TRANSPOSED (vertical, single-column-paper fit): rows = metrics (+ WINS row); cols = encoders
    # (short labels) + a WINNER col. Taller-than-wide → fits one paper column far better than the
    # old 11-wide layout.
    text_rows, colour_rows, winner_coords = [], [], []
    csv_rows = [["metric"] + [_short_label(e) for e in ordered] + ["WINNER"]]
    for (key, _f, _o, direction, _y, _c, _l) in cat:
        cells, ccols, wins_ci = [], [], []
        for ci, e in enumerate(ordered):
            be = metrics[key]["by_encoder"].get(e)
            if be is None:
                cells.append("N/A"); ccols.append(_cell_colour(None, key, direction)); continue
            val, ciw = be
            star = ""
            if e != frozen:
                dvf = _delta_v_vs_frozen(metrics[key]["deltas"], e, frozen)
                if dvf and _ci_excludes_zero(dvf[1], dvf[2]):
                    star = "*"
            cells.append(f"{val:.3f}\n±{ciw:.3f}{star}")            # value + CI wrapped to 2 lines
            ccols.append(_cell_colour(val, key, direction))
            if winner.get(key) == e:
                wins_ci.append(ci)
        win_cell = (_wrap_name(winner[key]) if winner.get(key)
                    else ("tie" if direction in ("higher", "lower") else "—"))  # dead-heat duel → tie
        csv_rows.append([key] + cells + [win_cell])                # CSV keeps ALL metrics incl signed 'order'
        if direction not in ("higher", "lower"):
            continue                                               # signed (order): diagnostic-only → CSV, not drawn
        ri = len(text_rows)                                        # figure row index (scorable rows only)
        winner_coords.extend((ri, ci) for ci in wins_ci)
        cells.append(win_cell); ccols.append((1.0, 0.96, 0.78, 1.0))
        text_rows.append(cells); colour_rows.append(ccols)

    row_labels = [f"{c[0]} {_DIR_TAG[c[3]]}" for c in scorable]
    col_labels = [_short_label(e) for e in ordered] + ["WINNER"]
    fig, ax = plt.subplots(figsize=(2.0 + 1.05 * len(col_labels), 1.8 + 0.62 * len(row_labels)))
    ax.axis("off")
    tbl = ax.table(cellText=text_rows, cellColours=colour_rows,
                   rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 2.0)
    for (ri, ci) in winner_coords:                              # spotlight winning cells (+1 for header row):
        cell = tbl[(ri + 1, ci)]                                # bold + thick blue outline (font unchanged → CI stays readable)
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#1a3cff")
        cell.set_linewidth(3.0)
    ax.set_title("HERO scorecard (vertical) — value ± BCa 95% CI · colour = per-metric min-max "
                 "(green = best) · * = Δ-vs-frozen CI excludes 0\n"
                 f"{len(scorable)} metrics × {len(ordered)} encoders · WINNER col + BLUE BOX = "
                 f"surgery-vs-vanilla-cont-SSL champion duel (tie = paired 95% CI overlaps 0) · {boot_str}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_table"))
    import csv
    with open(str(output_dir / artifact("m13_hero_table")), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  [hero-table] m13_hero_table.{{png,pdf,csv}} — {len(scorable)} metric rows drawn "
          f"(+signed 'order' in CSV only) × {len(col_labels)} cols (transposed) · WINNER col = champion duel (ties shown)")


def _winner_row(ax, tiesets, ny, mode):
    """Render the hero's bottom WINNER row in one of 3 honest styles and RETURN its height in grid
    units (so the caller extends ylim). tiesets[j] = {"sorted":[(e,v,ci) best-first], "tie":[e...
    95%-CI co-leaders], "leader":e, "lead_is_surg":bool}. "OURS" = the 4 GREEN surgery-novelty arms
    only (_OURS_GREEN) — argn (Auto-RGN baseline) and raw (surgery-on-raw control) are NOT credited
    as OURS even though their names contain 'surg' (the figure already colours them orange/brown).
    No style claims a significance the CIs don't support — ties are always marked; OURS (green) is
    highlighted only where it genuinely co-leads.
      · coleader_set     — stack ALL co-leaders, each in its OWN arm-colour (OURS green+bold); '+k more' if >4
      · surgery_coleader — name the best OURS co-leader (green); '~' unless it's the sole #1; else the true leader + 'OURS #rank'
      · tie_badge        — '~ tie (n)' when ≥2 co-leaders, else '<arm> WIN (sole)'; OURS tints the cell green"""
    ncol = max(tiesets, default=-1) + 1

    def _ours(e):
        return e.replace("vjepa_2_1_", "") in _OURS_GREEN

    def _empty(j):
        ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=(0.85, 0.85, 0.85, 0.35),
                               edgecolor="white", lw=1.0, zorder=2))
        ax.text(j, ny, "—", ha="center", va="center", fontsize=10, color="black", zorder=3)

    if mode == "coleader_set":
        nmax = max((len(ts["tie"]) for ts in tiesets.values()), default=1)
        shown = min(nmax, 4)
        dy, over = 0.30, (nmax > 4)
        row_h = 0.5 + (shown + (1 if over else 0)) * dy + 0.1
        for j in range(ncol):
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, row_h, facecolor=(0.96, 0.96, 0.96, 0.7),
                                   edgecolor="white", lw=1.0, zorder=2))
            ts = tiesets.get(j)
            if not ts:
                _empty(j)
                continue
            for k, e in enumerate(ts["tie"][:4]):         # each co-leader in its OWN arm colour; OURS bold
                ax.text(j, ny - 0.20 + k * dy, _short_label(e), ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if _ours(e) else "normal", color=_color_for(e, j), zorder=3)
            if len(ts["tie"]) > 4:
                ax.text(j, ny - 0.20 + 4 * dy, f"+{len(ts['tie']) - 4} more", ha="center", va="center",
                        fontsize=5.5, color="#999999", style="italic", zorder=3)
        return row_h

    if mode == "tie_badge":
        for j in range(ncol):
            ts = tiesets.get(j)
            if not ts:
                _empty(j)
                continue
            n = len(ts["tie"])
            sole_ours = (n == 1 and _ours(ts["leader"]))
            any_ours = any(_ours(e) for e in ts["tie"])
            face = (_SURG_GREEN if sole_ours else (0.80, 0.90, 0.80, 0.55) if any_ours
                    else (0.85, 0.85, 0.85, 0.45))
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=face,
                                   edgecolor="white", lw=1.0, zorder=2))
            if n == 1:
                ax.text(j, ny - 0.16, _short_label(ts["leader"]), ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white" if sole_ours else "black", zorder=3)
                ax.text(j, ny + 0.24, "WIN (sole)", ha="center", va="center", fontsize=6, color="black", zorder=3)
            else:
                ax.text(j, ny - 0.14, "~ tie", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="black", zorder=3)
                ax.text(j, ny + 0.24, f"n={n}", ha="center", va="center", fontsize=7, color="black", zorder=3)
        return 1.0

    # default: surgery_coleader
    for j in range(ncol):
        ts = tiesets.get(j)
        if not ts:
            _empty(j)
            continue
        ours = next((e for (e, _v, _c) in ts["sorted"] if _ours(e)), None)
        if ours is None or ours not in ts["tie"]:        # no OURS arm co-leads → name the true leader, honestly
            le = ts["leader"]
            ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=_color_for(le, j), alpha=0.5,
                                   edgecolor="white", lw=1.0, zorder=2))
            ax.text(j, ny - 0.16, _short_label(le), ha="center", va="center", fontsize=7,
                    fontweight="bold", color="black", zorder=3)
            if ours is not None:
                rank = [e for (e, _v, _c) in ts["sorted"]].index(ours) + 1
                ax.text(j, ny + 0.24, f"OURS #{rank}", ha="center", va="center", fontsize=6,
                        color=_SURG_GREEN, zorder=3)
            continue
        sole = (ts["tie"] == [ours])                     # OURS is the ONLY co-leader → a clear win
        ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=_SURG_GREEN, alpha=0.55,
                               edgecolor="white", lw=1.0, zorder=2))
        ax.text(j, ny - 0.16, ("" if sole else "~") + _short_label(ours), ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", zorder=3)
        ax.text(j, ny + 0.24, "clear #1" if sole else f"tie:{len(ts['tie'])}", ha="center",
                va="center", fontsize=6, color="black", zorder=3)
    return 1.0


def plot_hero_heatmap(metrics: dict, encoders: list, frozen: str, output_dir: Path):
    """B2: per-metric RAW-VALUE heatmap WITH numbers. TOP row = FROZEN baseline, then the contender arms,
    cols = hero metrics. Each cell PRINTS that arm's raw metric value 95% BCa CI [min, max] in native
    units — the paired surgery−pretrain comparison is NOT in the cells (it lives in the WINNER row + the
    dedicated m13_paired_diff_heatmap). Colour = good-oriented per-metric min-max WITH FROZEN INCLUDED, so
    a reader can read each arm's growth over the frozen baseline (green = best, red = worst), RdYlGn."""
    cat = [c for c in _hero_catalog(metrics) if c[3] in ("higher", "lower")]  # win/loss metrics only — signed (order) is diagnostic, lives in hero-table CSV
    contenders = [e for e in encoders if e != frozen]
    if not cat or not contenders:
        print("  [hero-heatmap] need a frozen baseline + ≥1 contender + metrics — skip")
        return
    rows = [frozen] + contenders                        # FROZEN baseline = row 0 (top), then the arms
    raw = np.full((len(rows), len(cat)), np.nan)        # RAW metric value (drives per-metric colour)
    cells = {}                                          # (i,j) -> (raw_lo, raw_hi) — the value's 95% CI
    for i, e in enumerate(rows):
        for j, (key, _fam, _on, direction, _yl, _cap, _lay) in enumerate(cat):
            be = metrics[key]["by_encoder"].get(e)
            if be is None:
                continue
            val, ciw = be                               # raw value + symmetric BCa half-width
            raw[i, j] = val
            cells[(i, j)] = (val - ciw, val + ciw)      # the value's 95% CI [min, max], native units
    norm = np.full_like(raw, np.nan)                    # colour = good-oriented per-metric min-max, FROZEN included → shows growth
    for j, (_k, _f, _o, direction, *_z) in enumerate(cat):
        col = raw[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            continue
        lo_, hi_ = float(fin.min()), float(fin.max())
        t = (col - lo_) / (hi_ - lo_) if hi_ > lo_ else col * 0.0 + 0.5  # 0..1, high raw = 1
        if direction == "lower":
            t = 1.0 - t                                 # low value = good = green
        norm[:, j] = 2.0 * t - 1.0                      # 0..1 → -1..1 for RdYlGn vmin/vmax = -1/+1
    fig, ax = plt.subplots(figsize=(3.0 + 1.75 * len(cat), 2.6 + 1.2 * (len(rows) + 1)))
    # alpha=0.6 → pastel RdYlGn (matches the hero TABLE); keeps red→green meaning while staying
    # light enough for plain BLACK BOLD numbers — no halo/outline needed (iter16 fix).
    im = ax.imshow(np.ma.masked_invalid(norm), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", alpha=0.6)
    ax.grid(False)                                      # iter16 fix: kill the distracting white gridlines
    ax.tick_params(length=0)
    ax.set_xticks(range(len(cat)))
    ax.set_xticklabels([f"{c[0]}\n{_DIR_TAG[c[3]]}" for c in cat], rotation=45, ha="right", fontsize=9)
    _WTICK = {"coleader_set": "CO-LEADERS\n(95%CI #1)", "surgery_coleader": "WINNER\n(surgery)",
              "tie_badge": "STAT #1\n(95%CI)"}
    ax.set_yticks(list(range(len(rows))) + [len(rows)])
    ax.set_yticklabels([f"{_display_label(frozen)}  (baseline)"]
                       + [_display_label(e) for e in contenders]
                       + [_WTICK.get(_WINNER_MODE, "WINNER")], fontsize=10)
    # iter18 2026-06-08 (user order): the bottom row reports the 95%-CI CO-LEADER set per metric —
    # EVERY arm statistically tied for #1 (its CI overlaps the point-best) — NOT a single point-best.
    # At POC the CIs are wide enough that a SURGERY arm co-leads EVERY metric, so the old single-name
    # row under-sold it. The blue SPOTLIGHT now boxes EVERY co-leader cell (surgery boxed on each
    # metric it ties). Three honest render styles (env HERO_WINNER_MODE) — none invents significance.
    tiesets = {}     # j -> {"sorted":[(e,v,ci) best-first], "tie":[e... CI-tied #1], "leader":e, "lead_is_surg":bool}
    for j, (key, _fam, _on, direction, *_z) in enumerate(cat):
        cands = [(e, metrics[key]["by_encoder"][e][0], metrics[key]["by_encoder"][e][1] or 0.0)
                 for e in contenders
                 if e in metrics[key]["by_encoder"]
                 and metrics[key]["by_encoder"][e][0] is not None
                 and np.isfinite(metrics[key]["by_encoder"][e][0])]
        if not cands:
            continue
        cands.sort(key=lambda x: x[1], reverse=(direction != "lower"))   # best raw value first
        le, lv, lci = cands[0]
        tie = [e for (e, v, ci) in cands if abs(v - lv) <= (ci + lci)]    # 95% CI overlaps the point-leader
        tiesets[j] = {"sorted": cands, "tie": tie, "leader": le,
                      "lead_is_surg": _arm_family(le) == "surgery"}
    _ri = {e: i for i, e in enumerate(rows)}             # display-row index (frozen=0, arms 1..n)
    spotlight = {(_ri[e], j) for j, ts in tiesets.items() for e in ts["tie"] if e in _ri}
    for (i, j), (rlo, rhi) in cells.items():
        win = (i, j) in spotlight
        # Each cell shows ONLY that arm's RAW metric value 95% CI [min, max].
        _ci = "nan" if (np.isnan(rlo) or np.isnan(rhi)) else f"[{_fmt_val(rlo)},\n{_fmt_val(rhi)}]"
        ax.text(j, i, _ci, ha="center", va="center",
                fontsize=11 if win else 9, fontweight="bold", color="black")
        if win:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a3cff", linewidth=3.0, zorder=4))
    ny = len(rows)
    row_h = _winner_row(ax, tiesets, ny, _WINNER_MODE)   # render the bottom row per mode → its height
    ax.set_ylim(ny - 0.5 + row_h, -0.5)                  # extend bottom to include the WINNER row(s)
    ax.axhline(0.5, color="#444444", lw=1.3, ls="--")    # separator BELOW the FROZEN baseline (row 0)
    ax.axhline(ny - 0.5, color="black", lw=1.5)           # separator above the WINNER row
    _TITLE_TAIL = {
        "coleader_set": "BOTTOM = CO-LEADER SET: every arm whose 95% CI overlaps the point-best "
                        "(statistically tied for #1), surgery in green · BLUE BOX = each co-leader cell",
        "surgery_coleader": "BOTTOM = best SURGERY arm among the 95%-CI co-leaders (green; ~ = it's a "
                            "statistical tie, not a significant win) · BLUE BOX = every co-leader cell",
        "tie_badge": "BOTTOM = statistical status: '~ tie (n)' when the top arms' 95% CIs overlap, "
                     "else 'WIN (sole)' · BLUE BOX = every co-leader cell (surgery boxed where it ties)"}
    ax.set_title("HERO — raw metric value  ·  per cell: that arm's value 95% BCa CI [min, max] · "
                 "TOP ROW = FROZEN baseline (read each arm's growth over it)\ncolour: red = worst, "
                 "green = best (per-metric min-max incl. frozen, good-oriented) · "
                 + _TITLE_TAIL.get(_WINNER_MODE, _TITLE_TAIL["coleader_set"]),
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="red = worst   →   green = best (per-metric normalized value, good-oriented)")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_raw_values"))
    print(f"  [hero-heatmap] m13_hero_raw_values.{{png,pdf}} — {len(rows)} rows (frozen + {len(contenders)} arms) × {len(cat)} (raw value CI)")


def plot_frozen_scorecard(metrics: dict, frozen_encoders: list, output_dir: Path, boot_str: str):
    """FROZEN-only leaderboard — ABSOLUTE head-metric values (NO Δ) for the image/video baselines +
    the V-JEPA frozen reference. rows = encoders (sorted best→worst on the 1st metric), cols = the
    metrics ALL frozen encoders share (the 3 head metrics; predictor metrics are V-JEPA-only → not
    comparable across baselines, excluded). Per-metric colour (green = best), best cell blue-boxed.
    Kept separate from the trained-arm hero views, which these baselines would clutter with N/A."""
    cat = _hero_catalog(metrics)
    cols = [c for c in cat if c[3] in ("higher", "lower")
            and all(metrics[c[0]]["by_encoder"].get(e) is not None for e in frozen_encoders)]
    if not cols or len(frozen_encoders) < _MIN_COMPARABLE:
        print(f"  [frozen-scorecard] skip — need ≥2 frozen encoders + ≥1 shared metric "
              f"(got {len(frozen_encoders)} enc, {len(cols)} shared metrics)")
        return
    cmap = plt.get_cmap("RdYlGn")
    colstat, winner = {}, {}
    for (key, _f, _o, direction, _y, _c, _l) in cols:
        vals = {e: metrics[key]["by_encoder"][e][0] for e in frozen_encoders}
        colstat[key] = (min(vals.values()), max(vals.values()))
        winner[key] = (max if direction == "higher" else min)(vals, key=vals.get)
    k0, d0 = cols[0][0], cols[0][3]                               # sort encoders by the 1st metric (good-first)
    ordered = sorted(frozen_encoders, key=lambda e: metrics[k0]["by_encoder"][e][0],
                     reverse=(d0 == "higher"))
    text_rows, colour_rows, win_coords = [], [], []
    csv_rows = [["encoder"] + [c[0] for c in cols]]
    for ri, e in enumerate(ordered):
        cells, ccols = [], []
        for ci, (key, _f, _o, direction, _y, _c, _l) in enumerate(cols):
            val, ciw = metrics[key]["by_encoder"][e]
            cells.append(f"{val:.3f}\n±{ciw:.3f}")
            vmin, vmax = colstat[key]
            t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
            if direction == "lower":
                t = 1.0 - t                                      # low value = good = green
            r, g, b, _ = cmap(t)
            ccols.append((r, g, b, 0.55))
            if winner[key] == e:
                win_coords.append((ri, ci))
        text_rows.append(cells); colour_rows.append(ccols)
        csv_rows.append([e] + cells)
    row_labels = [_display_label(e) for e in ordered]
    col_labels = [f"{c[0]}\n{_DIR_TAG[c[3]]}" for c in cols]
    fig, ax = plt.subplots(figsize=(3.0 + 2.1 * len(cols), 1.6 + 0.6 * len(ordered)))
    ax.axis("off")
    tbl = ax.table(cellText=text_rows, cellColours=colour_rows,
                   rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.2)
    for (ri, ci) in win_coords:                                  # spotlight per-metric best: bold + blue box
        cell = tbl[(ri + 1, ci)]
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#1a3cff"); cell.set_linewidth(3.0)
    ax.set_title("FROZEN encoders — ABSOLUTE head-metric values (no Δ) · value ± BCa 95% CI · "
                 "colour = per-metric min-max (green = best) · BLUE BOX = best per metric\n"
                 f"{len(ordered)} frozen encoders × {len(cols)} shared metrics · sorted by {k0} · {boot_str}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_frozen_scorecard"))
    import csv
    with open(str(output_dir / artifact("m13_frozen_scorecard")), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  [frozen-scorecard] m13_frozen_scorecard.{{png,pdf,csv}} — {len(ordered)} frozen × {len(cols)} metrics")


# ── §3.3d SURGERY-vs-PRETRAIN verdict views — make "who wins overall" unmistakable ──
# The hero heatmap colours per-column (per-metric winners) but never aggregates pretrain-vs-
# surgery. These 3 views answer it directly, ALL off ONE canonical tally (_family_verdict =
# best-surgery vs best-pretrain CHAMPION DUEL per metric, with explicit TIES) so their headline
# numbers are IDENTICAL — resolving the earlier discrepancy (the scoreboard's per-arm argmax had
# spuriously awarded dead-heat metrics to pretrain via alphabetical tie-breaking).

_PREDICTOR_KEYS = {"future_mse", "rollout", "causal", "tdist", "teacher_free", "maskratio", "order"}
_TIE_EPS = 1e-3   # champion-duel |Δ| below this = dead heat. Values separate cleanly here
                  # (real wins ≥ 0.009, ties ≤ 6e-5) — nothing sits near the threshold.
_SHORT_DESC = {   # ≤8-word plain-English blurb per metric (under the x-axis; no blank space)
    "action_top1":  "classify motion class\nfrom frozen features",
    "motion_cos":   "same-motion clips cluster\nvs different motions",
    "taxonomy_f1":  "read scene attributes\nfrom features",
    "future_mse":   "predictor guesses\nmasked next-frame tokens",
    "rollout":      "error growth over\nmulti-step rollout",
    "causal":       "predict 2nd half\nfrom 1st half",
    "tdist":        "accuracy decay vs\nprediction horizon",
    "teacher_free": "free-run minus\nteacher-forced gap",
    "maskratio":    "degradation under\nheavier masking",
    "order":        "reliance on\nframe order",
}


def _arm_family(enc: str) -> str:
    if "frozen" in enc:
        return "frozen"
    if "surgical" in enc or "surgery" in enc:
        return "surgery"
    if "pretrain" in enc:
        return "pretrain"
    # iter18 2026-06-08: FT-technique baselines (B1-B4: Full-FT / LP-FT / PEFT-LoRA-DoRA / CaSSLe /
    # EWC) are the ABLATION COMPARISON SET — they must join the hero `core` (table + raw-value
    # heatmap), not the external-baseline scorecard. Without this they classified as "other" and
    # were silently dropped from m13_hero_raw_values / m13_hero_table.
    if any(t in enc for t in ("full_ft", "lpft", "peft", "cassle", "ewc")):
        return "ft"
    return "other"


def _pick_frozen_ref(encoders):
    """Frozen REFERENCE = the same-backbone frozen baseline of the trained arms, NOT just the
    first alphabetical 'frozen'. Once cross-arch image/video baselines (e.g. lejepa_vitL_frozen)
    land in the shared roots, the old `next(e for e in sorted(encoders) if 'frozen' in e)` mis-won
    the sort (lejepa < vjepa). Derive the arms' backbone stem (common prefix of surgery+pretrain
    arms) and prefer '{stem}_frozen'; fall back to the alphabetical-first 'frozen' when no trained
    arms are present (pure-baseline render)."""
    frozens = sorted(e for e in encoders if "frozen" in e)
    if not frozens:
        return None
    arms = sorted(e for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft"))
    if arms:
        lo, hi = arms[0], arms[-1]                          # common prefix of the arm names
        i = 0
        while i < len(lo) and i < len(hi) and lo[i] == hi[i]:
            i += 1
        cand = f"{lo[:i].rstrip('_')}_frozen"               # e.g. 'vjepa_2_1_frozen' (or '..._vitg_frozen')
        if cand in encoders:
            return cand
    return frozens[0]


def _backbone_of(enc):
    """Map a RAW encoder key → its trained-backbone id (vjepa_2_1_vitG / vjepa_2_1_vitg /
    vjepa_2_0_vitg) via _canon (champion's legacy vjepa_2_1_<arm> → vjepa_2_1_vitG_) + the _BB_TAG
    prefixes — the SAME single source of backbone identity the display labels use. External baselines
    (dinov2 / lejepa / any non-_BB_TAG key) → None (not one of the 3 trained backbones)."""
    c = _canon(enc)
    for pre, _long, _short in _BB_TAG:
        if c.startswith(pre):
            return pre.rstrip("_")
    return None


def _backbones_present(encoders) -> list:
    """Distinct trained backbones with ≥1 surgery/pretrain arm, in _BB_TAG display order
    (ViT-G → ViT-g → ViT-g·2.0). Drives the per-backbone hero breakouts."""
    have = {_backbone_of(e) for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft")}
    return [pre.rstrip("_") for pre, _l, _s in _BB_TAG if pre.rstrip("_") in have]


def _good_orient(d, lo, hi, direction):
    """Orient (Δ, lo, hi) so POSITIVE = better: lower-better → negate + swap bounds; else as-is."""
    if direction == "lower":
        return -d, -hi, -lo
    return d, lo, hi


def _direction_of(key: str) -> str:
    return next(c[3] for c in _CATALOG if c[0] == key)


def _good_vs_frozen_ci(metrics, key, arm, frozen):
    """Good-oriented Δ(arm − frozen) as (g, glo, ghi) (positive = better), or None."""
    dv = _delta_v_vs_frozen(metrics[key]["deltas"], arm, frozen)
    return None if dv is None else _good_orient(dv[0], dv[1], dv[2], _direction_of(key))


def _good_vs_frozen(metrics, key, arm, frozen):
    g = _good_vs_frozen_ci(metrics, key, arm, frozen)
    return None if g is None else g[0]


def _family_champion(metrics, key, frozen, arms):
    """Best arm in `arms` by good-Δ-vs-frozen for this metric (None if none scorable)."""
    scored = [(a, _good_vs_frozen(metrics, key, a, frozen)) for a in arms]
    scored = [(a, g) for a, g in scored if g is not None]
    return max(scored, key=lambda t: t[1])[0] if scored else None


def _scorable_keys(metrics):
    """Hero metrics minus 'signed' (order's sign is reliance, not a win/loss)."""
    return [k for k, *_ in _hero_catalog(metrics) if _direction_of(k) != "signed"]


def _wrap_name(enc: str) -> str:
    """Full display name wrapped onto 2 lines (y-axis labels — readable, space-saving)."""
    parts = _display_label(enc).split(" ")
    if len(parts) <= 1:
        return parts[0]
    mid = (len(parts) + 1) // 2
    return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])


def _family_verdict(metrics, encoders, frozen):
    """CANONICAL surgery-vs-pretrain tally via champion duel (best surgery arm vs best pretrain
    arm) per metric, with explicit ties. Returns (n_surg, n_pre, n_tie,
    per{key→'surgery'|'pretrain'|'tie'}, arm_wins{arm→#outright metric wins}). Shared by all 3
    views so the headline is identical."""
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    ns = npr = nt = 0
    per, arm_wins = {}, {a: 0 for a in surg + pre}
    for key in _scorable_keys(metrics):
        sc, pc = _family_champion(metrics, key, frozen, surg), _family_champion(metrics, key, frozen, pre)
        if sc is None or pc is None:
            continue
        dv = _delta_v_vs_frozen(metrics[key]["deltas"], sc, pc)
        if dv is None:
            continue
        gd, glo, ghi = _good_orient(dv[0], dv[1], dv[2], _direction_of(key))
        # TIE = the PAIRED surgery−pretrain 95% BCa CI overlaps 0 → the two arms are statistically
        # indistinguishable on the SAME clips (the publishable test, NOT a mean point-estimate). A
        # degenerate (nan) CI falls back to the mean ε-guard. (iter17: was mean-only |gd|<_TIE_EPS.)
        is_tie = (abs(gd) < _TIE_EPS) if (np.isnan(glo) or np.isnan(ghi)) else (glo <= 0 <= ghi)
        if is_tie:
            per[key] = "tie"; nt += 1
        elif gd > 0:
            per[key] = "surgery"; ns += 1; arm_wins[sc] += 1
        else:
            per[key] = "pretrain"; npr += 1; arm_wins[pc] += 1
    return ns, npr, nt, per, arm_wins


def plot_scoreboard(metrics, encoders, frozen, output_dir):
    """#2 scoreboard: per-arm count of OUTRIGHT metric wins (champion duel; ties are NOT awarded
    to any arm — fixes the old alphabetical-tie-break that inflated pretrain) + the canonical
    SURGERY · PRETRAIN · TIE family banner shared with the other two views."""
    arms = [e for e in encoders if _arm_family(e) in ("surgery", "pretrain")]
    if not arms:
        print("  [scoreboard] no surgery/pretrain arms — skip"); return
    ns, npr, nt, _per, arm_wins = _family_verdict(metrics, encoders, frozen)
    order = sorted(arms, key=lambda a: arm_wins.get(a, 0))
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.55 * len(order) + 2)))
    y = np.arange(len(order))
    colors = [_color_for(a, i) for i, a in enumerate(order)]   # per-encoder colours (OURS green)
    ax.barh(y, [arm_wins.get(a, 0) for a in order], color=colors, alpha=0.85)
    for i, a in enumerate(order):
        ax.text(arm_wins.get(a, 0), i, f" {arm_wins.get(a, 0)}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([_wrap_name(a) for a in order], fontsize=8, rotation=20, va="center")
    ax.set_xlabel(f"# metrics won OUTRIGHT  (of {ns + npr + nt}; {nt} ties not awarded)")
    champ = "SURGERY" if ns > npr else ("VANILLA CONT-SSL" if npr > ns else "TIE")
    fig.suptitle(f"SCOREBOARD —  SURGERY {ns} · VANILLA CONT-SSL {npr} · TIE {nt}   (champion duel · winner: {champ})",
                 fontsize=13, fontweight="bold")
    ax.set_title("green = surgery arm · blue = vanilla continual-SSL arm · bar = outright metric wins by that arm", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, str(output_dir / "m13_scoreboard_surgery_vs_pretrain"))
    print(f"  [scoreboard] m13_scoreboard_surgery_vs_pretrain — surgery {ns} pretrain {npr} tie {nt}")


def plot_grouped_winner(metrics, encoders, frozen, output_dir):
    """#3 grouped heatmap: rows = PRETRAIN block then SURGERY block; cell = good-Δ-vs-frozen (bold)
    + its 95% CI on 2 lines; column-normalized colour; + SCORE column (mean per-metric-normalized,
    0–1) + WINNER row (canonical champion-duel: S/P/= ). Full wrapped y-names; ≤8-word metric
    blurbs under the x-axis. Block separators + verdict in title."""
    keys = _scorable_keys(metrics)
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    arms = pre + surg
    if not keys or not pre or not surg:
        print("  [grouped] need metrics + pretrain + surgery — skip"); return
    M = np.full((len(arms), len(keys)), np.nan)
    CI = {}
    for i, a in enumerate(arms):
        for j, k in enumerate(keys):
            gci = _good_vs_frozen_ci(metrics, k, a, frozen)
            if gci is not None:
                M[i, j] = gci[0]; CI[(i, j)] = (gci[1], gci[2])
    Mn = np.full_like(M, np.nan)
    for j in range(len(keys)):
        col = M[:, j]; lo, hi = np.nanmin(col), np.nanmax(col)
        Mn[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)
    score = np.nanmean(Mn, axis=1)
    _ns, _np, _nt, per, _aw = _family_verdict(metrics, encoders, frozen)
    # iter18 H3: strict — _family_verdict emits a verdict for every key; an
    # unknown verdict value must KeyError (a 4th category should crash, not
    # silently render '·').
    winner = [{"surgery": "S", "pretrain": "V", "tie": "="}[per[k]] for k in keys]
    nrow, ncol = len(arms) + 1, len(keys) + 1
    rgba = np.ones((nrow, ncol, 4)); cmap = plt.cm.RdYlGn
    for j in range(len(keys)):
        for i in range(len(arms)):
            if not np.isnan(Mn[i, j]):
                rgba[i, j] = cmap(Mn[i, j]); rgba[i, j, 3] = 0.6
    slo, shi = np.nanmin(score), np.nanmax(score)
    for i, s in enumerate(score):
        t = 0.5 if shi == slo else (s - slo) / (shi - slo)
        rgba[i, ncol - 1] = cmap(t); rgba[i, ncol - 1, 3] = 0.6
    for j, w in enumerate(winner):
        rgba[nrow - 1, j] = (0.17, 0.63, 0.17, 0.6) if w == "S" else (
            (0.84, 0.15, 0.16, 0.6) if w == "V" else (0.6, 0.6, 0.6, 0.35))
    fig, ax = plt.subplots(figsize=(max(11.0, 1.05 * ncol + 3), 0.95 * nrow + 3))
    ax.imshow(rgba, aspect="auto")
    # Spotlight (blue box + bigger Δ): a DECISIVE metric → its single winning arm; a TIE → BOTH family
    # champions (best surgery cell + best pretrain cell) — a tie shows TWO boxes, not one.
    spotlight = set()
    for j, k in enumerate(keys):
        if not np.any(~np.isnan(M[:, j])):
            continue
        if per.get(k) == "tie":
            for champ in (_family_champion(metrics, k, frozen, surg), _family_champion(metrics, k, frozen, pre)):
                if champ in arms:
                    spotlight.add((arms.index(champ), j))
        else:
            spotlight.add((int(np.nanargmax(M[:, j])), j))
    for i in range(len(arms)):
        for j in range(len(keys)):
            if not np.isnan(M[i, j]):
                win = (i, j) in spotlight              # decisive→winner; tie→both champions
                ax.text(j, i - 0.24, _fmt_compact(M[i, j]), ha="center", va="center",
                        fontsize=15 if win else 10, fontweight="bold", color="black")
                if (i, j) in CI:
                    lo, hi = CI[(i, j)]
                    ax.text(j, i + 0.2, f"[{_fmt_compact(lo)},\n{_fmt_compact(hi)}]", ha="center", va="center",
                            fontsize=6, color="black")
                if win:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#1a3cff", linewidth=3.0, zorder=4))
        ax.text(ncol - 1, i, f"{score[i]:.2f}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="black")
    for j, w in enumerate(winner):
        ax.text(j, nrow - 1, w, ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{k}  {_DIR_TAG[_direction_of(k)]}\n{_SHORT_DESC.get(k, '')}" for k in keys]
                       + ["SCORE\n(norm 0–1)"],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([_wrap_name(a) for a in arms] + ["WINNER"], fontsize=8, rotation=20, va="center")
    ax.axhline(len(pre) - 0.5, color="black", lw=2)
    ax.axhline(len(arms) - 0.5, color="black", lw=1)
    ax.axvline(len(keys) - 0.5, color="black", lw=1)
    verdict = "SURGERY" if _ns > _np else ("VANILLA CONT-SSL" if _np > _ns else "SPLIT")
    ax.set_title(f"Δ vs frozen (good-oriented, value + 95% CI) · VANILLA-CONT-SSL block / SURGERY block · "
                 f"SCORE = mean per-metric-normalized (0–1) · BLUE BOX + big Δ = best arm per metric\n"
                 f"WINNER row (champion duel — S=surgery V=vanilla-cont-SSL ==tie · tie = paired 95% CI overlaps 0):  "
                 f"surgery {_ns} · vanilla-cont-SSL {_np} · tie {_nt}  →  {verdict}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_grouped_winner_surgery_vs_pretrain"))
    print(f"  [grouped] m13_grouped_winner_surgery_vs_pretrain — surgery {_ns} pretrain {_np} tie {_nt}")


def plot_all_metric_bars(metrics: dict, output_dir: Path, boot_str: str) -> int:
    """One bar-with-CI panel per available catalog metric → <output_dir>/{head,predictor,encoder}/.
    Returns the number of panels written."""
    n = 0
    for key, family, out_name, direction, ylabel, caption, layman in _CATALOG:
        if direction == "signed":
            print(f"  [skip] {key}: signed (no better/worse direction) — diagnostic only, kept in CSV not plotted")
            continue
        md = metrics.get(key)
        if md is None:
            print(f"  [skip] {key}: source JSON absent")
            continue
        be, na = md["by_encoder"], md["na"]
        encoders = sorted(set(be) | na)
        if not encoders:
            print(f"  [skip] {key}: no encoders")
            continue
        vals = [be.get(e, (0.0, 0.0))[0] for e in encoders]
        errs = [be.get(e, (0.0, 0.0))[1] for e in encoders]
        sort_dir = "asc" if direction == "lower" else "desc"
        s_enc, s_vals, s_errs = _sort_by_metric(encoders, vals, errs, na, direction=sort_dir)
        order_word = "↑ lowest" if direction == "lower" else "↓ highest"
        badge = "" if direction == "signed" else direction
        sub = output_dir / _FAMILY_DIR[family]
        sub.mkdir(parents=True, exist_ok=True)
        _emit_bar(sub / out_name, s_enc, s_vals, s_errs, na, ylabel,
                  f"{key} — sorted {order_word} first", badge, caption, layman,
                  len(encoders), boot_str)
        n += 1
    return n


def plot_paired_forest(metrics, encoders, frozen, output_dir, boot_str):
    """Paired-difference FOREST plot: one row per metric = the champion-duel surgery−pretrain Δ + 95%
    BCa CI, on a per-metric SE-standardized x-axis so all metrics share ONE axis. Vertical line at 0 =
    no difference; shaded band = ±1.96 SE (the 95% 'not-significant' zone). A CI crossing 0 = TIE
    (grey); entirely right of 0 = surgery sig. better (green); entirely left = pretrain sig. better
    (red). Raw Δ [lo,hi] printed per row. Shows the DIFFERENCE directly (the convention for 'is A
    different from B across metrics' — forest plots; cf. Demšar 2006 JMLR critical-difference diagrams)
    so a tie is self-evident — no asking the reader to subtract two vs-frozen numbers."""
    surg = [e for e in encoders if _arm_family(e) == "surgery"]
    pre = [e for e in encoders if _arm_family(e) == "pretrain"]
    keys = _scorable_keys(metrics)
    if not surg or not pre or not keys:
        print("  [forest] need surgery + pretrain + metrics — skip")
        return
    rows = []
    for k in keys:
        sc = _family_champion(metrics, k, frozen, surg)
        pc = _family_champion(metrics, k, frozen, pre)
        if sc is None or pc is None:
            continue
        dv = _delta_v_vs_frozen(metrics[k]["deltas"], sc, pc)
        if dv is None:
            continue
        d, lo, hi = _good_orient(dv[0], dv[1], dv[2], _direction_of(k))
        rows.append((k, d, lo, hi))
    if not rows:
        print("  [forest] no paired deltas — skip")
        return
    clip, n = 5.0, len(rows)
    fig, ax = plt.subplots(figsize=(12.5, 0.62 * n + 2.6))
    ax.axvspan(-1.96, 1.96, color="#c8c8c8", alpha=0.30, zorder=0)        # 95% 'not significant' zone
    ax.axvline(0, color="black", lw=1.3, zorder=1)
    ns = npr = nt = 0
    for idx, (k, d, lo, hi) in enumerate(rows):
        if np.isnan(lo) or np.isnan(hi) or hi <= lo:                      # degenerate CI → mean-tie fallback
            tie = abs(d) < _TIE_EPS
            z = 0.0 if tie else (clip if d > 0 else -clip)
            zlo = zhi = z
        else:
            se = (hi - lo) / 3.9199                                       # 95% CI width = 2 × 1.96 SE
            z, zlo, zhi = d / se, lo / se, hi / se
            tie = lo <= 0 <= hi
        if tie:
            verdict, color = "tie", "#9e9e9e"; nt += 1
        elif d > 0:
            verdict, color = "SURGERY", "#2ca02c"; ns += 1
        else:
            verdict, color = "vanilla cont-SSL", "#d62728"; npr += 1
        zc = max(-clip, min(clip, z))
        zloc = max(-clip, min(clip, zlo))
        zhic = max(-clip, min(clip, zhi))
        ax.errorbar([zc], [idx], xerr=[[max(0.0, zc - zloc)], [max(0.0, zhic - zc)]], fmt="o",
                    color=color, ecolor=color, elinewidth=2.6, capsize=4, markersize=8, zorder=3)
        _ci_txt = "(≈ equal, no spread)" if (np.isnan(lo) or np.isnan(hi)) else f"[{_fmt_fine(lo)}, {_fmt_fine(hi)}]"
        ax.text(clip + 0.4, idx, f"{_fmt_fine(d)}  {_ci_txt}   {verdict}",
                ha="left", va="center", fontsize=8, fontweight="bold", color=color)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r[0]}  {_DIR_TAG[_direction_of(r[0])]}" for r in rows], fontsize=9)
    ax.invert_yaxis()                                                     # first metric at the top
    ax.set_xlim(-clip - 0.6, clip + 6.8)
    ax.set_xticks([-clip, -1.96, 0, 1.96, clip])
    ax.set_xticklabels(["≤−5", "−1.96", "0", "+1.96", "≥+5"], fontsize=8)
    ax.set_xlabel("standardized surgery − vanilla cont-SSL effect (Δ / SE) · shaded = ±1.96 (95% not-significant) · 0 = no difference",
                  fontsize=9)
    _bb = _backbone_of(surg[0]) if surg else None
    _label = _BB_LABEL.get(_bb, _display_label(frozen))
    _v = "SURGERY" if ns > npr else ("VANILLA CONT-SSL" if npr > ns else "SPLIT")
    ax.set_title(f"PAIRED forest — surgery − vanilla cont-SSL (champion duel) · {_label}\n"
                 f"surgery {ns} · vanilla cont-SSL {npr} · tie {nt}  →  {_v}   (green = surgery sig · "
                 f"red = vanilla cont-SSL sig · grey = tie, CI crosses 0) · {boot_str}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_paired_forest_surgery_vs_pretrain"))
    print(f"  [forest] m13_paired_forest_surgery_vs_pretrain.{{png,pdf}} — {n} metrics · "
          f"surgery {ns} pretrain {npr} tie {nt}")


def plot_paired_diff_heatmap(metrics, encoders, output_dir, boot_str):
    """COMBINED, appealing surgery−pretrain view: a metrics × backbones grid (like the hero heatmaps)
    where each cell is the champion-duel PAIRED Δ (surgery − pretrain, good-oriented, + = surgery) — the
    DIFFERENCE itself, not two vs-frozen numbers — coloured RdYlGn by the standardized effect (deep
    green = surgery sig. better, deep red = pretrain, pale grey = TIE / 95% CI crosses 0), with the Δ +
    95% CI + verdict printed per cell, and a per-backbone S/P/tie summary row. The 'effect-size +
    significance heatmap' convention: dense + colourful AND a tie is self-evident."""
    bbs = _backbones_present(encoders)
    keys = _scorable_keys(metrics)
    if not bbs or not keys:
        print("  [paired-heatmap] no backbones / metrics — skip")
        return
    nk, nb = len(keys), len(bbs)
    grid, tally, labels = {}, {j: [0, 0, 0] for j in range(nb)}, {}
    for j, bb in enumerate(bbs):
        be = [e for e in encoders if _backbone_of(e) == bb]
        frz = _pick_frozen_ref(be)
        surg = [e for e in be if _arm_family(e) == "surgery"]
        pre = [e for e in be if _arm_family(e) == "pretrain"]
        labels[j] = "\n".join(_BB_LABEL.get(bb, bb).split(" · ")[:3])
        for i, k in enumerate(keys):
            sc = _family_champion(metrics, k, frz, surg) if surg else None
            pc = _family_champion(metrics, k, frz, pre) if pre else None
            if sc is None or pc is None:
                continue
            dv = _delta_v_vs_frozen(metrics[k]["deltas"], sc, pc)
            if dv is None:
                continue
            d, lo, hi = _good_orient(dv[0], dv[1], dv[2], _direction_of(k))
            if np.isnan(lo) or np.isnan(hi):
                tie, z = abs(d) < _TIE_EPS, 0.0
            else:
                se = (hi - lo) / 3.9199
                z, tie = (d / se if se > 0 else 0.0), (lo <= 0 <= hi)
            grid[(i, j)] = (d, lo, hi, tie, z)
            tally[j][0 if (not tie and d > 0) else (1 if (not tie and d < 0) else 2)] += 1
    fig, ax = plt.subplots(figsize=(3.0 + 2.8 * nb, 2.2 + 0.98 * nk))
    cmap = plt.cm.RdYlGn
    for (i, j), (d, lo, hi, tie, z) in grid.items():
        yy = nk - 1 - i
        if tie:
            face = (0.90, 0.90, 0.87, 1.0)
        else:
            r, g, b, _a = cmap(0.5 + 0.5 * max(-1.0, min(1.0, z / 4.0)))
            face = (r, g, b, 0.80)
        ax.add_patch(Rectangle((j, yy), 1, 1, facecolor=face, edgecolor="white", lw=1.6))
        ci = "≈ equal" if (np.isnan(lo) or np.isnan(hi)) else f"[{_fmt_fine(lo)}, {_fmt_fine(hi)}]"
        ax.text(j + 0.5, yy + 0.66, _fmt_fine(d), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(j + 0.5, yy + 0.40, ci, ha="center", va="center", fontsize=7)
        ax.text(j + 0.5, yy + 0.15, "TIE" if tie else ("SURGERY" if d > 0 else "VANILLA CONT-SSL"),
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="black")
    for j in range(nb):
        ns, npr, nt = tally[j]
        ax.add_patch(Rectangle((j, -1), 1, 1, facecolor=(0.96, 0.96, 0.96, 1.0), edgecolor="white", lw=1.6))
        ax.text(j + 0.5, -0.5, f"Surgery {ns}\nVanilla cont-SSL {npr}\ntie {nt}", ha="center", va="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, nb)
    ax.set_ylim(-1, nk)
    ax.set_xticks([j + 0.5 for j in range(nb)])
    ax.set_xticklabels([labels[j] for j in range(nb)], fontsize=8, fontweight="bold")
    ax.set_yticks([nk - 1 - i + 0.5 for i in range(nk)] + [-0.5])
    # NO ↑/↓ "better" tag: the cell Δ is already good-oriented (surgery − pretrain), so + = surgery won,
    # − = pretrain won regardless of the raw metric's direction — the per-metric arrow would mislead here.
    ax.set_yticklabels([keys[i] for i in range(nk)] + ["WINS"], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("PAIRED  surgery − vanilla cont-SSL  (champion duel) across backbones · per cell: Δ + 95% BCa CI\n"
                 "deep green = surgery sig. better · deep red = vanilla cont-SSL · pale grey = TIE (95% CI crosses 0) · "
                 + boot_str, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_paired_diff_heatmap"))
    print(f"  [paired-heatmap] m13_paired_diff_heatmap.{{png,pdf}} — {nk} metrics × {nb} backbones")


def _emit_hero_suite(metrics, core, frozen, out_dir, boot_str):
    """The 4 surgery-vs-pretrain hero views for ONE encoder set, written into out_dir: hero-table +
    Δ-vs-frozen heatmap always; scoreboard + grouped-winner only when BOTH families are present.
    Called once for the COMBINED set (all backbones → eval/) and once per backbone (eval/<backbone>/),
    so combined and per-backbone share ONE code path (identical semantics, no drift). Caller
    guarantees `frozen` ∈ core and len(core) ≥ 2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hero_table(metrics, core, frozen, out_dir, boot_str)
    plot_hero_heatmap(metrics, core, frozen, out_dir)
    # §3.3d — surgery-vs-pretrain verdict views (only when both families are present)
    if any(_arm_family(e) == "surgery" for e in core) and any(_arm_family(e) == "pretrain" for e in core):
        plot_scoreboard(metrics, core, frozen, out_dir)
        plot_grouped_winner(metrics, core, frozen, out_dir)
        plot_paired_forest(metrics, core, frozen, out_dir, boot_str)


def _vstack_panels(output_dir, backbones, src_name, out_name):
    """COMBINED overview = the per-backbone <src_name>.png panels APPENDED VERTICALLY into one PNG with
    a labelled header band per backbone. A pure image concat — NO cross-backbone math (each panel keeps
    its own arms vs its OWN frozen). Panels stacked top→bottom in the given order."""
    panels = [(bb, output_dir / bb / f"{src_name}.png") for bb in backbones]
    panels = [(bb, Image.open(str(p)).convert("RGB")) for bb, p in panels if p.exists()]
    if not panels:
        print(f"  [stack] no per-backbone {src_name} panels present — skip")
        return
    w = max(im.width for _, im in panels)
    # Big LABELLED header band per panel so each heatmap's BACKBONE + model size is unmistakable.
    fpath = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    longest = max((_BB_LABEL.get(bb, bb) for bb, _ in panels), key=len)
    fsize = 110
    font = ImageFont.truetype(fpath, fsize)
    while fsize > _FONT_MIN_PT and font.getlength(longest) > w * 0.94:       # auto-shrink to fit the panel width
        fsize -= 4
        font = ImageFont.truetype(fpath, fsize)
    band_h = int(fsize * 1.8)
    total_h = sum(band_h + im.height for _, im in panels)
    canvas = Image.new("RGB", (w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for bb, im in panels:
        label = _BB_LABEL.get(bb, bb)
        draw.rectangle([0, y, w - 1, y + band_h - 1], fill=(18, 28, 64))         # dark navy banner
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, y + (band_h - th) // 2 - bbox[1]), label, fill="white", font=font)
        y += band_h
        canvas.paste(im, (0, y))           # left-aligned; a narrower panel just keeps white margin
        y += im.height
    canvas.save(str(output_dir / f"{out_name}.png"))
    canvas.save(str(output_dir / f"{out_name}.pdf"))               # both .png & .pdf (CLAUDE.md plot rule)
    print(f"  [stack] {out_name}.{{png,pdf}} — vertical append of {len(panels)} LABELLED backbone panels "
          f"({', '.join(bb for bb, _ in panels)}) · NO averaging")


# ── CLI ──────────────────────────────────────────────────────────────

def _skip_encoders_from_env(encoders):
    """Encoder names to HIDE from the m13 plots — honors ITER18_SKIP_ARMS (the run's --skip-arms),
    matching by arm-suffix with the surgery↔surgical rename so 'surgery_noDI_head' hides the encoder
    'vjepa_2_1[_vitg]_surgical_noDI_head'. Empty when the env isn't set (plot everything). The §3
    finale already passes a skip-filtered --encoders list; this is the belt-and-suspenders for the
    live preview + direct m13 runs."""
    raw = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    if not raw:
        return set()
    _norm = lambda s: s.replace("surgical", "surgery")           # noqa: E731 (one-liner rename)
    # .strip("\"'") per token: a direct-prompt paste of the runbook's ITER18_SKIP_ARMS=\"$SKIP\" watch
    # form bakes LITERAL quotes into the value → '"cassle_encoder' / 'surgery_noDI_head"' won't match.
    skip = {_norm(a.strip().strip("\"'")) for a in raw.split() if a.strip().strip("\"'")}
    _pres = ("vjepa_2_1_vitG_", "vjepa_2_1_vitg_", "vjepa_2_0_vitg_", "vjepa_2_1_vitL_", "vjepa_2_1_")
    out = set()
    for e in encoders:
        suf = next((e[len(p):] for p in _pres if e.startswith(p)), e)
        if _norm(suf) in skip:
            out.add(e)
    return out


# ═══ metrics_watch regeneration (iter18 2026-06-12, user order) ═══════════════════════════
# SELF-CONTAINED copy of scripts/iter18_poc_metrics.py's figure + data-dump generation, so the
# probe_plot/metrics_watch/<BACKBONE>/ artifacts (train_trajectories / kept_scorecard /
# eval_scorecard .png+.pdf + train_metrics/eval_metrics .json+.csv) stay reproducible from
# src/ ALONE after the iter18 scheduler scripts move to scripts/legacy/. No import from
# scripts/ — the needed DAG constants are duplicated below (they are FROZEN once iter18 ends;
# this copy is the durable record). Invoke:
#   python src/m13_eval_plot.py --POC --output-dir outputs/poc/probe_plot \
#       --outputs-root outputs/poc --metrics-watch-out outputs/poc/probe_plot/metrics_watch \
#       --metrics-watch-only
# Backbone via ITER18_BACKBONE env, hidden arms via ITER18_SKIP_ARMS env (same contract as
# the watch scripts; the scheduler-log fallback is dropped — env is the durable source).
_MW_BACKBONE = os.environ.get("ITER18_BACKBONE", "vjepa_2_1_vitG")
_MW_TRAIN_ORDER = [
    "pretrain_encoder",
    "surgery_3stage_DI_encoder", "surgery_noDI_encoder",
    "surgery_3stage_DI_head", "surgery_noDI_head",
    "surgical_autorgn_encoder", "surgery_raw_encoder",
    "full_ft_encoder", "lpft_encoder",
    "peft_lora_encoder", "peft_dora_encoder",
    "cassle_encoder", "ewc_encoder",
]
_MW_HEAD_ARMS = {"surgery_3stage_DI_head", "surgery_noDI_head"}
_MW_ARM2ENC = {
    "pretrain_encoder":          "pretrain_encoder",
    "surgery_3stage_DI_encoder": "surgical_3stage_DI_encoder",
    "surgery_noDI_encoder":      "surgical_noDI_encoder",
    "surgery_3stage_DI_head":    "surgical_3stage_DI_head",
    "surgery_noDI_head":         "surgical_noDI_head",
    "surgical_autorgn_encoder":  "surgical_autorgn_encoder",
    "surgery_raw_encoder":       "surgery_raw_encoder",
    "full_ft_encoder":           "full_ft_encoder",
    "lpft_encoder":              "lpft_encoder",
    "peft_lora_encoder":         "peft_lora_encoder",
    "peft_dora_encoder":         "peft_dora_encoder",
    "cassle_encoder":            "cassle_encoder",
    "ewc_encoder":               "ewc_encoder",
}
_MW_ARM2DIR = {
    "pretrain_encoder":          "m09a_pretrain_encoder",
    "surgery_3stage_DI_encoder": "m09c_surgery_3stage_DI_encoder",
    "surgery_noDI_encoder":      "m09c_surgery_noDI_encoder",
    "surgery_3stage_DI_head":    "m09c_surgery_3stage_DI_head",
    "surgery_noDI_head":         "m09c_surgery_noDI_head",
    "surgical_autorgn_encoder":  "m09e_autorgn_encoder",
    "surgery_raw_encoder":       "m09c_surgery_raw_encoder",
    "full_ft_encoder":           "m09f_full_ft_encoder",
    "lpft_encoder":              "m09f_lpft_encoder",
    "peft_lora_encoder":         "m09b_peft_lora_encoder",
    "peft_dora_encoder":         "m09b_peft_dora_encoder",
    "cassle_encoder":            "m09d_cassle_encoder",
    "ewc_encoder":               "m09d_ewc_encoder",
}
_MW_PT_FAMILIES = ["rollout", "causal", "tdist", "maskratio", "order", "teacher_free"]
_MW_ET_FLAT = ["aot", "tov", "pace"]
_MW_FRESH_LOG_S = 1800


def _mw_enc_name(arm_enc):
    """Champion 2B vitG drops its size tag; other backbones keep the full name (iter17 rule)."""
    prefix = "vjepa_2_1" if _MW_BACKBONE == "vjepa_2_1_vitG" else _MW_BACKBONE
    return f"{prefix}_{arm_enc}"


def _mw_jload(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mw_jsonl(p: Path) -> list:
    try:
        text = p.read_text()
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return rows     # record mid-write under a live run — show what's complete
    return rows


def _mw_skip_arms():
    """Arms hidden from the GRAPHS (data files keep everything) — ITER18_SKIP_ARMS env only."""
    env = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    return {tok.strip().strip("\"'") for tok in env.split() if tok.strip().strip("\"'")}


def _mw_arm_train_val_n(outputs_root: Path, mtag: str, arm: str):
    """(n_train, n_val) parsed from the arm's newest scheduler log when logs/ still exists —
    graceful (None, None) once the iter18 logs are archived; the json then records null."""
    logs = sorted((q for q in Path("logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log")
                   if q.exists()), key=lambda q: q.stat().st_mtime)
    if not logs:
        return None, None
    try:
        head = logs[-1].read_text(errors="replace")[:16000]
    except OSError:
        return None, None
    m = re.search(r"train/val split: ([\d,]+) train / ([\d,]+) val", head)
    if m:
        return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    m = re.search(r"Train clips: ([\d,]+) \| Val clips: ([\d,]+)", head)
    if m:
        return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    tr = re.search(r"Loaded subset: ([\d,]+) clip keys from train_pool", head)
    va = (re.search(r"Loaded subset: ([\d,]+) clip keys from val_split", head)
          or re.search(r"Loaded val subset: ([\d,]+) clips", head))
    return (int(tr.group(1).replace(",", "")) if tr else None,
            int(va.group(1).replace(",", "")) if va else None)


def _mw_train_blocks(outputs_root: Path, mtag: str):
    """One block per arm: every probe checkpoint + the selector-replay verdicts (🎯/✋/·)."""
    blocks = []
    for arm in _MW_TRAIN_ORDER:
        d = outputs_root / _MW_BACKBONE / _MW_ARM2DIR[arm]
        hist = _mw_jsonl(d / "probe_history.jsonl")
        summ = _mw_jload(d / artifact("training_summary"))
        logs = sorted((q for q in Path("logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log")
                       if q.exists()), key=lambda q: q.stat().st_mtime)
        log_fresh = bool(logs) and (time.time() - logs[-1].stat().st_mtime) < _MW_FRESH_LOG_S
        status = "✅" if summ else ("🔄" if (hist or log_fresh) else "⬚")
        n_tr, n_va = _mw_arm_train_val_n(outputs_root, mtag, arm)
        rows, kept_i = [], None
        best = float("inf")
        for i, r in enumerate(hist):
            if arm in _MW_HEAD_ARMS:
                verdict = "·"
            elif r.get("future_l1") is not None and r["future_l1"] < best:
                best, kept_i, verdict = r["future_l1"], i, "🎯"
            else:
                verdict = "✋"
            rows.append((r, verdict))
        head_best = ""
        if arm in _MW_HEAD_ARMS and summ and "best_val_loss" in summ:
            head_best = f"sel=head-vloss {summ['best_val_loss']:.3f}@ep{summ.get('best_epoch', '?')}"
        blocks.append({"arm": arm, "st": status, "n_tr": n_tr, "n_va": n_va,
                       "rows": rows, "kept_i": kept_i, "head_best": head_best})
    return blocks


def _mw_eval_rows(outputs_root: Path):
    """Per-encoder (mean, ci_half) per metric from the eval artifacts — the _raw payload the
    figures + json/csv consume (the watch script's terminal display strings are not files)."""
    encs = [_mw_enc_name("frozen")] + [_mw_enc_name(e) for e in _MW_ARM2ENC.values()]
    rows = []
    for enc in encs:
        act = _mw_jload(outputs_root / artifact("probe_action_dir") / enc / artifact("test_metrics"))
        tax = _mw_jload(outputs_root / artifact("probe_taxonomy_dir") / enc / artifact("test_metrics"))
        mc = _mw_jload(outputs_root / artifact("probe_motion_cos_dir") / enc / "intra_inter_ratio.json")
        fm = _mw_jload(outputs_root / artifact("probe_future_mse_dir") / enc / "aggregate_mse.json")
        pt = {fam: _mw_jload(outputs_root / artifact("predictor_temporal_dir") / enc / f"aggregate_{fam}.json")
              for fam in _MW_PT_FAMILIES}
        et = {fam: _mw_jload(outputs_root / artifact("encoder_temporal_dir") / enc / f"aggregate_{fam}.json")
              for fam in _MW_ET_FLAT}
        tcc = _mw_jload(outputs_root / artifact("encoder_temporal_dir") / enc / "aggregate_tcc.json") or {}
        tax_macro = None
        if tax and isinstance(tax.get("dims"), dict) and tax["dims"]:
            vals = [v["test_mean"] for v in tax["dims"].values() if "test_mean" in v]
            tax_macro = sum(vals) / len(vals) if vals else None
        n_test = next((src["n_test"] for src in (act, fm, mc) if src and "n_test" in src), None)

        def _half(d, ci_key):
            return d.get(ci_key, {}).get("ci_half") if (d and isinstance(d.get(ci_key), dict)) else None

        rows.append({
            "_enc_full": enc,
            "n_te": ("—" if n_test is None else str(n_test), ""),
            "_raw": {
                "act":  (act["top1_acc"] if act and "top1_acc" in act else None, _half(act, "top1_ci")),
                "tax":  (tax_macro, None),
                "mcos": (mc["score_mean"] if mc and "score_mean" in mc else None, _half(mc, "score_ci")),
                "fut":  (fm["mse_mean"] if fm and "mse_mean" in fm else None, _half(fm, "mse_ci")),
                **{fam: ((pt[fam]["mean"], _half(pt[fam], "ci")) if pt[fam] and "mean" in pt[fam] else (None, None))
                   for fam in _MW_PT_FAMILIES},
                **{fam: ((et[fam]["mean"], _half(et[fam], "ci")) if et[fam] and "mean" in et[fam] else (None, None))
                   for fam in _MW_ET_FLAT},
                **{key: ((tcc[sub]["mean"], _half(tcc.get(sub), "ci"))
                         if isinstance(tcc.get(sub), dict) and "mean" in tcc[sub] else (None, None))
                   for key, sub in (("tcc_cycle", "cycle_back"), ("tcc_tau", "kendalls_tau"))},
            },
        })
    return rows


_MW_FAM_OURS = {"surgery_3stage_DI_encoder", "surgery_noDI_encoder",
                "surgery_3stage_DI_head", "surgery_noDI_head"}
_MW_TRAIN_STYLE = {  # arm → (short label, color, linestyle, linewidth); OURS = greens
    "pretrain_encoder":          ("vCSSL",     "#1565C0", "-",  1.8),
    "surgery_3stage_DI_encoder": ("s3DI-enc",  "#1B5E20", "-",  2.6),
    "surgery_noDI_encoder":      ("sNoDI-enc", "#43A047", "-",  2.6),
    "surgery_3stage_DI_head":    ("s3DI-hd",   "#81C784", ":",  1.8),
    "surgery_noDI_head":         ("sNoDI-hd",  "#A5D6A7", ":",  1.8),
    "surgical_autorgn_encoder":  ("autoRGN",   "#E65100", "--", 1.4),
    "surgery_raw_encoder":       ("s-RAW",     "#827717", "-.", 1.8),
    "full_ft_encoder":           ("fullFT",    "#F57C00", "--", 1.4),
    "lpft_encoder":              ("LP-FT",     "#8D6E63", "--", 1.4),
    "peft_lora_encoder":         ("LoRA",      "#78909C", "--", 1.4),
    "peft_dora_encoder":         ("DoRA",      "#546E7A", "--", 1.4),
    "cassle_encoder":            ("CaSSLe",    "#9E9D24", "--", 1.4),
    "ewc_encoder":               ("EWC",       "#6D4C41", "--", 1.4),
}
_MW_TRAIN_METRICS = [
    ("probe_top1",    "action top-1",  "higher"),
    ("motion_cos",    "motion-cos",    "higher"),
    ("future_l1",     "future L1",     "lower"),
    ("causal_l1",     "causal L1",     "lower"),
    ("maskratio",     "mask-ratio",    "lower"),
    ("val_jepa_loss", "val JEPA loss", "lower"),
]
_MW_EVAL_METRICS = [
    ("act", "action top-1", "higher"), ("tax", "taxonomy F1", "higher"),
    ("mcos", "motion-cos", "higher"), ("fut", "future MSE", "lower"),
    ("rollout", "rollout drift", "lower"), ("causal", "causal L1", "lower"),
    ("tdist", "t-dist", "lower"), ("maskratio", "mask-ratio slope", "lower"),
    ("teacher_free", "teacher-free", "lower"),
    ("aot", "arrow-of-time acc", "higher"), ("tov", "temporal-order acc", "higher"),
    ("pace", "pace acc", "higher"), ("tcc_cycle", "TCC cycle-back", "lower"),
    ("tcc_tau", "TCC Kendall τ", "higher"),
]
_MW_EVAL_RAW_KEYS = ["act", "tax", "mcos", "fut", "rollout", "causal", "tdist", "maskratio",
                     "order", "teacher_free", "aot", "tov", "pace", "tcc_cycle", "tcc_tau"]
_MW_TRAIN_PROBE_KEYS = ["step", "probe_top1", "motion_cos", "future_l1", "causal_l1",
                        "maskratio", "val_jepa_loss"]


def _mw_render_graphs(blocks, ev, out_dir: Path):
    """The 3 metrics_watch figures (verbatim recipe of the watch script — uses this module's
    own _bar_with_ci/_sort_by_metric, which the watch script itself imported from here)."""
    def _kept_row(b):
        if b["kept_i"] is not None:
            return b["rows"][b["kept_i"]][0]
        return b["rows"][-1][0] if b["rows"] else None

    # ── F1: train trajectories ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    arms_drawn = []
    for ax, (key, title, direction) in zip(axes.flat, _MW_TRAIN_METRICS):
        for b in blocks:
            sty = _MW_TRAIN_STYLE[b["arm"]]
            pts = sorted((r["step"], r[key]) for r, _ in b["rows"]
                         if r.get(key) is not None and r.get("step") is not None)
            if not pts:
                continue
            if b["arm"] not in arms_drawn:
                arms_drawn.append(b["arm"])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, sty[2], color=sty[1], lw=sty[3], marker="o", ms=3.5)
            kr = _kept_row(b)
            if b["kept_i"] is not None and kr.get(key) is not None:
                ax.plot(kr["step"], kr[key], marker="*", ms=15, color=sty[1],
                        mec="black", mew=0.8, ls="none", zorder=5)
        arrow = "↑ better" if direction == "higher" else "↓ better"
        ax.set_title(f"{title}  ({arrow})", fontweight="bold", fontsize=13)
        ax.set_xlabel("global step", fontsize=10)
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], color=_MW_TRAIN_STYLE[a][1], ls=_MW_TRAIN_STYLE[a][2],
                          lw=_MW_TRAIN_STYLE[a][3], marker="o", ms=4, label=_MW_TRAIN_STYLE[a][0])
               for a in _MW_TRAIN_STYLE if a in arms_drawn]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                   frameon=True, fontsize=11)
    fig.suptitle("iter18 TRAIN probe checkpoints — every arm, every probe · star marker = KEPT ckpt"
                 " · OURS = greens (solid enc / dotted head)", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.13, hspace=0.35, wspace=0.25)
    save_fig(fig, str(out_dir / "train_trajectories"))
    plt.close(fig)

    # ── F2: KEPT-checkpoint scorecard + verdict strip ──
    _CI_KEY = {"probe_top1": "top1_ci_half", "motion_cos": "motion_cos_ci_half",
               "future_l1": "future_l1_ci_half"}
    _Z95 = 1.96

    def _row_ci(row, key):
        ck = _CI_KEY.get(key)
        if ck and row.get(ck) is not None:
            return float(row[ck])
        if key == "probe_top1" and row.get("n_probe_clips"):
            p_, n_ = float(row["probe_top1"]), int(row["n_probe_clips"])
            return _Z95 * (p_ * (1 - p_) / n_) ** 0.5
        return None

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.38], hspace=0.5, wspace=0.45)
    axes6 = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    verdicts = []
    for ax, (key, title, direction) in zip(axes6, _MW_TRAIN_METRICS):
        entries = []
        for b in blocks:
            row = _kept_row(b)
            if row is not None and row.get(key) is not None:
                entries.append((b["arm"], float(row[key]), _row_ci(row, key)))
        if not entries:
            ax.axis("off")
            ax.set_title(f"{title} — no data yet", fontsize=12)
            verdicts.append((title, None, None, None, None, None))
            continue
        entries.sort(key=lambda t: t[1], reverse=(direction == "higher"))
        labels = [_MW_TRAIN_STYLE[a][0] + (" (hd)" if a.endswith("_head") else "")
                  for a, _, _ in entries]
        colors = [_MW_TRAIN_STYLE[a][1] for a, _, _ in entries]
        vals = [v for _, v, _ in entries]
        errs = [(e or 0.0) for _, _, e in entries]
        ys = np.arange(len(entries))[::-1]
        bars = ax.barh(ys, vals, color=colors, alpha=0.9, height=0.65,
                       xerr=errs, capsize=3, error_kw={"lw": 1.1, "ecolor": "#222"})
        for y, (arm, v, e), b_ in zip(ys, entries, bars):
            if arm in _MW_FAM_OURS:
                b_.set_edgecolor("black")
                b_.set_linewidth(1.8)
            ax.text(v + (e or 0.0), y, f" {v:.4f}" + (f"±{e:.3f}" if e else ""),
                    va="center", fontsize=8)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        lo = min(v - (e or 0.0) for _, v, e in entries)
        hi = max(v + (e or 0.0) for _, v, e in entries)
        pad = (hi - lo) * 0.15 or abs(hi) * 0.05 or 0.01
        ax.set_xlim(lo - pad, hi + pad * 4.0)
        arrow = "↑" if direction == "higher" else "↓"
        ax.set_title(f"{title} {arrow} · best: {labels[0]}", fontweight="bold", fontsize=12)
        ours = [(v, e) for a, v, e in entries if a in _MW_FAM_OURS]
        other = [(v, e) for a, v, e in entries if a not in _MW_FAM_OURS]
        if ours and other:
            pick = max if direction == "higher" else min
            (bo, eo), (bx, ex) = pick(ours), pick(other)
            win = bo > bx if direction == "higher" else bo < bx
            verdicts.append((title, win, bo, bx, eo, ex))
        else:
            verdicts.append((title, None, None, None, None, None))
    axv = fig.add_subplot(gs[2, :])
    axv.axis("off")
    n = len(verdicts)
    for i, (title, win, bo, bx, eo, ex) in enumerate(verdicts):
        x0 = i / n
        if win is None:
            col, txt = "#ECEFF1", f"{title}\n(awaiting data)"
        else:
            tie = (eo is not None and ex is not None and abs(bo - bx) <= (eo + ex))
            col = "#FFF9C4" if tie else ("#C8E6C9" if win else "#FFCDD2")
            tag = "≈ TIE (CIs overlap)" if tie else ("OURS LEAD" if win else "OTHERS LEAD")
            txt = f"{title}\nOURS {bo:.4f} · other {bx:.4f}\n{tag}"
        axv.add_patch(plt.Rectangle((x0 + 0.004, 0.08), 1 / n - 0.008, 0.84,
                                    facecolor=col, edgecolor="#555",
                                    transform=axv.transAxes))
        axv.text(x0 + 1 / (2 * n), 0.5, txt, transform=axv.transAxes,
                 ha="center", va="center", fontsize=10, fontweight="bold")
    axv.text(0.0, 1.06, "VERDICT — best OURS (surgery; black-edged bars) vs best OTHER · "
                        "whiskers = 95% CI (per-clip BCa; legacy top-1 rows: binomial approx; "
                        "causal/mask-ratio/val-loss have no per-clip data → no whisker)",
             transform=axv.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.suptitle("iter18 KEPT-checkpoint scorecard — the encoder each arm EXPORTS to eval",
                 fontsize=15, fontweight="bold")
    save_fig(fig, str(out_dir / "kept_scorecard"))
    plt.close(fig)

    # ── F3: EVAL scorecard (14 panels on a 4×4 grid) ──
    have_any = any(r["_raw"][k][0] is not None for r in ev for k, _, _ in _MW_EVAL_METRICS)
    if not have_any:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "no EVAL artifacts yet —\nper-encoder evals launch as each arm finishes training",
                ha="center", va="center", fontsize=16, fontweight="bold", color="#666")
        save_fig(fig, str(out_dir / "eval_scorecard"))
        plt.close(fig)
        return
    fig, axes14 = plt.subplots(4, 4, figsize=(30, 22))
    for ax in axes14.flat[len(_MW_EVAL_METRICS):]:
        ax.axis("off")
    for ax, (k, title, direction) in zip(axes14.flat, _MW_EVAL_METRICS):
        encs = [r["_enc_full"] for r in ev]
        vals = [r["_raw"][k][0] for r in ev]
        errs = [(r["_raw"][k][1] or 0.0) for r in ev]
        na = {e for e, v in zip(encs, vals) if v is None}
        vals = [0.0 if v is None else v for v in vals]
        s_enc, s_val, s_err = _sort_by_metric(encs, vals, errs, na,
                                              "desc" if direction == "higher" else "asc")
        _bar_with_ci(ax, s_enc, s_val, s_err, ylabel=title,
                     title=title, na_set=na, direction=direction)
    fig.suptitle("iter18 EVAL scorecard — per-encoder TEST artifacts · 95% BCa CI where available",
                 fontsize=16, fontweight="bold")
    fig.subplots_adjust(hspace=0.6, wspace=0.3, top=0.94)
    save_fig(fig, str(out_dir / "eval_scorecard"))
    plt.close(fig)


def _mw_dump(blocks, ev, out_dir: Path):
    """train_metrics.{json,csv} + eval_metrics.{json,csv} — same schema as the watch script."""
    import csv
    train = [{"arm": b["arm"], "status": b["st"], "n_train": b["n_tr"], "n_val": b["n_va"],
              "head_best": b["head_best"], "kept_idx": b["kept_i"],
              "probes": [{**{k: r.get(k) for k in _MW_TRAIN_PROBE_KEYS},
                          "verdict": v, "kept": (i == b["kept_i"])}
                         for i, (r, v) in enumerate(b["rows"])]}
             for b in blocks]
    (out_dir / "train_metrics.json").write_text(json.dumps(train, indent=1))
    with open(out_dir / "train_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ckpt", *_MW_TRAIN_PROBE_KEYS, "verdict", "kept"])
        for t in train:
            for i, pr in enumerate(t["probes"]):
                w.writerow([t["arm"], i + 1, *[pr.get(k) for k in _MW_TRAIN_PROBE_KEYS],
                            pr["verdict"], pr["kept"]])
    ev_json = [{"encoder": r["_enc_full"], "n_test": r["n_te"][0],
                **{k: {"mean": r["_raw"][k][0], "ci_half": r["_raw"][k][1]}
                   for k in _MW_EVAL_RAW_KEYS if k in r["_raw"]}}
               for r in ev]
    (out_dir / "eval_metrics.json").write_text(json.dumps(ev_json, indent=1))
    with open(out_dir / "eval_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "n_test", *[c for k in _MW_EVAL_RAW_KEYS for c in (k, k + "_ci")]])
        for r in ev:
            raw = r["_raw"]
            cells = [c for k in _MW_EVAL_RAW_KEYS for c in raw.get(k, (None, None))]
            w.writerow([r["_enc_full"], r["n_te"][0], *cells])


def regen_metrics_watch(outputs_root: Path, out_base: Path, mtag: str):
    """Regenerate ALL metrics_watch artifacts into out_base/<BACKBONE>/ from the eval + train
    artifacts under outputs_root — no scripts/ dependency. Data files keep EVERY arm;
    graphs hide ITER18_SKIP_ARMS (same contract as the watch script)."""
    out_dir = out_base / _MW_BACKBONE
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = _mw_skip_arms()
    skip_encs = {_mw_enc_name(_MW_ARM2ENC[a]) for a in skip if a in _MW_ARM2ENC}
    blocks = _mw_train_blocks(outputs_root, mtag)
    ev = _mw_eval_rows(outputs_root)
    _mw_dump(blocks, ev, out_dir)            # FULL — every arm, incl. skipped
    vis_blocks = [b for b in blocks if b["arm"] not in skip]
    vis_ev = [r for r in ev if r["_enc_full"] not in skip_encs]
    _mw_render_graphs(vis_blocks, vis_ev, out_dir)
    if skip:
        print(f"  [metrics-watch] ⏷ {sorted(skip)} hidden from graphs · still in the json/csv")
    print(f"  [metrics-watch] regenerated → {out_dir}/ · "
          f"{{train_trajectories,kept_scorecard,eval_scorecard}}.{{png,pdf}} + "
          f"{{train,eval}}_metrics.{{json,csv}}")


def main():
    p = argparse.ArgumentParser(
        description="m13 — 14-metric bar-with-CI viz for the probe eval suite (hero: §3.3c).")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--action-probe-root", type=Path, default=None)
    p.add_argument("--motion-cos-root",   type=Path, default=None)
    p.add_argument("--future-mse-root",   type=Path, default=None)
    p.add_argument("--taxonomy-root",     type=Path, default=None,
                   help="m12c probe_taxonomy output dir (per_dim_acc.json) — OPTIONAL (graceful).")
    p.add_argument("--predictor-temporal-root", type=Path, default=None,
                   help="m12e output dir (predictor_temporal_per_variant.json) — OPTIONAL.")
    p.add_argument("--encoder-temporal-root",   type=Path, default=None,
                   help="m12f output dir (encoder_temporal_per_variant.json) — OPTIONAL.")
    p.add_argument("--output-dir",        type=Path, required=True)
    p.add_argument("--reference-hero", action="append", default=None, metavar="BACKBONE=PNG",
                   help="Paste a pre-made per-backbone Δ-vs-frozen hero into eval/<BACKBONE>/ verbatim "
                        "(e.g. a prior-iter champion whose data isn't in THIS run) and include it in the "
                        "combined vertical stack. Repeatable. Path supplied via CLI — never hardcoded.")
    p.add_argument("--metrics-watch-out", type=Path, default=None,
                   help="Regenerate the metrics_watch artifacts (3 figures + train/eval_metrics "
                        "json/csv) into <this dir>/<ITER18_BACKBONE>/ — self-contained, no "
                        "scripts/iter18_* dependency. Requires --outputs-root.")
    p.add_argument("--outputs-root", type=Path, default=None,
                   help="Run outputs root (e.g. outputs/poc) — m09 train dirs + eval artifact "
                        "subdirs live under it. Required by --metrics-watch-out.")
    p.add_argument("--metrics-watch-only", action="store_true",
                   help="Exit after the metrics_watch regeneration (skip the m13 eval/ plots).")
    add_wandb_args(p)
    args = p.parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # ── metrics_watch regeneration (self-contained; see the _mw_* section above) ──
    if args.metrics_watch_out is not None:
        if args.outputs_root is None:
            sys.exit("ERROR: --metrics-watch-out requires --outputs-root (e.g. outputs/poc)")
        init_style()
        regen_metrics_watch(args.outputs_root, args.metrics_watch_out, mode.lower())
        if args.metrics_watch_only:
            return
    elif args.metrics_watch_only:
        sys.exit("ERROR: --metrics-watch-only requires --metrics-watch-out")

    sub_dir = args.output_dir / "eval"
    if sub_dir.exists():
        nfile = sum(1 for _ in sub_dir.rglob("*") if _.is_file())
        print(f"  [m13_eval_plot] wiping eval/ — {nfile} stale file(s)")
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = sub_dir

    # The 3 legacy headline roots are REQUIRED (FAIL LOUD); temporal/taxonomy are graceful.
    missing = [name for name, val in [
        ("--action-probe-root", args.action_probe_root),
        ("--motion-cos-root",   args.motion_cos_root),
        ("--future-mse-root",   args.future_mse_root),
    ] if val is None]
    if missing:
        sys.exit(f"ERROR: m13 requires: {', '.join(missing)}")

    boot_str = f"{N_BOOTSTRAP // 1000} K bootstrap" if N_BOOTSTRAP >= _K_FMT_MIN else f"{N_BOOTSTRAP} bootstrap"

    wb = init_wandb("m13_eval_plot", mode, config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()
        srcs = {
            "action":   _load_json(args.action_probe_root / artifact("probe_paired_delta"), "Stage 4"),
            "motion":   _load_json(args.motion_cos_root / artifact("probe_motion_cos_paired"), "Stage 7"),
            "future":   _load_json(args.future_mse_root / artifact("probe_future_mse_per_variant"), "Stage 9"),
            "taxonomy": _opt_json(args.taxonomy_root / artifact("per_dim_acc")) if args.taxonomy_root else None,
            "pred":     _opt_json(args.predictor_temporal_root / artifact("predictor_temporal_per_variant"))
                        if args.predictor_temporal_root else None,
            "enc":      _opt_json(args.encoder_temporal_root / artifact("encoder_temporal_per_variant"))
                        if args.encoder_temporal_root else None,
        }
        # Encoder union across all present sources (no hardcoded list).
        encoders = sorted(
            set(srcs["action"].get("by_encoder", {}))
            | set(srcs["motion"].get("by_encoder", {}))
            | {v for v, e in srcs["future"].get("by_variant", {}).items() if isinstance(e, dict)}
        )
        if not encoders:
            sys.exit("FATAL: no encoders found in the action/motion/future JSONs")
        _hide = _skip_encoders_from_env(encoders)   # honor ITER18_SKIP_ARMS in the PLOTS
        if _hide:
            print(f"  [--skip-arms] hiding {len(_hide)} arm(s) from the plots: "
                  f"{sorted(_short_label(e) for e in _hide)}")
            encoders = [e for e in encoders if e not in _hide]
            if not encoders:
                sys.exit("FATAL: all encoders skipped via ITER18_SKIP_ARMS — nothing to plot")

        metrics = _load_all_metrics(srcs, encoders)
        pbar = make_pbar(total=len(_CATALOG), desc="m13_eval_plot", unit="panel")
        n_panels = plot_all_metric_bars(metrics, args.output_dir, boot_str)
        pbar.update(len(_CATALOG))
        pbar.close()
        plotted = sorted(k for k in metrics if _direction_of(k) != "signed")
        signed = sorted(k for k in metrics if _direction_of(k) == "signed")
        print(f"  {n_panels}/{len(_CATALOG)} bar panels written ({len(plotted)} plotted): {plotted}"
              + (f"  ·  signed (CSV-only, not plotted): {signed}" if signed else ""))

        # ── HERO (B1 table-with-CI + B2 Δ-vs-frozen heatmap) — plan §7.3 ──
        # baseline = runtime-discovered 'frozen' encoder (no hardcoded name); needs ≥1 contender.
        frozen = _pick_frozen_ref(encoders)
        # Split: the trained arms + their same-backbone frozen reference = the 'core' hero views.
        # The external image/video baselines (frozen/other) would clutter those views with mostly-N/A
        # predictor cells, so they get a SEPARATE absolute-value scorecard (no Δ) instead.
        core = [e for e in encoders if _arm_family(e) in ("surgery", "pretrain", "ft") or e == frozen]
        frozen_only = [e for e in encoders if _arm_family(e) in ("frozen", "other")]
        # Reference heroes: BACKBONE=PNG to paste verbatim (a prior-iter champion not evaluated here).
        ref_heroes = {}
        for s in (args.reference_hero or []):
            if "=" not in s:
                sys.exit(f"ERROR: --reference-hero must be BACKBONE=PNG, got {s!r}")
            bb, png = s.split("=", 1)
            ref_heroes[bb] = png
        if frozen and len(core) >= _MIN_COMPARABLE:
            # PER-BACKBONE views ONLY — each backbone's arms vs its OWN frozen → eval/<backbone>/.
            # We deliberately do NOT build a cross-backbone combined grid: the backbones have DIFFERENT
            # frozen baselines, so averaging / mixing their Δs is meaningless. The combined overview is a
            # pure VERTICAL IMAGE STACK of these panels (below) — append, never average.
            stacked = []
            for bb in _backbones_present(encoders):
                bb_enc = [e for e in encoders if _backbone_of(e) == bb]
                bb_frozen = _pick_frozen_ref(bb_enc)
                bb_core = [e for e in bb_enc if _arm_family(e) in ("surgery", "pretrain", "ft") or e == bb_frozen]
                if bb_frozen and len(bb_core) >= _MIN_COMPARABLE:
                    print(f"  [per-backbone] {bb} → {args.output_dir.name}/{bb}/  "
                          f"({len(bb_core)} arms vs {_short_label(bb_frozen)})")
                    _emit_hero_suite(metrics, bb_core, bb_frozen, args.output_dir / bb, boot_str)
                    stacked.append(bb)
                else:
                    print(f"  [per-backbone] {bb} — skipped (frozen={bb_frozen}, {len(bb_core)} core arms)")
            # Paste any reference (prior-iter) heroes into their eval/<backbone>/ verbatim, then stack.
            for bb, png in ref_heroes.items():
                src = Path(png)
                if not src.exists():
                    print(f"  [reference-hero] {bb}: {png} not found — skip"); continue
                dst = args.output_dir / bb / "m13_hero_raw_values.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dst))
                src_pdf = src.with_suffix(".pdf")             # bring the matching .pdf too (both png & pdf)
                if src_pdf.exists():
                    shutil.copy(str(src_pdf), str(dst.with_suffix(".pdf")))
                print(f"  [reference-hero] {bb} ← {png}")
                if bb not in stacked:
                    stacked.append(bb)
            # COMBINED = the per-backbone panels APPENDED VERTICALLY (in _BB_TAG order), NO averaging.
            bb_order = [pre.rstrip("_") for pre, _lt, _st in _BB_TAG]
            _present = [b for b in bb_order if b in stacked]
            _vstack_panels(args.output_dir, _present, "m13_hero_raw_values",
                           "m13_hero_raw_values")
            # COMBINED surgery-vs-pretrain overview = a colourful paired-diff HEATMAP (metrics ×
            # backbones), NOT the sparse stacked forests. Per-backbone forests still live in eval/<bb>/.
            plot_paired_diff_heatmap(metrics, encoders, args.output_dir, boot_str)
        else:
            print(f"  [hero] skipped — needs a 'frozen' baseline + ≥2 core encoders (got {core})")
        # FROZEN-only absolute scorecard (image/video baselines + the V-JEPA frozen reference).
        if len(frozen_only) >= _MIN_COMPARABLE:
            plot_frozen_scorecard(metrics, frozen_only, args.output_dir, boot_str)

        # wandb metric upload — generic prefix + encoder name (NO hardcoded keys).
        wb_metrics = {"n_encoders": len(encoders), "n_metric_panels": n_panels}
        for mkey, md in metrics.items():
            for e, (val, _ci) in md["by_encoder"].items():
                wb_metrics[f"{mkey}__{e}"] = float(val)
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
