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
  (§3.3c-hero will add m13_hero_table + m13_hero_surgery_vs_frozen.)

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
import shutil
import sys
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
    "frozen": "frozen", "pretrain_encoder": "pretrain encoder", "pretrain_2X_encoder": "pretrain_2X encoder",
    "pretrain_head": "pretrain head", "surgical_3stage_DI_encoder": "surgery 3stage_DI encoder",
    "surgical_noDI_encoder": "surgery noDI encoder", "surgical_3stage_DI_head": "surgery 3stage_DI head",
    "surgical_noDI_head": "surgery noDI head",
}
_ARM_SHORT = {
    "frozen": "frozen", "pretrain_encoder": "pre-enc", "pretrain_2X_encoder": "pre-2X", "pretrain_head": "pre-hd",
    "surgical_3stage_DI_encoder": "s3DI-enc", "surgical_noDI_encoder": "sNoDI-enc",
    "surgical_3stage_DI_head": "s3DI-hd", "surgical_noDI_head": "sNoDI-hd",
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


def _color_for(enc: str, idx: int) -> str:
    """Pick a color for an encoder bar (canonical map first, else deterministic rotation)."""
    if enc in ENCODER_COLORS:
        return ENCODER_COLORS[enc]
    if enc.startswith("vjepa"):
        return ENCODER_COLORS["vjepa"]
    fallback_key = _FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)]
    return COLORS.get(fallback_key, COLORS["gray"])


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
    """Sign-prefixed compact number for tight heatmap cells."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 100:
        return f"{x:+.0f}"
    if ax >= 1:
        return f"{x:+.2f}"
    if ax >= 0.01:
        return f"{x:+.3f}"
    return f"{x:+.4f}"


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
    # even crown the frozen baseline). Decisive metric → the winning arm; dead-heat (|Δ|<ε) → tie
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
                 f"surgery-vs-pretrain champion duel (tie if within ε) · {boot_str}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_table"))
    import csv
    with open(str(output_dir / "m13_hero_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  [hero-table] m13_hero_table.{{png,pdf,csv}} — {len(scorable)} metric rows drawn "
          f"(+signed 'order' in CSV only) × {len(col_labels)} cols (transposed) · WINNER col = champion duel (ties shown)")


def plot_hero_heatmap(metrics: dict, encoders: list, frozen: str, output_dir: Path):
    """B2: direction-normalized Δ-vs-frozen heatmap WITH numbers. rows=contenders, cols=hero
    metrics. Colour = sign-corrected Δ (positive=better; lower-better metrics negated),
    per-column normalized to [-1,1] for cross-metric comparability (RdYlGn). Each cell now
    PRINTS the sign-corrected Δ and its 95% BCa CI [lo, hi] in the metric's native units
    (iter16 fix — was colour-only); BOLD when the CI excludes 0 (significant vs frozen)."""
    cat = [c for c in _hero_catalog(metrics) if c[3] in ("higher", "lower")]  # win/loss metrics only — signed (order) is diagnostic, lives in hero-table CSV
    contenders = [e for e in encoders if e != frozen]
    if not cat or not contenders:
        print("  [hero-heatmap] need a frozen baseline + ≥1 contender + metrics — skip")
        return
    raw = np.full((len(contenders), len(cat)), np.nan)
    cells = {}                                          # (i,j) -> (sc, clo, chi, sig)
    for i, e in enumerate(contenders):
        for j, (key, _fam, _on, direction, _yl, _cap, _lay) in enumerate(cat):
            dvf = _delta_v_vs_frozen(metrics[key]["deltas"], e, frozen)
            if dvf is None:
                continue
            d, lo, hi = dvf
            sc, clo, chi = (d, lo, hi) if direction != "lower" else (-d, -hi, -lo)  # +=better, bounds too
            raw[i, j] = sc
            cells[(i, j)] = (sc, clo, chi, _ci_excludes_zero(lo, hi))
    norm = np.full_like(raw, np.nan)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        fin = col[np.isfinite(col)]
        amax = float(np.abs(fin).max()) if fin.size else 0.0
        norm[:, j] = col / amax if amax > 0 else col * 0.0
    fig, ax = plt.subplots(figsize=(3.0 + 1.75 * len(cat), 2.6 + 1.2 * (len(contenders) + 1)))
    # alpha=0.6 → pastel RdYlGn (matches the hero TABLE); keeps red→green meaning while staying
    # light enough for plain BLACK BOLD numbers — no halo/outline needed (iter16 fix).
    im = ax.imshow(np.ma.masked_invalid(norm), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", alpha=0.6)
    ax.grid(False)                                      # iter16 fix: kill the distracting white gridlines
    ax.tick_params(length=0)
    ax.set_xticks(range(len(cat)))
    ax.set_xticklabels([f"{c[0]}\n{_DIR_TAG[c[3]]}" for c in cat], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(list(range(len(contenders))) + [len(contenders)])
    ax.set_yticklabels([_display_label(e) for e in contenders] + ["WINNER"], fontsize=10)
    # Plain BLACK BOLD numbers (no halo): Δ on top, CI below. SPOTLIGHT (thick blue outline +
    # enlarged Δ font) = the champion-duel WINNING ARM per metric — SAME _family_verdict as the
    # WINNER row / hero table / scoreboard, NOT raw best-Δ. Tie metric → no box (no single winner).
    _surg = [e for e in contenders if _arm_family(e) == "surgery"]
    _pre = [e for e in contenders if _arm_family(e) == "pretrain"]
    _per = _family_verdict(metrics, encoders, frozen)[3]
    _row = {e: i for i, e in enumerate(contenders)}
    col_winner = {}                                     # j (metric col) -> i (winning-arm row); decisive only
    for j, (key, *_r) in enumerate(cat):
        v = _per.get(key)
        champ = (_family_champion(metrics, key, frozen, _surg) if v == "surgery"
                 else _family_champion(metrics, key, frozen, _pre) if v == "pretrain" else None)
        if champ in _row:
            col_winner[j] = _row[champ]
    for (i, j), (sc, clo, chi, _sig) in cells.items():
        win = col_winner.get(j) == i
        ax.text(j, i - 0.26, _fmt_compact(sc), ha="center", va="center",
                fontsize=18 if win else 13, fontweight="bold", color="black")
        ax.text(j, i + 0.18, f"[{_fmt_compact(clo)},\n{_fmt_compact(chi)}]", ha="center", va="center",
                fontsize=8, fontweight="bold", color="black")
        if win:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a3cff", linewidth=3.0, zorder=4))
    # WINNER row (champion duel — best-surgery vs best-pretrain per metric), mirroring the grouped
    # plot's [S/P/=] band. One extra row BELOW the contenders; signed metrics (order) aren't a
    # win/loss → neutral "·". Verdict single-sourced via _family_verdict (same tally as scoreboard).
    ny = len(contenders)
    _verdict_per = _per                                 # reuse (computed above for the spotlight)
    _VC = {"surgery": ((0.17, 0.63, 0.17, 0.6), "S"), "pretrain": ((0.84, 0.15, 0.16, 0.6), "P"),
           "tie": ((0.6, 0.6, 0.6, 0.35), "=")}
    for j, c in enumerate(cat):
        vcol, vlab = _VC.get(_verdict_per.get(c[0]), ((0.85, 0.85, 0.85, 0.35), "·"))
        ax.add_patch(Rectangle((j - 0.5, ny - 0.5), 1, 1, facecolor=vcol, edgecolor="white",
                               lw=1.0, zorder=2))
        ax.text(j, ny, vlab, ha="center", va="center", fontsize=14, fontweight="bold",
                color="black", zorder=3)
    ax.set_ylim(ny + 0.5, -0.5)                         # extend bottom to include the WINNER row
    ax.axhline(ny - 0.5, color="black", lw=1.5)         # separator above the WINNER row
    ax.set_title(f"HERO — Δ vs {_display_label(frozen)}  ·  per cell: Δ (top) and 95% BCa CI (bottom), "
                 "sign-corrected so + = better\ncolour: red = worst, green = best (per-metric min-max) · "
                 "column badge ↑/↓ = better-direction · BLUE BOX + big Δ = champion-duel winning arm "
                 "(none on ties — matches WINNER row)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="red = worst   →   green = best (per-metric normalized Δ)")
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m13_hero_surgery_vs_frozen"))
    print(f"  [hero-heatmap] m13_hero_surgery_vs_frozen.{{png,pdf}} — {len(contenders)} × {len(cat)} (Δ+CI, bold black)")


def plot_frozen_scorecard(metrics: dict, frozen_encoders: list, output_dir: Path, boot_str: str):
    """FROZEN-only leaderboard — ABSOLUTE head-metric values (NO Δ) for the image/video baselines +
    the V-JEPA frozen reference. rows = encoders (sorted best→worst on the 1st metric), cols = the
    metrics ALL frozen encoders share (the 3 head metrics; predictor metrics are V-JEPA-only → not
    comparable across baselines, excluded). Per-metric colour (green = best), best cell blue-boxed.
    Kept separate from the trained-arm hero views, which these baselines would clutter with N/A."""
    cat = _hero_catalog(metrics)
    cols = [c for c in cat if c[3] in ("higher", "lower")
            and all(metrics[c[0]]["by_encoder"].get(e) is not None for e in frozen_encoders)]
    if not cols or len(frozen_encoders) < 2:
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
    with open(str(output_dir / "m13_frozen_scorecard.csv"), "w", newline="") as f:
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
    arms = sorted(e for e in encoders if _arm_family(e) in ("surgery", "pretrain"))
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
    have = {_backbone_of(e) for e in encoders if _arm_family(e) in ("surgery", "pretrain")}
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
        gd = _good_orient(dv[0], dv[1], dv[2], _direction_of(key))[0]
        if abs(gd) < _TIE_EPS:
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
    colors = ["#2ca02c" if _arm_family(a) == "surgery" else "#1f77b4" for a in order]
    ax.barh(y, [arm_wins.get(a, 0) for a in order], color=colors, alpha=0.85)
    for i, a in enumerate(order):
        ax.text(arm_wins.get(a, 0), i, f" {arm_wins.get(a, 0)}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([_wrap_name(a) for a in order], fontsize=8, rotation=20, va="center")
    ax.set_xlabel(f"# metrics won OUTRIGHT  (of {ns + npr + nt}; {nt} ties not awarded)")
    champ = "SURGERY" if ns > npr else ("PRETRAIN" if npr > ns else "TIE")
    fig.suptitle(f"SCOREBOARD —  SURGERY {ns} · PRETRAIN {npr} · TIE {nt}   (champion duel · winner: {champ})",
                 fontsize=13, fontweight="bold")
    ax.set_title("green = surgery arm · blue = pretrain arm · bar = outright metric wins by that arm", fontsize=9)
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
    winner = [{"surgery": "S", "pretrain": "P", "tie": "="}.get(per.get(k, "tie"), "·") for k in keys]
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
            (0.84, 0.15, 0.16, 0.6) if w == "P" else (0.6, 0.6, 0.6, 0.35))
    fig, ax = plt.subplots(figsize=(max(11.0, 1.05 * ncol + 3), 0.95 * nrow + 3))
    ax.imshow(rgba, aspect="auto")
    col_winner = {j: int(np.nanargmax(M[:, j])) for j in range(len(keys)) if np.any(~np.isnan(M[:, j]))}
    for i in range(len(arms)):
        for j in range(len(keys)):
            if not np.isnan(M[i, j]):
                win = col_winner.get(j) == i           # per-metric best arm → spotlight (blue box + bigger Δ)
                ax.text(j, i - 0.24, _fmt_compact(M[i, j]), ha="center", va="center",
                        fontsize=15 if win else 10, fontweight="bold", color="black")
                if (i, j) in CI:
                    lo, hi = CI[(i, j)]
                    ax.text(j, i + 0.2, f"[{lo:+.3f},\n{hi:+.3f}]", ha="center", va="center",
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
    verdict = "SURGERY" if _ns > _np else ("PRETRAIN" if _np > _ns else "SPLIT")
    ax.set_title(f"Δ vs frozen (good-oriented, value + 95% CI) · PRETRAIN block / SURGERY block · "
                 f"SCORE = mean per-metric-normalized (0–1) · BLUE BOX + big Δ = best arm per metric\n"
                 f"WINNER row (champion duel — S=surgery, P=pretrain, ==tie):  "
                 f"surgery {_ns} · pretrain {_np} · tie {_nt}  →  {verdict}", fontsize=10)
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


def _vstack_heroes(output_dir, backbones, out_name):
    """COMBINED overview = the per-backbone Δ-vs-frozen heatmaps APPENDED VERTICALLY into one PNG.
    A pure image concat of the already-saved eval/<backbone>/m13_hero_surgery_vs_frozen.png files —
    NO cross-backbone math: each panel keeps its own arms vs its OWN frozen (averaging Δs across
    backbones with different frozen baselines is meaningless). Panels stacked top→bottom in order."""
    panels = [(bb, output_dir / bb / "m13_hero_surgery_vs_frozen.png") for bb in backbones]
    panels = [(bb, Image.open(str(p)).convert("RGB")) for bb, p in panels if p.exists()]
    if not panels:
        print("  [stack] no per-backbone heroes present — skip combined stack")
        return
    w = max(im.width for _, im in panels)
    # Big LABELLED header band per panel so each heatmap's BACKBONE + model size is unmistakable.
    fpath = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    longest = max((_BB_LABEL.get(bb, bb) for bb, _ in panels), key=len)
    fsize = 110
    font = ImageFont.truetype(fpath, fsize)
    while fsize > 28 and font.getlength(longest) > w * 0.94:       # auto-shrink to fit the panel width
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
    add_wandb_args(p)
    args = p.parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

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

    boot_str = f"{N_BOOTSTRAP // 1000} K bootstrap" if N_BOOTSTRAP >= 1000 else f"{N_BOOTSTRAP} bootstrap"

    wb = init_wandb("m13_eval_plot", mode, config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()
        srcs = {
            "action":   _load_json(args.action_probe_root / "probe_paired_delta.json", "Stage 4"),
            "motion":   _load_json(args.motion_cos_root / "probe_motion_cos_paired.json", "Stage 7"),
            "future":   _load_json(args.future_mse_root / "probe_future_mse_per_variant.json", "Stage 9"),
            "taxonomy": _opt_json(args.taxonomy_root / "per_dim_acc.json") if args.taxonomy_root else None,
            "pred":     _opt_json(args.predictor_temporal_root / "predictor_temporal_per_variant.json")
                        if args.predictor_temporal_root else None,
            "enc":      _opt_json(args.encoder_temporal_root / "encoder_temporal_per_variant.json")
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
        core = [e for e in encoders if _arm_family(e) in ("surgery", "pretrain") or e == frozen]
        frozen_only = [e for e in encoders if _arm_family(e) in ("frozen", "other")]
        # Reference heroes: BACKBONE=PNG to paste verbatim (a prior-iter champion not evaluated here).
        ref_heroes = {}
        for s in (args.reference_hero or []):
            if "=" not in s:
                sys.exit(f"ERROR: --reference-hero must be BACKBONE=PNG, got {s!r}")
            bb, png = s.split("=", 1)
            ref_heroes[bb] = png
        if frozen and len(core) >= 2:
            # PER-BACKBONE views ONLY — each backbone's arms vs its OWN frozen → eval/<backbone>/.
            # We deliberately do NOT build a cross-backbone combined grid: the backbones have DIFFERENT
            # frozen baselines, so averaging / mixing their Δs is meaningless. The combined overview is a
            # pure VERTICAL IMAGE STACK of these panels (below) — append, never average.
            stacked = []
            for bb in _backbones_present(encoders):
                bb_enc = [e for e in encoders if _backbone_of(e) == bb]
                bb_frozen = _pick_frozen_ref(bb_enc)
                bb_core = [e for e in bb_enc if _arm_family(e) in ("surgery", "pretrain") or e == bb_frozen]
                if bb_frozen and len(bb_core) >= 2:
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
                dst = args.output_dir / bb / "m13_hero_surgery_vs_frozen.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dst))
                src_pdf = src.with_suffix(".pdf")             # bring the matching .pdf too (both png & pdf)
                if src_pdf.exists():
                    shutil.copy(str(src_pdf), str(dst.with_suffix(".pdf")))
                print(f"  [reference-hero] {bb} ← {png}")
                if bb not in stacked:
                    stacked.append(bb)
            # COMBINED = the per-backbone heroes APPENDED VERTICALLY (in _BB_TAG order), NO averaging.
            bb_order = [pre.rstrip("_") for pre, _lt, _st in _BB_TAG]
            _vstack_heroes(args.output_dir, [b for b in bb_order if b in stacked],
                           "m13_grouped_winner_surgery_vs_pretrain")
        else:
            print(f"  [hero] skipped — needs a 'frozen' baseline + ≥2 core encoders (got {core})")
        # FROZEN-only absolute scorecard (image/video baselines + the V-JEPA frozen reference).
        if len(frozen_only) >= 2:
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
