#!/usr/bin/env python3
"""iter18 live METRICS table — EVERY in-training probe checkpoint + every EVAL metric.

Sibling of scripts/iter18_poc_status.py (which tracks job STATE/ETA — this one tracks NUMBERS).
Read-only: no uploads, no plot rebuilds, no GPU — safe under `watch` next to a live run.

Sources (single source of truth = the pipeline's own artifacts; clip counts are EXTRACTED
live, never hardcoded):
  TRAIN · outputs/<mode>/vjepa_2_1_vitG/<m09dir>/probe_history.jsonl  (ALL probe checkpoints)
        · outputs/<mode>/vjepa_2_1_vitG/<m09dir>/training_summary.json (✅ done + head best)
        · logs/iter18_ngpu_<mode>_train_<arm>_*.log                   (per-arm n_train/n_val)
        · data/<local_data>/{train_pool,val_split,test_split}.json    (global split sizes, counted)
  EVAL  · probe_action/<enc>/test_metrics.json            → action top-1 (±CI) + n_test
        · probe_taxonomy/<enc>/test_metrics.json          → taxonomy macro test_mean over dims
        · probe_motion_cos/<enc>/intra_inter_ratio.json   → motion-cos score (±CI)
        · probe_future_mse/<enc>/aggregate_mse.json       → future-frame MSE
        · predictor_temporal/<enc>/aggregate_{rollout,causal,tdist,maskratio,order,teacher_free}

USAGE:
  python -u scripts/iter18_poc_metrics.py                  # POC
  python -u scripts/iter18_poc_metrics.py --mode SANITY
  watch -n60 'python -u scripts/iter18_poc_metrics.py'     # live, refresh every 60s
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from iter18_poc_ngpu import ARM2DIR, ARM2ENC, BACKBONE, enc_name, enc_prefix  # noqa: E402  (canonical DAG names + backbone-aware encoder naming)
from utils.config import get_local_data_dir              # noqa: E402  (yaml-driven data dir)
from utils.data_paths import artifact                    # noqa: E402  (canonical artifact names)

# Runbook §2 arm order (same as iter18_poc_status.py).
TRAIN_ORDER = [
    "pretrain_encoder",
    "surgery_3stage_DI_encoder", "surgery_noDI_encoder",
    "surgery_3stage_DI_head", "surgery_noDI_head",
    "surgical_autorgn_encoder", "surgery_raw_encoder",
    "full_ft_encoder", "lpft_encoder",
    "peft_lora_encoder", "peft_dora_encoder",
    "cassle_encoder", "ewc_encoder",
]
_HEAD_ARMS = {"surgery_3stage_DI_head", "surgery_noDI_head"}
# predictor_temporal aggregate families (aggregate_<name>.json each, headline key "mean").
_PT_FAMILIES = ["rollout", "causal", "tdist", "maskratio", "order", "teacher_free"]
_LOG_HEAD_BYTES = 16000   # the split-count prints land in the first few KB of each arm log
_FRESH_LOG_S = 1800       # a per-job log touched within this window = arm is live
                          # (the quietest phase — 451-clip probe ENCODE — writes nothing
                          #  for ~15 min on a busy GPU; verified noDI_head 11:40→11:56)


def _jload(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl(p: Path) -> list:
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
                # a probe record mid-write under a live run — show what's complete
                return rows
    return rows


def _f(v, nd=4):
    """Format a float compactly; '—' when absent."""
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _ci(d, mean_key, ci_key, nd=3):
    """'mean±half' from a metrics dict whose CI sub-dict carries ci_half."""
    if not d or mean_key not in d:
        return "—"
    half = d.get(ci_key, {}).get("ci_half") if isinstance(d.get(ci_key), dict) else None
    return f"{d[mean_key]:.{nd}f}" + (f"±{half:.{nd}f}" if half is not None else "")


def _ci2(d, mean_key, ci_key, nd=3):
    """(mean_str, '±half') — the 95% CI on its OWN line UNDER the mean (iter18 2026-06-08, user
    order: stack the range on a second row). ('—','') when the metric is absent."""
    if not d or mean_key not in d:
        return ("—", "")
    half = d.get(ci_key, {}).get("ci_half") if isinstance(d.get(ci_key), dict) else None
    return (f"{d[mean_key]:.{nd}f}", f"±{half:.{nd}f}" if half is not None else "")


def split_sizes() -> str:
    """Global split sizes COUNTED live from the split artifacts (never hardcoded)."""
    dd = get_local_data_dir()
    out = []
    for label, key in [("train_pool", "train_pool"), ("val", "val_split"), ("test", "test_split")]:
        blob = _jload(dd / artifact(key))
        # train_pool.json = {"clip_keys": [...]}; val/test_split.json = plain lists.
        keys = blob.get("clip_keys") if isinstance(blob, dict) else blob
        out.append(f"{label}={len(keys)}" if isinstance(keys, list) else f"{label}=—")
    return " · ".join(out)


def arm_train_val_n(mtag, arm):
    """Per-arm effective (n_train, n_val), parsed live from the arm's newest scheduler log.
    Formats seen across families (all printed by the trainers themselves):
      'train/val split: 7,021 train / 451 val'            (m09c factor arms: pool ∩ manifest)
      'Train clips: 7,724 | Val clips: 451'                (m09a)
      '[POC] Loaded subset: 451 clip keys from val_split'  (m09c2 heads; train from train_pool line)
    """
    logs = sorted((p for p in (REPO / "logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log") if p.exists()),
                  key=lambda p: p.stat().st_mtime)   # if p.exists(): skip DANGLING symlinks (moved logs) — .stat() on one crashes the whole watch
    if not logs:
        return None, None
    try:
        head = logs[-1].read_text(errors="replace")[:_LOG_HEAD_BYTES]
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


def train_blocks(mtag):
    """One block per arm: meta (status, n_train/n_val) + ONE ROW PER PROBE CHECKPOINT with a
    selector verdict per row (🎯 promoted / ✋ held / ←KEPT = the checkpoint that survived)."""
    blocks = []
    for arm in TRAIN_ORDER:
        d = REPO / f"outputs/{mtag}/{BACKBONE}/{ARM2DIR[arm]}"
        hist = _jsonl(d / "probe_history.jsonl")
        summ = _jload(d / artifact("training_summary"))
        # 🔄 also when the arm is TRAINING but hasn't reached its first probe yet
        # (first probe ≈ step 109 ≈ ~55 min in): a per-job log written in the last
        # few minutes means a live process, even with zero probe records.
        logs = sorted((q for q in (REPO / "logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log") if q.exists()),
                      key=lambda q: q.stat().st_mtime)   # if q.exists(): skip DANGLING symlinks (moved logs)
        log_fresh = bool(logs) and (time.time() - logs[-1].stat().st_mtime) < _FRESH_LOG_S
        status = "✅" if summ else ("🔄" if (hist or log_fresh) else "⬚")
        n_tr, n_va = arm_train_val_n(mtag, arm)
        # Selector replay — enc arms (incl. pretrain) keep the lowest future_l1 seen so far;
        # head arms select on head-val_loss (not a probe metric) → rows are diagnostics ("·").
        rows, kept_i = [], None
        best = float("inf")
        for i, r in enumerate(hist):
            if arm in _HEAD_ARMS:
                verdict = "·"
            elif r.get("future_l1") is not None and r["future_l1"] < best:
                best, kept_i, verdict = r["future_l1"], i, "🎯"
            else:
                verdict = "✋"
            rows.append((r, verdict))
        head_best = ""
        if arm in _HEAD_ARMS and summ and "best_val_loss" in summ:
            head_best = f"sel=head-vloss {summ['best_val_loss']:.3f}@ep{summ.get('best_epoch', '?')}"
        blocks.append({"arm": arm, "st": status, "n_tr": n_tr, "n_va": n_va,
                       "rows": rows, "kept_i": kept_i, "head_best": head_best})
    return blocks


def skip_arms(mtag):
    """Arms to HIDE from the printed TABLES + the 3 graphs (NOT the data files — those keep
    everything, user order 2026-06-08). Source: ITER18_SKIP_ARMS env (explicit, space-separated),
    else the run's own '[--skip-arms] dropped [...]' banner in the latest scheduler log → the watch
    auto-honors whatever the run skipped, no extra flag. Empty when neither exists (show all)."""
    env = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    if env:
        # strip stray quote chars per token: the runbook watch line uses ITER18_SKIP_ARMS=\"$SKIP\"
        # (correct inside watch "..."); pasted DIRECTLY at a prompt the \" become LITERAL quotes that
        # bash bakes into the value → tokens "cassle_encoder … surgery_noDI_head" never match the arms.
        return {tok.strip().strip("\"'") for tok in env.split() if tok.strip().strip("\"'")}
    cands = [p for p in (REPO / "logs").glob(f"iter18_ngpu_{mtag}*.log")
             if not any(s in p.name for s in ("_train_", "_eval_", "_pt_", "_s3_"))]
    if not cands:
        return set()
    try:
        txt = max(cands, key=lambda p: p.stat().st_mtime).read_text(errors="replace")
    except OSError:
        return set()
    m = re.search(r"\[--skip-arms\] dropped \[(.*?)\]", txt)
    return set(re.findall(r"'([^']+)'", m.group(1))) if m else set()


def eval_rows(mtag):
    base = REPO / f"outputs/{mtag}"
    encs = [enc_name("frozen")] + [enc_name(e) for e in ARM2ENC.values()]
    rows = []
    for enc in encs:
        act = _jload(base / artifact("probe_action_dir") / enc / artifact("test_metrics"))
        tax = _jload(base / artifact("probe_taxonomy_dir") / enc / artifact("test_metrics"))
        mc = _jload(base / artifact("probe_motion_cos_dir") / enc / "intra_inter_ratio.json")
        fm = _jload(base / artifact("probe_future_mse_dir") / enc / "aggregate_mse.json")
        pt = {fam: _jload(base / artifact("predictor_temporal_dir") / enc / f"aggregate_{fam}.json")
              for fam in _PT_FAMILIES}
        tax_macro = None
        if tax and isinstance(tax.get("dims"), dict) and tax["dims"]:
            vals = [v["test_mean"] for v in tax["dims"].values() if "test_mean" in v]
            tax_macro = sum(vals) / len(vals) if vals else None
        n_test = next((src["n_test"] for src in (act, fm, mc) if src and "n_test" in src), None)

        def _half(d, ci_key):
            return d.get(ci_key, {}).get("ci_half") if (d and isinstance(d.get(ci_key), dict)) else None

        rows.append({
            # iter18 2026-06-08: every metric cell is a (mean, '±ci') PAIR — the table renders the
            # mean on one line and the 95% CI range on the line below it.
            "enc": enc.replace(enc_prefix() + "_", ""),
            "n_te": ("—" if n_test is None else str(n_test), ""),
            "act": _ci2(act, "top1_acc", "top1_ci"),
            "tax": (_f(tax_macro, 3), ""),
            "mcos": _ci2(mc, "score_mean", "score_ci"),
            "fut": _ci2(fm, "mse_mean", "mse_ci"),
            # 95% BCa CI (bootstrap_ci on the per-clip array) ships in aggregate_<fam>.json's
            # "ci" sub-dict — mean on top, ±ci_half on the line below (like the other metrics).
            **{fam: _ci2(pt[fam], "mean", "ci", 4) for fam in _PT_FAMILIES},
            # iter18 graphs (2026-06-07): raw numerics alongside the display strings —
            # the table loop only reads the display keys, so its output is byte-identical.
            "_enc_full": enc,
            "_raw": {
                "act":  (act["top1_acc"] if act and "top1_acc" in act else None, _half(act, "top1_ci")),
                "tax":  (tax_macro, None),
                "mcos": (mc["score_mean"] if mc and "score_mean" in mc else None, _half(mc, "score_ci")),
                "fut":  (fm["mse_mean"] if fm and "mse_mean" in fm else None, _half(fm, "mse_ci")),
                **{fam: ((pt[fam]["mean"], _half(pt[fam], "ci")) if pt[fam] and "mean" in pt[fam] else (None, None))
                   for fam in _PT_FAMILIES},
            },
        })
    return rows


# ── matplotlib graphs (iter18, 2026-06-07) — the two tables, DRAWN ─────────
# Reuses src/m13_eval_plot.py primitives (the §3 finale's plot module): _bar_with_ci
# (CI bars + N/A hatching + direction badges) and _sort_by_metric for the EVAL scorecard;
# utils.plots init_style/save_fig for publication style + png&pdf. Three figures into
# outputs/<mode>/probe_plot/metrics_watch/ on every invocation (cheap, CPU-only, ~3 s):
#   train_trajectories — every probe checkpoint of every arm, per metric (★ = KEPT ckpt)
#   kept_scorecard     — each arm's KEPT-checkpoint value per metric, sorted, with an
#                        OURS-vs-BEST-OTHER verdict strip (the paper question, per metric)
#   eval_scorecard     — per-encoder eval artifacts with 95% CIs (graceful before evals)
_FAM_OURS = {"surgery_3stage_DI_encoder", "surgery_noDI_encoder",
             "surgery_3stage_DI_head", "surgery_noDI_head"}
_TRAIN_STYLE = {  # arm → (short label, color, linestyle, linewidth); OURS = greens
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
_TRAIN_METRICS = [  # (probe-row key, panel title, direction) — mirrors the TRAIN table columns
    ("probe_top1",    "action top-1",  "higher"),
    ("motion_cos",    "motion-cos",    "higher"),
    ("future_l1",     "future L1",     "lower"),
    ("causal_l1",     "causal L1",     "lower"),
    ("maskratio",     "mask-ratio",    "lower"),
    ("val_jepa_loss", "val JEPA loss", "lower"),
]
_EVAL_METRICS = [  # (_raw key, panel title, direction) — mirrors the EVAL table columns
    ("act", "action top-1", "higher"), ("tax", "taxonomy F1", "higher"),
    ("mcos", "motion-cos", "higher"), ("fut", "future MSE", "lower"),
    ("rollout", "rollout drift", "lower"), ("causal", "causal L1", "lower"),
    ("tdist", "t-dist", "lower"), ("maskratio", "mask-ratio slope", "lower"),
    ("teacher_free", "teacher-free", "lower"),
]


def render_metric_graphs(blocks, ev, out_dir):
    """All three figures. Lazy imports so --no-plots costs nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from m13_eval_plot import _bar_with_ci, _sort_by_metric
    from utils.plots import init_style, save_fig
    init_style()

    def _kept_row(b):
        """The row eval inherits: enc arms → KEPT ckpt; head arms → last probe (diagnostic)."""
        if b["kept_i"] is not None:
            return b["rows"][b["kept_i"]][0]
        return b["rows"][-1][0] if b["rows"] else None

    # ── F1: train trajectories (every probe checkpoint, per metric) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    arms_drawn = []
    for ax, (key, title, direction) in zip(axes.flat, _TRAIN_METRICS):
        for b in blocks:
            sty = _TRAIN_STYLE[b["arm"]]
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
    handles = [plt.Line2D([], [], color=_TRAIN_STYLE[a][1], ls=_TRAIN_STYLE[a][2],
                          lw=_TRAIN_STYLE[a][3], marker="o", ms=4, label=_TRAIN_STYLE[a][0])
               for a in _TRAIN_STYLE if a in arms_drawn]
    if handles:   # at run START no arm has a probe checkpoint yet → handles=[] → fig.legend(ncol=0) CRASHES
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                   frameon=True, fontsize=11)
    fig.suptitle("iter18 TRAIN probe checkpoints — every arm, every probe · star marker = KEPT ckpt"
                 " · OURS = greens (solid enc / dotted head)", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.13, hspace=0.35, wspace=0.25)
    save_fig(fig, str(out_dir / "train_trajectories"))
    plt.close(fig)

    # ── F2: KEPT-checkpoint scorecard (95% CI whiskers) + verdict strip ──
    # CI sources, honest only (iter18 2026-06-07): rows written by the CI-enabled
    # producer carry per-clip BCa half-widths (probe_trio); LEGACY rows (pre-CI)
    # have none — for those, top-1 gets the analytic binomial half-width from
    # (p, n_probe_clips); metrics without per-clip data get NO whisker.
    _CI_KEY = {"probe_top1": "top1_ci_half", "motion_cos": "motion_cos_ci_half",
               "future_l1": "future_l1_ci_half"}
    _Z95 = 1.96

    def _row_ci(row, key):
        ck = _CI_KEY.get(key)
        if ck and row.get(ck) is not None:
            return float(row[ck])                      # stored per-clip BCa CI
        if key == "probe_top1" and row.get("n_probe_clips"):
            p, n_ = float(row["probe_top1"]), int(row["n_probe_clips"])
            return _Z95 * (p * (1 - p) / n_) ** 0.5    # legacy rows: binomial approx
        return None

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.38], hspace=0.5, wspace=0.45)
    axes6 = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    verdicts = []
    for ax, (key, title, direction) in zip(axes6, _TRAIN_METRICS):
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
        labels = [_TRAIN_STYLE[a][0] + (" (hd)" if a.endswith("_head") else "")
                  for a, _, _ in entries]
        colors = [_TRAIN_STYLE[a][1] for a, _, _ in entries]
        vals = [v for _, v, _ in entries]
        errs = [(e or 0.0) for _, _, e in entries]
        ys = np.arange(len(entries))[::-1]
        bars = ax.barh(ys, vals, color=colors, alpha=0.9, height=0.65,
                       xerr=errs, capsize=3, error_kw={"lw": 1.1, "ecolor": "#222"})
        for y, (arm, v, e), b_ in zip(ys, entries, bars):
            if arm in _FAM_OURS:
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
        ours = [(v, e) for a, v, e in entries if a in _FAM_OURS]
        other = [(v, e) for a, v, e in entries if a not in _FAM_OURS]
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
            # CI-aware verdict: when both CIs are known and the intervals overlap,
            # the lead is statistically a TIE at POC N — say so (no fake winners).
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

    # ── F3: EVAL scorecard (per-encoder artifacts, 95% CI via m13's bar) ──
    have_any = any(r["_raw"][k][0] is not None for r in ev for k, _, _ in _EVAL_METRICS)
    if not have_any:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "no EVAL artifacts yet —\nper-encoder evals launch as each arm finishes training",
                ha="center", va="center", fontsize=16, fontweight="bold", color="#666")
        save_fig(fig, str(out_dir / "eval_scorecard"))
        plt.close(fig)
        return
    fig, axes9 = plt.subplots(3, 3, figsize=(26, 17))
    for ax, (k, title, direction) in zip(axes9.flat, _EVAL_METRICS):
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


# Data behind the two tables/graphs → json + csv (iter18 2026-06-08, user order). Lands in the SAME
# backbone-keyed metrics_watch/<backbone>/ folder so the plots and the numbers that produced them
# travel together (self-contained per backbone).
_EVAL_RAW_KEYS = ["act", "tax", "mcos", "fut", "rollout", "causal", "tdist", "maskratio", "order", "teacher_free"]
_TRAIN_PROBE_KEYS = ["step", "probe_top1", "motion_cos", "future_l1", "causal_l1", "maskratio", "val_jepa_loss"]


def _dump_metric_data(blocks, ev, out_dir):
    """Write train_metrics.{json,csv} (every probe checkpoint per arm) + eval_metrics.{json,csv}
    (per-encoder mean ± 95%CI per metric) into out_dir — the data SOURCE for the 3 graphs."""
    import csv
    # ── TRAIN: every probe checkpoint per arm ──
    train = [{"arm": b["arm"], "status": b["st"], "n_train": b["n_tr"], "n_val": b["n_va"],
              "head_best": b["head_best"], "kept_idx": b["kept_i"],
              "probes": [{**{k: r.get(k) for k in _TRAIN_PROBE_KEYS},
                          "verdict": v, "kept": (i == b["kept_i"])}
                         for i, (r, v) in enumerate(b["rows"])]}
             for b in blocks]
    (out_dir / "train_metrics.json").write_text(json.dumps(train, indent=1))
    with open(out_dir / "train_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ckpt", *_TRAIN_PROBE_KEYS, "verdict", "kept"])
        for t in train:
            for i, p in enumerate(t["probes"]):
                w.writerow([t["arm"], i + 1, *[p.get(k) for k in _TRAIN_PROBE_KEYS], p["verdict"], p["kept"]])
    # ── EVAL: per-encoder (mean, ci_half) per metric, from the row's _raw payload ──
    ev_json = [{"encoder": r["_enc_full"], "n_test": r["n_te"][0],
                **{k: {"mean": r["_raw"][k][0], "ci_half": r["_raw"][k][1]}
                   for k in _EVAL_RAW_KEYS if k in r["_raw"]}}
               for r in ev]
    (out_dir / "eval_metrics.json").write_text(json.dumps(ev_json, indent=1))
    with open(out_dir / "eval_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "n_test", *[c for k in _EVAL_RAW_KEYS for c in (k, k + "_ci")]])
        for r in ev:
            raw = r["_raw"]
            cells = [c for k in _EVAL_RAW_KEYS for c in raw.get(k, (None, None))]
            w.writerow([r["_enc_full"], r["n_te"][0], *cells])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the matplotlib graphs (terminal tables only)")
    args = ap.parse_args()
    mtag = args.mode.lower()

    # Arms to HIDE from the tables + graphs (data files below keep ALL of them). enc-name form too.
    skip = skip_arms(mtag)
    skip_encs = ({enc_name(ARM2ENC[a]) for a in skip if a in ARM2ENC}
                 | ({enc_name("frozen")} if "frozen" in skip else set()))

    bar = "─"
    # ── TRAIN table: every probe checkpoint per arm ──
    AW, C, SV = 29, 9, 7
    hdr = ["st", "ckpt", "step", "top1↑", "m_cos↑", "fut_l1↓", "caus↓", "maskr", "vjepa↓"]
    print(f"═══ iter18 {args.mode} TRAIN metrics · EVERY probe checkpoint (probe_history.jsonl) ═══")
    print(f"  splits (counted live from {get_local_data_dir().name}): {split_sizes()}")
    print("┌" + bar * AW + "┬" + "┬".join(bar * C for _ in hdr) + "┬" + bar * SV + "┐")
    print("│" + " arm · n_train/n_val".ljust(AW) + "│" + "│".join(h.center(C) for h in hdr)
          + "│" + "sel".center(SV) + "│")
    print("├" + bar * AW + "┼" + "┼".join(bar * C for _ in hdr) + "┼" + bar * SV + "┤")
    blocks = train_blocks(mtag)
    for b in blocks:
        if b["arm"] in skip:        # hidden from the TABLE (still in train_metrics.{json,csv})
            continue
        nt = "—" if b["n_tr"] is None else f"{b['n_tr']:,}"
        nv = "—" if b["n_va"] is None else f"{b['n_va']:,}"
        label = f"{b['arm']}"
        sub = f"  └ {nt}/{nv}" + (f" · {b['head_best']}" if b["head_best"] else "")
        if not b["rows"]:
            print("│ " + label.ljust(AW - 1) + "│" + "│".join(("—" if h != "st" else b["st"]).center(C)
                  for h in hdr) + "│" + "—".center(SV) + "│")
            print("│ " + sub.ljust(AW - 1) + "│" + "│".join(" " * C for _ in hdr) + "│" + " " * SV + "│")
            continue
        for i, (r, verdict) in enumerate(b["rows"]):
            sel = verdict + ("←KEPT" if i == b["kept_i"] else "")
            cells = [(b["st"] if i == 0 else "").center(C - 1),
                     f"{i + 1}/{len(b['rows'])}".center(C),
                     str(r.get("step", "—")).center(C),
                     _f(r.get("probe_top1")).center(C), _f(r.get("motion_cos")).center(C),
                     _f(r.get("future_l1")).center(C), _f(r.get("causal_l1")).center(C),
                     _f(r.get("maskratio")).center(C), _f(r.get("val_jepa_loss")).center(C)]
            print("│ " + (label if i == 0 else (sub if i == 1 else "")).ljust(AW - 1)
                  + "│" + "│".join(cells) + "│" + sel.center(SV - (1 if "🎯" in sel or "✋" in sel else 0)) + "│")
        if len(b["rows"]) == 1:
            print("│ " + sub.ljust(AW - 1) + "│" + "│".join(" " * C for _ in hdr) + "│" + " " * SV + "│")
    print("└" + bar * AW + "┴" + "┴".join(bar * C for _ in hdr) + "┴" + bar * SV + "┘")

    # ── EVAL table: headline number per metric family per encoder ──
    ev = eval_rows(mtag)
    EW = 28
    # columns are narrow again — each row prints the mean on line 1 and the ±95%CI on line 2.
    cols = [("n_te", "n_te", 6), ("act_top1↑", "act", 11), ("tax_F1↑", "tax", 8),
            ("m_cos↑", "mcos", 10), ("fut_mse↓", "fut", 10), ("rollout↓", "rollout", 10),
            ("causal↓", "causal", 10), ("tdist↓", "tdist", 10), ("maskr↓", "maskratio", 10),
            ("order", "order", 10), ("t_free↓", "teacher_free", 10)]
    print(f"\n═══ iter18 {args.mode} EVAL metrics · per-encoder artifacts (mean / ±95%CI · — = not computed) ═══")
    print("┌" + bar * EW + "┬" + "┬".join(bar * w for _, _, w in cols) + "┐")
    print("│" + " encoder".ljust(EW) + "│" + "│".join(h.center(w) for h, _, w in cols) + "│")
    print("├" + bar * EW + "┼" + "┼".join(bar * w for _, _, w in cols) + "┤")
    for r in ev:
        if r["_enc_full"] in skip_encs:   # hidden from the TABLE (still in eval_metrics.{json,csv})
            continue
        # line 1 = means; line 2 = ±95%CI (printed only when at least one CI exists for this row)
        print("│ " + r["enc"].ljust(EW - 1) + "│"
              + "│".join(str(r[k][0]).center(w) for _, k, w in cols) + "│")
        bots = [str(r[k][1]).center(w) for _, k, w in cols]
        if any(b.strip() for b in bots):
            print("│ " + " " * (EW - 1) + "│" + "│".join(bots) + "│")
    print("└" + bar * EW + "┴" + "┴".join(bar * w for _, _, w in cols) + "┘")
    print("\n  legend: ↑ higher better · ↓ lower better · ✅ trained · 🔄 training · ⬚ not started"
          "\n  sel: 🎯 promoted · ✋ held (worse than best) · ←KEPT = the exported ckpt · "
          "head arms select on head-val_loss (· rows = diagnostics)")

    # ── graphs + data dump (iter18 2026-06-08, user order): BACKBONE-keyed subfolder so the 2B and
    #    1B runs write disjoint dirs (no overwrite), AND the json+csv data SOURCE behind both tables
    #    lands beside the graphs → each metrics_watch/<backbone>/ folder is self-contained. ──
    rel = f"outputs/{mtag}/probe_plot/metrics_watch/{BACKBONE}"
    out_dir = REPO / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_metric_data(blocks, ev, out_dir)   # FULL — data files keep EVERY arm (incl. skipped)
    if skip:
        print(f"  ⏷ {sorted(skip)} hidden from tables+graphs · still in {{train,eval}}_metrics.{{json,csv}}")
    if not args.no_plots:
        vis_blocks = [b for b in blocks if b["arm"] not in skip]            # graphs honor --skip-arms
        vis_ev = [r for r in ev if r["_enc_full"] not in skip_encs]
        render_metric_graphs(vis_blocks, vis_ev, out_dir)
        print(f"  🖼  graphs + data → {rel}/ · "
              f"{{train_trajectories,kept_scorecard,eval_scorecard}}.{{png,pdf}} + {{train,eval}}_metrics.{{json,csv}}")
    else:
        print(f"  📄 data → {rel}/{{train,eval}}_metrics.{{json,csv}}  (--no-plots: graphs skipped)")


if __name__ == "__main__":
    main()
