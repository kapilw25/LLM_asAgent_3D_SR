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
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from iter18_poc_ngpu import ARM2DIR, ARM2ENC, BACKBONE  # noqa: E402  (canonical DAG names)
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
    logs = sorted((REPO / "logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log"),
                  key=lambda p: p.stat().st_mtime)
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
        logs = sorted((REPO / "logs").glob(f"iter18_ngpu_{mtag}_train_{arm}_*.log"),
                      key=lambda q: q.stat().st_mtime)
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


def eval_rows(mtag):
    base = REPO / f"outputs/{mtag}"
    encs = ["vjepa_2_1_frozen"] + [f"vjepa_2_1_{e}" for e in ARM2ENC.values()]
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
        rows.append({
            "enc": enc.replace("vjepa_2_1_", ""),
            "n_te": "—" if n_test is None else str(n_test),
            "act": _ci(act, "top1_acc", "top1_ci"),
            "tax": _f(tax_macro, 3),
            "mcos": _ci(mc, "score_mean", "score_ci"),
            "fut": _ci(fm, "mse_mean", "mse_ci"),
            **{fam: _f(pt[fam]["mean"], 4) if pt[fam] else "—" for fam in _PT_FAMILIES},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    args = ap.parse_args()
    mtag = args.mode.lower()

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
    for b in train_blocks(mtag):
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
    cols = [("n_te", "n_te", 6), ("act_top1↑", "act", 13), ("tax_F1↑", "tax", 8),
            ("m_cos↑", "mcos", 14), ("fut_mse↓", "fut", 14), ("rollout↓", "rollout", 9),
            ("causal↓", "causal", 9), ("tdist↓", "tdist", 9), ("maskr↓", "maskratio", 9),
            ("order", "order", 9), ("t_free↓", "teacher_free", 9)]
    print(f"\n═══ iter18 {args.mode} EVAL metrics · per-encoder artifacts (— = not computed yet) ═══")
    print("┌" + bar * EW + "┬" + "┬".join(bar * w for _, _, w in cols) + "┐")
    print("│" + " encoder".ljust(EW) + "│" + "│".join(h.center(w) for h, _, w in cols) + "│")
    print("├" + bar * EW + "┼" + "┼".join(bar * w for _, _, w in cols) + "┤")
    for r in ev:
        print("│ " + r["enc"].ljust(EW - 1) + "│"
              + "│".join(str(r[k]).center(w) for _, k, w in cols) + "│")
    print("└" + bar * EW + "┴" + "┴".join(bar * w for _, _, w in cols) + "┘")
    print("\n  legend: ↑ higher better · ↓ lower better · ✅ trained · 🔄 training · ⬚ not started"
          "\n  sel: 🎯 promoted · ✋ held (worse than best) · ←KEPT = the exported ckpt · "
          "head arms select on head-val_loss (· rows = diagnostics)")


if __name__ == "__main__":
    main()
