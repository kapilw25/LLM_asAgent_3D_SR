#!/usr/bin/env python3
"""iter18 POC/SANITY live status — emoji TABLE + live-calibrated ETA per job and for the whole run.

Adapted from scripts/legacy/iter17_poc_status.py for the iter18 DAG (ONE backbone, 13 train arms
+ 14 eval jobs; pretrain_encoder is the root for all 12 other arms). Imports build_jobs() from
iter18_poc_ngpu.py so the job list is the SINGLE SOURCE OF TRUTH (it can never drift from the
scheduler). Classifies every job ✅/🔄/⬚/❌ from the main log's GPU ◀/✓/✗ markers.

ETA: a forward DAG simulation over the GPU pool. Per-arm durations are OBSERVED, not hardcoded —
  · a completed job's measured wall sets that arm-class's estimate (mean of completed peers),
  · a RUNNING train's total = static step ledger ÷ its own live rate (see REAL-ETA below),
  · a RUNNING/PENDING eval's total = the eval STAGE ledger: current-stage remainder
    (live clip bar) + Σ measured walls of the queued stages (stamp-banner timestamps),
  · only an arm-class with NO completion yet AND no parseable progress falls back to a prior (below),
    seeded from the 06-05/06-06 measured runs and replaced the moment real data arrives.

AUTO-BACKUP (POC only): while you watch, this backs up outputs/poc to HF every UPLOAD_EVERY_MIN
  minutes (`upload`, reuse mode — iter18 2026-06-07: now mirrors EVERY file incl. the resume
  checkpoints *ckpt_latest/stage*.pt, so a box migration needs no extra upload; HF dedups
  unchanged files. The full-fidelity `_full-*.tar` shards + `_full-manifest.json` on the remote
  are PROTECTED from its mirror-cleanup) and once more when the run finishes, so the paid node
  can be killed right after completion. A missing HF_TOKEN now FAILS the backup loudly (rc=1 →
  ❌ line) — it skipped silently as "✅" for 20 h on 06-06/07.

USAGE:
  python -u scripts/iter18_poc_status.py                 # latest POC main log
  python -u scripts/iter18_poc_status.py --mode SANITY
  python -u scripts/iter18_poc_status.py --log logs/iter18_ngpu_poc_20260606_101530.log
  watch -n60 'python -u scripts/iter18_poc_status.py'    # live, refresh every 60s
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import json  # noqa: E402
from iter18_poc_ngpu import (  # noqa: E402  (canonical DAG — single source for naming + jobs)
    ARM2DIR, ARM2ENC, BACKBONE, PT_METRICS, S3_SKIP_PERENC, build_jobs, enc_name, enc_prefix)
from utils.config import get_pipeline_config, load_merged_config  # noqa: E402  (trainers' own loader)

EMOJI = {"done": "✅", "running": "🔄", "pending": "⬚", "failed": "❌"}
_MIN_EVAL_POINTS = 5     # eval rate needs a few clips before extrapolating
_SIM_GUARD = 10000       # DAG-sim infinite-loop backstop
# Runbook §2 order: novelty ×4 + control + baselines + pretrain root first.
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
# ── REAL-ETA WORKLOAD LEDGER (iter18 2026-06-07) ──────────────────────────
# The total work of every arm is DETERMINED by yaml + split artifacts — guessing it
# from log archaeology (priors, plan banners, hand-kept stage maps) is what caused the
# all-night ETA drift (hidden noDI stage2, hidden lpft FT stage, hopeful 5.2h priors).
# This ledger computes each arm's EXACT optimizer-step plan from the SAME single
# sources the trainers read, mirroring their formulas line-by-line:
#   · merged cfg via utils.config.load_merged_config (the trainers' own loader)
#   · m09a1:514-515   spe = max(1, n_pool // batch);            total = spe × max_epochs
#   · m09c2:525       spe = max(1, ceil(n_pool / batch))  (head arms)
#   · m09c1:811-812   spe = n_factor // batch  (n_factor = pool ∩ factor_manifest)
#   · m09c1:1270      stage_steps = max(int(total × max_epochs_pct), 1) per surgery stage
#   · m09c1:700-710   LP-FT stage0 PREPENDED at lp_ft_stage0.max_epochs_pct of total
#   · probe schedule  probe_every = spe // saves_per_epoch; per stage
#                     n = max(1, round(stage_steps / probe_every))  (the exact schedule)
# ETA then = (ledger_total − progress) ÷ measured rate + calibrated overheads — the
# field-standard "remaining work / measured throughput", never extrapolated totals.
_LEDGER_CACHE = {}


def _build_ledger(mtag):
    """{arm: {stages:[int], total:int, n_probes:int}} — static plan per arm."""
    if mtag in _LEDGER_CACHE:
        return _LEDGER_CACHE[mtag]
    pcfg = get_pipeline_config()
    model_cfg = pcfg["backbone_model_configs"][BACKBONE]
    local = Path(pcfg["data"]["local_data_dir"])
    pool = set(json.loads((local / "train_pool.json").read_text())["clip_keys"])
    fm_raw = json.loads((local / pcfg["data"]["factor_subdir"] / "factor_manifest.json").read_text())
    fm = set(fm_raw.keys()) if isinstance(fm_raw, dict) else {
        x["clip_key"] if isinstance(x, dict) else x for x in fm_raw}
    n_factor = len(pool & fm)
    led = {}
    for arm in TRAIN_ORDER:
        cfg = load_merged_config(model_cfg, pcfg["arm_train_configs"][arm])
        opt = cfg["optimization"]
        batch = opt["batch_size"][mtag] if isinstance(opt["batch_size"], dict) else opt["batch_size"]
        me = opt["max_epochs"]
        max_epochs = me[mtag] if isinstance(me, dict) else me
        sp = cfg["checkpoint"]["saves_per_epoch"]
        saves = sp[mtag] if isinstance(sp, dict) else sp
        if arm == "pretrain_encoder":
            spe = max(1, len(pool) // batch)                       # m09a1:514
            stages = [spe * max_epochs]
        elif arm in _HEAD_ARMS:
            spe = max(1, (len(pool) + batch - 1) // batch)         # m09c2:525 (ceil)
            stages = [spe * max_epochs]
        else:
            spe = n_factor // batch                                # m09c1:811
            total = spe * max_epochs
            lp = cfg["surgery"]["lp_ft_stage0"]
            stages = ([max(int(total * lp["max_epochs_pct"]), 1)] if lp["enabled"] else [])
            stages += [max(int(total * s["max_epochs_pct"]), 1)    # m09c1:1270
                       for s in cfg["surgery"]["stages"]]
        probe_every = max(1, spe // saves)
        n_probes = sum(max(1, round(st / probe_every)) for st in stages)
        led[arm] = {"stages": stages, "total": sum(stages), "n_probes": n_probes}
    _LEDGER_CACHE[mtag] = led
    return led


# Per-stage FINAL tqdm bars ("surgery:<name>: 100%|…| S/S [H:MM:SS<00:00" — tqdm omits
# the hours field under 1h, so HH is optional) — their elapsed sums to the arm's pure
# in-loop wall; (job wall − Σ) = startup + stage-end probes + finalize = the overhead.
_RE_FINAL_BAR = re.compile(r"surgery:\S+\s+100%\|[^|]*\|\s*(\d+)/\1\s*\[(?:(\d+):)?(\d+):(\d+)<")
_RE_LIVE_BAR = re.compile(r"surgery:\S+\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:(\d+):)?(\d+):(\d+)<")


def _bar_secs(h, m, s):
    return (int(h) if h else 0) * 3600 + int(m) * 60 + int(s)


def _calibrate(jobs, done, launched, mtag, ledger):
    """Measured (pure step rate, per-pad overhead) from THIS run's completed train arms.
    pure_rate = Σ(final-bar elapsed)/executed_steps; overhead = wall − Σ(bar elapsed),
    split over (n executed stages + 2) pads (each stage-end probe + startup + finalize)."""
    rates, pads, fins = [], [], []
    if not ledger:
        return {"ledger": ledger, "rate": None, "pad": _FINALIZE_PAD_S,
                "finalize": None, "mtag": mtag}
    for jid, t in done.items():
        if t == "resume" or not jid.startswith("T:") or jid not in launched:
            continue
        arm = _arm_of(jid)
        cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
        if not cands:
            continue
        try:
            full = cands[-1].read_text(errors="replace")
        except OSError:
            continue
        wall = (_sod(t) - _sod(launched[jid])) % 86400
        # set(): tqdm prints the 100% bar line TWICE (last update + close) — without
        # dedupe every stage double-counts (06-07: rate read 128 s/step = 2× true, pad 0).
        bars = [_bar_secs(h, m, s) for _, h, m, s in set(_RE_FINAL_BAR.findall(full))]
        rm = re.findall(r"Resumed from step (\d+)", full)
        executed = ledger[arm]["total"] - (int(rm[-1]) if rm else 0)
        if not bars or executed <= 0:
            continue
        rates.append((sum(bars), executed))
        pads.append(max(wall - sum(bars), 0) / (len(bars) + 2))
        # MEASURED finalize: ✓stamp − newest stage-ckpt mtime (saved right before
        # finalize). 06-07 measured 23m/32m/37m/67m — replaces the fake 15m constant.
        mt = _newest_stage_ckpt_mtime(arm, mtag)
        if mt is not None:
            fins.append((_sod(t) - _epoch_sod(mt)) % 86400)
    pads.sort()
    fins.sort()
    pooled = (sum(b for b, _ in rates) / sum(e for _, e in rates)) if rates else None
    return {"ledger": ledger, "mtag": mtag,
            "rate": pooled,    # pooled Σbar-seconds / Σexecuted-steps across completions
            "pad": pads[len(pads) // 2] if pads else _FINALIZE_PAD_S,
            "finalize": fins[len(fins) // 2] if fins else None}


_STAGE_NOTE = {}   # jid → "sN/M" live stage marker, filled by _running_total, shown in the cell
# COLD-START fallback for the end-of-training finalize (stage-end probe-trio on 451 clips
# + best-ckpt reload + export). 06-07 MEASURED on this run's completed arms (✓stamp −
# last-stage-ckpt mtime): 23m / 32m / 37m / 67m — the old 15m constant was 2-4× fake.
# Used ONLY before any in-log completion; afterwards calib["finalize"] (live median) wins,
# and the remaining-finalize display counts DOWN from it anchored on the ckpt mtime.
_FINALIZE_PAD_S = 35 * 60


def _newest_stage_ckpt_mtime(arm, mtag):
    """mtime (epoch) of the arm's newest *ckpt_stage*.pt — written immediately before
    finalize starts, so (✓stamp − it) = measured finalize, (now − it) = time IN finalize."""
    d = REPO / f"outputs/{mtag}/{BACKBONE}/{ARM2DIR[arm]}"
    cks = sorted((p for p in d.glob("*ckpt_stage*.pt") if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling/cleared ckpt symlinks
    return cks[-1].stat().st_mtime if cks else None


def _epoch_sod(epoch):
    import time as _t
    g = _t.gmtime(epoch)
    return g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec


# ── EVAL REAL-ETA STAGE LEDGER (iter18 2026-06-07) ────────────────────────
# A per-encoder eval is a FIXED stage chain (run_eval.sh per-encoder loop):
#   2 features → 3 probe-train → 3.5 taxonomy (= stage id 11) → 5 motion-feat
#   → 6 cosine → 8 future_mse → 8b predictor_temporal
# The old estimator extrapolated ONLY the current stage's clip bar and capped the
# job at a hardcoded 3 h — once elapsed passed 3 h the cell froze at the 60s floor
# ("🔄 3h05m·~1m00s" while s7/7 had ~8m left AND nothing counted queued stages).
# run_eval.sh stamps "═══ HH:MM:SS · STAGE <id>" before every stage (stamp(),
# run_eval.sh:311), so completed stage walls are MEASURED from consecutive banner
# timestamps (the "DONE · total wall" banner closes the last). A stage NO eval has
# completed yet borrows the live full-stage projection (clip-bar total × recent
# rate) from any sibling currently inside it; a stage no eval has even reached
# falls back to the whole-job prior's per-stage share — replaced the moment any
# sibling reaches it (the same self-correcting contract as the train priors).
EVAL_PLAN = [("3.5" if s == "11" else s) for s in S3_SKIP_PERENC.split(",")]
_RE_STAMP = re.compile(r"═══ (\d\d:\d\d:\d\d) ·\s*(?:STAGE ([\w.]+)|(DONE))")
_RE_CLIP_BAR = re.compile(r"(\d+)/(\d+) \[[^\]]*recent=([\d.]+)s/clip")


def _runtime_extra_skip():
    """EVAL_PLAN display-ids dropped at runtime by run_eval.sh's EXTRA_SKIP_STAGES env /
    logs/.eval_extra_skip sentinel (iter18 2026-06-08) — so a mid-run `echo '3,11' >
    logs/.eval_extra_skip` (skip STAGE 3 action-probe-train + STAGE 3.5 taxonomy) makes the ETA
    DROP by those stages without a scheduler restart. Skip-id 11 maps to plan id '3.5'."""
    raw = os.environ.get("EXTRA_SKIP_STAGES", "")
    f = REPO / "logs" / ".eval_extra_skip"
    if not raw and f.exists():
        try:
            raw = f.read_text().strip()
        except OSError:
            raw = ""
    return {("3.5" if s.strip() == "11" else s.strip()) for s in raw.split(",") if s.strip()}


def _eval_plan_for(jid, mtag):
    """Stage plan for one per-encoder E: eval — stages 2-8 ONLY (iter18 2026-06-07: Stage 8b
    is now 6 separate P: metric jobs, estimated via _pt_total — so 8b is dropped here to avoid
    double-counting). Stage 8 is ALSO dropped when the encoder's predictor-bearing best ckpt is
    PROVABLY absent (student_encoder.pt exists, m09{a,c}_ckpt_best.pt doesn't) — mirrors
    run_eval.sh's Stage-8 preflight. A still-training arm keeps the full 2-8 plan."""
    plan = [s for s in EVAL_PLAN if s != "8b"]    # E: runs 2,3,3.5,5,6,8 — never 8b
    plan = [s for s in plan if s not in _runtime_extra_skip()]   # runtime taxonomy/probe-skip → ETA drops
    enc = jid.split(f"E:{enc_prefix()}_", 1)[-1]
    if enc == "frozen":
        return plan               # Meta ckpt always carries the predictor
    arm = next(a for a, e in ARM2ENC.items() if e == enc)
    d = REPO / f"outputs/{mtag}/{BACKBONE}/{ARM2DIR[arm]}"
    best = "m09a_ckpt_best.pt" if ARM2DIR[arm].startswith("m09a_") else "m09c_ckpt_best.pt"
    if (d / "student_encoder.pt").exists() and not (d / best).exists():
        return [s for s in plan if s != "8"]
    return plan


def _eval_calibrate(jobs, mtag, now_s):
    """(medians, state, plans) for the eval stage ledger.
    medians: stage → median measured wall, pooled over EVERY eval log segment of this
      run (banner-timestamp diffs; interrupted segments contribute their closed stages).
      Stages with no closed wall get the median live projection (bar total × recent).
    state[jid] (running evals only): {cur, in, rem_cur} — current stage, seconds inside
      it, and its live-bar remainder (None when the stage has no recent= clip bar; the
      bar is read ONLY from bytes after the last banner, so a finished stage's stale
      bar can never masquerade as the current one's)."""
    walls, projs, state, plans = {}, {}, {}, {}
    for jid, j in jobs.items():
        if _arm_of(jid) != "eval" or jid.startswith("P:"):
            continue                  # P: (Stage-8b metric) jobs use _pt_total, not the stage ledger
        plans[jid] = _eval_plan_for(jid, mtag)
        cands = sorted((p for p in REPO.glob(j["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
        for p in cands:
            try:
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            seq = [( _sod(m.group(1)), m.group(2) or "DONE", m.end())
                   for m in _RE_STAMP.finditer(txt)
                   if (m.group(2) in plans[jid]) or m.group(3)]
            for (t1, s1, _), (t2, _s2, __) in zip(seq, seq[1:]):
                walls.setdefault(s1, []).append((t2 - t1) % 86400)
            if p is not cands[-1] or not seq or seq[-1][1] == "DONE":
                continue
            cur = seq[-1][1]
            es = {"cur": cur, "in": (now_s - seq[-1][0]) % 86400, "rem_cur": None}
            pb = _RE_CLIP_BAR.findall(txt[seq[-1][2]:])
            if pb:
                c, tot, r = int(pb[-1][0]), int(pb[-1][1]), float(pb[-1][2])
                if tot and c >= _MIN_EVAL_POINTS:
                    es["rem_cur"] = (tot - c) * r
                    projs.setdefault(cur, []).append(tot * r)
            state[jid] = es
    med = {s: sorted(v)[len(v) // 2] for s, v in walls.items()}
    for s, v in projs.items():
        med.setdefault(s, sorted(v)[len(v) // 2])
    return med, state, plans
# COLD-START priors (seconds), used ONLY until a live measurement (a completion, or a running job's
# step progress) replaces them. Empirical: 06-06 1× pretrain 4h32m; 06-05 enc arms ~5h10m;
# head arms ~58m; eval = per-encoder stages only (2,3,11,5,6,8,8b) — self-corrects on 1st completion.
PRIOR = {arm: (int(4.6 * 3600) if arm == "pretrain_encoder"
               else int(1.0 * 3600) if arm in _HEAD_ARMS
               else int(5.2 * 3600))
         for arm in TRAIN_ORDER}
PRIOR["eval"] = int(0.75 * 3600)
# SANITY arms are ~5-8 min each (06-06 measured: pretrain 6m20s, frozen eval 5m) — the POC
# priors above are ~50× too big there and made the first SANITY ETA read 10h46m. Mode-scaled
# priors fix the cold start; live measurements still override both the moment they exist.
PRIOR_SANITY = {k: 7 * 60 for k in PRIOR}
# COLD prior for ONE Stage-8b metric job (P:) — used ONLY until that metric's first completion
# populates pt_med (per-metric). The whole-eval prior (est["eval"], ~3.4h incl. 8b) is a terrible
# per-metric prior, so use a dedicated one: at bs=16 the 6 metrics run ~10-25 min each (teacher_free
# longest). 20 min is conservative (safe-high) and self-corrects within ~1 metric completion.
_PT_COLD_PRIOR = {"poc": 20 * 60, "sanity": 60}

# Back up outputs/poc to HF this often (minutes) WHILE the run goes, so the final backup at the end
# is tiny and the paid node can be killed right away. Driven by the 60s `watch` refresh + a stamp file.
UPLOAD_EVERY_MIN = 45   # iter18 2026-06-06: full-artifact backups are heavier — user chose 45m
# 2026-06-07: re-enabled after the manual every-file mirror completed (08:53 run:
# "Upload complete: 1781s", 12 churned files transferred, rest deduped). Flip True
# only while a MANUAL upload is in flight (concurrent commits race).
# 2026-06-08: DISABLED by user — manual `python src/utils/hf_outputs.py upload outputs/poc`
# is running; the watch's auto-backup would race its commit (the rc=1 042554 failure was
# exactly this collision). Flip back to False once the manual upload finishes.
AUTO_BACKUP_DISABLED = True
# Rebuild the §3-style preview plots from whatever evals are DONE this often (minutes). CPU-only.
PLOT_EVERY_MIN = 15


def _latest_log(mtag):
    """Latest MAIN scheduler log — excludes the per-job logs the scheduler itself writes
    (iter18_ngpu_<mtag>_train_*/_eval_*/_pt_*/_s3_*). Matches both the B5 main tee
    (iter18_ngpu_poc_<ts>.log) and variants like _regate_/_only_pretrain_.
    iter18 2026-06-07: _pt_ (the metric-parallel P: job logs) MUST be excluded too — they are
    NEWER than the main tee and carry no GPU ◀/✓ markers, so picking one made every job read as
    pending (all train cells showed ⬚ despite the resume-skip of 10 trained arms)."""
    cands = [p for p in (REPO / "logs").glob(f"iter18_ngpu_{mtag}*.log")
             if p.exists() and not any(seg in p.name for seg in ("_train_", "_eval_", "_pt_", "_s3_"))]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _arm_consumed(jobs, mtag):
    """{jid: Σ seconds across ALL of that job's per-arm log files} (iter18 2026-06-07:
    '✅ (cached)' hid that a resume-skipped arm may have burned 12h+ in earlier runs).
    One log per LAUNCH — mid-run interruptions create several logs per arm (full_ft has
    4: _185942, _234435, _001232, _002339). Each segment = (log mtime − the UTC start
    timestamp in its filename); mtime = the last byte tee wrote = completion/kill moment.
    Symlinked logs (moved to result_outputs/) stat-follow to the target — mtimes intact.
    A LIVE arm's open segment is included but unused (the 🔄 cell path never reads it)."""
    totals = {}
    del mtag  # jid→log template already carries the mode tag
    for jid, j in jobs.items():
        secs = 0
        for p in REPO.glob(j["log"].format(ts="*")):
            m = re.search(r"_(\d{8}_\d{6})\.log$", p.name)
            if not m:
                continue
            start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            try:
                secs += max(p.stat().st_mtime - start.timestamp(), 0)
            except OSError:
                continue
        if secs:
            totals[jid] = secs
    return totals


def _sod(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def _dur(secs):
    secs = int(round(secs))
    if secs < 0:
        secs += 86400
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _arm_of(jid):
    return jid.split(":")[2] if jid.startswith("T:") else "eval"


def _tail(path, nbytes=8000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - nbytes))
            return f.read().decode(errors="replace")
    except Exception:
        return ""


def _eval_total(jid, elapsed, prior, ecalib):
    """TOTAL wall estimate for ONE eval job from the stage ledger (med, state, plans).
      · RUNNING (jid in state): elapsed + current-stage remainder (live clip bar, else
        median-minus-time-in-stage) + Σ medians of the QUEUED stages — the queued 8b
        (m12e ~2.2 h on 1825 clips) is now counted instead of capped away.
      · PENDING: Σ medians over the full per-encoder stage plan.
    A stage with no measured wall AND no live projection borrows the per-stage share of
    the whole-job prior (self-corrects the moment any sibling walks that stage)."""
    med, state, plans = ecalib
    plan = plans.get(jid, EVAL_PLAN)
    share = prior / max(len(EVAL_PLAN), 1)

    def w(s):
        return med.get(s, share)

    es = state.get(jid)
    if es is None:                                  # pending — whole plan
        return sum(w(s) for s in plan)
    cur = es["cur"]
    i = plan.index(cur) if cur in plan else 0
    rem_cur = es["rem_cur"] if es["rem_cur"] is not None else max(w(cur) - es["in"], 60)
    _STAGE_NOTE[jid] = f"s{i + 1}/{len(plan)}"
    return elapsed + rem_cur + sum(w(s) for s in plan[i + 1:])


def _pt_total(jobs, jid, elapsed, prior, pt_med):
    """TOTAL wall for a Stage-8b single-metric P: job (iter18 2026-06-07). Each P: log is ONE
    metric's 0→1825 m12e run, so the single clip bar (cur/tot/`recent=`) is an exact estimate —
    no stage ledger needed. Pending (no bar yet) → that metric's measured median (pt_med), else
    the prior. Metrics differ a lot in cost (teacher_free ≫ causal) so pt_med is keyed per-metric."""
    metric = jid.rsplit(":", 1)[-1]
    cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
    txt = _tail(cands[-1]) if cands else ""
    cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
    rr = re.findall(r"recent=([\d.]+)s/clip", txt)
    if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= _MIN_EVAL_POINTS:
        cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
        return elapsed + (tot - cur) * rate
    return max(pt_med.get(metric, prior), elapsed + 300)


def _running_total(jobs, jid, elapsed, prior, calib, ecalib):
    """Total-duration estimate for a RUNNING job.
      · TRAIN — REAL ETA (iter18 2026-06-07): remaining = (ledger_total − progress) ×
        live step rate + calibrated pads for the remaining stage-end probes + finalize.
        Total work comes from the STATIC ledger (yaml+artifacts) — never extrapolated,
        no hidden stages possible. Rate comes from the CURRENT stage's own bar.
      · EVAL — current-stage remainder + Σ queued-stage medians (the eval stage ledger,
        _eval_total); no more current-stage-only blindness or 3 h cap.
      · fallback — prior, capped (only before any in-log data exists)."""
    cap = max(prior * 2.5, elapsed + 600)
    if _arm_of(jid) == "eval":
        return _eval_total(jid, elapsed, prior, ecalib)
    tmpl = jobs[jid]["log"]
    cands = sorted((p for p in REPO.glob(tmpl.format(ts="*")) if p.exists()),
                   key=lambda p: p.stat().st_mtime)   # if p.exists(): skip a log deleted mid-glob (dangling) — .stat() on it crashes the whole watch
    txt = _tail(cands[-1]) if cands else ""
    led = (calib["ledger"] or {}).get(_arm_of(jid))
    if jid.startswith("T:") and led and cands:
        try:
            full = cands[-1].read_text(errors="replace")
        except OSError:
            full = txt
        stages, total = led["stages"], led["total"]
        rm = re.findall(r"Resumed from step (\d+)", full)
        base = int(rm[-1]) if rm else 0
        cont = re.findall(r"stage (\d+) continues at local step (\d+)", full)
        bar_init = int(cont[-1][1]) if cont else 0   # mid-stage anchor: bar starts at L
        if cont:
            base -= bar_init                         # L is already inside the bar's cur
        done_now = sum(int(n) for n in re.findall(r"Stage \w+ complete: (\d+) steps", full))
        # current stage index in the LEDGER (resumed prefix + stages completed this run)
        pre, k = 0, 0
        while k < len(stages) and pre + stages[k] <= base:
            pre += stages[k]
            k += 1
        idx = min(k + len(re.findall(r"Stage \w+ complete:", full)), len(stages) - 1)
        _STAGE_NOTE[jid] = f"s{idx + 1}/{len(stages)}"
        bar = _RE_LIVE_BAR.findall(txt)
        contrib, live_rate = 0, None
        if bar:
            cur, stage_tot = int(bar[-1][0]), int(bar[-1][1])
            bsec = _bar_secs(*bar[-1][2:])
            n_full_bars = len(set(_RE_FINAL_BAR.findall(full)))
            n_complete = len(re.findall(r"Stage \w+ complete:", full))
            if cur < stage_tot:
                contrib = cur                          # live mid-stage bar
                # RECENT rate from the last distinct bar pair (reacts to contention
                # easing within minutes); whole-bar average as fallback (lags hours).
                pairs = [(int(c), _bar_secs(*hms)) for c, _t, *hms in bar]
                for (c1, e1), (c2, e2) in zip(pairs, pairs[1:]):
                    if c2 > c1 and e2 > e1:
                        live_rate = (e2 - e1) / (c2 - c1)
                if live_rate is None and cur - bar_init > 0 and bsec > 0:
                    live_rate = bsec / (cur - bar_init)
            elif n_full_bars > n_complete:
                # current stage's bar hit 100% but its "complete" line hasn't printed
                # yet (stage-end probe running) → its steps aren't in done_now yet.
                contrib = stage_tot
        progress = min(base + done_now + contrib, total)
        steps_left = total - progress
        # remaining pads: one per not-yet-passed stage end + 1 finalize-equivalent
        ends_left = sum(1 for i, s in enumerate(stages)
                        if sum(stages[:i + 1]) > progress) + 1
        rate = live_rate or calib["rate"] or (prior / max(total, 1))
        if steps_left <= 0:
            # FINALIZE running (stage-end probe-trio + export). Remaining = measured
            # median finalize MINUS time already spent in it (anchored on the last
            # stage-ckpt mtime) — counts down honestly instead of a frozen "~15m".
            med = calib["finalize"] or _FINALIZE_PAD_S
            mt = _newest_stage_ckpt_mtime(_arm_of(jid), calib["mtag"])
            in_fin = ((_sod(datetime.now(timezone.utc).strftime("%H:%M:%S"))
                       - _epoch_sod(mt)) % 86400) if mt else 0
            _STAGE_NOTE[jid] = (_STAGE_NOTE.get(jid) or "") + "·final"
            return elapsed + max(med - in_fin, 300)
        return elapsed + steps_left * rate + ends_left * calib["pad"]
    return min(max(prior, elapsed + 300), cap)


def _hf_backup():
    """Light `upload outputs/poc` in reuse mode, non-interactive (HF_UPLOAD_MODE=reuse so the
    delete/reuse gate never blocks an unattended watch). The remote _full-*.tar shards +
    _full-manifest.json are protected from the light mirror-cleanup. Returns (rc, log_name)."""
    ulog = REPO / "logs" / f"upload_outputs_poc_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    env = dict(os.environ, HF_UPLOAD_MODE="reuse", PYTHONPATH="src")
    with open(ulog, "wb") as fh:
        rc = subprocess.run(
            ["python", "-u", "src/utils/hf_outputs.py", "upload", "outputs/poc"],
            cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    return rc, ulog.name


def maybe_backup(fully_done, mtag):
    """Keep outputs/poc backed up on HF every UPLOAD_EVERY_MIN minutes + once more at the finish.
    Returns one plain-English status line to print. SANITY runs are never backed up."""
    if AUTO_BACKUP_DISABLED:
        return ("  ⏸️  HF auto-backup PAUSED (manual upload in flight — flip "
                "AUTO_BACKUP_DISABLED=False in iter18_poc_status.py to re-enable)")
    if mtag != "poc":
        return "  ⏫ HF backup: skipped (SANITY outputs are throwaway)"
    last = REPO / "logs" / ".upload_outputs_poc.LAST"
    lock = REPO / "logs" / ".upload_outputs_poc.LOCK"
    done_flag = REPO / "logs" / ".upload_outputs_poc.DONE"
    now_ts = datetime.now(timezone.utc).timestamp()
    if done_flag.exists():
        return "  ✅ run finished and fully backed up to HF — you can kill the vast.ai node any time."
    # stale-lock clear: 4× the cadence (not 2×) — the first every-file mirror moves the
    # ~346G resume anchors and can legitimately run ~2-3 h; clearing its lock early would
    # start a CONCURRENT upload of the same tree (HF commit races).
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 4 * UPLOAD_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < UPLOAD_EVERY_MIN and not fully_done:
        return f"  ⏫ HF backup: last {int(age_min)}m ago · next in ~{max(int(UPLOAD_EVERY_MIN - age_min), 0)}m"
    try:
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  ⏫ HF backup already running — leaving it."
    try:
        last.write_text("")
        rc, name = _hf_backup()
        if rc != 0:
            return f"  ❌ HF backup FAILED rc={rc} (see {name}) — it will try again next backup."
        if fully_done:
            done_flag.write_text(name)
            return f"  ✅ FINAL HF backup done ({name}) — you can kill the vast.ai node now."
        return f"  ✅ HF backup done ({name}) · next in ~{UPLOAD_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


def maybe_plot(mtag, mode):
    """Every PLOT_EVERY_MIN min, rebuild the shared paired-Δ + m13 plots from the evals done SO FAR —
    a live partial hero table to screen-share while the run finishes. Mirrors the scheduler's §3
    finale EXACTLY (same run_eval invocation, same SKIP_STAGES) so there is one recipe, not two.
    SAFE: paired stages always re-aggregate; the final §3 rebuilds the COMPLETE plots at the end.
    With few evals done, run_eval may exit non-zero (missing caches) — reported, harmless."""
    last = REPO / "logs" / ".plot_preview.LAST"
    lock = REPO / "logs" / ".plot_preview.LOCK"
    now_ts = datetime.now(timezone.utc).timestamp()
    hero = REPO / f"outputs/{mtag}/probe_plot/eval/m13_hero_table.png"
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 2 * PLOT_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < PLOT_EVERY_MIN:
        return (f"  🖼  preview: rebuilt {int(age_min)}m ago · next in ~{max(int(PLOT_EVERY_MIN - age_min), 0)}m"
                f" → outputs/{mtag}/probe_plot/eval/m13_hero_table.png")
    try:
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  🖼  preview already rebuilding — leaving it."
    try:
        last.write_text("")
        all_encs = " ".join([enc_name("frozen")] + [enc_name(e) for e in ARM2ENC.values()])
        chain = (f"SKIP_STAGES={S3_SKIP_PERENC} CACHE_POLICY_ALL=1 "
                 f"./scripts/run_eval.sh --{mode} --encoders \"{all_encs}\"")
        plog = REPO / "logs" / f"plot_preview_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
        with open(plog, "wb") as fh:
            rc = subprocess.run(chain, shell=True, executable="/bin/bash", cwd=str(REPO),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        if hero.exists():
            return f"  🖼  preview REBUILT (rc={rc}) → outputs/{mtag}/probe_plot/eval/ · next in ~{PLOT_EVERY_MIN}m"
        return f"  🖼  preview rc={rc} — partial/blocked, see {plog.name} · next in ~{PLOT_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    ap.add_argument("--log", default=None,
                    help="explicit main log (default: latest logs/iter18_ngpu_<mode>*.log "
                         "excluding per-job _train_/_eval_/_s3_ logs)")
    args = ap.parse_args()

    jobs, mtag = build_jobs(args.mode)
    log = Path(args.log) if args.log else _latest_log(mtag)
    if log is None or not log.exists():
        sys.exit(f"No {args.mode} main log found (logs/iter18_ngpu_{mtag}*.log). "
                 f"Has the scheduler started?")

    text = log.read_text(errors="replace")

    # iter18 2026-06-08: the run's backbone is in the banner (backbone=…). If THIS watch pane was
    # launched without the matching ITER18_BACKBONE, our imported BACKBONE (+ all job-ids/paths)
    # are for the wrong family → every cell would read pending. Warn LOUDLY rather than mislead.
    _bm = re.search(r"backbone=(\S+)", text)
    if _bm and _bm.group(1) != BACKBONE:
        print(f"  ⚠️  BACKBONE MISMATCH — run is '{_bm.group(1)}' but this watch pane is '{BACKBONE}'. "
              f"Re-run:  ITER18_BACKBONE={_bm.group(1)} python -u scripts/iter18_poc_status.py", flush=True)

    # An --only run restricts the DAG — mirror that restriction so this table matches reality.
    om = re.search(r"\[--only\] restricted to \[(.*?)\]", text)
    only_arms = set(re.findall(r"'([^']+)'", om.group(1))) if om else None
    if only_arms:
        keep = {f"T:{BACKBONE}:{a}" for a in only_arms}
        jobs = {jid: j for jid, j in jobs.items() if jid in keep}

    # --skip-arms runs drop arms (train+eval) from the DAG — mirror it (iter18 2026-06-07).
    sm = re.search(r"\[--skip-arms\] dropped \[(.*?)\]", text)
    if sm:
        skip = set(re.findall(r"'([^']+)'", sm.group(1)))
        drop = ({f"T:{BACKBONE}:{a}" for a in skip}
                | {f"E:{enc_name(ARM2ENC[a])}" for a in skip if a in ARM2ENC}
                | {f"P:{enc_name(ARM2ENC[a])}:{m}" for a in skip if a in ARM2ENC for m in PT_METRICS})
        jobs = {jid: j for jid, j in jobs.items() if jid not in drop}

    launched = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ◀ (\S+)", text)}
    done = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✓ (\S+)", text)}
    failed = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✗ (\S+)", text)}
    rm = re.search(r"\[resume --cache 1\] skipping \d+ already-trained arms: \[(.*?)\]", text)
    if rm:
        # iter18 prints BARE arm names (iter17 printed bb:arm) → rebuild the full jid.
        for tok in re.findall(r"'([^']+)'", rm.group(1)):
            done.setdefault(f"T:{BACKBONE}:{tok}", "resume")
    # P: (Stage-8b single-metric) jobs are 'done' when their aggregate_<metric>.json exists on disk.
    # The scheduler resume-skips them silently (no per-job GPU ✓ marker logged), so mirror that from
    # disk — and it also surfaces 8b metrics produced by a prior monolithic eval as already-done.
    for jid in jobs:
        if jid.startswith("P:"):
            enc_nm, metric = jid[2:].rsplit(":", 1)   # NOT 'enc_name' — that's the imported helper (shadowing it makes it a main()-local → UnboundLocalError)
            if (REPO / f"outputs/{mtag}/predictor_temporal/{enc_nm}/aggregate_{metric}.json").exists():
                done.setdefault(jid, "resume")
    # iter18 2026-06-08: drop any log-parsed jid NOT in THIS run's job set. A watch pane reading a
    # DIFFERENT-backbone or OLD (pre-banner) log carries foreign jids (e.g. T:vjepa_2_1_vitG:… while
    # this pane built vjepa_2_1_vitg jobs); feeding one into jobs[jid] downstream → KeyError. The
    # backbone-mismatch warning above flags the case when the banner records backbone=.
    launched = {j: t for j, t in launched.items() if j in jobs}
    done = {j: t for j, t in done.items() if j in jobs}
    failed = {j: t for j, t in failed.items() if j in jobs}
    gm = re.search(r"gpus=(\d+)", text)
    gpus = int(gm.group(1)) if gm else 4
    s3 = re.search(r"§3 rc=(-?\d+)", text)
    only_complete = "--only run complete" in text
    now_hms = datetime.now(timezone.utc).strftime("%H:%M:%S")
    now_s = _sod(now_hms)

    def classify(jid):
        if jid in done:
            return "done"
        if jid in failed:
            return "failed"
        if jid in launched:
            return "running"
        return "pending"

    def elapsed(jid):
        return (now_s - _sod(launched[jid])) % 86400 if jid in launched else 0

    # ── REAL-ETA inputs (iter18 2026-06-07): static workload ledger + calibration ──
    try:
        ledger = _build_ledger(mtag)
    except Exception as e:   # the watch must survive a ledger failure — fall back LOUDLY
        print(f"  ⚠️  workload ledger FAILED ({type(e).__name__}: {e}) — ETA on priors", flush=True)
        ledger = None
    calib = _calibrate(jobs, done, launched, mtag, ledger)
    # eval REAL-ETA: per-stage medians + running-eval stage state + per-job stage plans.
    ecalib = _eval_calibrate(jobs, mtag, now_s)

    # evals stay duration-class-based (homogeneous); their measured mean replaces the prior.
    est = dict(PRIOR if mtag == "poc" else PRIOR_SANITY)
    measured = {}
    for jid, t in done.items():
        if t != "resume" and jid in launched:
            measured.setdefault(_arm_of(jid), []).append((_sod(t) - _sod(launched[jid])) % 86400)
    for arm, vals in measured.items():
        est[arm] = sum(vals) / len(vals)

    # Σ secs per jid across all log segments — reused for the P: per-metric medians here and the
    # table's consumed/✅ cells + Σ TOTAL row below.
    consumed = _arm_consumed(jobs, mtag)
    # P: (Stage-8b single-metric) per-metric medians from COMPLETED P: jobs → priors for pending P:
    # (metrics differ a lot in cost — teacher_free ≫ causal — so keep them per-metric).
    _pt_by = {}
    for jid in jobs:
        if jid.startswith("P:") and classify(jid) == "done" and jid in consumed:
            _pt_by.setdefault(jid.rsplit(":", 1)[-1], []).append(consumed[jid])
    pt_med = {m: sorted(v)[len(v) // 2] for m, v in _pt_by.items()}

    # ── per-job remaining time ──
    remaining = {}
    for jid in jobs:
        st = classify(jid)
        arm = _arm_of(jid)
        prior = est.get(arm, est["eval"])
        if st in ("done", "failed"):
            remaining[jid] = 0.0
        elif jid.startswith("P:"):
            # Stage-8b single-metric job — estimated from its OWN m12e clip bar (running) or the
            # per-metric median (pending), NOT the stage ledger. Cold prior is _PT_COLD_PRIOR
            # (per-metric), NOT est["eval"] (the whole-eval time, ~10× too big for one metric).
            el = elapsed(jid) if st == "running" else 0.0
            tot = _pt_total(jobs, jid, el, _PT_COLD_PRIOR.get(mtag, 20 * 60), pt_med)
            remaining[jid] = max(tot - el, 60) if st == "running" else tot
        elif st == "running":
            remaining[jid] = max(_running_total(jobs, jid, elapsed(jid), prior, calib, ecalib)
                                 - elapsed(jid), 60)
        elif arm == "eval":
            # PENDING E: eval = stages 2-8 plan × measured/projected stage walls (Stage 8b is the
            # separate P: jobs above) — its OWN honest total, not a class prior.
            remaining[jid] = _eval_total(jid, 0.0, prior, ecalib)
        elif jid.startswith("T:") and ledger and calib["rate"] is not None:
            # PENDING train = full ledger plan × measured pure rate + every pad —
            # the REAL ETA (known work ÷ measured throughput), not a hopeful prior.
            led = ledger[arm]
            remaining[jid] = (led["total"] * calib["rate"]
                              + (len(led["stages"]) + 2) * calib["pad"])
        else:
            remaining[jid] = prior

    # ── forward DAG sim over the GPU pool → finish time (secs from now) per job ──
    done_set = {j for j in jobs if classify(j) in ("done", "failed")}
    running = {j for j in jobs if classify(j) == "running"}
    pending = {j for j in jobs if classify(j) == "pending"}
    finish = {j: 0.0 for j in done_set}
    run_end = {j: remaining[j] for j in running}
    free = max(gpus - len(running), 0)
    t = 0.0
    guard = 0
    while (running or pending) and guard < _SIM_GUARD:
        guard += 1
        for j in [x for x in jobs if x in pending and jobs[x]["deps"] <= done_set]:
            if free <= 0:
                break
            pending.discard(j)
            running.add(j)
            run_end[j] = t + remaining[j]
            free -= 1
        if not running:
            break
        nxt = min(running, key=lambda j: run_end[j])
        t = run_end[nxt]
        running.discard(nxt)
        done_set.add(nxt)
        finish[nxt] = t
        free += 1
    # §3 finale: MEASURED from the 15-min preview runs (maybe_plot — the EXACT same
    # run_eval recipe/stages). A preview lasting >120s had real eval caches to aggregate,
    # so its wall ≈ the finale's; the 7s previews (caches missing, early exit) are
    # excluded. Until a substantive preview exists, the documented prior is used and
    # the printout SAYS so — no unlabeled fake numbers (user order, 06-07).
    pv = []
    for p in (REPO / "logs").glob("plot_preview_2*.log"):
        if not p.exists():        # skip dangling symlinks
            continue
        m2 = re.search(r"_(\d{8}_\d{6})\.log$", p.name)
        if m2:
            st = datetime.strptime(m2.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            d = p.stat().st_mtime - st.timestamp()
            if d > 120:
                pv.append(d)
    pv.sort()
    if pv:
        s3_pad, s3_src = pv[len(pv) // 2], "measured"
    else:
        s3_pad, s3_src = (15 if mtag == "poc" else 4) * 60, "prior, unmeasured"
    eta_secs = (max(finish.values()) if finish else 0) + s3_pad

    # ── render: emoji table (rows = arm, cols = train | eval) ──
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for jid in jobs:
        counts[classify(jid)] += 1

    # `consumed` (Σ GPU-seconds per jid across ALL log segments) was computed above for pt_med.

    def cell(jid):
        if jid not in jobs:
            return "—"
        st = classify(jid)
        if st == "done":
            # TOTAL duration consumed across ALL the arm's log segments (every launch,
            # incl. interrupted ones) — one number, no per-run breakdown (user 06-07).
            tot = consumed.get(jid, 0)
            if tot:
                return f"✅ {_dur(tot)}"
            if done.get(jid) == "resume":
                return "✅ (prior run)"
            return f"✅ {_dur((_sod(done[jid]) - _sod(launched[jid])) % 86400)}"
        if st == "failed":
            return "❌ FAILED"
        if st == "running":
            note = _STAGE_NOTE.get(jid)
            stg = f"·{note}" if note else ""
            return f"🔄 {_dur(elapsed(jid))}{stg}·~{_dur(finish[jid])}"
        return f"⬚ ~{_dur(finish[jid])}"

    def _eval_group_cell(enc):   # param is 'enc' (a full encoder name) — NOT 'enc_name' (shadowing the imported helper is the bug that bit twice)
        # iter18 2026-06-07: an encoder's eval = ONE E: job (stages 2-8) + 6 P: jobs (Stage 8b, one
        # metric each). They run in PARALLEL across the GPU pool, so the group's remaining is the
        # MAX finish, not the sum. `·8b D✓R▶/6` = D done, R running of the 6 metric jobs — so the
        # one rolled-up cell reconciles with the 🔄 job counter (3 parallel metrics here = 3▶).
        group = [j for j in ([f"E:{enc}"] + [f"P:{enc}:{m}" for m in PT_METRICS]) if j in jobs]
        if not group:
            return "—"
        sts = [classify(j) for j in group]
        if all(s == "done" for s in sts):
            return f"✅ {_dur(sum(consumed.get(j, 0) for j in group))}"
        if any(s == "failed" for s in sts):
            return "❌ FAILED"
        n8 = sum(1 for j in group if j.startswith("P:"))
        n8done = sum(1 for j in group if j.startswith("P:") and classify(j) == "done")
        n8run = sum(1 for j in group if j.startswith("P:") and classify(j) == "running")
        rem = max((finish.get(j, 0.0) for j in group), default=0.0)
        glyph = "🔄" if any(s == "running" for s in sts) else "⬚"
        tag = (f"·8b {n8done}✓{n8run}▶/{n8}" if n8 else "")
        return f"{glyph}{tag}·~{_dur(rem)}"

    def kemoji(arm):
        if arm.startswith("pretrain"):
            return "🚂"
        if arm.startswith(("surgery", "surgical")):
            return "🔧"
        return "🔩"   # FT-technique baselines (full_ft/lpft/peft/cassle/ewc)

    SW, CW = 32, 22
    bar = "─"
    print(f"═══ iter18 {args.mode} · {log.name} · now {now_hms} UTC · {gpus} GPU ═══")
    print("┌" + bar * SW + "┬" + bar * CW + "┬" + bar * CW + "┐")
    print("│" + " arm".ljust(SW) + "│" + " train".ljust(CW) + "│" + " eval".ljust(CW) + "│")
    print("├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    print("│ " + "📊 frozen (eval-only)".ljust(SW - 1) + "│" + " —".ljust(CW) + "│ "
          + _eval_group_cell(enc_name("frozen")).ljust(CW - 1) + "│")
    for arm in TRAIN_ORDER:
        tj, ej = f"T:{BACKBONE}:{arm}", f"E:{enc_name(ARM2ENC[arm])}"
        if tj not in jobs and ej not in jobs:
            continue
        print("│ " + f"{kemoji(arm)} {arm}".ljust(SW - 1) + "│ "
              + cell(tj).ljust(CW - 1) + "│ " + _eval_group_cell(enc_name(ARM2ENC[arm])).ljust(CW - 1) + "│")

    # Σ TOTAL row (iter18 2026-06-07): per column, completed compute (consumed across
    # every log segment) + estimated remaining for running/pending jobs — the full
    # GPU-time bill of the table, done + still-to-come.
    def _col_total(kind):
        tot = 0.0
        for jid in jobs:
            if (_arm_of(jid) == "eval") != (kind == "eval"):
                continue
            st = classify(jid)
            if st in ("done", "failed"):
                tot += consumed.get(jid, 0) or (
                    (_sod(done[jid]) - _sod(launched[jid])) % 86400
                    if jid in done and done[jid] != "resume" and jid in launched else 0)
            elif st == "running":
                tot += consumed.get(jid, elapsed(jid)) + remaining[jid]
            else:
                tot += remaining[jid]
        return tot

    t_tot, e_tot = _col_total("train"), _col_total("eval")
    print("├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    print("│ " + "Σ TOTAL (done + estimated)".ljust(SW - 1) + "│ "
          + f"Σ {_dur(t_tot)}".ljust(CW - 1) + "│ " + f"Σ {_dur(e_tot)}".ljust(CW - 1) + "│")
    print("└" + bar * SW + "┴" + bar * CW + "┴" + bar * CW + "┘")
    print(f"  Σ compute bill (train+eval, done+estimated): ~{_dur(t_tot + e_tot)} GPU-time")

    # ── Stage-8b metric fan: encoder × 6 predictor-temporal metrics, LIVE grid (iter18 2026-06-07) ──
    # Makes the metric-parallel fan-out visible — each cell = the P:<enc>:<metric> job's state.
    # "2-8" = the encoder's E: job (the non-8b eval stages: features/probe/taxonomy/motion/future).
    _GLYPH = {"done": "✓", "running": "▶", "pending": "·", "failed": "✗"}
    _ABBR = {"rollout": "roll", "causal": "caus", "tdist": "tdis",
             "teacher_free": "t-fr", "maskratio": "mask", "order": "ordr"}
    enc_rows = ([("frozen", enc_name("frozen"))]
                + [(a, enc_name(ARM2ENC[a])) for a in TRAIN_ORDER
                   if f"E:{enc_name(ARM2ENC[a])}" in jobs])
    EW, PW = 26, 5

    def _gl(jid):
        return _GLYPH[classify(jid)] if jid in jobs else " "
    g_top = "┌" + bar * EW + "┬" + bar * 5 + ("┬" + bar * PW) * len(PT_METRICS) + "┐"
    g_mid = "├" + bar * EW + "┼" + bar * 5 + ("┼" + bar * PW) * len(PT_METRICS) + "┤"
    g_bot = "└" + bar * EW + "┴" + bar * 5 + ("┴" + bar * PW) * len(PT_METRICS) + "┘"
    print(f"\n  Stage-8b metric fan (encoder × {len(PT_METRICS)} metrics · ✓ done · ▶ run · · pend)")
    print("  " + g_top)
    print("  │" + " encoder".ljust(EW) + "│" + "2-8".center(5)
          + "│" + "│".join(_ABBR[m].center(PW) for m in PT_METRICS) + "│")
    print("  " + g_mid)
    for label, enc in enc_rows:
        cells = "│".join(_gl(f"P:{enc}:{m}").center(PW) for m in PT_METRICS)
        print("  │ " + label.ljust(EW - 1) + "│" + _gl(f"E:{enc}").center(5) + "│" + cells + "│")
    print("  " + g_bot)
    _pall = [j for j in jobs if j.startswith("P:")]
    _pc = {k: sum(1 for j in _pall if classify(j) == k) for k in ("done", "running", "pending")}
    print(f"  {_pc['done']}✓ done · {_pc['running']}▶ running · {_pc['pending']}· pending"
          f"  of {len(_pall)} metric jobs")

    # ── summary + run ETA ──
    end_utc = datetime.now(timezone.utc) + timedelta(seconds=eta_secs)
    end_pdt = end_utc - timedelta(hours=7)   # user reads PDT
    print(f"\n  {counts['done']}✅  {counts['running']}🔄  {counts['pending']}⬚  "
          f"{counts['failed']}❌  / {len(jobs)} jobs")
    fully_done = bool(s3 and s3.group(1) == "0") or only_complete
    settled = counts["done"] + counts["failed"] == len(jobs)
    plot_msg = (maybe_plot(mtag, args.mode)
                if (not settled or counts["failed"]) and not fully_done and not only_arms else None)
    backup_msg = maybe_backup(fully_done, mtag)
    if fully_done:
        print("\a\n" + "█" * 78)
        print(f"  🏁 RUN COMPLETE — plots in outputs/{mtag}/probe_plot/eval/"
              if not only_complete else "  🏁 --only RUN COMPLETE (no evals/finale by design)")
        print(backup_msg)
        print("█" * 78)
    elif settled and counts["failed"]:
        print("\a\n" + "█" * 78)
        print(f"  ⚠️  ALL {len(jobs)} JOBS SETTLED · {counts['failed']}❌ FAILED — DO NOT KILL YET.")
        print("  The scheduler skips the §3 finale on any failure. Fix, then resume the survivors:")
        print(f"       python -u scripts/iter18_poc_ngpu.py --mode {args.mode} --gpus {gpus} --cache 1")
        if plot_msg:
            print(plot_msg)
        print(backup_msg)
        print("█" * 78)
    elif settled:
        print(f"  🏁 all {len(jobs)} jobs settled · §3 finale {'🔄 running' if '§3 finale' in text else '⬚ pending'}")
        print(backup_msg)
    else:
        print(f"  🏁 run ETA  ~{_dur(eta_secs)} from now  →  {end_utc:%H:%M} UTC · "
              f"{end_pdt:%H:%M} PDT  (incl. §3 finale ~{_dur(s3_pad)} · {s3_src})")
        if plot_msg:
            print(plot_msg)
        print(backup_msg)
    if counts["failed"]:
        print("  ❌ failed (recover with --cache 1): "
              + ", ".join(j for j in jobs if classify(j) == "failed"))
    print("\n  legend: 🚂 pretrain · 🔧 surgery · 🔩 FT baseline · 📊 eval │ "
          "✅ done · 🔄 running · ⬚ pending · ❌ failed")


if __name__ == "__main__":
    main()
