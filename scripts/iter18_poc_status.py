#!/usr/bin/env python3
"""iter18 POC/SANITY live status — emoji TABLE + live-calibrated ETA per job and for the whole run.

Adapted from scripts/legacy/iter17_poc_status.py for the iter18 DAG (ONE backbone, 13 train arms
+ 14 eval jobs; pretrain_encoder is the root for all 12 other arms). Imports build_jobs() from
iter18_poc_ngpu.py so the job list is the SINGLE SOURCE OF TRUTH (it can never drift from the
scheduler). Classifies every job ✅/🔄/⬚/❌ from the main log's GPU ◀/✓/✗ markers.

ETA: a forward DAG simulation over the GPU pool. Per-arm durations are OBSERVED, not hardcoded —
  · a completed job's measured wall sets that arm-class's estimate (mean of completed peers),
  · a RUNNING job's total is extrapolated from its own per-job-log progress (step tqdm / recent rate),
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
from iter18_poc_ngpu import ARM2ENC, BACKBONE, S3_SKIP_PERENC, build_jobs  # noqa: E402  (canonical DAG)
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
    rates, pads = [], []
    if not ledger:
        return {"ledger": ledger, "rate": None, "pad": _FINALIZE_PAD_S}
    for jid, t in done.items():
        if t == "resume" or not jid.startswith("T:") or jid not in launched:
            continue
        arm = _arm_of(jid)
        cands = sorted(REPO.glob(jobs[jid]["log"].format(ts="*")), key=lambda p: p.stat().st_mtime)
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
    pads.sort()
    pooled = (sum(b for b, _ in rates) / sum(e for _, e in rates)) if rates else None
    return {"ledger": ledger,
            "rate": pooled,    # pooled Σbar-seconds / Σexecuted-steps across completions
            "pad": pads[len(pads) // 2] if pads else _FINALIZE_PAD_S}


_STAGE_NOTE = {}   # jid → "sN/M" live stage marker, filled by _running_total, shown in the cell
# End-of-training finalize for a multi-stage arm (stage-end probe-trio on 451 clips +
# best-ckpt reload + student_encoder export) — DI measured ~15m at POC. Shown once every
# stage's steps are done; "remaining ~1m00s" during a 15-min finalize was a display lie.
_FINALIZE_PAD_S = 15 * 60
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

# Back up outputs/poc to HF this often (minutes) WHILE the run goes, so the final backup at the end
# is tiny and the paid node can be killed right away. Driven by the 60s `watch` refresh + a stamp file.
UPLOAD_EVERY_MIN = 45   # iter18 2026-06-06: full-artifact backups are heavier — user chose 45m
# 2026-06-07 TEMP (user order): auto-backup PAUSED — the user runs the first every-file
# mirror MANUALLY (python -u src/utils/hf_outputs.py upload outputs/poc …); two concurrent
# upload_folder commits on the same tree race (412 retries / doubled bandwidth).
# FLIP BACK TO False once the manual upload completes.
AUTO_BACKUP_DISABLED = True
# Rebuild the §3-style preview plots from whatever evals are DONE this often (minutes). CPU-only.
PLOT_EVERY_MIN = 15


def _latest_log(mtag):
    """Latest MAIN scheduler log — excludes the per-job logs the scheduler itself writes
    (iter18_ngpu_<mtag>_train_*/_eval_*/_s3_*). Matches both the B5 main tee
    (iter18_ngpu_poc_<ts>.log) and variants like _regate_/_only_pretrain_."""
    cands = [p for p in (REPO / "logs").glob(f"iter18_ngpu_{mtag}*.log")
             if "_train_" not in p.name and "_eval_" not in p.name and "_s3_" not in p.name]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


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


def _running_total(jobs, jid, elapsed, prior, calib):
    """Total-duration estimate for a RUNNING job.
      · TRAIN — REAL ETA (iter18 2026-06-07): remaining = (ledger_total − progress) ×
        live step rate + calibrated pads for the remaining stage-end probes + finalize.
        Total work comes from the STATIC ledger (yaml+artifacts) — never extrapolated,
        no hidden stages possible. Rate comes from the CURRENT stage's own bar.
      · EVAL — from the current stage's RECENT rate ('recent=<R>s/clip').
      · fallback — prior, capped (only before any in-log data exists)."""
    cap = max(prior * 2.5, elapsed + 600)
    tmpl = jobs[jid]["log"]
    cands = sorted(REPO.glob(tmpl.format(ts="*")), key=lambda p: p.stat().st_mtime)
    txt = _tail(cands[-1]) if cands else ""
    if _arm_of(jid) == "eval" and txt:
        cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
        rr = re.findall(r"recent=([\d.]+)s/clip", txt)
        if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= _MIN_EVAL_POINTS:
            cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
            return min(elapsed + (tot - cur) * rate, 3 * 3600)
        return min(max(prior, elapsed + 300), cap)
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
            return elapsed + _FINALIZE_PAD_S          # finalize (probe+export) running
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
        all_encs = " ".join(["vjepa_2_1_frozen"] + [f"vjepa_2_1_{e}" for e in ARM2ENC.values()])
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

    # An --only run restricts the DAG — mirror that restriction so this table matches reality.
    om = re.search(r"\[--only\] restricted to \[(.*?)\]", text)
    only_arms = set(re.findall(r"'([^']+)'", om.group(1))) if om else None
    if only_arms:
        keep = {f"T:{BACKBONE}:{a}" for a in only_arms}
        jobs = {jid: j for jid, j in jobs.items() if jid in keep}

    launched = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ◀ (\S+)", text)}
    done = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✓ (\S+)", text)}
    failed = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✗ (\S+)", text)}
    rm = re.search(r"\[resume --cache 1\] skipping \d+ already-trained arms: \[(.*?)\]", text)
    if rm:
        # iter18 prints BARE arm names (iter17 printed bb:arm) → rebuild the full jid.
        for tok in re.findall(r"'([^']+)'", rm.group(1)):
            done.setdefault(f"T:{BACKBONE}:{tok}", "resume")
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

    # evals stay duration-class-based (homogeneous); their measured mean replaces the prior.
    est = dict(PRIOR if mtag == "poc" else PRIOR_SANITY)
    measured = {}
    for jid, t in done.items():
        if t != "resume" and jid in launched:
            measured.setdefault(_arm_of(jid), []).append((_sod(t) - _sod(launched[jid])) % 86400)
    for arm, vals in measured.items():
        est[arm] = sum(vals) / len(vals)

    # ── per-job remaining time ──
    remaining = {}
    for jid in jobs:
        st = classify(jid)
        arm = _arm_of(jid)
        prior = est.get(arm, est["eval"])
        if st in ("done", "failed"):
            remaining[jid] = 0.0
        elif st == "running":
            remaining[jid] = max(_running_total(jobs, jid, elapsed(jid), prior, calib)
                                 - elapsed(jid), 60)
        elif jid.startswith("T:") and ledger and calib["rate"] is not None:
            # PENDING train = full ledger plan × measured pure rate + every pad —
            # the REAL ETA (known work ÷ measured throughput), not a hopeful prior.
            led = ledger[arm]
            remaining[jid] = (led["total"] * calib["rate"]
                              + (len(led["stages"]) + 2) * calib["pad"])
        else:
            remaining[jid] = prior

    # Lift each PENDING eval to the slowest RUNNING eval's remaining (fresh evals all hit the same
    # slowest stage; a resume-contaminated tiny prior would otherwise freeze the run ETA).
    run_eval_rem = [remaining[j] for j in jobs if classify(j) == "running" and _arm_of(j) == "eval"]
    if run_eval_rem:
        floor = max(run_eval_rem)
        for jid in jobs:
            if classify(jid) == "pending" and _arm_of(jid) == "eval":
                remaining[jid] = max(remaining[jid], floor)

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
    # §3 finale pad: paired-Δ + m13 over all encoders ≈ 15 min at POC (10K BCa over ~9k clips),
    # ≈ 4 min at SANITY (20 clips) — the flat 15 made a finished SANITY read "ETA 16m".
    s3_pad = (15 if mtag == "poc" else 4) * 60
    eta_secs = (max(finish.values()) if finish else 0) + s3_pad

    # ── render: emoji table (rows = arm, cols = train | eval) ──
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for jid in jobs:
        counts[classify(jid)] += 1

    def cell(jid):
        if jid not in jobs:
            return "—"
        st = classify(jid)
        if st == "done":
            d = "(cached)" if done.get(jid) == "resume" else _dur((_sod(done[jid]) - _sod(launched[jid])) % 86400)
            return f"✅ {d}"
        if st == "failed":
            return "❌ FAILED"
        if st == "running":
            note = _STAGE_NOTE.get(jid)
            stg = f"·{note}" if note else ""
            return f"🔄 {_dur(elapsed(jid))}{stg}·~{_dur(finish[jid])}"
        return f"⬚ ~{_dur(finish[jid])}"

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
          + cell("E:vjepa_2_1_frozen").ljust(CW - 1) + "│")
    for arm in TRAIN_ORDER:
        tj, ej = f"T:{BACKBONE}:{arm}", f"E:vjepa_2_1_{ARM2ENC[arm]}"
        if tj not in jobs and ej not in jobs:
            continue
        print("│ " + f"{kemoji(arm)} {arm}".ljust(SW - 1) + "│ "
              + cell(tj).ljust(CW - 1) + "│ " + cell(ej).ljust(CW - 1) + "│")
    print("└" + bar * SW + "┴" + bar * CW + "┴" + bar * CW + "┘")

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
              f"{end_pdt:%H:%M} PDT  (incl. §3 finale ~{s3_pad // 60}m)")
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
