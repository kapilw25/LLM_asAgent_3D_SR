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
  minutes (light `upload`, reuse mode — now carries ALL result artifacts incl. ckpt_best — HF dedups unchanged files; the full-fidelity `_full-*.tar`
  shards + `_full-manifest.json` on the remote are PROTECTED from its mirror-cleanup) and once more
  when the run finishes, so the paid node can be killed right after completion.

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
from iter18_poc_ngpu import ARM2ENC, BACKBONE, S3_SKIP_PERENC, build_jobs  # noqa: E402  (canonical DAG)

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
# Factor arms run SEQUENTIAL STAGES (one tqdm bar each: 4 stages DI / 3 noDI / 4 raw) —
# a per-stage bar must never be extrapolated to the whole job. Value = total stage count
# (incl. LP-FT stage0): plan banners print only at STAGE START, so unstarted stages are
# invisible in the log — the count lets the estimator pad them in (06-06: noDI sat at
# stage1 bar 100% and read "~1m00s" while its whole 219-step stage2 hadn't begun).
_MULTI_STAGE_ARMS = {"surgery_3stage_DI_encoder": 4, "surgery_noDI_encoder": 3,
                     "surgery_raw_encoder": 4,
                     # lpft = LP warmup (43 steps) + FT stage (06-07: its stage0 bar read
                     # "~36m" for a ~5h arm). Until the FT banner prints, the pad
                     # under-sizes stage1 (uses the 43-step banner) — exact after ~25m.
                     "lpft_encoder": 2}
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


def _running_total(jobs, jid, elapsed, prior):
    """Total-duration estimate for a RUNNING job, read from its own per-job log:
      · TRAIN — extrapolate from the whole-job step tqdm `cur/tot [` (m09a is step-based; the
        m09c/e/f/b/d per-stage bars also match — the LAST bar in the tail is the live one, and
        within a stage the fraction is still a usable rate signal, capped at prior×2.5).
      · EVAL — from the current stage's RECENT rate ('recent=<R>s/clip'): remaining ≈ (tot-cur)×R
        (tqdm's overall 'remaining' is cold-start-inflated; the recent window tracks the live rate).
      · else — prior, capped at prior×2.5."""
    cap = max(prior * 2.5, elapsed + 600)
    tmpl = jobs[jid]["log"]
    cands = sorted(REPO.glob(tmpl.format(ts="*")), key=lambda p: p.stat().st_mtime)
    txt = _tail(cands[-1]) if cands else ""
    # iter18 fix (2026-06-06): tqdm extrapolation is only valid for SINGLE-BAR jobs
    # (pretrain, heads, autorgn/full_ft/lpft/peft/cassle/ewc — one whole-job bar).
    # The multi-STAGE factor arms emit one bar PER stage, so "67/219 [" of stage 1
    # made a ~5h arm read "~40m" (and its sibling "~57m" off a different stage) —
    # they fall through to the class prior until a completion calibrates them.
    if jid.startswith("T:") and txt and _arm_of(jid) not in _MULTI_STAGE_ARMS:
        # --cache 1 resume offset: total ≈ elapsed × (tot - s0)/(cur - s0).
        rm = re.findall(r"Resumed from step (\d+)", txt)
        s0 = int(rm[-1]) if rm else 0
        ms = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
        if ms:
            cur, tot = int(ms[-1][0]), int(ms[-1][1])
            # strict cur < tot: a COMPLETE bar (e.g. probe_decode 451/451) is a finished
            # sub-task, extrapolating from it pinned head ETAs to `elapsed` (the 51m vs
            # 1h34m sibling asymmetry).
            if s0 < cur < tot:
                return min(max(elapsed * (tot - s0) / (cur - s0), elapsed + 120), cap)
    if _arm_of(jid) == "eval" and txt:
        cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
        rr = re.findall(r"recent=([\d.]+)s/clip", txt)
        if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= _MIN_EVAL_POINTS:
            cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
            return min(elapsed + (tot - cur) * rate, 3 * 3600)
    # Multi-stage arm progress (2026-06-06 fix: concurrent factor-streaming runs 60-140
    # s/step vs the 27 s/step solo prior, so the prior-clamp printed 'remaining ~5m' at
    # elapsed 10h). Honest estimate from the arm's OWN log: completed-stage step counts +
    # live stage bar position + the stage plan banners give true global progress; the last
    # recent= window gives the true current rate.
    if _arm_of(jid) in _MULTI_STAGE_ARMS and cands:
        try:
            full = cands[-1].read_text(errors="replace")
        except OSError:
            full = txt
        n_total = _MULTI_STAGE_ARMS[_arm_of(jid)]
        done_list = re.findall(r"Stage \w+ complete: (\d+) steps", full)
        done_steps = sum(int(n) for n in done_list)
        planned = [int(n) for n in re.findall(r"\| (\d+) steps \| warmup", full)]
        # surgery: train-step bars ONLY (desc "surgery:<stage>") — the generic cur/tot
        # pattern also matched the probe-trio's 451-clip bar and polluted the estimate.
        bar = re.findall(r"surgery:\S+\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[", txt)
        rr = re.findall(r"recent=([\d.]+)s/step", full)
        # --cache 1 resume: the new log carries neither the skipped stages' "complete"
        # lines nor their plan banners — fold them in from the resume prints.
        rm = re.findall(r"Resumed from step (\d+)", full)
        base = int(rm[-1]) if rm else 0
        sk = re.findall(r"skipping stages <= (\d+)", full)
        n_prior = (int(sk[-1]) + 1) if sk else 0
        cont = re.findall(r"stage (\d+) continues at local step (\d+)", full)
        if cont and not sk:        # mid-stage _latest anchor: bar starts at initial=L,
            n_prior = int(cont[-1][0])     # so L is already inside the bar's cur —
            base -= int(cont[-1][1])       # don't count it twice
        # pad one entry per not-yet-started stage with the last banner's size
        # (post-stage0 recipe stages are equal-sized: 50/50 noDI, 30/30 DI/raw tail).
        missing = n_total - n_prior - len(planned)
        if planned and missing > 0:
            planned = planned + [planned[-1]] * missing
        if bar:
            cur, stage_tot = int(bar[-1][0]), int(bar[-1][1])
            # stale-bar guard: when every STARTED stage already printed its "complete"
            # line, the last bar belongs to a finished stage (between-stages probe is
            # running) — its steps are already inside done_steps; contribution = 0.
            live = len(done_list) < (len(planned) - max(missing, 0))
            contrib = min(cur, stage_tot) if live else 0
        else:
            # between-stages the probe-trio bar spam can push the last "surgery:" bar
            # out of the tail window — the plan + completed counts alone still give an
            # honest estimate (contribution 0 = stage boundary).
            cur = stage_tot = contrib = 0
        if planned or bar:
            gstep = base + done_steps + contrib
            total = (base + sum(planned)) if planned else None
            _STAGE_NOTE[jid] = f"s{min(n_prior + len(done_list) + 1, n_total)}/{n_total}"
            if total and total > gstep:
                steps_left = total - gstep
            elif bar and stage_tot > cur:
                steps_left = stage_tot - cur
            else:
                # every planned stage's steps are done → end-of-training finalize
                # (stage-end probe + best-ckpt reload + student_encoder export).
                return elapsed + _FINALIZE_PAD_S
            rate = float(rr[-1]) if rr else (elapsed / max(done_steps + contrib, 1))
            return min(elapsed + steps_left * rate, elapsed + 12 * 3600)
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
    if mtag != "poc":
        return "  ⏫ HF backup: skipped (SANITY outputs are throwaway)"
    last = REPO / "logs" / ".upload_outputs_poc.LAST"
    lock = REPO / "logs" / ".upload_outputs_poc.LOCK"
    done_flag = REPO / "logs" / ".upload_outputs_poc.DONE"
    now_ts = datetime.now(timezone.utc).timestamp()
    if done_flag.exists():
        return "  ✅ run finished and fully backed up to HF — you can kill the vast.ai node any time."
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 2 * UPLOAD_EVERY_MIN * 60:
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

    # ── live-calibrated per-arm-class duration estimate ──
    est = dict(PRIOR if mtag == "poc" else PRIOR_SANITY)
    measured = {}
    for jid, t in done.items():
        if t != "resume" and jid in launched:
            measured.setdefault(_arm_of(jid), []).append((_sod(t) - _sod(launched[jid])) % 86400)
    # enc-class arms share one duration profile → pool their completions so the 2nd wave's
    # estimates sharpen from the 1st wave (head arms + pretrain + eval stay their own class).
    enc_class = [a for a in TRAIN_ORDER if a not in _HEAD_ARMS and a != "pretrain_encoder"]
    enc_vals = [v for a in enc_class for v in measured.get(a, [])]
    for arm, vals in measured.items():
        est[arm] = sum(vals) / len(vals)
    for arm in enc_class:
        if arm not in measured and enc_vals:
            est[arm] = sum(enc_vals) / len(enc_vals)

    # ── per-job remaining time ──
    remaining = {}
    for jid in jobs:
        st = classify(jid)
        prior = est.get(_arm_of(jid), est["eval"])
        if st in ("done", "failed"):
            remaining[jid] = 0.0
        elif st == "running":
            remaining[jid] = max(_running_total(jobs, jid, elapsed(jid), prior) - elapsed(jid), 60)
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
