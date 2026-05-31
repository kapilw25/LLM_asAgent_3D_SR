#!/usr/bin/env python3
"""iter17 POC/SANITY live status — emoji TABLE + live-calibrated ETA per job and for the whole run.

Imports build_jobs() from iter17_poc_ngpu.py so the job list is the SINGLE SOURCE OF TRUTH (it can
never drift from the scheduler). Classifies every job ✅/🔄/⬚/❌ from the main log's GPU ◀/✓/✗ markers.

ETA: a forward DAG simulation over the 8-GPU pool. Per-arm durations are OBSERVED, not hardcoded —
  · a completed job's measured wall sets that arm-type's estimate (mean of completed peers),
  · a RUNNING job's total is extrapolated from its own per-job-log progress (Epoch e/N or STAGE s/N),
  · only an arm-type with NO completion yet AND no parseable progress falls back to a prior (below),
    seeded from the 1× node run + iter16 and replaced the moment the first such job reports progress.
So the ETA reflects the latest real throughput and sharpens as the run advances.

AUTO-BACKUP: while you watch, this also backs up outputs/poc to HF every UPLOAD_EVERY_MIN minutes
  (reuse mode — HF skips files that haven't changed) and once more when the run finishes, so you can
  kill the paid node right after completion instead of paying for one big end-of-run upload.

USAGE:
  python -u scripts/iter17_poc_status.py                 # latest POC log
  python -u scripts/iter17_poc_status.py --mode SANITY
  python -u scripts/iter17_poc_status.py --log logs/ngpu_POC_20260531_003848.log
  watch -n60 'python -u scripts/iter17_poc_status.py'    # live, refresh every 60s
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
from iter17_poc_ngpu import build_jobs, ARM2ENC  # noqa: E402  (canonical job DAG)

EMOJI = {"done": "✅", "running": "🔄", "pending": "⬚", "failed": "❌"}
TRAIN_ORDER = ["pretrain_encoder", "pretrain_2X_encoder", "pretrain_head",
               "surgery_3stage_DI_encoder", "surgery_noDI_encoder",
               "surgery_3stage_DI_head", "surgery_noDI_head"]
# COLD-START priors (seconds), used ONLY until a live measurement (a completion, or a running job's
# epoch/stage progress) replaces them. Empirical: from the 1× node run + iter16 (PE 3h57m, P2X ~7.5h).
PRIOR = {
    "pretrain_encoder": 4 * 3600,
    "pretrain_2X_encoder": int(7.5 * 3600),
    "pretrain_head": int(1.5 * 3600),
    "surgery_3stage_DI_encoder": int(3.8 * 3600),
    "surgery_noDI_encoder": int(3.8 * 3600),
    "surgery_3stage_DI_head": int(1.5 * 3600),
    "surgery_noDI_head": int(1.5 * 3600),
    "eval": int(2.5 * 3600),   # POC eval = multi-stage over ~10k clips (feat+probes+temporal); self-corrects on 1st completion
}

# Back up outputs/poc to HF this often (in minutes) WHILE the run is going, so the final backup at the
# end is tiny and the paid node can be killed right away. Driven by the 60s `watch` refresh + a stamp file.
UPLOAD_EVERY_MIN = 15


def _latest_log(mode):
    cands = sorted((REPO / "logs").glob(f"ngpu_{mode}_*.log"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


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
    """Total-duration estimate for a RUNNING job, read from its own log:
      · PRETRAIN — extrapolate from the whole-job STEP/Epoch tqdm (epochs ~equal → fraction meaningful).
      · EVAL — from the current stage's RECENT rate ('recent=<R>s/clip'): remaining ≈ (tot-cur)×R. This
        tracks the live bottleneck (e.g. m12e predictor-rollout ~50min). tqdm's OVERALL 'remaining' is
        cold-start-inflated (m12f reads 1h40m while recent=0.6s/clip → ~16m), so the recent rate is used
        (it matches tqdm's own 'ETA0.3h'). The prior is the mean of COMPLETED evals — but those are
        resume-SKIPPED (~70s) → it used to pin every running eval to a useless ~5m floor that never moved.
      · else (surgery) — prior, capped at prior×2.5."""
    cap = max(prior * 2.5, elapsed + 600)
    if _arm_of(jid).startswith("pretrain"):
        tmpl = jobs[jid]["log"]
        cands = sorted(REPO.glob(tmpl.format(ts="*")), key=lambda p: p.stat().st_mtime)
        if cands:
            txt = _tail(cands[-1])
            # --cache 1 resume: the job restarted at step s0, so `elapsed` only covers (cur - s0) steps.
            # total ≈ elapsed × (tot - s0)/(cur - s0) corrects for that offset — else an 82%-done resumed
            # pretrain_2X reads as the full ~7.5h prior and inflates the whole-run ETA by hours.
            rm = re.findall(r"Resumed from step (\d+)", txt)
            s0 = int(rm[-1]) if rm else 0
            ms = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)          # whole-job STEP tqdm (pretrain is step-based)
            if ms:
                cur, tot = int(ms[-1][0]), int(ms[-1][1])
                if s0 < cur <= tot:
                    return min(max(elapsed * (tot - s0) / (cur - s0), elapsed + 120), cap)
            me = re.findall(r"[Ee]poch[ :]+(\d+)\s*/\s*(\d+)", txt)
            if me:
                cur, tot = int(me[-1][0]), int(me[-1][1])
                if 0 < cur <= tot:
                    return min(max(elapsed * tot / cur, elapsed + 120), cap)
    if _arm_of(jid) == "eval":
        tmpl = jobs[jid]["log"]
        cands = sorted(REPO.glob(tmpl.format(ts="*")), key=lambda p: p.stat().st_mtime)
        if cands:
            txt = _tail(cands[-1])
            # current stage bar: '<cur>/<tot> [… recent=<R>s/clip …]'. Use the RECENT rate, NOT tqdm's
            # overall 'remaining' (cold-start-inflated: m12f reads 1h40m while recent=0.6s/clip → ~16m,
            # which matches tqdm's own 'ETA0.3h'). remaining ≈ (tot-cur) × recent_rate; capped at 3h.
            cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
            rr = re.findall(r"recent=([\d.]+)s/clip", txt)
            if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= 5:
                cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
                return min(elapsed + (tot - cur) * rate, 3 * 3600)
    return min(max(prior, elapsed + 300), cap)


def _hf_backup():
    """Upload outputs/poc to HF in reuse mode, non-interactive. Returns (returncode, log_filename).
    reuse = re-upload over what's already on HF; HF skips files that haven't changed. HF_UPLOAD_MODE
    is set so it never stops to ask delete-vs-reuse (which would hang an unattended run)."""
    ulog = REPO / "logs" / f"upload_outputs_poc_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    env = dict(os.environ, HF_UPLOAD_MODE="reuse", PYTHONPATH="src")
    with open(ulog, "wb") as fh:
        rc = subprocess.run(
            ["python", "-u", "src/utils/hf_outputs.py", "upload", "outputs/poc"],
            cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    return rc, ulog.name


def maybe_backup(fully_done):
    """Keep outputs/poc backed up on HF every UPLOAD_EVERY_MIN minutes while the run goes, plus one more
    time when it finishes — so the paid node can be killed right after completion instead of waiting on
    one big end-of-run upload. The 60s `watch` refresh drives this; a stamp file enforces the interval.
    Returns one plain-English status line to print."""
    last = REPO / "logs" / ".upload_outputs_poc.LAST"      # mtime = when the last backup started
    lock = REPO / "logs" / ".upload_outputs_poc.LOCK"      # present while a backup is running
    done_flag = REPO / "logs" / ".upload_outputs_poc.DONE"  # written after the final (post-finish) backup
    now_ts = datetime.now(timezone.utc).timestamp()
    if done_flag.exists():
        return "  ✅ run finished and fully backed up to HF — you can kill the vast.ai node any time."
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 2 * UPLOAD_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)                       # a backup was hard-killed long ago → clear it
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < UPLOAD_EVERY_MIN and not fully_done:
        return f"  ⏫ HF backup: last {int(age_min)}m ago · next in ~{max(int(UPLOAD_EVERY_MIN - age_min), 0)}m"
    try:                                                   # only one backup at a time
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  ⏫ HF backup already running — leaving it."
    try:
        last.write_text("")                                # start the clock now → next backup ~UPLOAD_EVERY_MIN later
        rc, name = _hf_backup()
        if rc != 0:
            return f"  ❌ HF backup FAILED rc={rc} (see {name}) — it will try again next backup."
        if fully_done:
            done_flag.write_text(name)
            return f"  ✅ FINAL HF backup done ({name}) — you can kill the vast.ai node now."
        return f"  ✅ HF backup done ({name}) · next in ~{UPLOAD_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


# Rebuild the §G plots from whatever evals are DONE this often (minutes) — a LIVE preview to show in a
# meeting while the run finishes. CPU-only (paired-Δ aggregation + m13). The final §3 rebuilds it complete.
PLOT_EVERY_MIN = 5


def maybe_plot(mtag, mode):
    """Every PLOT_EVERY_MIN min, rebuild the §G plots from the evals done SO FAR (CPU: paired-Δ + m13) →
    a live partial hero table you can screen-share in a meeting. SAFE: paired_delta always re-aggregates
    (no cache skip), so the scheduler's final §3 rebuilds the COMPLETE §G at the end — this preview can't
    corrupt it. The 60s `watch` + a stamp file drive the cadence; the chain uses ';' so an early metric
    with too few done evals won't block — m13 is graceful and plots whatever JSONs exist."""
    last = REPO / "logs" / ".plot_preview.LAST"
    lock = REPO / "logs" / ".plot_preview.LOCK"
    now_ts = datetime.now(timezone.utc).timestamp()
    hero = REPO / f"outputs/{mtag}/probe_plot/eval/m13_hero_table.png"
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 2 * PLOT_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < PLOT_EVERY_MIN:
        return (f"  🖼  §G preview: rebuilt {int(age_min)}m ago · next in ~{max(int(PLOT_EVERY_MIN - age_min), 0)}m"
                f" → outputs/{mtag}/probe_plot/eval/m13_hero_table.png")
    try:
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  🖼  §G preview already rebuilding — leaving it."
    try:
        last.write_text("")
        out = f"outputs/{mtag}"
        chain = " ; ".join([   # same chain the scheduler runs at §3, into the SAME probe_plot/eval/ dir
            f"python -u src/m12a_action_top1.py --{mode} --stage paired_delta --output-root {out}/probe_action --cache-policy 1 --no-wandb",
            f"python -u src/m12b_motion_cos.py  --{mode} --stage paired_delta --output-root {out}/probe_motion_cos --cache-policy 1 --no-wandb",
            f"python -u src/m12c_taxonomy_f1.py --{mode} --stage paired_delta --features-root {out}/probe_action --output-root {out}/probe_taxonomy --cache-policy 1 --no-wandb",
            f"python -u src/m12d_future_mse.py  --{mode} --stage paired_per_variant --output-root {out}/probe_future_mse --cache-policy 1 --no-wandb",
            f"python -u src/m13_eval_plot.py --{mode} --action-probe-root {out}/probe_action "
            f"--motion-cos-root {out}/probe_motion_cos --future-mse-root {out}/probe_future_mse "
            f"--taxonomy-root {out}/probe_taxonomy --predictor-temporal-root {out}/predictor_temporal "
            f"--encoder-temporal-root {out}/encoder_temporal --output-dir {out}/probe_plot --no-wandb",
        ])
        plog = REPO / "logs" / f"plot_preview_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
        with open(plog, "wb") as fh:
            rc = subprocess.run(chain, shell=True, executable="/bin/bash", cwd=str(REPO),
                                env=dict(os.environ, PYTHONPATH="src"),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        if hero.exists():
            return f"  🖼  §G preview REBUILT (rc={rc}) → outputs/{mtag}/probe_plot/eval/ · next in ~{PLOT_EVERY_MIN}m"
        return f"  🖼  §G preview rc={rc} — partial/blocked, see {plog.name} · next in ~{PLOT_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    ap.add_argument("--log", default=None, help="explicit main log (default: latest logs/ngpu_<mode>_*.log)")
    args = ap.parse_args()

    jobs, mtag = build_jobs(args.mode)
    log = Path(args.log) if args.log else _latest_log(args.mode)
    if log is None or not log.exists():
        sys.exit(f"No {args.mode} log found (logs/ngpu_{args.mode}_*.log). Has the scheduler started?")

    text = log.read_text(errors="replace")
    launched = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ◀ (\S+)", text)}
    done = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✓ (\S+)", text)}
    failed = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✗ (\S+)", text)}
    rm = re.search(r"\[resume --cache 1\] skipping \d+ already-trained arms: \[(.*?)\]", text)
    if rm:
        for tok in re.findall(r"'([^']+)'", rm.group(1)):
            done.setdefault(f"T:{tok}", "resume")
    gpus = int((re.search(r"gpus=(\d+)", text) or [0, "8"])[1]) if re.search(r"gpus=(\d+)", text) else 8
    all_passed = "all 30 jobs PASSED" in text or "all jobs settled" in text
    s3 = re.search(r"§3 rc=(-?\d+)", text)
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

    # ── live-calibrated per-arm duration estimate ──
    est = dict(PRIOR)
    measured = {}
    for jid, t in done.items():
        if t != "resume" and jid in launched:
            measured.setdefault(_arm_of(jid), []).append((_sod(t) - _sod(launched[jid])) % 86400)
    for arm, vals in measured.items():
        est[arm] = sum(vals) / len(vals)          # mean of completed peers (observed)

    # ── per-job remaining time ──
    remaining = {}
    for jid in jobs:
        st = classify(jid)
        prior = est.get(_arm_of(jid), PRIOR["eval"])
        if st in ("done", "failed"):
            remaining[jid] = 0.0
        elif st == "running":
            remaining[jid] = max(_running_total(jobs, jid, elapsed(jid), prior) - elapsed(jid), 60)
        else:
            remaining[jid] = prior

    # A FRESH eval's m12e (predictor rollout) is ~50min — far above the resume-contaminated eval prior
    # (~70s, mean of resume-skipped evals). Lift each PENDING eval to the slowest RUNNING eval's remaining:
    # the pending surgical-head evals hit the SAME m12e wall, so without this the run ETA reads a fixed
    # ~21m that never moves while the real work grinds. (No-op once a fresh eval completes and est rises.)
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
    while (running or pending) and guard < 10000:
        guard += 1
        for j in [x for x in jobs if x in pending and jobs[x]["deps"] <= done_set]:
            if free <= 0:
                break
            pending.discard(j); running.add(j); run_end[j] = t + remaining[j]; free -= 1
        if not running:
            break
        nxt = min(running, key=lambda j: run_end[j])
        t = run_end[nxt]
        running.discard(nxt); done_set.add(nxt); finish[nxt] = t; free += 1
    eta_secs = (max(finish.values()) if finish else 0) + 15 * 60      # + §3 (CPU ~15 min)

    # ── render: emoji table (rows = step, cols = backbone) ──
    bbs = sorted({jid.split(":")[1] for jid in jobs if jid.startswith("T:")})
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for jid in jobs:
        counts[classify(jid)] += 1

    def cell(jid):
        st = classify(jid)
        if st == "done":
            d = "(cached)" if done.get(jid) == "resume" else _dur((_sod(done[jid]) - _sod(launched[jid])) % 86400)
            return f"✅ {d}"
        if st == "failed":
            return "❌ FAILED"
        if st == "running":
            return f"🔄 {_dur(elapsed(jid))}·~{_dur(finish[jid])}"
        return f"⬚ ~{_dur(finish[jid])}"

    def kemoji(arm):
        return "🚂" if arm.startswith("pretrain") else "🔧" if arm.startswith(("surger", "surgical")) else "📊"

    rows = [(f"{kemoji(a)} {a}", lambda bb, a=a: f"T:{bb}:{a}") for a in TRAIN_ORDER]
    rows += [(f"📊 eval:{e}", lambda bb, e=e: f"E:{bb}_{e}")
             for e in ["frozen"] + [ARM2ENC[a] for a in TRAIN_ORDER]]

    SW, CW = 34, 22
    bar = "─"
    print(f"═══ iter17 {args.mode} · {log.name} · now {now_hms} UTC · {gpus} GPU ═══")
    print("┌" + bar * SW + "┬" + (bar * CW + "┬") * (len(bbs) - 1) + bar * CW + "┐")
    print("│" + " step".ljust(SW) + "│" + "│".join(" " + b.replace("vjepa_2_", "2.").ljust(CW - 1) for b in bbs) + "│")
    print("├" + bar * SW + "┼" + (bar * CW + "┼") * (len(bbs) - 1) + bar * CW + "┤")
    for label, jidfn in rows:
        line = "│ " + label.ljust(SW - 1) + "│"
        for bb in bbs:
            line += " " + cell(jidfn(bb)).ljust(CW - 1) + "│"
        print(line)
    print("└" + bar * SW + "┴" + (bar * CW + "┴") * (len(bbs) - 1) + bar * CW + "┘")

    # ── summary + run ETA ──
    end_utc = datetime.now(timezone.utc) + timedelta(seconds=eta_secs)
    end_pdt = end_utc - timedelta(hours=7)
    print(f"\n  {counts['done']}✅  {counts['running']}🔄  {counts['pending']}⬚  "
          f"{counts['failed']}❌  / {len(jobs)} jobs")
    # ── HF BACKUP: keep outputs/poc on HF every UPLOAD_EVERY_MIN min + one final backup at the end, so
    #    the paid node can be killed right after completion (not after one big end-of-run upload). ──
    fully_done = bool(s3 and s3.group(1) == "0")
    settled = counts["done"] + counts["failed"] == len(jobs)
    # §G live preview: rebuild from done-so-far evals WHILE the run is going (the meeting case), and in the
    # settled-with-failures case where the scheduler SKIPS §3. Skip it once §3 is the scheduler's job
    # (settled-clean / fully_done) so this preview never races the authoritative final build.
    plot_msg = maybe_plot(mtag, args.mode) if (not settled or counts["failed"]) and not fully_done else None
    backup_msg = maybe_backup(fully_done)
    if fully_done:
        print("\a\n" + "█" * 78)
        print(f"  🏁 RUN COMPLETE — §G plots in outputs/{mtag}/probe_plot/eval/")
        print(backup_msg)
        print("█" * 78)
    elif settled and counts["failed"]:
        print("\a\n" + "█" * 78)
        print(f"  ⚠️  ALL {len(jobs)} JOBS SETTLED · {counts['failed']}❌ FAILED — DO NOT KILL YET.")
        print("  The scheduler skips §3 on any failure. Re-run to finish the failed jobs (now fixed)")
        print("  + build §G, THEN it does a final backup + you kill:")
        print(f"       python -u scripts/iter17_poc_ngpu.py --mode {args.mode} --gpus {gpus} --cache 1")
        print(plot_msg)
        print(backup_msg)
        print("█" * 78)
    elif settled:
        print(f"  🏁 all {len(jobs)} jobs settled · §3 aggregate {'🔄 running' if all_passed else '⬚ pending'}")
        print(backup_msg)
    else:
        print(f"  🏁 run ETA  ~{_dur(eta_secs)} from now  →  {end_utc:%H:%M} UTC · "
              f"{end_pdt:%H:%M} PDT  (incl. §3 ~15m)")
        print(plot_msg)
        print(backup_msg)
    if counts["failed"]:
        print("  ❌ failed (recovered on the final --cache 1): "
              + ", ".join(j for j in jobs if classify(j) == "failed"))
    print("\n  legend: 🚂 pretrain · 🔧 surgery · 📊 eval │ ✅ done · 🔄 running · ⬚ pending · ❌ failed")


if __name__ == "__main__":
    main()
