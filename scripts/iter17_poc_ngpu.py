#!/usr/bin/env python3
"""iter17 POC — N-GPU ARM-LEVEL scheduler. Both backbones, full pipeline, max parallelism.

WHY: the backbone-level orchestrator (iter17_poc_2gpu.sh) pins ONE backbone per GPU → wall =
per-backbone time (~30h). For an 8x node we want ARM-level fan-out: every train-arm and every
eval-encoder is a node in a DAG, scheduled onto the next free GPU as soon as its deps are met.
On 8 GPUs the wall collapses to ~max(pretrain_2X 7.5h, pretrain_encoder→surgery 7.8h) + eval
(pipelined ~2-3h) + §3 ≈ ~10-11h.

DAG (30 jobs for 2 backbones):
  TRAIN (14):
    pretrain_encoder      [SEED for vitg: also writes shared labels/splits; others gate on them]
    pretrain_2X_encoder   [root]   pretrain_head [root]
    surgery_{3stage_DI,noDI}_{encoder,head}  [dep: same-bb pretrain_encoder — surgery_init]
  EVAL (16, per encoder, per-encoder stages only — aggregate deferred to §3):
    <bb>_frozen                         [dep: labels only]
    <bb>_<arm>                          [dep: that arm's TRAIN job]

SAFETY (shared disk): train splits/pool are ATOMIC-written (clip_splits/probe_train_subset);
shared LABELS are bootstrap-if-missing (NOT atomic) → only the SEED creates them, everything else
gates on their existence. Eval writes per-encoder dirs (distinct) + SKIPS the shared paired-Δ /
labels / plots; the combined paired-Δ + m13 run ONCE at the end. m12c deletes probe heads after
metrics (bounds disk). ENCODER_CKPT is overridden to a present ckpt (vitG default may be deleted).

Usage:
  python -u scripts/iter17_poc_ngpu.py [--mode POC|SANITY] [--gpus 8] [--cache 2|1] [--dry-run]
  (SANITY = tiny data smoke; POC = real 10k numbers. --dry-run prints the schedule, launches nothing.)

RESUME after a failure: re-launch with --cache 1. The scheduler skips TRAIN arms that already
exported student_encoder.pt and unblocks their evals immediately; eval jobs fast-resume internally
(m12a/c skip stages whose test_metrics.json exists). Only the failed job(s) + their dependents re-run.
PREFLIGHT: aborts on low disk; WARNS if cpu-cores < gpus×6 (TAR-decode contention slows the ETA).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKBONES = ["vjepa_2_1_vitg", "vjepa_2_0_vitg"]
# (run_train arm  →  eval-encoder suffix)  — note surgery→surgical rename in the registry.
ARM2ENC = {
    "pretrain_encoder":          "pretrain_encoder",
    "pretrain_2X_encoder":       "pretrain_2X_encoder",
    "pretrain_head":             "pretrain_head",
    "surgery_3stage_DI_encoder": "surgical_3stage_DI_encoder",
    "surgery_noDI_encoder":      "surgical_noDI_encoder",
    "surgery_3stage_DI_head":    "surgical_3stage_DI_head",
    "surgery_noDI_head":         "surgical_noDI_head",
}
SURGERY_ARMS = {a for a in ARM2ENC if a.startswith("surgery_")}
# run_train arm → on-disk m09 output dir (for --resume done-marker: student_encoder.pt).
ARM2DIR = {
    "pretrain_encoder":          "m09a_pretrain_encoder",
    "pretrain_2X_encoder":       "m09a_pretrain_2X_encoder",
    "pretrain_head":             "m09a_pretrain_head",
    "surgery_3stage_DI_encoder": "m09c_surgery_3stage_DI_encoder",
    "surgery_noDI_encoder":      "m09c_surgery_noDI_encoder",
    "surgery_3stage_DI_head":    "m09c_surgery_3stage_DI_head",
    "surgery_noDI_head":         "m09c_surgery_noDI_head",
}
EVAL_SKIP = "1,4,7,9,9b,9c,12,13,10"   # per-encoder stages only; aggregate+labels+plots → §3
ENC_CKPT_OVERRIDE = "checkpoints/vjepa2_1_vitg_384.pt"  # present ckpt to satisfy run_eval preflight


def ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_jobs(mode):
    """Return dict id→job. job = {id,kind,cmd,deps:set,needs_labels:bool,log}."""
    mflag, mtag = f"--{mode}", mode.lower()
    jobs = {}
    seed_id = f"T:{BACKBONES[0]}:pretrain_encoder"   # vitg pretrain_encoder seeds the shared labels
    # INTERLEAVE backbones (arm-major, bb-inner) so the dict-order = launch-order alternates vitg/2.0.
    # On a --cache 1 resume surgery is already unblocked → ~10 arms are ready at once; bb-major order
    # filled all 8 GPUs with BACKBONES[0] first and SERIALIZED the 2nd backbone behind it (iter17 3rd
    # run). Arm-major balances the pool across both backbones. The 1st cache-2 run is unaffected (its
    # surgery still gates on pretrain, so only the roots are ready at t=0). Deps/labels/seed unchanged.
    for arm in ARM2ENC:
        for bb in BACKBONES:
            jid = f"T:{bb}:{arm}"
            deps = {f"T:{bb}:pretrain_encoder"} if arm in SURGERY_ARMS else set()
            jobs[jid] = dict(
                id=jid, kind="train", deps=deps, needs_labels=(jid != seed_id),
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} BACKBONE={bb} CACHE_POLICY_ALL={{cache}} "
                     f"./scripts/run_train.sh {arm} {mflag}"),
                log=f"logs/iter17_ngpu_{mtag}_train_{bb}_{arm}_{{ts}}.log")
    for arm, enc in [("frozen", "frozen")] + [(a, e) for a, e in ARM2ENC.items()]:
        for bb in BACKBONES:
            enc_name = f"{bb}_{enc}"
            jid = f"E:{enc_name}"
            deps = set() if arm == "frozen" else {f"T:{bb}:{arm}"}
            jobs[jid] = dict(
                id=jid, kind="eval", deps=deps, needs_labels=True,
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} ENCODER_CKPT={ENC_CKPT_OVERRIDE} "
                     f"ENCODERS={enc_name} SKIP_STAGES={EVAL_SKIP} CACHE_POLICY_ALL={{cache}} "
                     f"./scripts/run_eval.sh {mflag}"),
                log=f"logs/iter17_ngpu_{mtag}_eval_{enc_name}_{{ts}}.log")
    return jobs, mtag


def labels_ready(mtag):
    return ((REPO / f"outputs/{mtag}/probe_action/action_labels.json").exists()
            and (REPO / f"outputs/{mtag}/probe_taxonomy/taxonomy_labels.json").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--cache", choices=["1", "2"], default="2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs, mtag = build_jobs(args.mode)

    # disk-preflight: both backbones' POC ≈ 2×(71G ckpts + ~6G taxonomy w/ head-cleanup) + ~79G
    # baseline ≈ ~230G; require a margin. Abort BEFORE any GPU work on a too-small node.
    import shutil
    free_gb = shutil.disk_usage(str(REPO)).free / 1e9
    # --cache 1 (resume) re-writes only the REMAINING eval outputs (~20-30G); the 142G of ckpts +
    # the done evals are already on disk, so it fits beside a 141G eval frame cache. Fresh runs need 250G.
    req_gb = (60 if args.cache == "1" else 250) if args.mode == "POC" else 30
    print(f"  [disk-preflight] free={free_gb:.0f}G · required≈{req_gb}G", flush=True)
    if free_gb < req_gb and not args.dry_run:
        sys.exit(f"FATAL: insufficient disk — {free_gb:.0f}G free < {req_gb}G for {args.mode}. "
                 f"Use a bigger node or free space.")

    # cpu-core check: each concurrent job spawns ~6-8 CPU threads (TAR decode + producer). At full
    # fan-out that's gpus×~6 cores; below that, decode I/O contends and the ~10-11h ETA slips. WARN only.
    import os
    cores = os.cpu_count() or 0
    need_cores = args.gpus * 6
    if cores < need_cores:
        print(f"  ⚠️  [cpu-preflight] {cores} cores < ~{need_cores} (gpus×6) — TAR-decode will contend at "
              f"full fan-out; wall-time may exceed the ~10-11h estimate. Use a higher-core node.",
              flush=True)
    else:
        print(f"  [cpu-preflight] {cores} cores ≥ ~{need_cores} (gpus×6) — OK", flush=True)

    done, failed, running = set(), {}, {}   # running: gpu→(jid, Popen, log)

    # --resume: when --cache 1, skip TRAIN arms that already exported student_encoder.pt (their eval
    # dependents then unblock immediately; eval jobs themselves fast-resume internally under cache=1
    # — m12a/c skip stages whose test_metrics.json already exists). --cache 2 (default) re-runs all.
    if args.cache == "1":
        for jid, j in jobs.items():
            if j["kind"] != "train":
                continue
            _, bb, arm = jid.split(":")
            if (REPO / f"outputs/{mtag}/{bb}/{ARM2DIR[arm]}/student_encoder.pt").exists():
                done.add(jid)
        if done:
            print(f"  [resume --cache 1] skipping {len(done)} already-trained arms: "
                  f"{sorted(d.split(':', 1)[1] for d in done)}", flush=True)

    free = list(range(args.gpus))

    print(f"═══ iter17 N-GPU scheduler · mode={args.mode} · gpus={args.gpus} · cache={args.cache} · "
          f"{len(jobs)} jobs ({sum(j['kind']=='train' for j in jobs.values())} train + "
          f"{sum(j['kind']=='eval' for j in jobs.values())} eval) ═══", flush=True)

    def ready(jid):
        j = jobs[jid]
        if jid in done or jid in failed or any(jid == r[0] for r in running.values()):
            return False
        if not j["deps"] <= done:
            return False
        if j["needs_labels"] and not labels_ready(mtag):
            return False
        return True

    if args.dry_run:
        print("[dry-run] dependency-ordered launch plan (no GPU work):")
        sim_done = set()
        wave = 0
        while len(sim_done) < len(jobs):
            launchable = [jid for jid in jobs if jid not in sim_done and jobs[jid]["deps"] <= sim_done]
            if not launchable:
                print(f"  !! stuck — unmet deps: {[j for j in jobs if j not in sim_done]}"); sys.exit(1)
            print(f"  wave {wave} ({len(launchable)} ready): {sorted(launchable)}")
            sim_done |= set(launchable); wave += 1
        print(f"[dry-run] {len(jobs)} jobs across {wave} dependency waves. OK.")
        return

    t0 = time.time()
    while len(done) + len(failed) < len(jobs):
        # launch ready jobs onto free GPUs
        for jid in list(jobs):
            if not free or not ready(jid):
                continue
            g = free.pop(0)
            log = jobs[jid]["log"].format(ts=ts())
            cmd = jobs[jid]["cmd"].format(gpu=g, cache=args.cache)
            print(f"[{now()}] GPU{g} ◀ {jid}  → {log}", flush=True)
            lf = open(REPO / log, "w")
            p = subprocess.Popen(cmd, shell=True, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)
            running[g] = (jid, p, lf)
        # reap finished
        for g in list(running):
            jid, p, lf = running[g]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[g]; free.append(g)
            if rc == 0:
                done.add(jid); print(f"[{now()}] GPU{g} ✓ {jid}  ({len(done)}/{len(jobs)} done)", flush=True)
            else:
                failed[jid] = rc; print(f"[{now()}] GPU{g} ✗ {jid} rc={rc} — see its log", flush=True)
        if running or any(ready(j) for j in jobs):
            time.sleep(10)
        elif free and not running:   # nothing ready, nothing running → deadlock (failed dep blocks rest)
            break

    el = (time.time() - t0) / 3600
    print(f"═══ all jobs settled in {el:.1f}h · done={len(done)} failed={len(failed)} ═══", flush=True)
    if failed:
        print("FAILED jobs (their dependents were skipped):")
        for jid, rc in failed.items():
            print(f"  ✗ {jid} (rc={rc})")
        print("Fix + re-run with --cache 1 to resume the survivors; NOT running §3.")
        sys.exit(1)

    # ── §3 combined aggregate (CPU, once over ALL encoders) ──
    print(f"═══ all 30 jobs PASSED — running §3 combined paired-Δ + m13 plots ({now()}) ═══", flush=True)
    out = f"outputs/{mtag}"
    # m12a/b/c/d build the paired-Δ JSONs m13 reads; m13 sys.exit()s if ANY is absent, so the chain
    # MUST be `&&` (not `;`) AND run under `pipefail` so a §3 failure is not MASKED by the trailing
    # `| tee`. The prior `;` + tee-masking let an aborted m13 report "§3 rc=0" with ZERO plots written
    # because m12d_future_mse (--stage paired_per_variant → probe_future_mse_per_variant.json) was
    # never run. predictor/encoder temporal need NO paired step (m13 reads their per-encoder aggregates
    # directly). Subshell `( … )` so the WHOLE chain's stdout is tee'd, not just m13's.
    s3 = " && ".join([
        f"python -u src/m12a_action_top1.py --{args.mode} --stage paired_delta --output-root {out}/probe_action --cache-policy 1 --no-wandb",
        f"python -u src/m12b_motion_cos.py  --{args.mode} --stage paired_delta --output-root {out}/probe_motion_cos --cache-policy 1 --no-wandb",
        f"python -u src/m12c_taxonomy_f1.py --{args.mode} --stage paired_delta --features-root {out}/probe_action --output-root {out}/probe_taxonomy --cache-policy 1 --no-wandb",
        f"python -u src/m12d_future_mse.py  --{args.mode} --stage paired_per_variant --output-root {out}/probe_future_mse --cache-policy 1 --no-wandb",
        f"python -u src/m13_eval_plot.py --{args.mode} --action-probe-root {out}/probe_action "
        f"--motion-cos-root {out}/probe_motion_cos --future-mse-root {out}/probe_future_mse "
        f"--taxonomy-root {out}/probe_taxonomy --predictor-temporal-root {out}/predictor_temporal "
        f"--encoder-temporal-root {out}/encoder_temporal --output-dir {out}/probe_plot --no-wandb",
    ])
    rc = subprocess.call(
        f"export PYTHONPATH=src ; set -o pipefail ; ( {s3} ) 2>&1 | tee logs/iter17_ngpu_{mtag}_s3_{ts()}.log",
        shell=True, executable="/bin/bash", cwd=str(REPO))
    print(f"═══ §3 rc={rc} · total wall {el:.1f}h + §3 · §G plots → {out}/probe_plot/eval/ ═══", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
