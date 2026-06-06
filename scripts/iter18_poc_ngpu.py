#!/usr/bin/env python3
"""iter18 POC — N-GPU ARM-LEVEL scheduler (2×/4×/8× single-node). One backbone (vitG 2B), 13 train
arms + 14 eval jobs as a DAG, each job pinned to the next free GPU via CUDA_VISIBLE_DEVICES.

Adapted from scripts/legacy/iter17_poc_ngpu.py with the iter18 DAG:
  - ONE backbone: vjepa_2_1_vitG (BACKBONES axis dropped).
  - pretrain_encoder is the ROOT for *ALL* 12 other train arms (its m09a_ckpt_best.pt is every
    arm's --init-from-ckpt via run_train.sh SURGERY_INIT) — iter17 only gated surgery on it.
  - 13 train arms = §2 of iter/iter18_ablations_FTtechniues/runbook.md (novelty ×4 + control +
    8 FT-technique baselines + pretrain).
  - per-encoder eval jobs run ONLY stages 2,3,11,5,6,8,8b (SKIP_STAGES drops the shared stages);
    the §3 finale is ONE run_eval over ALL encoders with the per-encoder stages skipped → it runs
    the shared paired-Δ/plot stages (1,4,12,13,7,9,9b,10) off the per-encoder caches.

WALL-TIME at POC (7,724 clips, ~26.6 s/step ≈ 3.5-4.5 h per encoder arm, ~2-3 h per head arm):
  1 GPU ≈ 50 h  ·  2 GPU ≈ ~28 h  ·  4 GPU ≈ ~17 h  (pretrain is the serial 3.6 h prefix)

SAFETY (same as iter17): split/pool JSONs are atomic-written + deterministic (concurrent
run_train.sh regenerations produce identical bytes); shared labels are created by the SEED
(pretrain_encoder) only — every other job gates on the label files existing. Eval jobs write
per-encoder dirs (disjoint). RESUME: --cache 1 skips train arms whose student_encoder.pt exists.

USAGE (run inside tmux on the multi-GPU box; data + env must already be provisioned):
  python -u scripts/iter18_poc_ngpu.py --mode POC    --gpus 4 --cache 2              # fresh
  python -u scripts/iter18_poc_ngpu.py --mode POC    --gpus 4 --cache 1              # resume
  python -u scripts/iter18_poc_ngpu.py --mode SANITY --gpus 2 --cache 2 --dry-run    # plan only
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKBONE = "vjepa_2_1_vitG"

# (run_train arm → eval-encoder suffix). NOTE the surgery_→surgical_ rename for the factor arms +
# autorgn; the iter18 baselines keep their arm name verbatim (registry rows added 2026-06-04).
ARM2ENC = {
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
# run_train arm → on-disk m09 output dir (resume done-marker: student_encoder.pt). Verified against
# the 2026-06-04 SANITY outputs (outputs/sanity/vjepa_2_1_vitG/<dir>/).
ARM2DIR = {
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
# Stage split (verified against scripts/run_eval.sh should_skip gates, 2026-06-04):
#   per-encoder stages: 2 features · 3 probe · 11 taxonomy-train · 5/6 motion_cos · 8 future_mse
#                       · 8b predictor_temporal
#   shared stages:      1 labels · 4 action-paired · 12 taxonomy-paired · 13 taxonomy-plot
#                       · 7 motion-paired · 9 future-paired · 9b predictor-paired · 10 m13 plots
EVAL_SKIP_SHARED = "1,4,12,13,7,9,9b,10"   # per-encoder jobs skip the shared stages
S3_SKIP_PERENC = "2,3,11,5,6,8,8b"         # the §3 finale skips the per-encoder stages


def ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_jobs(mode):
    """dict id→job. job = {id,kind,cmd,deps:set,needs_labels:bool,log}."""
    mflag, mtag = f"--{mode}", mode.lower()
    jobs = {}
    seed_id = f"T:{BACKBONE}:pretrain_encoder"   # seeds shared labels + everyone's init ckpt
    for arm in ARM2ENC:
        jid = f"T:{BACKBONE}:{arm}"
        # iter18: EVERY non-pretrain arm inits from pretrain's m09a_ckpt_best.pt → all dep on it.
        deps = set() if arm == "pretrain_encoder" else {seed_id}
        jobs[jid] = dict(
            id=jid, kind="train", deps=deps, needs_labels=(jid != seed_id),
            cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} BACKBONE={BACKBONE} CACHE_POLICY_ALL={{cache}} "
                 f"./scripts/run_train.sh {arm} {mflag}"),
            log=f"logs/iter18_ngpu_{mtag}_train_{arm}_{{ts}}.log")
    # eval jobs: frozen + pretrain references + the 12 contender arms (= m13 needs all three families)
    for arm, enc in [("frozen", "frozen")] + list(ARM2ENC.items()):
        enc_name = f"vjepa_2_1_{enc}"
        jid = f"E:{enc_name}"
        deps = set() if arm == "frozen" else {f"T:{BACKBONE}:{arm}"}
        jobs[jid] = dict(
            id=jid, kind="eval", deps=deps, needs_labels=True,
            cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_SHARED} "
                 f"CACHE_POLICY_ALL={{cache}} ./scripts/run_eval.sh {mflag} --encoders {enc_name}"),
            log=f"logs/iter18_ngpu_{mtag}_eval_{enc_name}_{{ts}}.log")
    return jobs, mtag


def labels_ready(mtag):
    return ((REPO / f"outputs/{mtag}/probe_action/action_labels.json").exists()
            and (REPO / f"outputs/{mtag}/probe_taxonomy/taxonomy_labels.json").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY"], default="POC")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--cache", choices=["1", "2"], default="2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, metavar="ARM",
                    help="run ONLY these train arms (no eval jobs, no §3 finale). Deps outside the "
                         "subset are trusted to the operator — run_train FATALs on a missing init "
                         "ckpt anyway. Used by runbook §0.A5 (1× box trains just pretrain_encoder) "
                         "and for single-arm crash re-runs.")
    args = ap.parse_args()

    jobs, mtag = build_jobs(args.mode)

    if args.only:
        bad = [a for a in args.only if a not in ARM2ENC]
        if bad:
            sys.exit(f"FATAL: unknown arm(s) {bad} — choices: {sorted(ARM2ENC)}")
        keep = {f"T:{BACKBONE}:{a}" for a in args.only}
        jobs = {jid: j for jid, j in jobs.items() if jid in keep}
        for j in jobs.values():
            j["deps"] &= keep   # outside-subset deps: operator's responsibility (FAIL LOUD in run_train)
        print(f"  [--only] restricted to {sorted(args.only)} — eval jobs + §3 finale SKIPPED", flush=True)

    # disk-preflight: 13 arms × ~14G ckpts ≈ 185G + eval artifacts ≈ ~210G; require margin.
    import shutil
    free_gb = shutil.disk_usage(str(REPO)).free / 1e9
    req_gb = (80 if args.cache == "1" else 250) if args.mode == "POC" else 30
    print(f"  [disk-preflight] free={free_gb:.0f}G · required≈{req_gb}G", flush=True)
    if free_gb < req_gb and not args.dry_run:
        sys.exit(f"FATAL: insufficient disk — {free_gb:.0f}G free < {req_gb}G for {args.mode}.")

    # cpu-preflight: each job spawns ~6-8 CPU threads (TAR decode + factor-streaming workers).
    import os
    cores = os.cpu_count() or 0
    need_cores = args.gpus * 6
    if cores < need_cores:
        print(f"  ⚠️  [cpu-preflight] {cores} cores < ~{need_cores} (gpus×6) — TAR-decode will "
              f"contend at full fan-out; wall-time will slip.", flush=True)
    else:
        print(f"  [cpu-preflight] {cores} cores ≥ ~{need_cores} (gpus×6) — OK", flush=True)

    done, failed, running = set(), {}, {}   # running: gpu→(jid, Popen, logfile)

    if args.cache == "1":
        for jid, j in jobs.items():
            if j["kind"] != "train":
                continue
            _, bb, arm = jid.split(":")
            if (REPO / f"outputs/{mtag}/{bb}/{ARM2DIR[arm]}/student_encoder.pt").exists():
                done.add(jid)
        if done:
            print(f"  [resume --cache 1] skipping {len(done)} already-trained arms: "
                  f"{sorted(d.split(':')[2] for d in done)}", flush=True)

    free = list(range(args.gpus))
    print(f"═══ iter18 N-GPU scheduler · mode={args.mode} · gpus={args.gpus} · cache={args.cache} · "
          f"{len(jobs)} jobs ({sum(j['kind'] == 'train' for j in jobs.values())} train + "
          f"{sum(j['kind'] == 'eval' for j in jobs.values())} eval) ═══", flush=True)

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
        sim_done, wave = set(), 0
        while len(sim_done) < len(jobs):
            launchable = [j for j in jobs if j not in sim_done and jobs[j]["deps"] <= sim_done]
            if not launchable:
                print(f"  !! stuck — unmet deps: {[j for j in jobs if j not in sim_done]}")
                sys.exit(1)
            print(f"  wave {wave} ({len(launchable)} ready): {sorted(launchable)}")
            sim_done |= set(launchable)
            wave += 1
        print(f"[dry-run] {len(jobs)} jobs across {wave} dependency waves. OK.")
        return

    t0 = time.time()
    while len(done) + len(failed) < len(jobs):
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
        for g in list(running):
            jid, p, lf = running[g]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[g]
            free.append(g)
            if rc == 0:
                done.add(jid)
                print(f"[{now()}] GPU{g} ✓ {jid}  ({len(done)}/{len(jobs)} done)", flush=True)
            else:
                failed[jid] = rc
                print(f"[{now()}] GPU{g} ✗ {jid} rc={rc} — see its log", flush=True)
        if running or any(ready(j) for j in jobs):
            time.sleep(10)
        elif free and not running:   # nothing ready, nothing running → a failed dep blocks the rest
            break

    el = (time.time() - t0) / 3600
    print(f"═══ jobs settled in {el:.1f}h · done={len(done)} failed={len(failed)} ═══", flush=True)
    if failed:
        print("FAILED jobs (their dependents were skipped):")
        for jid, rc in failed.items():
            print(f"  ✗ {jid} (rc={rc})")
        print("Fix + re-run with --cache 1 to resume the survivors; NOT running the §3 finale.")
        sys.exit(1)

    if args.only:
        print(f"═══ --only run complete ({sorted(args.only)}) — §3 finale skipped by design ═══",
              flush=True)
        return

    # ── §3 finale: ONE run_eval over ALL encoders, per-encoder stages skipped (cached) →
    #    runs the shared paired-Δ + m13 stages with every encoder present. cache=1 keeps caches.
    all_encs = " ".join(["vjepa_2_1_frozen"] + [f"vjepa_2_1_{e}" for e in ARM2ENC.values()])
    s3_log = f"logs/iter18_ngpu_{mtag}_s3_{ts()}.log"
    s3 = (f"SKIP_STAGES={S3_SKIP_PERENC} CACHE_POLICY_ALL=1 "
          f"./scripts/run_eval.sh --{args.mode} --encoders \"{all_encs}\"")
    print(f"═══ all jobs PASSED — §3 finale (shared paired-Δ + m13) → {s3_log} ═══", flush=True)
    rc = subprocess.call(f"set -o pipefail ; ( {s3} ) 2>&1 | tee {s3_log}",
                         shell=True, executable="/bin/bash", cwd=str(REPO))
    print(f"═══ §3 rc={rc} · total wall {el:.1f}h + §3 · plots → outputs/{mtag}/probe_plot/eval/ ═══",
          flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
