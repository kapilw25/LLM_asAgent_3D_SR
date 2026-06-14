#!/usr/bin/env python3
"""iter18 POC — N-GPU ARM-LEVEL scheduler (2×/4×/8× single-node). One backbone (vitG 2B), 13 train
arms + 14 eval jobs as a DAG, each job pinned to the next free GPU via CUDA_VISIBLE_DEVICES.

Adapted from scripts/legacy/iter17_poc_ngpu.py with the iter18 DAG:
  - ONE backbone: vjepa_2_1_vitG (BACKBONES axis dropped).
  - pretrain_encoder is the ROOT for *ALL* 12 other train arms (its m09a_ckpt_best.pt is every
    arm's --init-from-ckpt via run_train.sh SURGERY_INIT) — iter17 only gated surgery on it.
  - 13 train arms = §2 of iter/iter18_ablations_FTtechniues/runbook.md (novelty ×4 + control +
    8 FT-technique baselines + pretrain).
  - per-encoder eval jobs run ONLY stages 2,3,11,5,6,8,8b,8c (SKIP_STAGES drops the shared stages);
    the §3 finale is ONE run_eval over ALL encoders with the per-encoder stages skipped → it runs
    the shared paired-Δ/plot stages (1,4,12,13,7,9,9b,9c,10) off the per-encoder caches.

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
import contextlib
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))   # iter18 (2026-06-14): arm roster single-source (configs/arm_registry.yaml)
from utils.arm_registry import arm2enc as _arm2enc, arm2dir as _arm2dir  # noqa: E402
# iter18 2026-06-08: BACKBONE is the run's encoder family, switchable via env so the SAME scheduler
# runs the 2B champion (vjepa_2_1_vitG, default) or the 1B scale-axis backbone (vjepa_2_1_vitg). The
# status tool reads it back from the run banner (or this same env) so its job-ids match. Set it once
# in BOTH the launch and the watch panes (or rely on the banner parse in the status tool).
BACKBONE = os.environ.get("ITER18_BACKBONE", "vjepa_2_1_vitG")


def enc_prefix():
    """iter17 multi-backbone naming: the CHAMPION 2B vitG drops its size tag → eval encoders are the
    bare 'vjepa_2_1_<arm>'; every OTHER backbone keeps its full name ('vjepa_2_1_vitg_<arm>',
    'vjepa_2_0_vitg_<arm>') so per-backbone eval dirs + the m13 stacked hero never collide/overwrite."""
    return "vjepa_2_1" if BACKBONE == "vjepa_2_1_vitG" else BACKBONE


def enc_name(arm_enc):
    """Full eval-encoder name for the current BACKBONE, e.g. 'vjepa_2_1_frozen' (2B) /
    'vjepa_2_1_vitg_frozen' (1B). Single source for both this scheduler and the status tool."""
    return f"{enc_prefix()}_{arm_enc}"

# (run_train arm → eval-encoder suffix). NOTE the surgery_→surgical_ rename for the factor arms +
# autorgn; the iter18 baselines keep their arm name verbatim (registry rows added 2026-06-04).
# iter18 (2026-06-14): the arm roster + arm→encoder→dir mapping moved to configs/arm_registry.yaml
# (SINGLE SOURCE — was re-typed here AND in run_train.sh/run_eval.sh/the plot scripts; that scatter
# caused 6 whack-a-mole FATALs in one session). arm2enc()/arm2dir() return the scheduler==true arms,
# byte-equal to the former literals (parity-tested). Add a new arm = ONE entry in the yaml.
ARM2ENC = _arm2enc()   # run_train arm → eval encoder-token  (surgery_* → surgical_* swap lives in the yaml)
ARM2DIR = _arm2dir()   # run_train arm → on-disk m09 output dir (resume done-marker: student_encoder.pt)
# Stage split (verified against scripts/run_eval.sh should_skip gates, 2026-06-04; 8c/9c iter18):
#   per-encoder stages: 2 features · 3 probe · 11 taxonomy-train · 5/6 motion_cos · 8 future_mse
#                       · 8b predictor_temporal · 8c encoder_temporal (m12f aot/tov/pace/tcc)
#   shared stages:      1 labels · 4 action-paired · 12 taxonomy-paired · 13 taxonomy-plot
#                       · 7 motion-paired · 9 future-paired · 9b predictor-paired
#                       · 9c encoder-temporal-paired · 10 m13 plots
EVAL_SKIP_SHARED = "1,4,12,13,7,9,9b,9c,10"  # per-encoder jobs skip the shared stages (9c = m12f paired)
S3_SKIP_PERENC = "2,3,11,5,6,8,8b,8c"        # the §3 finale skips the per-encoder stages
# iter18 2026-06-07 (metric-parallel Stage 8b): Stage 8b (m12e predictor_temporal, ~2.2 h, the eval
# bottleneck) is fanned into 6 independent single-metric jobs across the GPU pool → wall = slowest
# metric (~teacher_free), not the sum of six. The per-encoder E: job then runs stages 2-8 only.
# iter18 m12f revival: Stage 8c (encoder_temporal, 4 metrics) fans the same way into F: jobs.
EVAL_SKIP_PERENC_NO8B = EVAL_SKIP_SHARED + ",8b,8c"          # E: job — per-encoder stages EXCEPT the fans
EVAL_SKIP_ONLY_8B = "2,3,11,5,6,8,8c," + EVAL_SKIP_SHARED    # P: job — skip all but 8b
EVAL_SKIP_ONLY_8C = "2,3,11,5,6,8,8b," + EVAL_SKIP_SHARED    # F: job — skip all but 8c
PT_METRICS = ["rollout", "causal", "tdist", "teacher_free", "maskratio", "order"]  # mirrors m12e.METRICS keys
ET_METRICS = ["aot", "tov", "pace", "tcc"]                   # mirrors m12f.METRICS keys


def cpu_slots(n):
    """Private CPU-set per GPU slot (iter18 2026-06-07): each arm is launched under
    `taskset -c <slice>` owning 64/n physical cores PLUS their SMT siblings — no core
    migration, no cross-arm L3/CCD thrash, no SMT-sibling interference (the residual
    single-node concurrency tax left after worker autotune). This box is single-NUMA
    (lscpu: 1 node), so the cpuset is the applicable subset of the standard numactl
    prescription; the binding is inherited by every DataLoader/ProcessPool worker.
    Returns [None]*n (unpinned, status quo) when lscpu/taskset are unavailable."""
    import shutil as _sh
    try:
        if not _sh.which("taskset"):
            raise RuntimeError("taskset not on PATH")
        out = subprocess.run(["lscpu", "-p=CPU,CORE"], capture_output=True,
                             text=True, check=True, timeout=10).stdout
        core2cpus = {}
        for ln in out.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            cpu, core = map(int, ln.split(",")[:2])
            core2cpus.setdefault(core, []).append(cpu)
        cores = sorted(core2cpus)
        per = len(cores) // n
        if per < 1:
            raise RuntimeError(f"{len(cores)} physical cores < {n} slots")
        slots = []
        for g in range(n):
            chunk = cores[g * per:(g + 1) * per] if g < n - 1 else cores[(n - 1) * per:]
            cpus = sorted(c for k in chunk for c in core2cpus[k])
            slots.append(",".join(map(str, cpus)))
        return slots
    except Exception as e:
        print(f"  [cpuset] pinning disabled ({type(e).__name__}: {e}) — arms launch unpinned",
              flush=True)
        return [None] * n


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
        # iter18 (2026-06-13) lever #1: WiSE-FT is NOT trained — its "train" job is a post-hoc weight
        # merge (OURS encoder × FROZEN base V-JEPA). Depends on the OURS train job (needs its
        # student_encoder.pt). NO literals: alpha ← pipeline.yaml wiseft.alpha; the frozen base ckpt is
        # resolved per backbone via backbone_model_configs.<BACKBONE> → model.checkpoint_path (same
        # registry run_train.sh uses). The resume-guard still skips it if the merge already ran.
        if arm == "surgical_3stage_DI_wiseft_encoder":
            _ours_dir = ARM2DIR["surgery_3stage_DI_encoder"]
            _wf_dir = ARM2DIR["surgical_3stage_DI_wiseft_encoder"]
            _yx = "scripts/lib/yaml_extract.py"
            _alpha = f"$({_yx} configs/pipeline.yaml wiseft.alpha)"
            _frozen = f"$({_yx} $({_yx} configs/pipeline.yaml backbone_model_configs.{BACKBONE}) model.checkpoint_path)"
            jobs[jid] = dict(
                id=jid, kind="train", deps={f"T:{BACKBONE}:surgery_3stage_DI_encoder"}, needs_labels=False,
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} {{pin}}python -u src/utils/wiseft_merge.py "
                     f"--alpha {_alpha} --frozen-ckpt {_frozen} "
                     f"--surgery-ckpt outputs/{mtag}/{BACKBONE}/{_ours_dir}/student_encoder.pt "
                     f"--out-dir outputs/{mtag}/{BACKBONE}/{_wf_dir}"),
                log=f"logs/iter18_ngpu_{mtag}_wiseft_{{ts}}.log")
            continue
        # iter18: EVERY non-pretrain arm inits from pretrain's m09a_ckpt_best.pt → all dep on it.
        deps = set() if arm == "pretrain_encoder" else {seed_id}
        jobs[jid] = dict(
            id=jid, kind="train", deps=deps, needs_labels=(jid != seed_id),
            cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} NGPU_CONCURRENCY={{conc}} BACKBONE={BACKBONE} CACHE_POLICY_ALL={{cache}} "
                 f"{{pin}}./scripts/run_train.sh {arm} {mflag}"),
            log=f"logs/iter18_ngpu_{mtag}_train_{arm}_{{ts}}.log")
    # eval jobs (iter18 2026-06-07 metric-parallel): per encoder = ONE E: job (stages 2-8) + 6 P:
    # jobs (Stage 8b, one predictor_temporal metric each, fanned across GPUs). E: and all 6 P: gate
    # on the SAME dep (the encoder's TRAIN job, or none for frozen) — NOT on E: — so Stage 8b runs
    # concurrently with stages 2-8. Each job gets its OWN deps set (set(deps)); --only mutates deps
    # in place, so a shared set object would cross-contaminate the 7 sibling jobs.
    for arm, enc in [("frozen", "frozen")] + list(ARM2ENC.items()):
        encn = enc_name(enc)
        deps = set() if arm == "frozen" else {f"T:{BACKBONE}:{arm}"}
        ejid = f"E:{encn}"
        jobs[ejid] = dict(
            id=ejid, kind="eval", deps=set(deps), needs_labels=True,
            cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_PERENC_NO8B} "
                 f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
            log=f"logs/iter18_ngpu_{mtag}_eval_{encn}_{{ts}}.log")
        for m in PT_METRICS:
            pjid = f"P:{encn}:{m}"
            jobs[pjid] = dict(
                id=pjid, kind="eval", deps=set(deps), needs_labels=True,
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_ONLY_8B} PT_METRIC={m} "
                     f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
                log=f"logs/iter18_ngpu_{mtag}_pt_{encn}_{m}_{{ts}}.log")
        # iter18 m12f revival (#1): Stage 8c fans into 4 F: jobs (one encoder_temporal metric
        # each), gated on TRAIN like P: — runs concurrent with stages 2-8. If a F: job lands
        # before this encoder's Stage-2 features exist, m12f's share-features falls back to
        # fresh identity forwards with an explicit reason (correct either way; in practice the
        # queue drains E: stage 2 long before F: slots open, so the share usually hits).
        for m in ET_METRICS:
            fjid = f"F:{encn}:{m}"
            jobs[fjid] = dict(
                id=fjid, kind="eval", deps=set(deps), needs_labels=True,
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_ONLY_8C} ET_METRIC={m} "
                     f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
                log=f"logs/iter18_ngpu_{mtag}_et_{encn}_{m}_{{ts}}.log")
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
    ap.add_argument("--skip-arms", nargs="+", default=None, metavar="ARM",
                    help="iter18 2026-06-07 (user order: park CaSSLe, 213.8 s/step measured): drop "
                         "these train arms AND their eval jobs from the DAG; the §3 finale covers "
                         "only the remaining encoders. Skipped arms keep every on-disk anchor — "
                         "rerun later WITHOUT this flag + --cache 1 to train them and rebuild the "
                         "finale over all 14 encoders.")
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

    if args.skip_arms:
        if args.only:
            sys.exit("FATAL: --skip-arms and --only are mutually exclusive.")
        bad = [a for a in args.skip_arms if a not in ARM2ENC]
        if bad:
            sys.exit(f"FATAL: unknown arm(s) {bad} — choices: {sorted(ARM2ENC)}")
        if "pretrain_encoder" in args.skip_arms:
            sys.exit("FATAL: cannot skip pretrain_encoder — it is every arm's init dependency.")
        drop = ({f"T:{BACKBONE}:{a}" for a in args.skip_arms}
                | {f"E:{enc_name(ARM2ENC[a])}" for a in args.skip_arms}
                | {f"P:{enc_name(ARM2ENC[a])}:{m}" for a in args.skip_arms for m in PT_METRICS}
                | {f"F:{enc_name(ARM2ENC[a])}:{m}" for a in args.skip_arms for m in ET_METRICS})
        jobs = {jid: j for jid, j in jobs.items() if jid not in drop}
        print(f"  [--skip-arms] dropped {sorted(args.skip_arms)} (train+eval+8b/8c-metrics) — §3 finale "
              f"will cover the remaining encoders only", flush=True)

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
        n_train_done = len(done)
        if n_train_done:
            print(f"  [resume --cache 1] skipping {n_train_done} already-trained arms: "
                  f"{sorted(d.split(':')[2] for d in done)}", flush=True)
        # P: (Stage-8b single-metric) jobs are done when their aggregate_<metric>.json exists — skip
        # them on resume so a restart never relaunches the ~84 no-op m12e processes. (E: jobs re-run
        # but cache-skip their stages internally via CACHE_POLICY_ALL=1, so they stay schedulable.)
        pt_done = 0
        for jid in jobs:
            if not jid.startswith("P:"):
                continue
            enc_nm, metric = jid[2:].rsplit(":", 1)   # NOT 'enc_name' — that's the module helper; a main()-local shadows it → UnboundLocalError at the skip-arms block
            if (REPO / f"outputs/{mtag}/predictor_temporal/{enc_nm}/aggregate_{metric}.json").exists():
                done.add(jid)
                pt_done += 1
        if pt_done:
            print(f"  [resume --cache 1] skipping {pt_done} already-done Stage-8b metric jobs",
                  flush=True)
        # F: (Stage-8c encoder_temporal) jobs — same done-marker pattern (m12f writes
        # aggregate_<metric>.json per metric; tcc's is aggregate_tcc.json).
        et_done = 0
        for jid in jobs:
            if not jid.startswith("F:"):
                continue
            enc_nm, metric = jid[2:].rsplit(":", 1)
            if (REPO / f"outputs/{mtag}/encoder_temporal/{enc_nm}/aggregate_{metric}.json").exists():
                done.add(jid)
                et_done += 1
        if et_done:
            print(f"  [resume --cache 1] skipping {et_done} already-done Stage-8c metric jobs",
                  flush=True)

    # CPU-set pinning (iter18 2026-06-07): one private core slice per GPU slot —
    # NGPU_CPUSET tells stream_autotune the cores are private (divide by 1, not conc).
    slots = cpu_slots(args.gpus)
    if slots[0]:
        print("  [cpuset] GPU slots pinned: " +
              " · ".join(f"GPU{g}→{s.split(',')[0]}..{s.split(',')[-1]}({len(s.split(','))}t)"
                         for g, s in enumerate(slots)), flush=True)

    def pin_for(g):
        return (f"NGPU_CPUSET={len(slots[g].split(','))} taskset -c {slots[g]} "
                if slots[g] else "")

    free = list(range(args.gpus))
    print(f"═══ iter18 N-GPU scheduler · mode={args.mode} · backbone={BACKBONE} · gpus={args.gpus} · "
          f"cache={args.cache} · {len(jobs)} jobs ({sum(j['kind'] == 'train' for j in jobs.values())} "
          f"train + {sum(j['kind'] == 'eval' for j in jobs.values())} eval) ═══", flush=True)

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
    # FAIL-FAST (iter18 2026-06-08, user order): the FIRST job failure aborts the WHOLE run — stop
    # launching new jobs, SIGTERM every still-running job, and exit(1) — mirroring run_train.sh /
    # run_eval.sh's own `set -euo pipefail`. The OLD behaviour (keep launching all 95, report the
    # failures only at the very end, then skip §3) buried the first failure under ~80 later launches,
    # which read as "still making progress" — misleading. Now the run stops AT the first ✗.
    abort = None
    while len(done) + len(failed) < len(jobs):
        for jid in list(jobs):
            if abort or not free or not ready(jid):
                continue
            g = free.pop(0)
            log = jobs[jid]["log"].format(ts=ts())
            cmd = jobs[jid]["cmd"].format(gpu=g, cache=args.cache, conc=args.gpus, pin=pin_for(g))
            print(f"[{now()}] GPU{g} ◀ {jid}  → {log}", flush=True)
            lf = open(REPO / log, "w")
            # start_new_session: each job is its OWN process group, so a fail-fast abort can SIGTERM
            # the whole tree (shell + python + DataLoader/decode workers), not just the wrapper shell.
            p = subprocess.Popen(cmd, shell=True, cwd=str(REPO), stdout=lf,
                                 stderr=subprocess.STDOUT, start_new_session=True)
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
                if abort is None:
                    abort = jid          # first failure → trip the fail-fast abort
        if abort is not None:
            if running:
                print(f"  ⛔ FAIL-FAST: {abort} failed (rc={failed[abort]}) — SIGTERM "
                      f"{len(running)} still-running job(s) + aborting the run", flush=True)
                for _g, (_jid, _p, _lf) in list(running.items()):
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.killpg(os.getpgid(_p.pid), signal.SIGTERM)   # whole process group
                    _lf.close()
                running.clear()
            break
        if running or any(ready(j) for j in jobs):
            time.sleep(10)
        elif free and not running:   # nothing ready, nothing running → a failed dep blocks the rest
            break

    el = (time.time() - t0) / 3600
    if abort is not None:
        print(f"═══ ⛔ FAIL-FAST ABORT after {el:.1f}h · {abort} failed (rc={failed[abort]}) · "
              f"{len(done)} job(s) done before the abort · §3 finale NOT run ═══", flush=True)
        print(f"  ✗ first failure: {abort} — read its per-job log (named on its GPU ◀ line above).")
        print(f"  Fix the cause, then re-run with --cache 1 to keep the {len(done)} completed job(s).")
        sys.exit(1)
    print(f"═══ jobs settled in {el:.1f}h · done={len(done)} failed={len(failed)} ═══", flush=True)

    if args.only:
        print(f"═══ --only run complete ({sorted(args.only)}) — §3 finale skipped by design ═══",
              flush=True)
        return

    # ── §3 finale: ONE run_eval over ALL encoders, per-encoder stages skipped (cached) →
    #    runs the shared paired-Δ + m13 stages with every encoder present. cache=1 keeps caches.
    #    --skip-arms: the finale covers only the encoders whose evals ran (paired-Δ/m13
    #    handle encoder subsets — the live 15-min preview exercises this same path daily).
    _skip = set(args.skip_arms or [])
    all_encs = " ".join([enc_name("frozen")]
                        + [enc_name(e) for a, e in ARM2ENC.items() if a not in _skip])
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
