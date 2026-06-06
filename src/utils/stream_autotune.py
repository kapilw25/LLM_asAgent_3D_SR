"""Runtime CPU-worker auto-tune for streaming dataloaders (iter18, 2026-06-06).

WHY: configs/train/surgery_base.yaml used to FIX factor_streaming.num_workers per mode
(poc: 4). On the 4× box, 3-4 concurrent arms each capped at 4 workers starved their GPUs
(60-140 s/step vs 27 s/step solo; box load 5.7 of 128 cores — the per-arm cap was the
bottleneck, not the machine). A fixed number can never fit 1×/2×/4×/8× boxes at once.

Train yamls now declare `num_workers: auto` (or keep an explicit int as an override).
This resolver turns "auto" into a number from the box's LIVE resources:

    workers = clamp( cores // concurrency − reserve_per_proc, min_workers, max_workers )
    workers = min( workers,  cgroup_ram × ram_headroom_pct / concurrency / ram_per_worker_gb )

· cores        = os.cpu_count() (container-visible CPUs)
· concurrency  = how many sibling training processes share the box:
                 NGPU_CONCURRENCY env (fed by scripts/iter18_poc_ngpu.py per job)
                 → else the box's GPU count (one arm per GPU is this repo's pattern)
                 → else 1 (solo run)
· all tuning constants live in configs/pipeline.yaml streaming.worker_autotune —
  NO numbers in this file (CLAUDE.md no-hardcode).

USAGE (each m09* trainer, right after flattening the mode-gated yaml value):
    from utils.stream_autotune import resolve_stream_workers
    fs_cfg["num_workers"] = resolve_stream_workers(fs_cfg["num_workers"], label="m09c1")
"""
import os
import shutil
import subprocess

from utils.config import get_cgroup_memory_gb, get_pipeline_config


def _detect_concurrency() -> int:
    """How many sibling trainers share this box. Scheduler-fed env wins; else one arm
    per GPU (this repo's launch pattern); else solo."""
    env = os.environ.get("NGPU_CONCURRENCY", "").strip()
    if env:
        return max(int(env), 1)
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
        n = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
        if n:
            return n
    return 1


def resolve_stream_workers(cfg_value, label: str) -> int:
    """Resolve a yaml num_workers value: int = explicit override (returned as-is),
    "auto" = computed from live cores/concurrency/RAM. FAIL LOUD on anything else."""
    if isinstance(cfg_value, int):
        return cfg_value
    if cfg_value != "auto":
        raise ValueError(
            f"[{label}] factor_streaming.num_workers must be an int or 'auto', "
            f"got {cfg_value!r} — declare it in the train yaml.")

    at = get_pipeline_config()["streaming"]["worker_autotune"]
    cores = os.cpu_count()
    if not cores:
        raise RuntimeError(f"[{label}] os.cpu_count() returned {cores!r} — cannot auto-tune; "
                           f"set an explicit num_workers int in the train yaml.")
    conc = _detect_concurrency()

    by_cpu = cores // conc - at["reserve_per_proc"]
    workers = max(min(by_cpu, at["max_workers"]), at["min_workers"])
    ram_gb = get_cgroup_memory_gb()
    if ram_gb != float("inf"):
        by_ram = int(ram_gb * at["ram_headroom_pct"] / conc / at["ram_per_worker_gb"])
        workers = max(min(workers, by_ram), at["min_workers"])
    print(f"  [stream-autotune {label}] cores={cores} · concurrency={conc} "
          f"(src={'env' if os.environ.get('NGPU_CONCURRENCY') else 'gpu-count'}) · "
          f"cgroup_ram={'inf' if ram_gb == float('inf') else f'{ram_gb:.0f}G'} "
          f"→ num_workers={workers}", flush=True)
    return workers
