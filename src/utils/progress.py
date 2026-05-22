"""Shared tqdm progress bar factory for all src/m*.py scripts.

Every script must show: total count, current progress, rate, ETA.
Use make_pbar() instead of raw tqdm() to ensure consistent format.

USAGE:
    from utils.progress import make_pbar
    pbar = make_pbar(total=115687, desc="m06_metrics", unit="clip")
    for batch in data:
        process(batch)
        pbar.update(len(batch))
    pbar.close()
"""
from tqdm import tqdm


def make_pbar(total: int, desc: str, unit: str = "item",
              initial: int = 0) -> tqdm:
    """Create a standardized tqdm progress bar with ETA.

    Args:
        total: Total number of items to process.
        desc: Short description (e.g., "m06_metrics", "m05_vjepa").
        unit: Unit name (e.g., "clip", "step", "video", "shard").
        initial: Starting count (for checkpoint resume).

    Returns:
        tqdm progress bar instance. Caller must call .update(n) and .close().
    """
    # iter16 (2026-05-21): smoothing=0 → tqdm uses total/elapsed → honest
    # aggregate ETA on bursty workloads (m04d incident: 4.70clip/s headline hid
    # 1.10 clip/s reality + ~25 hr ETA shown as 5:52). Default smoothing=0.3 is
    # an EWMA over recent samples that captures intra-burst rate when the GPU
    # alternates with I/O / decode / disk waits — banned by src/CLAUDE.md
    # "no false advertising / windowed rate" rules. Applies to every make_pbar
    # caller (m04*, m09*, m10*, m11*, probe_*) in one place.
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
