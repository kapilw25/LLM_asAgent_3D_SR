"""D_I tube cache — DataLoader-level cache-fill + COLD-vs-WARM speedup (Test C-pre / D).

Drives the REAL StreamingFactorDataset (the exact class the m09c trainer uses) over the
intervene arm's merged config, with D_I forced (mode_mixture I=1.0) so every emitted item
exercises stream_interaction_tubes. Confirms:
  · the tube_cache dir FILLS with .npz files during the cold pass (cache write path),
  · a WARM re-run over the same clips reads those .npz (no MP4 decode / mining),
  · per-item D_I time COLD (empty cache) vs WARM (populated) — the honest speedup ratio,
  · multi-worker (fork) writes are atomic (no corrupt .npz; all loadable).

This is CPU-side (no GPU) — it measures the EXACT work the cache removes from the
DataLoader worker (decode + mining), which is what the cache targets. The GPU-bound vs
data-bound question (does the cache help end-to-end?) is answered by run_train SANITY
(Test C) + the cold/warm step-time there; here we isolate the data-side delta.

    python -u scripts/legacy/tests_streaming/test_tube_cache_loader.py <cache_dir> [n_items] [n_workers]
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from torch.utils.data import DataLoader   # noqa: E402

from utils.config import get_pipeline_config, load_merged_config   # noqa: E402
from utils import data_paths   # noqa: E402
from utils.training import (   # noqa: E402
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
)

MODEL_CFG = PROJECT_ROOT / "configs/model/vjepa2_1.yaml"
TRAIN_CFG = PROJECT_ROOT / "configs/train/surgery_3stage_DI_intervene_encoder.yaml"


def _drain(ds, batch_size, n_workers, n_items, label):
    """Iterate n_items from a DataLoader over ds; return (count, n_DI, elapsed_s)."""
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=n_workers,
        prefetch_factor=2 if n_workers > 0 else None,
        persistent_workers=False,
        worker_init_fn=_streaming_worker_init if n_workers > 0 else None,
    )
    seen = 0
    n_di = 0
    t0 = time.perf_counter()
    for batch in loader:
        fts = batch["factor_type"]
        n_di += sum(1 for f in fts if f == "D_I")
        seen += len(fts)
        if seen >= n_items:
            break
    elapsed = time.perf_counter() - t0
    del loader
    print(f"  [{label}] drained {seen} items ({n_di} D_I) in {elapsed:.2f}s "
          f"→ {elapsed/max(seen,1)*1000:.1f} ms/item")
    return seen, n_di, elapsed


def _make_ds(cfg, mp4_index, mask_index, manifest, cache_dir, base_seed):
    """Build a StreamingFactorDataset forcing D_I (mode_mixture I=1.0).

    factor_cfg is for D_L/D_A only; with I=1.0 no layout/agent view is sampled,
    so an empty dict is fine (the D_I branch reads interaction_cfg, not factor_cfg)."""
    interaction_cfg = dict(cfg["interaction_mining"])
    interaction_cfg["tube_cache"] = {"enabled": True, "dir": str(cache_dir)}
    return StreamingFactorDataset(
        mp4_index=mp4_index, mask_index=mask_index, factor_manifest=manifest,
        factor_cfg={},
        mode_mixture={"L": 0.0, "A": 0.0, "I": 1.0},
        num_frames=cfg["data"]["num_frames"], crop_size=cfg["model"]["crop_size"],
        base_seed=base_seed, steps_per_epoch=None,
        interaction_cfg=interaction_cfg, raw_replay_pct=0.0,
    )


def main():
    cache_dir = Path(sys.argv[1])
    n_items = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    cache_dir.mkdir(parents=True, exist_ok=True)

    pcfg = get_pipeline_config()
    local_data = pcfg["data"]["local_data_dir"]
    cfg = load_merged_config(str(MODEL_CFG), str(TRAIN_CFG))

    print(f"local_data={local_data} · cache_dir={cache_dir} · "
          f"n_items={n_items} · n_workers={n_workers}")
    manifest_path = data_paths.factor_manifest_path(local_data)
    masks_dir = data_paths.masks_dir(local_data)
    print(f"manifest={manifest_path}")
    mp4_index, mask_index, manifest = build_streaming_indices(
        manifest_path, masks_dir, str(local_data))
    di_keys = [k for k, v in manifest.items() if v.get("has_D_I")]
    print(f"  D_I-eligible clips in manifest: {len(di_keys)}")
    if not di_keys:
        print("FATAL: no has_D_I clips in manifest — cannot exercise the tube path.")
        sys.exit(1)

    batch_size = min(8, n_items)

    # ── COLD: empty cache → decode + mine every D_I item, write .npz. ──
    n_before = len(list(cache_dir.glob("*.npz")))
    ds_cold = _make_ds(cfg, mp4_index, mask_index, manifest, cache_dir, base_seed=4242)
    _, di_cold, t_cold = _drain(ds_cold, batch_size, n_workers, n_items, "COLD")
    n_after_cold = len(list(cache_dir.glob("*.npz")))
    print(f"  cache .npz: {n_before} → {n_after_cold} (cold pass wrote "
          f"{n_after_cold - n_before})")

    # ── Validate every written .npz loads (atomic-write integrity). ──
    bad = []
    for f in cache_dir.glob("*.npz"):
        try:
            d = np.load(f, allow_pickle=False)
            _ = [d[f"t{i}"] for i in range(int(d["n"]))]
        except Exception as e:
            bad.append((f.name, repr(e)))
    if bad:
        print(f"  ❌ {len(bad)} corrupt cache files: {bad[:3]}")
    else:
        print(f"  ✅ all {n_after_cold} cache .npz load cleanly (atomic write OK)")

    # ── WARM: same seed → same clip sequence → all HITs (no decode/mining). ──
    ds_warm = _make_ds(cfg, mp4_index, mask_index, manifest, cache_dir, base_seed=4242)
    _, di_warm, t_warm = _drain(ds_warm, batch_size, n_workers, n_items, "WARM")

    ratio = t_cold / max(t_warm, 1e-9)
    print("\n" + "=" * 60)
    print("D_I DATA-SIDE SPEEDUP (decode+mining removed by cache)")
    print(f"  COLD: {t_cold:.2f}s  ·  WARM: {t_warm:.2f}s  ·  ratio: {ratio:.2f}×")
    print(f"  (n_items={n_items}, D_I cold={di_cold}/warm={di_warm}, "
          f"workers={n_workers})")
    ok = (n_after_cold > n_before) and not bad
    if ok:
        print("  ✅ cache fills + reads cleanly; ratio above is the data-side gain.")
        sys.exit(0)
    print("  ❌ cache did not fill or had corrupt files.")
    sys.exit(1)


if __name__ == "__main__":
    main()
