"""D_I interaction-tube READ-THROUGH CACHE — determinism / result-safety test (CPU).

The critical red-flag check for iter18's tube cache: prove that the cache returns the
SAME tube list as the cold-compute path (so the caller's random pick is byte-identical →
training is bit-equivalent), and that any cfg-field change MISSES the cache (no stale
reuse). Mirrors the harness of test_parity_D_I.py but exercises the cache layer added to
src/utils/factor_streaming.py:stream_interaction_tubes.

Steps:
  1. Build the merged intervene config the SAME way the trainer does (load_merged_config
     + the tube_cache.dir injection that utils.m09_common.merge_m09_common_config applies),
     but point the cache dir at a FRESH tmp dir (never touches the real corpus cache).
  2. Find one real D_I clip (mask.npz with non-empty interactions_json + bbox/centroid).
  3. (A) COLD: cache empty → stream_interaction_tubes computes + writes one .npz.
     (B) WARM: same call → cache HIT, must be bitwise-identical (len, shapes, np.array_equal).
     (C) MISS: flip tube_margin_pct → different cfg-hash → recompute → a 2nd .npz appears.
  4. Cross-check the cached list also equals a direct _compute_interaction_tubes (no-cache).

Exit 0 = all green; non-zero on ANY mismatch (with clip_key + tube idx + diff stats).

    python -u scripts/legacy/tests_streaming/test_tube_cache.py 2>&1 | tee logs/tube_cache_det.log
"""
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import get_pipeline_config, load_merged_config   # noqa: E402
from utils import data_paths   # noqa: E402
from utils.factor_streaming import (   # noqa: E402
    stream_interaction_tubes,
    _compute_interaction_tubes,
    _tube_cache_key,
)
from utils.data_download import _get_clip_key   # noqa: E402

MODEL_CFG = PROJECT_ROOT / "configs/model/vjepa2_1.yaml"
TRAIN_CFG = PROJECT_ROOT / "configs/train/surgery_3stage_DI_intervene_encoder.yaml"
NUM_FRAMES_FALLBACK = 16


def _build_mp4_index(local_data: Path) -> dict:
    """{clip_key: (tar_path, base)} — same scheme as build_streaming_indices."""
    index = {}
    tars = data_paths.find_video_shards(str(local_data))
    for tar_path in tars:
        with tarfile.open(tar_path, "r") as tar:
            entries = {}
            for member in tar.getmembers():
                base = member.name.rsplit(".", 1)[0]
                ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                entries.setdefault(base, {})[ext] = member
            for base, parts in entries.items():
                if "json" not in parts or "mp4" not in parts:
                    continue
                clip_key = _get_clip_key(tar.extractfile(parts["json"]).read())
                if clip_key not in index:
                    index[clip_key] = (str(tar_path), base)
    return index


def _find_di_clip(masks_dir: Path, mp4_index: dict):
    """First clip (sorted) whose mask.npz has usable interactions AND an mp4."""
    for npz in sorted(masks_dir.glob("*.npz")):
        try:
            d = np.load(npz, allow_pickle=False)
        except Exception:
            continue
        if "interactions_json" not in d.files:
            continue
        inter = json.loads(str(d["interactions_json"]))
        if not inter:
            continue
        has_bbox = ("per_object_bboxes_json" in d.files
                    and json.loads(str(d["per_object_bboxes_json"])))
        has_cent = ("centroids_json" in d.files
                    and json.loads(str(d["centroids_json"])))
        if not (has_bbox or has_cent):
            continue
        # mask filename = <safe_clip_key>.npz; recover clip_key candidates.
        safe = npz.name[:-len(".npz")]
        for ck, (_tar, _base) in mp4_index.items():
            if ck.replace("/", "__") == safe:
                return ck, npz, len(inter)
    return None, None, None


def _assert_identical(a, b, label):
    if len(a) != len(b):
        print(f"  ❌ {label}: tube COUNT differs ({len(a)} vs {len(b)})")
        return False
    ok = True
    for j, (ta, tb) in enumerate(zip(a, b)):
        if ta.shape != tb.shape:
            print(f"  ❌ {label}: tube {j} SHAPE differs {ta.shape} vs {tb.shape}")
            ok = False
        elif not np.array_equal(ta, tb):
            diff = np.abs(ta.astype(int) - tb.astype(int))
            print(f"  ❌ {label}: tube {j} BYTES differ shape={ta.shape} "
                  f"max_diff={diff.max()} mean_diff={diff.mean():.4f}")
            ok = False
    if ok:
        sizes = [t.shape for t in a]
        print(f"  ✅ {label}: {len(a)} tubes bitwise-identical · shapes={sizes}")
    return ok


def main():
    pcfg = get_pipeline_config()
    local_data = Path(pcfg["data"]["local_data_dir"])
    masks_dir = data_paths.masks_dir(str(local_data))
    print(f"local_data = {local_data}")
    print(f"masks_dir  = {masks_dir}")

    cfg = load_merged_config(str(MODEL_CFG), str(TRAIN_CFG))
    interaction_cfg = cfg["interaction_mining"]
    if not interaction_cfg["enabled"]:
        print(f"FATAL: {TRAIN_CFG.name} has interaction_mining.enabled=false")
        sys.exit(1)
    if "tube_cache" not in interaction_cfg:
        print("FATAL: interaction_mining.tube_cache block missing from merged cfg "
              "(surgery_base.yaml edit not applied?)")
        sys.exit(1)
    num_frames = cfg["data"]["num_frames"] if "num_frames" in cfg.get("data", {}) \
        else NUM_FRAMES_FALLBACK
    print(f"num_frames = {num_frames} · interaction_cfg keys = {sorted(interaction_cfg)}")

    print("Building mp4 index from TAR shards...")
    mp4_index = _build_mp4_index(local_data)
    print(f"  mp4 index: {len(mp4_index)} clips")

    clip_key, mask_npz, n_inter = _find_di_clip(masks_dir, mp4_index)
    if clip_key is None:
        print("FATAL: no D_I clip found (mask.npz with interactions + bbox/centroid + mp4).")
        sys.exit(1)
    print(f"  D_I clip: {clip_key}  (n_interactions={n_inter})")

    tar_path, base = mp4_index[clip_key]
    with tarfile.open(tar_path, "r") as tar:
        mp4_bytes = tar.extractfile(tar.getmember(f"{base}.mp4")).read()

    all_ok = True
    with tempfile.TemporaryDirectory(prefix="tube_cache_test_") as cache_dir, \
         tempfile.TemporaryDirectory(prefix="tube_cache_tmp_") as tmp:
        # Inject the cache dir EXACTLY like merge_m09_common_config does — but into
        # a fresh tmp dir, so we never read/write the real corpus cache.
        interaction_cfg["tube_cache"]["dir"] = cache_dir
        cache_root = Path(cache_dir)

        # ── (0) Ground truth: a PURE compute (cache logic bypassed entirely). ──
        tubes_pure = _compute_interaction_tubes(
            mp4_bytes, mask_npz, interaction_cfg, num_frames, tmp, clip_key)
        print(f"\n[0] PURE compute: {len(tubes_pure)} tubes")
        if not tubes_pure:
            print("FATAL: chosen clip produced 0 tubes — cannot test result-safety.")
            sys.exit(1)

        # ── (A) COLD: cache empty → compute + write. ──
        t0 = time.perf_counter()
        tubes_cold = stream_interaction_tubes(
            mp4_bytes=mp4_bytes, mask_npz_path=mask_npz,
            interaction_cfg=interaction_cfg, num_frames=num_frames,
            tmp_dir=tmp, clip_key=clip_key)
        t_cold = time.perf_counter() - t0
        n_files_after_cold = len(list(cache_root.glob("*.npz")))
        print(f"[A] COLD call: {len(tubes_cold)} tubes · {t_cold*1000:.1f} ms · "
              f"cache now holds {n_files_after_cold} .npz")
        all_ok &= _assert_identical(tubes_pure, tubes_cold, "PURE == COLD")
        if n_files_after_cold != 1:
            print(f"  ❌ expected exactly 1 cache file after cold, got {n_files_after_cold}")
            all_ok = False

        # ── (B) WARM: same call → HIT, must be bitwise-identical. ──
        t0 = time.perf_counter()
        tubes_warm = stream_interaction_tubes(
            mp4_bytes=mp4_bytes, mask_npz_path=mask_npz,
            interaction_cfg=interaction_cfg, num_frames=num_frames,
            tmp_dir=tmp, clip_key=clip_key)
        t_warm = time.perf_counter() - t0
        print(f"[B] WARM call: {len(tubes_warm)} tubes · {t_warm*1000:.1f} ms "
              f"(cold {t_cold*1000:.1f} ms → {t_cold/max(t_warm,1e-9):.1f}× on this 1-clip call)")
        all_ok &= _assert_identical(tubes_cold, tubes_warm, "COLD == WARM (cache HIT)")

        # ── (C) MISS on cfg change: flip tube_margin_pct → new hash → recompute. ──
        key_before = _tube_cache_key(clip_key, num_frames, interaction_cfg)
        orig_margin = interaction_cfg["tube_margin_pct"]
        interaction_cfg["tube_margin_pct"] = orig_margin + 0.05
        key_after = _tube_cache_key(clip_key, num_frames, interaction_cfg)
        tubes_flipped = stream_interaction_tubes(
            mp4_bytes=mp4_bytes, mask_npz_path=mask_npz,
            interaction_cfg=interaction_cfg, num_frames=num_frames,
            tmp_dir=tmp, clip_key=clip_key)
        n_files_after_flip = len(list(cache_root.glob("*.npz")))
        print(f"[C] cfg FLIP tube_margin_pct {orig_margin}→{orig_margin+0.05}: "
              f"key {key_before} → {key_after} · "
              f"cache now holds {n_files_after_flip} .npz · {len(tubes_flipped)} tubes")
        if key_before == key_after:
            print("  ❌ cache key did NOT change on tube_margin_pct flip — STALE-REUSE BUG")
            all_ok = False
        elif n_files_after_flip != 2:
            print(f"  ❌ expected 2 cache files after flip (miss→new), got {n_files_after_flip}")
            all_ok = False
        else:
            print("  ✅ cfg change MISSED the cache → recomputed into a 2nd file (no stale reuse)")
        interaction_cfg["tube_margin_pct"] = orig_margin  # restore

        # ── (D) num_frames also changes the key (decode arg). ──
        key_nf2 = _tube_cache_key(clip_key, num_frames + 1, interaction_cfg)
        if key_nf2 == key_before:
            print("  ❌ cache key did NOT change on num_frames flip — STALE-REUSE BUG")
            all_ok = False
        else:
            print(f"  ✅ num_frames {num_frames}→{num_frames+1} changes key "
                  f"({key_before} → {key_nf2})")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL GREEN — tube cache is RESULT-SAFE (cached == uncached, "
              "cfg changes miss).")
        sys.exit(0)
    print("❌ FAILED — tube cache is NOT result-safe; do not enable.")
    sys.exit(1)


if __name__ == "__main__":
    main()
