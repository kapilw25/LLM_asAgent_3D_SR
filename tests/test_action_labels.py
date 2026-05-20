"""Self-contained test runner for src/utils/action_labels.py.

iter16 M5 (2026-05-20): verifies _extract_video_id + video-disjoint
stratified_split. Runs without pytest dependency — uses plain asserts + a
__main__ collector that discovers `test_*` functions and reports per-test
PASS / FAIL / ERROR.

USAGE:
    python -u tests/test_action_labels.py 2>&1 | tee logs/test_action_labels.log
"""
import sys
import traceback
from collections import defaultdict
from pathlib import Path

# Make src/ importable when this file is run directly from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.action_labels import _extract_video_id, stratified_split  # noqa: E402


# ── Synthetic-data fixtures ───────────────────────────────────────────

def _synth_records(n_videos_per_class: int, n_clips_per_video: int,
                   n_classes: int):
    """Build a balanced synthetic record list.

    Each class has `n_videos_per_class` distinct videos × `n_clips_per_video`
    clips per video. Class-balanced; deterministic clip_keys.
    """
    records = []
    for cls_id in range(n_classes):
        for vid_idx in range(n_videos_per_class):
            video_id = f"vid_c{cls_id}_v{vid_idx:02d}"
            for clip_idx in range(n_clips_per_video):
                records.append({
                    "clip_key": f"tier1/city/walk/{video_id}/{video_id}-{clip_idx:03d}.mp4",
                    "class": f"cls_{cls_id}",
                    "class_id": cls_id,
                })
    return records


# ── _extract_video_id tests ───────────────────────────────────────────

def test_extract_video_id_happy_5part():
    """tier1/tier2 5-part clip_key → video_id = parts[-2]."""
    assert _extract_video_id(
        "tier2/bhopal/rain/--pBu8H35ro/--pBu8H35ro-004.mp4"
    ) == "--pBu8H35ro"
    assert _extract_video_id(
        "tier1/mumbai/walking/DFQO7Zn-KM4/DFQO7Zn-KM4-1000.mp4"
    ) == "DFQO7Zn-KM4"


def test_extract_video_id_happy_4part_goa():
    """goa/monuments 4-part clip_key (no tier prefix) → still parts[-2]."""
    assert _extract_video_id(
        "goa/walking/04YKvC8kAgI/04YKvC8kAgI-009.mp4"
    ) == "04YKvC8kAgI"
    assert _extract_video_id(
        "monuments/red_fort/abc123/abc123-001.mp4"
    ) == "abc123"


def test_extract_video_id_short_raises():
    """<3-part clip_key → ValueError with diagnostic."""
    try:
        _extract_video_id("a/b.mp4")
    except ValueError as e:
        assert "2 parts" in str(e), f"expected diagnostic in msg, got: {e}"
        return
    raise AssertionError("expected ValueError on 2-part clip_key")


def test_extract_video_id_empty_raises():
    """Empty clip_key → ValueError."""
    try:
        _extract_video_id("")
    except ValueError as e:
        assert "1 parts" in str(e), f"expected diagnostic in msg, got: {e}"
        return
    raise AssertionError("expected ValueError on empty clip_key")


# ── stratified_split tests ────────────────────────────────────────────

def test_stratified_split_video_disjoint():
    """30v × 10c × 5cls synthetic corpus → no video straddles splits."""
    records = _synth_records(n_videos_per_class=6, n_clips_per_video=10,
                              n_classes=5)
    splits = stratified_split(records, train_pct=0.60, val_pct=0.20,
                              seed=99, min_per_split=5)

    vmap = defaultdict(set)
    for k, s in splits.items():
        vmap[_extract_video_id(k)].add(s)
    straddlers = {v: list(sp) for v, sp in vmap.items() if len(sp) > 1}
    assert not straddlers, f"video-id leakage: {straddlers}"

    assert len(splits) == len(records), \
        f"expected {len(records)} clips assigned, got {len(splits)}"


def test_stratified_split_class_proportions():
    """Per-class train/val/test counts within ±5% of target ratios."""
    n_classes = 5
    records = _synth_records(n_videos_per_class=10, n_clips_per_video=20,
                              n_classes=n_classes)
    train_pct, val_pct = 0.60, 0.20
    test_pct = 1.0 - train_pct - val_pct
    splits = stratified_split(records, train_pct=train_pct, val_pct=val_pct,
                              seed=99, min_per_split=5)

    # Per-class actual vs expected
    per_cls = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    cls_total = defaultdict(int)
    for r in records:
        cls = r["class"]
        per_cls[cls][splits[r["clip_key"]]] += 1
        cls_total[cls] += 1

    tolerance = 0.05
    for cls, counts in per_cls.items():
        total = cls_total[cls]
        for split, target_pct in [("train", train_pct), ("val", val_pct),
                                   ("test", test_pct)]:
            actual_pct = counts[split] / total
            deviation = abs(actual_pct - target_pct)
            assert deviation < tolerance, (
                f"class={cls} split={split}: target={target_pct:.2f} "
                f"actual={actual_pct:.2f} dev={deviation:.3f} > {tolerance}"
            )


def test_stratified_split_deterministic():
    """Same seed → byte-identical split dict."""
    records = _synth_records(n_videos_per_class=8, n_clips_per_video=10,
                              n_classes=4)
    splits_a = stratified_split(records, train_pct=0.60, val_pct=0.20,
                                seed=42, min_per_split=5)
    splits_b = stratified_split(records, train_pct=0.60, val_pct=0.20,
                                seed=42, min_per_split=5)
    assert splits_a == splits_b, "same seed produced different splits"


def test_stratified_split_seed_sensitive():
    """Different seed → different split (sanity: ≥10% clip reassignment)."""
    records = _synth_records(n_videos_per_class=8, n_clips_per_video=10,
                              n_classes=4)
    splits_99  = stratified_split(records, train_pct=0.60, val_pct=0.20,
                                   seed=99, min_per_split=5)
    splits_100 = stratified_split(records, train_pct=0.60, val_pct=0.20,
                                   seed=100, min_per_split=5)
    diff = sum(1 for k in splits_99 if splits_99[k] != splits_100.get(k))
    diff_pct = diff / len(splits_99)
    assert diff_pct >= 0.10, (
        f"seed change only reassigned {diff_pct:.1%} of clips — "
        f"expected ≥10% (sanity check on RNG)"
    )


def test_stratified_split_infeasible_single_video_class():
    """Class with all clips in 1 video → cannot span 3 splits → ValueError."""
    # 4 normal classes × 5 videos × 10 clips PLUS a single-video class.
    records = _synth_records(n_videos_per_class=5, n_clips_per_video=10,
                              n_classes=4)
    # Append a pathological class: 1 video, 20 clips, class_id=4.
    for clip_idx in range(20):
        records.append({
            "clip_key": f"tier1/city/walk/pathological_vid/pathological_vid-{clip_idx:03d}.mp4",
            "class": "single_video_class",
            "class_id": 4,
        })

    try:
        stratified_split(records, train_pct=0.60, val_pct=0.20,
                         seed=99, min_per_split=5)
    except ValueError as e:
        msg = str(e)
        assert "min_per_split" in msg, f"expected min_per_split in msg, got: {msg}"
        assert "Videos-per-class" in msg, f"expected diagnostic, got: {msg}"
        return
    raise AssertionError("expected ValueError on single-video class")


# ── Integration: real eval_10k corpus ─────────────────────────────────

def test_integration_eval_10k_video_disjoint():
    """End-to-end on data/eval_10k_local — must produce zero straddlers.

    Skipped if motion_features.npy isn't on disk (this test is fixture-
    sensitive; CI may not have the 33 GB local-data bundle).
    """
    motion_features = (Path(__file__).resolve().parent.parent
                       / "data" / "eval_10k_local"
                       / "m04d_motion_features" / "motion_features.npy")
    eval_subset = (Path(__file__).resolve().parent.parent
                   / "data" / "eval_10k_local" / "eval_10k.json")
    if not motion_features.exists() or not eval_subset.exists():
        print(f"      SKIP: {motion_features} or {eval_subset} missing")
        return "skip"

    from utils.action_labels import load_subset_with_labels
    records, _ = load_subset_with_labels(
        eval_subset, motion_features, min_clips_per_class=34,
    )
    splits = stratified_split(records, train_pct=0.70, val_pct=0.15,
                              seed=99, min_per_split=5)

    # Coverage
    assert len(splits) == len(records), \
        f"coverage gap: {len(splits)} != {len(records)}"

    # Video-disjointness
    vmap = defaultdict(set)
    for k, s in splits.items():
        vmap[_extract_video_id(k)].add(s)
    straddlers = {v: list(sp) for v, sp in vmap.items() if len(sp) > 1}
    assert not straddlers, (
        f"video-id leakage on real corpus: {len(straddlers)} straddlers — "
        f"first 5: {dict(list(straddlers.items())[:5])}"
    )

    # Ratio sanity
    counts = {"train": 0, "val": 0, "test": 0}
    for s in splits.values():
        counts[s] += 1
    total = sum(counts.values())
    print(f"      eval_10k: {total} clips, "
          f"train/val/test = {counts['train']}/{counts['val']}/{counts['test']} "
          f"({counts['train']/total:.1%}/{counts['val']/total:.1%}/{counts['test']/total:.1%})")


# ── Test runner ───────────────────────────────────────────────────────

def _main():
    test_fns = {name: fn for name, fn in globals().items()
                if name.startswith("test_") and callable(fn)}
    print(f"Running {len(test_fns)} tests from tests/test_action_labels.py")
    print("=" * 70)

    n_pass = n_fail = n_skip = 0
    failures = []
    for name in sorted(test_fns.keys()):
        try:
            result = test_fns[name]()
            if result == "skip":
                print(f"  [SKIP] {name}")
                n_skip += 1
            else:
                print(f"  [PASS] {name}")
                n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            n_fail += 1
            failures.append((name, str(e)))
        except Exception as e:
            print(f"  [ERR ] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            n_fail += 1
            failures.append((name, f"{type(e).__name__}: {e}"))

    print("=" * 70)
    print(f"Result: {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"of {len(test_fns)} total")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _main()
