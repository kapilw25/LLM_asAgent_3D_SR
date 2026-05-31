"""Shared video I/O functions — clip key reconstruction, HF streaming, video decoding.

Moved from m05_vjepa_embed.py to eliminate cross-imports between m*.py scripts.
All pipeline scripts (m04d, m05, m05b, m05c, m09, m10, m11) import from here.

USAGE:
    from utils.video_io import get_clip_key, decode_video_bytes, create_stream
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from utils.config import HF_DATASET_REPO

# Video decoder: torchcodec (GPU NVDEC) segfaults on TAR-extracted mp4 bytes (SIGSEGV,
# no Python traceback). Disabled until upstream fix. PyAV works reliably.
# Incident: MEMORY.md "torchcodec can't read TAR files" — confirmed 2026-04-13.
_USE_TORCHCODEC = False
VideoDecoder = None

if not _USE_TORCHCODEC:
    try:
        import av
    except ImportError:
        print("FATAL: Neither torchcodec nor av available for video decoding")
        print("Install with: pip install av")
        sys.exit(1)


def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF WebDataset example metadata.

    Key format: "{section}/{video_id}/{source_file}"
    Example: "goa/walking/U2hl1v8xxlE/U2hl1v8xxlE-092.mp4"
    """
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


def create_stream(skip_count: int = 0, local_data: str = None):
    """Create streaming dataset from HF or local WebDataset shards.

    Args:
        skip_count: Number of examples to skip (for resume).
        local_data: Path to local WebDataset dir (from m00d). If None, streams from HF.

    Returns:
        IterableDataset: HuggingFace streaming dataset (undecoded bytes).
    """
    from datasets import load_dataset

    if local_data:
        from utils.data_paths import video_shard_glob  # iter17: subdir-or-root subset-*.tar
        ds = load_dataset("webdataset", data_files=video_shard_glob(local_data),
                          split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _load_torchcodec(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using torchcodec (GPU NVDEC, 5-10x faster than PyAV CPU)."""
    decoder = VideoDecoder(video_path)
    total_frames = decoder.metadata.num_frames if hasattr(decoder.metadata, "num_frames") else 200
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = decoder.get_frames_at(indices=frame_indices.tolist())
    video_tensor = frames.data  # (T, C, H, W)
    del decoder
    if video_tensor.shape[0] < num_frames:
        pad_size = num_frames - video_tensor.shape[0]
        last_frame = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
    return video_tensor[:num_frames]


def _load_av(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using PyAV (CPU fallback)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 1
    total_frames = stream.frames
    if total_frames == 0:
        if stream.duration and stream.average_rate:
            total_frames = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        else:
            total_frames = 200
    indices = set(np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int).tolist())
    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices:
            img = frame.to_ndarray(format="rgb24")
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img_tensor)
        frame_idx += 1
        if len(frames) >= num_frames:
            break
    container.close()
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].clone())
        else:
            frames.append(torch.zeros((3, 256, 256), dtype=torch.uint8))
    return torch.stack(frames[:num_frames])


# ── Eval frame cache (iter17) ─────────────────────────────────────────────
# Every eval module (m12a action, m12d future, m12e predictor-temporal, m12f
# encoder-temporal) reaches frames through decode_to_tensor → THIS decode_video_bytes,
# which is also the only decode in frozen_features. The scheduler runs one run_eval.sh
# PER encoder, so 8 processes decode the SAME ~1.8k clips concurrently (~7 passes each),
# pinning the CPU (load 467/256) while the GPUs idle. When EVAL_FRAME_CACHE_DIR is set
# (run_eval.sh ONLY — never training, whose full-pool stream over many epochs would blow
# the cache), the decoded native (T,C,H,W) uint8 tensor is memoized to disk keyed by
# (clip_key, num_frames). Decode is deterministic (linspace sampling + fixed decoder), so
# a hit returns a byte-identical tensor; decode_to_tensor still runs its crop/normalize
# AFTER, so metrics are unchanged. Unset env → every line below is skipped → zero effect
# on training / data-prep / the in-flight run (which uses the old run_eval.sh).
_FRAME_CACHE_ENV = "EVAL_FRAME_CACHE_DIR"
_FRAME_CACHE_MIN_FREE_ENV = "EVAL_FRAME_CACHE_MIN_FREE_GB"


def _frame_cache_path(key: str, num_frames: int):
    """On-disk cache path for (clip_key, num_frames), or None when the cache is disabled."""
    cache_dir = os.environ.get(_FRAME_CACHE_ENV)
    if not cache_dir:
        return None
    safe = key.replace("/", "_").replace("\\", "_")
    if safe.endswith(".mp4"):
        safe = safe[:-4]
    return os.path.join(cache_dir, f"{safe}__nf{int(num_frames)}.npy")


def _frame_cache_load(cache_path: str):
    """Cached (T,C,H,W) uint8 tensor, or None on miss / unreadable. Unreadable = a
    half-written file from a peer still mid-decode → treat as a miss and decode fresh
    (correctness-preserving: a bad cache file never contaminates the returned frames)."""
    if not os.path.exists(cache_path):
        return None
    try:
        return torch.from_numpy(np.load(cache_path))
    except Exception:
        return None  # partial/corrupt cache file → fall through to a fresh decode


def _frame_cache_store(cache_path: str, frames) -> None:
    """Atomically memoize the decoded tensor. The cache is a pure speed optimization, so a
    write failure WARNs and continues — the decode already returned the correct frames.
    Atomic = unique temp in the same dir + os.replace (POSIX-atomic); racing writers write
    identical bytes so last-rename-wins is safe. A free-disk floor stops the cache from
    filling the disk on large (FULL) runs: it just stops growing, reads still hit."""
    try:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        min_free_gb = float(os.environ.get(_FRAME_CACHE_MIN_FREE_ENV, "20"))
        if shutil.disk_usage(cache_dir).free < min_free_gb * (1024 ** 3):
            return  # disk getting tight → stop adding entries (existing ones still serve hits)
        arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception as e:
        print(f"  WARN: frame-cache store failed ({cache_path}): {e}")


def decode_video_bytes(mp4_bytes: bytes, tmp_dir: str, key: str,
                       num_frames: int) -> torch.Tensor:
    """Write mp4 bytes to temp file, decode frames, delete temp file.

    Args:
        mp4_bytes: Raw MP4 bytes from WebDataset TAR.
        tmp_dir: Temp directory for writing intermediate files.
        key: Clip key (used for safe filename).
        num_frames: Number of frames to extract (uniformly sampled).

    Returns:
        torch.Tensor: (T, C, H, W) uint8 tensor, or None if decode fails.

    When EVAL_FRAME_CACHE_DIR is set, the (key, num_frames) decode is memoized to disk
    (see the eval-frame-cache note above) so concurrent per-encoder evals decode each clip
    ONCE instead of re-decoding it for every encoder/metric.
    """
    cache_path = _frame_cache_path(key, num_frames)
    if cache_path is not None:
        cached = _frame_cache_load(cache_path)
        if cached is not None:
            return cached
    safe_key = key.replace("/", "_").replace("\\", "_")
    # HF stream clip keys end in ".mp4" (e.g. "tier1/.../abc-119.mp4"); strip
    # before re-appending so we don't write/read "abc-119.mp4.mp4". The doubled
    # extension is consistent across write+read in the same call (path is reused),
    # so it's not the ENOENT cause — but the doubled name made the explora v10
    # decode-failure logs misleading. Idempotent: keys without .mp4 unaffected.
    if safe_key.endswith(".mp4"):
        safe_key = safe_key[:-4]
    tmp_path = os.path.join(tmp_dir, f"{safe_key}.mp4")
    result = None
    try:
        # iter17 fix: the streaming __iter__ holds ONE TemporaryDirectory across all its yields;
        # under the DataLoader's multi-worker/prefetch lifecycle that dir can be torn down mid-decode
        # → ENOENT on the write below → return None → stream_factor FATALs the whole surgery training.
        # Recreate it (idempotent, cheap) so a transient temp-dir race is survived, not fatal.
        os.makedirs(tmp_dir, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)
        if _USE_TORCHCODEC:
            result = _load_torchcodec(tmp_path, num_frames)
        else:
            result = _load_av(tmp_path, num_frames)
    except Exception as e:
        print(f"  WARN: decode failed ({key}): {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                print(f"  WARN: failed to delete temp file {tmp_path}")
    if cache_path is not None and result is not None:
        _frame_cache_store(cache_path, result)
    return result
