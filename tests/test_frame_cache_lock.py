#!/usr/bin/env python3
"""iter17 — the per-clip lock in eval_frame_cache.get_or_build_frames makes N concurrent evals
decode each clip EXACTLY ONCE (no thundering-herd cold-wave). CPU-only, no GPU, no real video.

Run:  PYTHONPATH=src python -u tests/test_frame_cache_lock.py
"""
import os
import tempfile
import threading
import time

import torch

from utils import eval_frame_cache as efc


def main():
    nf = 16
    fake = torch.zeros((nf, 3, 8, 8), dtype=torch.uint8)
    n_decodes = [0]
    lock = threading.Lock()

    def fake_decode(mp4, td, key, num_frames):       # stand in for the real PyAV decode
        with lock:
            n_decodes[0] += 1
        time.sleep(0.1)                              # simulate a slow decode → forces the race
        return fake.clone()

    efc.decode_video_bytes = fake_decode             # patch the name get_or_build_frames calls

    with tempfile.TemporaryDirectory() as cd, tempfile.TemporaryDirectory() as td:
        os.environ["EVAL_FRAME_CACHE_DIR"] = cd
        os.environ["EVAL_FRAME_CACHE_MIN_FREE_GB"] = "0.001"
        n = 8
        results = [None] * n
        barrier = threading.Barrier(n)

        def worker(i):
            barrier.wait()                           # all n hit the COLD cache at the same instant
            results[i] = efc.get_or_build_frames(b"x", td, "same/clip/key.mp4", nf)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert n_decodes[0] == 1, f"LOCK FAILED: {n_decodes[0]} decodes (expected exactly 1)"
        assert all(r is not None and torch.equal(r, fake) for r in results), "a worker got wrong frames"
        npys = [f for f in os.listdir(cd) if f.endswith(".npy")]
        assert len(npys) == 1, f"expected exactly 1 cache file, got {npys}"

    print(f"LOCK TEST PASSED — {n} concurrent evals on a cold clip → {n_decodes[0]} decode, "
          f"all {n} got identical frames, 1 cache file.")


if __name__ == "__main__":
    main()
