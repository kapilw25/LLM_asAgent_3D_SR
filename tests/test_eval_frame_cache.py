#!/usr/bin/env python3
"""iter17 — unit test for the eval frame cache in utils.video_io.decode_video_bytes.

CPU-only, synthetic mp4 — no dataset, no GPU, cannot interfere with a live run. Validates:
  1. cache OFF (env unset)  → nothing written, normal decode.
  2. cache ON → MISS        → writes exactly one (clip,num_frames) .npy, == the OFF decode.
  3. cache ON → HIT         → returns a byte-identical tensor WITHOUT decoding (proved by
                              feeding garbage bytes on the 2nd call and still matching).
  4. distinct num_frames    → distinct cache entries (no collision).
  5. free-disk floor        → store skipped when min_free_gb is absurdly high; decode still ok.

Run:  PYTHONPATH=src python -u tests/test_eval_frame_cache.py
"""
import io
import os
import tempfile

import av
import numpy as np
import torch

from utils.video_io import decode_video_bytes


def _synth_mp4(n=20, hw=64):
    """Encode a tiny deterministic mp4 in memory (no files, no dataset)."""
    buf = io.BytesIO()
    c = av.open(buf, mode="w", format="mp4")
    s = c.add_stream("mpeg4", rate=10)
    s.width, s.height, s.pix_fmt = hw, hw, "yuv420p"
    for i in range(n):
        img = np.zeros((hw, hw, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 12) % 256
        img[i % hw, :, 1] = 255
        for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return buf.getvalue()


def main():
    mp4 = _synth_mp4()
    nf = 16
    garbage = b"not-an-mp4-at-all"
    os.environ.pop("EVAL_FRAME_CACHE_DIR", None)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as cd:
        def ncache():
            return sorted(f for f in os.listdir(cd) if f.endswith(".npy"))

        # 1. OFF → no writes
        a = decode_video_bytes(mp4, td, "sec/vid/clip.mp4", nf)
        assert a is not None, "decode returned None on a valid synthetic mp4"
        assert a.dtype == torch.uint8 and a.shape[0] == nf, f"bad decode {a.shape}/{a.dtype}"
        assert ncache() == [], "cache OFF must not write anything"

        # 2. ON → MISS writes one keyed file, identical to the OFF decode
        os.environ["EVAL_FRAME_CACHE_DIR"] = cd
        os.environ["EVAL_FRAME_CACHE_MIN_FREE_GB"] = "0.001"
        b = decode_video_bytes(mp4, td, "sec/vid/clip.mp4", nf)
        assert torch.equal(a, b), "MISS decode must equal OFF decode (determinism)"
        assert ncache() == ["sec_vid_clip__nf16.npy"], f"MISS wrote {ncache()}"

        # 3. HIT → ignores the garbage bytes, returns byte-identical cached frames
        h = decode_video_bytes(garbage, td, "sec/vid/clip.mp4", nf)
        assert torch.equal(a, h), "HIT must return cached frames without decoding"

        # 4. different num_frames → separate entry
        _ = decode_video_bytes(mp4, td, "sec/vid/clip.mp4", 8)
        assert "sec_vid_clip__nf8.npy" in ncache(), "num_frames must key a separate entry"

        # 5. free-disk floor blocks NEW stores but decode still succeeds
        os.environ["EVAL_FRAME_CACHE_MIN_FREE_GB"] = "9999999"
        before = ncache()
        g = decode_video_bytes(mp4, td, "other/key/x.mp4", nf)
        assert g is not None, "decode must succeed even when the disk floor blocks the store"
        assert ncache() == before, "free-disk floor must skip the store"

    print("ALL EVAL-FRAME-CACHE TESTS PASSED")


if __name__ == "__main__":
    main()
