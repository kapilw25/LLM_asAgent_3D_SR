# ⚡ GPU speedup ledger

> Every optimization in ≤2 lines [max 20 words epr line]. Line 1 = what it did. Line 2 (→) = the measured win.

---

## 🎬 Batched VLM tagging & embedding (m04–m08, Mar–Apr 2026)

**1. Batched `generate()`** — replaced sequential per-clip generate with one padded batched call plus adaptive sub-batching.
→ 0.09 → 1.70 clips/s · GPU 32% → 91% (18.3×).

**2. VRAM auto batch-sizer** — compute batch from live VRAM `(VRAM×80%−overhead)/per-clip`; geometric OOM backoff 64→1, grow/shrink at 65/85%.
→ one config scales across 24/40/80/96 GB cards, no hardcoded baselines.

**3. Producer/consumer pipeline** — CPU threads decode/preprocess N clips into a prefetch queue while the GPU runs generate.
→ keeps GPU fed; 4 batches buffered ahead.

**4. Local pre-download (m00d)** — download the 10K subset once to disk vs streaming 115K and filtering every step.
→ ~39h → ~6.5h · GPU idle 90% → 10%.

**5. CDN shard download (m00d v3)** — `hf_hub_download` per TAR shard instead of the throttled HF streaming API.
→ 10K clips in 23.8 min, no bandwidth throttle.

**6. m05c skip-deduped** — embed only the 5,105 dedup-unique clips, not all 10K.
→ ~2× less V-JEPA inference.

**7. Subsample 30K for CIs** — statistically-sufficient subset for bootstrap confidence intervals (common SSL practice).
→ ~3.8× on the m05c double-embedding; paper-safe.

**8. torch.compile (m05c)** — compile the frozen V-JEPA forward after `eval()`.
→ ~2× on the double-embed pass.

**9. Adapted-V-JEPA OOM fixes** — 16 not 64 frames, `sdp_kernel` nullcontext, fp16 input, compile warmup.
→ OOM at BS=176 → 15.3 clips/s, 115K in ~2.1h.

**10. m06 bootstrap vectorization** — Python for-loops → NumPy `np.take`/broadcasting/`cumsum` for per-clip metrics.
→ 113 min → ~5–15 min per encoder.

**11. Vectorized augmentation (m05c)** — 64 per-frame `F.interpolate` → one 4-D tensor op; removed ThreadPool ATen explosion.
→ 644 → ~20 threads, unstuck the 0% GPU stall.

**12. Anti-fragmentation** — `expandable_segments:True` + `cuda_cleanup()` + a fresh process per encoder.
→ kills the 64→1 batch death-spiral; VRAM stays stable.

---

## 🧪 iter19 eval pipeline (Jul 2026)

**13. DecodeFeeder (rung 1)** — N worker threads own decode→bounded ready_q, so decode never pauses during the forward.
→ square-wave GPU 17–75% gone · consumer wait 2000ms → 7ms.

**14. BatchFeeder (rung 2)** — builder thread stacks + pins batch k+1 while batch k forwards (depth-2 double buffer).
→ removes the residual idle troughs at the batch cadence.

**15. Upload-once dispatch (rung 3)** — `batch.to(cuda, non_blocking=True)` at the top; variant gathers then run on GPU.
→ pace 1.3 → 0.5 s/clip (×2.6); no 1-core CPU shuffling.

**16. Background ckpt writer** — moved m12f's GB-scale `torch.cat`+`np.savez` off the hot path to a join-guarded thread.
→ atomic publish unchanged; no per-checkpoint GPU stall.

**17. device_gather (on-device)** — upload feats once + `index_select` per chunk vs CPU-gather-then-`.to(cuda)`.
→ tcc pairwise 1h+ stall → 96s · frozen 3244s → 66s (~49×).

**18. h_full feature sharing** — reuse the encoder's full-sequence features across per-metric forwards instead of re-encoding.
→ eval 1.8×.

**19. Decode memoization** — bounded frame + tube LRU disk caches keyed by (clip, decode_T), 300G cap.
→ skip re-decode across processes and runs.

**20. CPU-loop vectorization** — tcc pair-enum via `np.triu_indices`; `temporal_token_idx` + `expand_mask` index arithmetic.
→ byte-identical; kills the python triple-loop + 30× smaller mask H2D.

**21. Rejected after measurement** — torch.compile (2B) and NVDEC (415 vs 395 ms on short clips) both measured slower.
→ documented as deliberately not taken; measure, don't assert.
