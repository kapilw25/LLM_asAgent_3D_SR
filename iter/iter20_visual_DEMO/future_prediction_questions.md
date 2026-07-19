# 🎬 iter20 — MOTION & PREDICTION quiz questions (VQA demo, OURS vs FROZEN)

| 🎯 | why the pivot (2026-07-15) |
|---|---|
| ask **only what OURS actually wins** — MOTION & PREDICTION — not scene/semantic | OURS **loses 0/15 scene-attribute** questions vs FROZEN (`probe_taxonomy`: scene_type 0.66 vs 0.71 …). Surgery specialized OURS for **motion/temporal**, trading static-scene reading. So the quiz must ask **how things MOVE** and **what happens NEXT**, where OURS is measurably better ([[project_iter20_demo_cosmos_impossible]]). |

## 📊 Ground truth — where OURS beats FROZEN (effect size = % of FROZEN mean, `forest_plot_frozen_mean`)

```text
OURS WINS ✅                                     OURS LOSES ❌ (do NOT quiz these)
────────────────────────────────────────        ──────────────────────────────────
motion-cosine separation      +1432.8%           arrow-of-time            −2.0%
mask-ratio robustness           +27.4%           rollout drift            −3.3%
action top-1 (motion class)     +16.2%           TCC Kendall τ            −5.1%
free-running exposure-bias      +13.3%           TCC cycle-back          −31.6%
future-frame MSE                +10.6%           L1-vs-Δt decay          −34.0%
causal future-block L1           +8.9%
playback-pace                    +5.9%
```

## 📋 The quiz questions (each maps to a metric OURS wins · honesty flag on verifiability)

```text
┌────┬──────────────────────────────────────────────────────────┬────────────────────────┬────────┬────────────────┐
│ #  │ QUIZ QUESTION (a layman reads it off the clip)           │ maps to (OURS win)      │ OURS Δ │ layman-verify? │
├────┼──────────────────────────────────────────────────────────┼────────────────────────┼────────┼────────────────┤
│    │ 🏃 MOTION — does the model perceive HOW things move                                                          │
│ 1  │ Which clip MOVES most like this one? (pick the match)    │ motion-cosine sep.      │+1432%  │ 🟡 partial ⭐   │
│ 2  │ How much MOTION do you SEE? still / slow / medium / fast │ visible-motion (frame-Δ)│ +3pp*  │ ✅ yes          │
│ 3  │ What's the motion PATTERN? (speed × direction)          │ action top-1            │ +16%   │ 🟡 partial      │
│ 4  │ Can you still judge the motion with 90% of it HIDDEN?    │ mask-ratio robustness   │ +27%   │ ✅ yes ⭐       │
├────┼──────────────────────────────────────────────────────────┼────────────────────────┼────────┼────────────────┤
│    │ 🔮 PREDICTION — does the model anticipate the FUTURE                                                         │
│ 5  │ What does the NEXT frame look like? (predict it)        │ future-frame MSE        │ +11%   │ 🟡 (decoder=fog)│
│ 6  │ Reconstruct the hidden 2nd HALF from the 1st half        │ causal future-block     │ +9%    │ 🟡 (decoder=fog)│
│ 7  │ Predicting from its OWN guesses — who drifts LESS?       │ exposure-bias gap       │ +13%   │ ⬜ technical    │
│ 8  │ Is this clip NORMAL speed or SPED-UP (2× / 4×)?         │ playback-pace           │ +6%    │ ✅ yes ⭐       │
├────┼──────────────────────────────────────────────────────────┼────────────────────────┼────────┼────────────────┤
│    │ ⏱️ TEMPORAL — order & timing (weaker; test before use)                                                       │
│ 9  │ Are these frames in the RIGHT order, or shuffled?       │ temporal-order (tov)    │ verify │ ✅ yes          │
└────┴──────────────────────────────────────────────────────────┴────────────────────────┴────────┴────────────────┘
   ⭐ = best NEW demo candidate   *visible-motion is a gate-tested +3pp (weak but honest & eye-verifiable)
```

## 🥇 The 3 strongest HONEST demo candidates (ranked)

| rank | question | why it's the best bet | status |
|---|---|---|---|
| 1 | **#8 "Normal speed or sped-up?"** (pace) | genuinely **layman-verifiable** (sped-up video looks jerky/fast) AND a real OURS win (+6%) — and I have NOT tested it yet | 🚦 gate next |
| 2 | **#1 "Which clip moves like this?"** (motion-cosine) | OURS's **biggest** win (+1432%); the `m16` retrieval demo is already built | ⚠️ caveat: the win is a camera-subtracted flow fingerprint → partly sub-perceptual |
| 3 | **#4 "Judge motion with 90% hidden"** (mask-ratio) | OURS +27%, and "still recognisable under heavy masking" is a visible, intuitive robustness story | 🚦 gate |

## 🔗 Clip sourcing (unchanged — the tunnel works)

- YouTube download from THIS box is IP-blocked; use the **SSH reverse-SOCKS tunnel** (`ssh -N -R 1080 root@<box>` on your Mac) → `yt-dlp --proxy socks5://127.0.0.1:1080`. Already validated: 51 clips in `data/youtube_demo/`.
- For **pace (#8)** we don't even need YouTube — we can take any WalkIndia clip and make the "sped-up" version ourselves (frame-drop), so ground truth is exact and free.

> ⚠️ Honesty rule (unchanged): every question is **GATED** before it renders — OURS must beat FROZEN on a
> held-out probe, and the ground truth must be eye-verifiable (no repeat of the fast/still anti-correlation, VM30).
