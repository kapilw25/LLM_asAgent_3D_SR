# 🧬 Why Surgery Loses on V-JEPA 2.0 (and Wins on Both 2.1 Backbones)

> 🧪 **Controlled comparison:** `2.1-vitg` vs `2.0-vitg` — identical arch (1B · 40 blk · 1408-dim),
> identical recipe (480 steps · 7021 clips · bs 32 · lr-peak 5e-5), identical data.
> Only the pretrained **checkpoint** differs → the cause is the **base weights**, not the recipe.

## 🎯 TL;DR

```text
┌─ 🎯 item ───────────────┬─ finding ──────────────────────────────────────────────┐
│ 🧪 controlled pair      │ 2.1-vitg vs 2.0-vitg: same arch/recipe/data, diff base  │
│ 🧬 root cause           │ the V-JEPA 2.0 pretrained CHECKPOINT (not recipe/arch)  │
│ 🩺 surgery training     │ HEALTHY on 2.0 (loss dropped, normal drift, no NaN)     │
│ 🔬 failure mode A       │ 2.0 encoder too FRAGILE -> surgery < frozen on 5 metrics│
│ 💪 failure mode B       │ 2.0 motion PLASTIC -> 2x raw compute beats surgery       │
│ 📜 paper framing        │ scope boundary: surgery wins where base dynamics good    │
└─────────────────────────┴──────────────────────────────────────────────────────────┘
```

## 📂 Evidence read (no guessing — every file parsed)

```text
┌─ 📄 source ─────────────────────────────┬ count ┬ used for ──────────────────────────┐
│ surgery training_summary.json           │  12   │ loss / probe-traj / drift / best-ckpt│
│ surgery loss_log.csv                    │  12   │ per-step loss_jepa, lr, grad_norm   │
│ surgery block_drift_history.json        │  12   │ per-block rel-L2 drift              │
│ pretrain folders (m09a)                 │   3   │ pretrain_2X / encoder / head        │
│ eval aggregate JSONs (m13 inputs)       │   5   │ action/motion/future/taxonomy/predict│
└─────────────────────────────────────────┴───────┴──────────────────────────────────────┘
```

## 🧊 Root cause — 2.0 frozen base is older, weaker, more plastic

```text
┌─ 🧊 FROZEN base ────┬ arch  ┬ action ┬ motion ┬ future(LO) ┬ causal(LO) ┬ taxonomy ┬ 🏅 ┐
│ 2.1-vitG [2B]       │ 48blk │  44.38 │ 0.0091 │   0.5571   │   0.5831   │  0.7934  │ 🥇 │
│ 2.1-vitg [1B]       │ 40blk │  42.41 │ 0.0070 │   0.6365   │   0.6551   │  0.8062  │ 🥈 │
│ 2.0-vitg [1B]       │ 40blk │  38.36 │ 0.0076 │   1.6455   │   1.6562   │  0.7782  │ 🔴 │
└─────────────────────┴───────┴────────┴────────┴────────────┴────────────┴──────────┴────┘
🔴 2.0 future/causal ~2.6x WORSE — independently confirmed by training probe (future_l1 ~1.72 vs ~0.63).
```

## 🔬 Mechanism A — 2.0 encoder too FRAGILE to adapt

```text
┌─ 🥶 best-surgery WORSE than its OWN frozen? ┬ # metrics ┬ which ─────────────────────────┬ 🚦 ┐
│ 2.1-vitG [2B]                               │     1     │ taxonomy (0.0001 = noise)      │ 🟢 │
│ 2.1-vitg [1B]                               │     2     │ taxonomy, tdist (tiny)         │ 🟢 │
│ 2.0-vitg [1B]                               │     5     │ taxonomy future causal tf mask │ 🔴 │
└─────────────────────────────────────────────┴───────────┴────────────────────────────────┴────┘
```

```text
┌─ 2.0-vitg metric ┬ 🧊 frozen ┬ 🧠 pretrain_HEAD ┬ 🔪 best surgery ┬ 🏆 ┐
│ future  (LO)     │  1.6455   │ 1.6458 (≈frozen) │     1.7153      │ 🔴 P │
│ causal  (LO)     │  1.6562   │ 1.6562 (=frozen) │     1.7186      │ 🔴 P │
└──────────────────┴───────────┴──────────────────┴─────────────────┴──────┘
👉 On 2.0 EVERY encoder-adapting arm degrades these — surgery 1.715 < pretrain_enc 1.756
   < pretrain_2X 1.784 (surgery is the LEAST damaging) — yet all lose to pretrain_HEAD,
   which wins precisely because it keeps the encoder FROZEN.
👉 On 2.1 the SAME adaptation HELPS: surgery future 0.628 < frozen 0.637. Touch 2.0 = hurt; touch 2.1 = help.
```

## 💪 Mechanism B — 2.0 MOTION plasticity captured by brute-force 2x compute

```text
┌─ motion_cos ─┬ 🧊 frozen ┬ pretrain_enc(1x) ┬ pretrain_2X(2x) ┬ 🔪 surgery(1x) ┬ 🏆 ┐
│ 2.1-vitg     │  0.0070   │     0.0420       │     0.0444      │     0.0627     │ 🟢 S │
│ 2.0-vitg     │  0.0076   │     0.1130       │     0.2283      │     0.1263     │ 🔴 P │
└──────────────┴───────────┴──────────────────┴─────────────────┴────────────────┴──────┘
👉 2.0 @ MATCHED 1x compute: surgery 0.126 ≈ pretrain_enc 0.113 (surgery slightly AHEAD).
   It only loses to pretrain_2X (964 vs 480 steps) — 2x raw SSL converts 2.0 plasticity to
   motion near-linearly (0.113 -> 0.228).
👉 2.1 is motion-SATURATED (2x compute 0.042 -> 0.044 does nothing) so surgery's structure wins.
```

## 🩺 Surgery training was HEALTHY on 2.0 (not a bug / divergence)

```text
┌─ 🩺 health check  (2.0 surgery_3stage_DI_encoder) ┬ value ───────────────┬ 🚦 ┐
│ loss_jepa  start -> end                           │ 0.5386 -> 0.4774     │ 🟢 │
│ loss drop (LARGEST of the 3 backbones)            │ 0.0612               │ 🟢 │
│ block-drift mean (end)                            │ 0.0034 (normal)      │ 🟢 │
│ probe top1  start -> peak -> end                  │ 0.337 -> 0.355 -> .346│ 🟢 │
│ early-stop triggered / NaN                        │ no / no              │ 🟢 │
│ action vs frozen (surgery DOES help here)         │ 50.25 vs 38.36       │ 🟢 │
└───────────────────────────────────────────────────┴──────────────────────┴────┘
```

## 🏆 Verdict per backbone

```text
┌─ backbone ────────┬ base quality ───────────┬ mean-gap (S/P) ┬ paired-CI (S/P/tie) ┬ 🏆 ┐
│ 2.1-vitG [2B]     │ best, adaptation-friendly│     4 / 2      │      4 / 0 / 5      │ 🥇 │
│ 2.1-vitg [1B]     │ good                     │     5 / 4      │      5 / 2 / 2      │ 🥇 │
│ 2.0-vitg [1B]     │ old / fragile / plastic  │     1 / 6      │      0 / 5 / 4      │ 🔴 │
└───────────────────┴──────────────────────────┴────────────────┴─────────────────────┴────┘
```

## 📜 Paper framing

```text
┌─ 📜 takeaway ───────────────┬ statement ──────────────────────────────────────────────────┐
│ ❌ NOT a bug                │ surgery trained cleanly on 2.0; the result is real           │
│ 🎯 scope boundary           │ surgery > pretrain holds where pretrained DYNAMICS are good  │
│ 🧊 2.0 optimal moves        │ (a) DON'T touch encoder [predictor]  (b) SPEND raw compute [motion]│
│ 🔪 surgery by design        │ adapts encoder @ 1x compute — neither 2.0-optimal move       │
│ 💡 modern regime            │ 2.1 = adaptation-friendly = exactly where world-models live  │
└─────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

---
🔁 **Reproduce:** the 5 m13-input JSONs (`outputs/poc/{probe_action,probe_motion_cos,probe_future_mse,probe_taxonomy,predictor_temporal}/`) + the 12 surgery folders (`outputs/poc/<backbone>/m09c_surgery_*`). Numbers above are direct reads, not estimates.
