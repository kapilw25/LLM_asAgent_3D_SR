# 🗂️ 263 jobs = 🏋️ 21 train + 📊 242 eval

## 🏋️ A · 21 TRAIN jobs

| 🏷️ group | # | arms |
|---|:--:|---|
| 🌱 pretrain (SEED) | 1 | `pretrain_encoder` &nbsp; ✅ resume-skipped |
| 🔬 surgery novelty | 2 | `surgery_3stage_DI_encoder`, `surgery_noDI_encoder` |
| 🧪 surgery control | 1 | `surgery_raw_encoder` |
| 🎯 surgery heads | 2 | `surgery_3stage_DI_head`, `surgery_noDI_head` |
| 🚀 surgery improvements | 4 | `…_replay25`, `…_diheavy`, `…_tccaux`, `…_intervene` |
| ⚙️ autorgn baseline | 1 | `surgical_autorgn_encoder` |
| 🔩 FT baselines | 6 | `full_ft`, `lpft`, `peft_lora`, `peft_dora`, `cassle`, `ewc` |
| 🔀 wiseft merges (no GPU) | 4 | `3stage_DI_wiseft`, `intervene_wiseft_{f30,f50,f70}` |
| **Σ TRAIN** | **21** | |

## 📊 B · 11 metric-jobs PER encoder

| 🏷️ tag | # | stage / suite | metrics |
|:--:|:--:|---|---|
| 🟦 `E:` | 1 | stages 2·3·11·5·6·8 | features + action-top1 + taxonomy-F1 + motion-cos + future-MSE |
| 🟨 `P:` | 6 | 8b predictor_temporal | rollout · causal · tdist · teacher_free · maskratio · order |
| 🟪 `F:` | 4 | 8c encoder_temporal | aot · tov · pace · tcc |
| **Σ** | **11** | | per encoder |

## 🎬 C · 22 eval encoders

> 🧊 `frozen` ➕ the 21 trained arms from **A** — all named `vjepa_2_1_vitg_<arm>`

| # | encoder | # | encoder |
|:--:|---|:--:|---|
| 1 | 🧊 `frozen` | 12 | ⚙️ `surgical_autorgn_encoder` |
| 2 | 🌱 `pretrain_encoder` | 13 | 🔩 `full_ft_encoder` |
| 3 | 🔬 `surgery_3stage_DI_encoder` | 14 | 🔩 `lpft_encoder` |
| 4 | 🔬 `surgery_noDI_encoder` | 15 | 🔩 `peft_lora_encoder` |
| 5 | 🧪 `surgery_raw_encoder` | 16 | 🔩 `peft_dora_encoder` |
| 6 | 🎯 `surgery_3stage_DI_head` | 17 | 🔩 `cassle_encoder` |
| 7 | 🎯 `surgery_noDI_head` | 18 | 🔩 `ewc_encoder` |
| 8 | 🚀 `…_replay25_encoder` | 19 | 🔀 `…_3stage_DI_wiseft_encoder` |
| 9 | 🚀 `…_diheavy_encoder` | 20 | 🔀 `…_intervene_wiseft_f30_encoder` |
| 10 | 🚀 `…_tccaux_encoder` | 21 | 🔀 `…_intervene_wiseft_f50_encoder` |
| 11 | 🚀 `…_intervene_encoder` | 22 | 🔀 `…_intervene_wiseft_f70_encoder` |

## 🧮 D · 242 EVAL jobs = 22 encoders × 11

| 🏷️ tag | per-encoder | total |
|:--:|:--:|:--:|
| 🟦 `E:` | 22 × 1 | 22 |
| 🟨 `P:` | 22 × 6 | 132 &nbsp; *(6 frozen ✅ done)* |
| 🟪 `F:` | 22 × 4 | 88 |
| **Σ EVAL** | | **242** |

## 🏁 GRAND TOTAL

| 🏋️ train | 📊 eval | 🟰 total | ➕ finale |
|:--:|:--:|:--:|---|
| 21 | 242 | **263** | §3 paired-Δ + m13 plots (after all 263 settle) |
