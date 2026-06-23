# 🗂️ 1B run roster — 💰 167 jobs (money-saver; full DAG = 263)

> Picked from the **2B results** (`eval_metrics`): keep the COMPETITORs + the **hero-covering** OURs, drop the
> redundant / non-hero arms. **NOT overtuned** — we reproduce the existing winners, we don't double-down a
> within-CI knob (inverted-U / overtuning risk — see the runbook audit).

## 🏆 OURs kept — by 2B hero metric

| keep | OUR | role | 🥇/🥈 2B hero metric |
|:--:|---|---|---|
| ✅ | 🔬 `surgery_3stage_DI_encoder` | **FLAGSHIP** (base / Δ-anchor) | — (the baseline every Δ is measured against) |
| ✅ | 🚀 `surgery_3stage_DI_intervene_encoder` | best improvement + wiseft base | 🥇 future-MSE · 🥈 causal-L1 |
| ✅ | ⬆️ `surgery_3stage_DI_diheavy_encoder` | improvement | 🥇 mask-ratio slope |
| ✅ | ✨ `surgical_intervene_wiseft_f30/f50/f70` | eval-only merge (FREE, no train) | 🥇 taxonomy·AoT·ToV·TCC · 🥈 rollout·t-dist·teacher-free·pace |
| ❌ | `surgery_noDI` · `…_tccaux` · `…_replay25` | non-hero / within-CI redundant | — (dropped: no distinct win) |

## ✅ KEPT roster — 14 encoders

| | 🏋️ Trained & Eval | 👁️ Eval-only |
|---|---|---|
| 🆚 **COMPETITORs** (8) | `pretrain`*, `autorgn`, `raw`, `full_ft`, `lpft`, `peft_lora`, `peft_dora` (7) | 🧊 `frozen` (1) |
| 🏆 **OURs** (6) | `flagship` · `intervene` · `diheavy` (3) | `wiseft_f30` · `wiseft_f50` · `wiseft_f70` (3, merge) |

> `*` `pretrain_encoder` = ✅ resume-skipped (1B seed already exists & matches the POC recipe)

## ❌ DROPPED via `--skip-arms` (8 arms)

| 🚫 always-skip (5) | 🚫 non-hero surgery (3) |
|---|---|
| `surgery_3stage_DI_head`, `surgery_noDI_head`, `cassle_encoder`, `ewc_encoder`, `surgical_3stage_DI_wiseft_encoder` | `surgery_noDI_encoder`, `surgery_3stage_DI_tccaux_encoder`, `surgery_3stage_DI_replay25_encoder` |

## 📊 11 metric-jobs PER encoder

| 🏷️ tag | # | stage / suite | metrics |
|:--:|:--:|---|---|
| 🟦 `E:` | 1 | stages 2·3·11·5·6·8 | features + action-top1 + taxonomy-F1 + motion-cos + future-MSE |
| 🟨 `P:` | 6 | 8b predictor_temporal | rollout · causal · tdist · teacher_free · maskratio · order |
| 🟪 `F:` | 4 | 8c encoder_temporal | aot · tov · pace · tcc |
| **Σ** | **11** | | per encoder |

## 🧮 job count

| roster | 🏋️ train | 📊 eval (×11) | 🟰 total |
|---|:--:|:--:|:--:|
| 💰 **1B kept** | 13 | 14 × 11 = **154** | **167** |
| (full DAG) | 21 | 242 | 263 |

> → **~37% fewer jobs**; GPU-training drops from ~16 → ~9 arms (`pretrain` resume-skipped in both). `🏁 §3 finale` (paired-Δ + m13 plots) runs after the 167 settle.
