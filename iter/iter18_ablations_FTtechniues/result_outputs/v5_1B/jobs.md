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

## 🧮 167-job derivation — per-encoder matrix

> Every kept encoder gets the **same 11-metric eval** (`E:`1 + `P:`6 + `F:`4). The 13 encoders that aren't the
> frozen base ALSO get **one `T:` job** (10 GPU-trains + 3 CPU WiSE-FT merges). `167 = 13 T: + 154 eval`.
> Verified from `iter18_poc_ngpu.py --mode POC --dry-run --skip-arms $SKIP` (4 dependency waves).

| # | 🎛️ encoder | class · `T:` kind | 🏋️ `T:` | 🟦 `E:` | 🟨 `P:` | 🟪 `F:` | Σ row |
|:--:|---|---|:--:|:--:|:--:|:--:|:--:|
| 1 | 🧊 `frozen` | COMP · — (base model, no train) | 0 | 1 | 6 | 4 | **11** |
| 2 | `pretrain` ♻️ | COMP · GPU-train (seed, resume-skipped) | 1 | 1 | 6 | 4 | **12** |
| 3 | `autorgn` | COMP · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 4 | 🟢 `raw` | COMP · GPU-train (surgery-on-raw ablation) | 1 | 1 | 6 | 4 | **12** |
| 5 | `full_ft` | COMP · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 6 | `lpft` | COMP · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 7 | `peft_lora` | COMP · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 8 | `peft_dora` | COMP · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 9 | 🔬 `flagship` | OUR · GPU-train (`surgery_3stage_DI`) | 1 | 1 | 6 | 4 | **12** |
| 10 | 🚀 `intervene` | OUR · GPU-train (WiSE-FT base) | 1 | 1 | 6 | 4 | **12** |
| 11 | ⬆️ `diheavy` | OUR · GPU-train | 1 | 1 | 6 | 4 | **12** |
| 12 | ✨ `wiseft_f30` | OUR · CPU merge (α=0.3, free) | 1 | 1 | 6 | 4 | **12** |
| 13 | ✨ `wiseft_f50` | OUR · CPU merge (α=0.5, free) | 1 | 1 | 6 | 4 | **12** |
| 14 | ✨ `wiseft_f70` | OUR · CPU merge (α=0.7, free) | 1 | 1 | 6 | 4 | **12** |
| **Σ** | **14 encoders** | 10 GPU-train · 3 merge | **13** | **14** | **84** | **56** | **167** |

> Row math: frozen `0+1+6+4 = 11`; each of the other 13 `1+1+6+4 = 12` → `11 + 13×12 = 11 + 156 = `**`167`**.

## 🧮 the 167 — arithmetic rollup

| 🏷️ | what | formula | jobs |
|:--:|---|---|:--:|
| 🏋️ `T:` | train/merge — all 14 enc **minus** frozen | 10 GPU-train + 3 WiSE-FT merge | **13** |
| 🟦 `E:` | encoder eval (features + action-top1 + taxonomy-F1 + motion-cos + future-MSE) | 14 × 1 | **14** |
| 🟨 `P:` | predictor_temporal (rollout·causal·tdist·teacher_free·maskratio·order) | 14 × 6 | **84** |
| 🟪 `F:` | encoder_temporal (aot·tov·pace·tcc) | 14 × 4 | **56** |
| 📊 | **eval subtotal** | 14 + 84 + 56 | **154** |
| 🟰 | **GRAND TOTAL** | 13 + 154 | **167** |

> vs **full DAG = 263** (22 enc · 21 `T:` + 242 eval) → **−96 jobs (−37%)**. Real GPU cost is even lower: of
> the 13 `T:`, `pretrain` is **resume-skipped** (seed exists) and the 3 wiseft are **CPU merges** → only **9
> GPU-trains** actually run. `🏁 §3 finale` (paired-Δ + m13 plots) runs after the 167 settle.
