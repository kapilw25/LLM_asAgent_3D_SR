# 🌐 An open-system diagnostic benchmark for **open-weight world models**

> 🔁 **Pivot.** This iteration stops being *"our surgery arm is best"* and becomes a **benchmark**:
> a **generation-free, probe-based diagnostic** that scores any **open-weight world model** on **15 metrics**,
> and publishes a **leaderboard** the field's checkpoints can join. **Our surgery / fine-tuning arms become
> reference rows on that leaderboard — not the headline.** The disjoint **fresh 10k** is the benchmark's **held-out test split**.

| | 🧪 Method paper (old) | 🌐 Benchmark (new) |
|---|---|---|
| **Subject under test** | our ~17 surgery / FT arms | **the field's open-weight WM checkpoints** (our arms = reference points) |
| **The claim** | *"surgery ≫ fine-tuning"* | *"here is a standardized diagnostic + a leaderboard"* |
| **The artifact** | a win | the **protocol + 15 metrics + applicability matrix + leaderboard** |
| **Adding a model** | retrain an arm | **inference only** — run its open encoder (+ open predictor) + fit a tiny linear probe · **no training** |

---

## ✅ Task checklist — ordered by ROI for AAAI (highest first = least effort × most acceptance)

> Legend: ✅ done · 🔧 code-done, data-blocked · ⬜ pending (cheap, do next) · ⛔ blocked (needs a GPU / external run)

| # | task | effort | status |
|---|---|---|---|
| 1 | 🔴 **Fix the 4 construction bugs** (probe-consistency · dup-row guard · relabel **concurrent** · metadata dedup) — *see ⬇ Critical fixes* | tiny | 🔧 2 done · 2 plan-level |
| 2 | **All 15 metrics** reported · `order` reported-not-ranked | tiny | ✅ |
| 3 | Validity **heatmap v1** (tooling, verdict pending v2) + external-anchor plot **code**, auto-refresh at real path | tiny | ✅ / 🔧 |
| 4 | **FDR per board** over valid (non-N/A) cells — averts "false-positive wins" reject | small | ⬜ |
| 5 | **Leakage audit** = source-video/clip-ID **+ ±30s metadata** disjointness | small | ⬜ |
| 6 | **Cross-domain panel** — Walking Tours, label-free metrics (kills "one domain") | medium | ⬜ |
| 7 | Wire **external WMs** → the **8-metric encoder board** (🔬 de-risk **DONE**: VideoMAE loads, fix-10) | high | ⛔ loader + eval |
| 8 | Construct-validity **v2** over the full board (real within≫between verdict) | high | ⛔ needs externals |
| 9 | **Concurrent-validity DATA** — SSv2/Kinetics attentive probe → `criterion.csv` | medium | ⛔ GPU run |
| 10 | ~~1B backbone retrain~~ → **CUT to method-appendix** (fix-7); benchmark's 1B = an **eval-only row** | — | ✂️ demoted |
| 11 | **CI-tightening** — reuse-heads cross-set full-10k eval | medium | ⬜ |

## 0️⃣ Build status — the validity study (this session)

| piece | status |
|---|---|
| ✅ **heatmap (v1) — tooling** | built · at the real location · regenerates with `poc_status --plots` · ⚠️ verdict **pending v2** (16 one-model arms = under-powered, fix-8) |
| ✅ **external-anchor plot (code)** | BUILT + wired (guarded) + smoke-tested: selftest ρ=0.94 [0.87,0.97] · synthetic ρ=0.72 · missing-file → clean SKIP |
| ⛔ **external-anchor DATA** | needs `criterion.csv` from a GPU probe (SSv2 = **convergent**; a downstream task = **true criterion**, fix-6) — **your run**. Until it exists, the pipeline **SKIPS** it (no-op) |
| ⛔ **v2 leaderboard** | needs external WM checkpoints scored (`load_external_wm` + eval) |

## 🔧 Critical fixes (verified + websearch-grounded — no hand-waving)

| # | 🐛 issue (reviewer-killer) | ✅ fix (grounded) | state |
|---|---|---|---|
| 🔴1 | **Probe mismatch** would rig the board | ✔ Verified our action/taxonomy probes **already use `AttentiveClassifier`** (`m12a`/`m12c`) — *not* linear. Real fix: score **externals** with the *identical* attentive head + recipe → **one probe for every row** (don't re-score our arms). | ⬜ at external-scoring |
| 🔴2 | **Fake-perfect correlations** from by-construction duplicate rows | ✔ Verified on the live matrix: **max \|off-diag ρ\| = 0.97, zero fake-perfect cells** (`*_head`/empty rows already dropped). Added an **explicit exact-duplicate-row guard** → stays clean as the board grows. | ✅ done |
| 🟠3 | **Mislabeled "criterion validity"** | ✔ Websearch: our metric vs an external **gold-standard of the SAME construct, same time = CONCURRENT validity** (a criterion subtype) — *not* reliability (repeated test), *not* predictive-criterion. **Relabeled the figure → "concurrent validity (criterion-related)".** True predictive-criterion would need a human-rating / downstream anchor (optional). | ✅ done (label) |
| 🟡4 | **FAISS-0.90 near-dup floods** on walking video | ✔ Websearch + our own rule: content cosine false-fires on same-camera consecutive clips. Audit disjointness on **source-video ID + ±30 s metadata** (matches CLAUDE.md *"hard-mode ±30s exclusion is metadata-based"*), **not** frame embeddings. §4c step-2 → optional (cross-source only, high threshold + rationale). | ⬜ at §4c |

> 📚 Fix sources: [concurrent vs criterion vs convergent validity (Scribbr)](https://www.scribbr.com/frequently-asked-questions/difference-convergent-concurrent-validity/) · [concurrent validity (Wikipedia)](https://en.wikipedia.org/wiki/Concurrent_validity) · [perceptual-hash vs embedding video dedup (MDPI)](https://www.mdpi.com/2079-9292/15/7/1493).

### 🔧 Critical fixes — Round 2 (verified + websearch-grounded)

| # | 🐛 mistake | ✅ fix | state |
|---|---|---|---|
| 🔴5 | **Self-contradiction:** caveats still promised *"max cross-set cosine < 0.90"* after §4c **deleted** the embedding audit | Report **only "0 shared clip IDs + 0 same-source-video within ±30 s"** (metadata) — deleted the cosine promise. | ✅ done |
| 🟠6 | **`act` vs SSv2 = near-circular** — both are attentive **action** probes → ρ high for the *trivial* reason (same construct + method) | Websearch: same-construct correlation = **CONVERGENT** validity (weakest evidence), **not** criterion. **Relabel act-vs-SSv2 → convergent.** True **criterion** = anchor a **predictor metric (`fut`/`causal`) to a DOWNSTREAM task** (different construct). | ✅ relabel · ⬜ criterion-anchor |
| 🟠7 | **1B retrain ≈ 0 benchmark value** — proving "surgery ≫ FT on a 2nd backbone" is a **method** result, the most expensive task for the least benchmark credit | **Cut the 1B 9-arm retrain from the benchmark** → **method-paper appendix**. The benchmark's "2nd backbone" is just **V-JEPA 2.0 1B as an eval-only row** (§1). | ✅ demoted |
| 🟡8 | **v1 heatmap verdict over-reaches** — within≫between on **16 one-model arms** is near-unfalsifiable | **Report v1 ρ but label the verdict "PENDING external encoders (v2)".** Never claim *validated families* off the 16-arm matrix. | ✅ relabel |

> 📚 Round-2 sources: [convergent validity (SimplyPsychology)](https://www.simplypsychology.org/convergent-validity-definition-and-examples.html) · [representation metrics predict downstream perf (arXiv 2205.07477)](https://arxiv.org/abs/2205.07477) · [criterion vs construct validity (Conjointly KB)](https://conjointly.com/kb/measurement-validity-types/).

### 🔧 Critical fixes — Round 3 (verified + proven)

| # | 🐛 mistake | ✅ fix | state |
|---|---|---|---|
| 🔴9 | **Bottom-line "still owed" still listed the 1B retrain** — contradicts fix-7 (cut everywhere else) | End the owed-list at ④ cross-domain + the `load_external_wm` loader; the **1B retrain is appendix, not owed.** | ✅ done |
| 🟠10 | **The whole paper assumes externals LOAD** — nothing proved a non-V-JEPA model yields a clean frozen encoder; if not, the board collapses to *"V-JEPA vs itself"* and there is no benchmark | **De-risked END-TO-END (this session):** `VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')` (non-V-JEPA masked-video-AE) → **loads + a clean `(1,768)` frozen feature in 7 s** on this box. ⚠️ HF 5.5.4 flagged a few attention-bias params newly-init → **pin `qkv_bias`/config** in the real `load_external_wm`. **The cross-model board is proven feasible, not assumed.** | ✅ proven |

> 📚 Round-3 sources: [VideoMAE (HF transformers)](https://huggingface.co/docs/transformers/model_doc/videomae) · [InternVideo2 (OpenGVLab)](https://huggingface.co/OpenGVLab/InternVideo2_Chat_8B_InternLM2_5).

---

## 🧩 §1 — Two leaderboards — and **which one IS the benchmark**

> 🏆 **The benchmark contribution = the 8-metric ENCODER board** — a **broad, cross-model** leaderboard of real open-weight WMs, scored by **frozen-encoder probing** (the accepted fair cross-architecture protocol). **Lead the paper with this.**
> 🔬 **The 15-metric *predictor* board is SECONDARY** — with **DINO-WM restrained to encoder-only** (its predictor adapter is a validity landmine, §-pt 3), the predictor board has **no external model left** → it is **V-JEPA-family + our arms only**: a **within-family deep-dive diagnostic, NOT a cross-model leaderboard.**
> 🔑 **Gate per board:** encoder board ⇒ open **encoder**; predictor board ⇒ open **encoder + masked-latent predictor**. Closed systems (Genie 3, Marble, Sora) → out by construction.

| 🤖 Model | 🔓 encoder open? | 🔓 predictor open? | 🎯 can score | priority |
|---|---|---|---|---|
| **V-JEPA 2.1 ViT-G (2B)** — home | ✅ | ✅ | 🟢 **all 15** | ⭐ base of our arms |
| **V-JEPA 2.0 ViT-g (1B)** | ✅ | ✅ | 🟢 **all 15** | ⭐ 2nd backbone |
| **V-JEPA 2 ViT-L / ViT-H** | ✅ | ✅ | 🟢 **all 15** | 🟢 easy extra rows (same family) |
| **V-JEPA (v1)** | ✅ | ✅ | 🟢 **all 15** | 🟢 easy extra row |
| **DINO-WM** (DINOv2 enc + future-patch predictor) | ✅ | ⛔ *adapter SKIPPED — validity landmine* | 🟡 **8 only** (restrained) | 🟢 encoder-board row (non-V-JEPA) |
| **NVIDIA Cosmos 3** (Predict-2.5, OpenMDW 1.1) | ✅ tokenizer | ❌ pixel generator | 🟡 **8 only** | ✅ accepted (encoder board) |
| **Microsoft WHAM / Muse** (`microsoft/wham`, 200M + 1.6B) | ✅ tokenizer | ❌ AR token gen | 🟡 **8 only** ⚠️ game-OOD | 🟢 encoder-board row |
| **Tencent HY-World 2.0** (`tencent/HY-World-2.0`) | ✅ | ❌ 3D recon/gen, not temporal | 🟡 **8 only** ⚠️ 3D-fit | 🟡 vet fit |
| **NVIDIA GR00T N1** (humanoid VLA, open) | ✅ vision enc | ❌ action gen | 🟡 **8 only** ⚠️ robot-OOD | 🟡 candidate |
| **Navigation World Model** (Meta, video+action) | ✅ | ❌ pixel gen | 🟡 **8 only** ⚠️ nav-OOD | 🟡 candidate |
| **DINOv2 · VideoMAE(v2) · InternVideo2** (video encoders) | ✅ | ❌ no predictor | 🟡 **8 only** | 🟢 strong encoder references |

> 🏆 **HEADLINE = the 8-metric encoder board (broad, cross-model):** V-JEPA family · DINO-WM · Cosmos · WHAM · GR00T · NWM · HY-World · strong video encoders + our arms — **a genuine cross-model leaderboard.**
> 🔬 **SECONDARY = the 7-metric predictor board (within-family):** only models with an open masked-latent predictor remain → **V-JEPA family + our arms.** Label it honestly as a **deep-dive diagnostic**, not a cross-model leaderboard.
> 🧱 **Probe protocol (fairness — cite it):** cross-architecture probing must be **fair** — *linear* probes **underestimate** masked-modeling encoders (V-JEPA, VideoMAE); **attentive probing** (freeze encoder + small attention head) is the field standard. **Document the probe; prefer attentive on the cross-model board.**
> ⚠️ **Domain caveat:** most 8-board models are **out-of-domain** for WalkIndia (game / robot / driving / 3D) → expect domain-shifted rows. That's an **informative result**, not a bug — and exactly what a cross-domain benchmark surfaces.
> 🚫 **Excluded (closed weights):** Genie 3 (DeepMind), Marble (World Labs), Sora, GAIA-2 (Wayve) — out of scope by construction (a *feature* of "open-system"). *(Vista, OpenViGA, LingBot-World = open driving-WMs → optional 8-board candidates.)*

---

## 🧾 §1b — Our reference rows: 16 arms on **V-JEPA 2.1 2B** (eval-only on `subset_10k_local`)

These are **already trained** on 2B — in the benchmark they're **reference rows**, scored **eval-only, no retraining**:

| 🥊 Competitors — non-ours (7) | 🏆 OURS — surgery family (9) |
|---|---|
| `pretrain_encoder` *(vanilla continual-SSL)* · `surgical_autorgn_encoder` · `surgery_raw_encoder` · `full_ft_encoder` · `lpft_encoder` · `peft_lora_encoder` · `peft_dora_encoder` | `surgery_3stage_DI_encoder` · `surgery_noDI_encoder` · `…_replay25_encoder` · `…_diheavy_encoder` · `…_tccaux_encoder` · `…_intervene_encoder` · `…_intervene_wiseft_f30` · `_f50` · `_f70` |

> ⚓ Plus **`vjepa_2_1_frozen`** = the anchor row. *(Dropped from the roster: the `*_head` arms — they duplicate the encoder on encoder-side metrics — and `cassle` / `ewc`, which are empty.)*

---

## 📐 §2 — The metrics: **15, not 9, not 14** + the `order` transparency fix

The iter18 JSON/CSV stores **15 metrics** (CSV = **32 cols** = 15 × (mean + ci) + `encoder` + `n_test`):

```text
act · tax · mcos · fut · rollout · causal · tdist · maskratio · order · teacher_free · aot · tov · pace · tcc_cycle · tcc_tau
```

### 🧱 Applicability matrix — which model can be scored on which metric (the construction-validity backbone)

| Tier | Metrics | Needs | Universal? |
|---|---|---|---|
| 🟦 **Tier E — encoder (8)** | `act` `tax` `mcos` · `aot` `tov` `pace` `tcc_tau` `tcc_cycle` | only a video→latent **encoder** | ✅ **any** open-weight WM with an encoder (V-JEPA, DINO-WM, Cosmos) |
| 🟥 **Tier P — predictor (7)** | `fut` `rollout` `causal` `tdist` `teacher_free` `maskratio` `order` | a **masked / future latent predictor** | ⚠️ JEPA-native; DINO-WM via adapter; **pixel generators (Cosmos) = N/A or adapter** |

> 🧷 A documented **model × tier** matrix with explicit **N/A** cells is what separates a *benchmark* from a *leaderboard hack* — it tells a reviewer exactly why a cell is empty (interface mismatch), not cherry-picking.

### 🚩 The `order` fix — report-or-justify every computed metric
**Verified in code:** `src/m13_eval_plot.py:614` → `continue  # signed (order): diagnostic-only → CSV, not drawn`; `:612` → `# CSV keeps ALL metrics incl signed 'order'`. So **`order` is computed + stored but silently dropped from the scorecard** (15 computed → 14 drawn).

- 🧪 **Method paper:** nobody cares — a hidden diagnostic is harmless.
- 🌐 **Benchmark paper:** a silently-dropped metric is a **construction-validity / transparency flag** — *"you computed 15, reported 14, why is one missing? cherry-picking?"*
- ✅ **Fix:** **report `order`** as a **signed diagnostic column** (outside the higher/lower-is-better ranking, since its sign — not magnitude — is the signal), with a one-line stated rationale ("signed frame-order reliance; reported, not ranked"). **Report it, or justify the cut in writing — never drop it in silence.**

---

## 🧪 §2b — Construct-validity study (**the single biggest missing piece**)

> ❓ A benchmark reviewer's hardest bar: *"does `causal` actually track causal understanding? does `act` track real action competence — or are these just **15 numbers of unknown meaning**?"* Today the metrics are **computed, not validated** — and this is **NOT yet on the "still owed" list.** It is now item **#1.** *(Messick / Kane construct validity; recent ML work [2510.23191](https://arxiv.org/abs/2510.23191), [2511.04703](https://arxiv.org/abs/2511.04703), [2603.15121](https://arxiv.org/abs/2603.15121).)*

Two studies — both **reuse data we already have**, so they're cheap:

| validity type | what it shows | how (reuses existing artifacts) |
|---|---|---|
| 🕸️ **Convergent / discriminant** (nomological net) | whether the 15 metrics cluster into the **3 declared families** | correlate the **arm × metric** matrix → heatmap. ⚠️ **v1 over 16 one-model arms is near-unfalsifiable (fix-8)** — report ρ but the **verdict is PENDING external encoders (v2)**; never claim *validated families* off the 16-arm matrix. **Near-free tooling, not yet evidence.** |
| 🎯 **Convergent + (true) criterion** (fix-6) | ⚠️ `act` vs an SSv2 **action**-probe = **CONVERGENT** only (same construct + method → ρ high *trivially*, near-circular — the **weakest** evidence). **True criterion** = anchor a **predictor metric (`fut`/`causal`) to a DOWNSTREAM task** of a *different* construct (the metric-predicts-downstream-perf design). | rank arms by the metric vs the external score → **Spearman ρ + 95% CI**. |

> ✅ **Minimum to ship:** the **convergent/discriminant heatmap** (reuses the matrix) **+ one criterion correlation**. Without it a reviewer dismisses the whole suite as "15 numbers of unknown meaning."

### 🔧 §2b build spec — the concrete `src/` task

```text
 eval_metrics.json  (≈16 arms × 15 metrics — ALREADY on disk, no GPU)
      │  orient every metric to "higher = better"  (flip the ↓ ones; 'order' is signed → set aside)
      ▼
 src/utils/validity.py     (🆕 pure-CPU reusable math · reuses utils/bootstrap)
      │  pairwise-complete Spearman corr (null-safe) + within/between-family summary + criterion ρ
      ▼
 src/m13_eval_plot.py      (✏️ owns the _CATALOG registry → passes family + direction IN)
      ▼
 m13_metric_validity.{png,pdf,csv}      +      m13_criterion_<metric>.{png,pdf,csv}
```

| 📄 file | new/edit | contents |
|---|---|---|
| `src/utils/validity.py` | 🆕 | `orient_higher_better(M, dirs)` · `pairwise_spearman(M)→(corr,n)` *(pairwise-complete — null cells from `cassle`/`ewc`/`*_head` don't crash it)* · `family_summary(corr, fam)→{within_ρ, between_ρ, gap, perm_p}` *(label-permutation test)* · `criterion_rho(x, y)→ρ ± BCa 95% CI` (via `utils/bootstrap`). + a CLI `main` for a standalone CSV smoke. **No `m*.py` import (rule 32)** — families + directions arrive as args. |
| `src/m13_eval_plot.py` | ✏️ | `plot_metric_validity(metrics_json, out_dir)` — build the arm×metric matrix, read **family + direction from `_CATALOG`** (the single metric-registry source), call `validity.py`, render **(a)** a family-ordered Spearman **heatmap** with the 3 family blocks outlined and **(b)** the **within-vs-between** headline number. `plot_criterion_validity(metrics_json, criterion_csv, metric_key, out_dir)` — rank-scatter + `ρ ± CI`. Wire both into the existing `_mw_dump` plot block (beside `plot_paper_scorecard`). |
| external-anchor input | 🆕 *(1 small eval)* | one external score per arm → `criterion.csv [encoder, ext_score]`. `act` vs SSv2 action-probe = **convergent** only (weak, fix-6). **For true criterion**, score a **downstream task** (different construct) and anchor a **predictor metric** (`fut`/`causal`) to it. The **only** piece needing extra compute; the heatmap needs none. |

**📐 Stats (honest).** Spearman (rank → scale/direction-robust); orient all to higher-better first · **N ≈ 16 arms** → individual cells noisy, headline = the 3-block clustering + within ≫ between gap + a label-permutation p · `order` (signed) **excluded** from orientation/ranking · **v1 = tooling over ≈16 one-model arms → NOT a validation; the verdict is PENDING (fix-8)** · **v2 = recompute over the FULL external leaderboard → the actual within≫between verdict.**

**✅ Smoke (CPU, no GPU).** `from utils.validity import pairwise_spearman, family_summary` on the existing `eval_metrics.csv` → prints `within_ρ / between_ρ / gap`; then `plot_metric_validity` on the v3 json → eyeball the 3 family blocks. **The entire heatmap half is GPU-free and reuses the matrix you already have** — only the one criterion anchor needs a small extra run.

---

## 🔬 §3 — QQ0 reframed: what the pivot does to each AAAI reviewer bite

| 🐛 Bite (method-paper) | 🗣️ Reviewer's line | 🔄 Under the **benchmark** pivot |
|---|---|---|
| 🎲 **Single seed** | *"Report ≥ 3 training seeds."* | 🟢 **Downgraded to a caveat** — a benchmark reports **score stability**, and eval-bootstrap over clips is the *right* uncertainty. No "beats baselines across seeds" claim → seed-variance stops being the blocker. |
| 🥊 **Model / FT coverage** (6 today) | *"Did you compare enough techniques?"* | 🟢 **Becomes a feature** — more models-under-test = better coverage. *"Only 2B"* is just *"current leaderboard rows,"* and **adding a model is inference, not retraining.** |
| 🧬 **Single backbone** | *"Does it hold on another backbone (1B ViT-g)?"* | 🟢 **Just another row.** The benchmark adds **V-JEPA 2.0 1B (eval-only)** + the whole external pool — coverage is **inference, not retraining**. *(Retraining our arms on 1B = a method claim → appendix, fix-7.)* |
| 🧮 **Multiple-comparison** | *"Some 'wins' are α=0.05 false positives."* | 🟡 **Required — and corrected RIGHT.** **Benjamini-Hochberg FDR per board** over the **actually-computed (model × metric) cells** (encoder board ≠ predictor board) — **not naive 15×N**: N/A cells lower the real count, naive correction **over-corrects + buries true effects.** |
| 🧪 **Metric validity** *(NEW)* | *"Do your 15 metrics measure what they claim — or are they 15 numbers of unknown meaning?"* | 🔴 **The hardest bar, and it was missing.** Ship a **construct-validity study** (§2b): convergent/discriminant heatmap + ≥1 criterion correlation. **#1 owed item.** |
| 🌍 **In-domain only** (all WalkIndia) | *"Does it hold off your clips?"* | 🟢 **Solved by reframing** — the contribution is the **protocol + metrics**; the disjoint 10k is the **held-out test**; *"all WalkIndia"* becomes a **stated domain-scope limit**, not a flaw. |
| 📉 **WiSE-FT overlaps baselines** | *"Not statistically significant."* | 🟢 **Becomes a result** — *"WiSE-FT overlaps baselines"* is a **legitimate reported diagnostic finding** in a benchmark; the fresh disjoint 10k just **hardens its stability.** |
| 📏 **Scale = 10k, not 115k** | *"Why only 10k?"* | ✅ **Declared scope** — the **held-out test split** of an openly-stated, compute-bounded protocol. |

> 🎯 **Punchline.** The pivot dissolves the **seed / coverage / domain / overlap** bites. What **actually remains**, in priority order: ① **construct-validity study** (§2b — the biggest, and it was *not* on the old list), ② **FDR per board** over valid cells, ③ the **leakage audit** (§4c — cheap), ④ a **cross-domain panel** (§5 — Walking Tours). *(The 1B retrain is **cut to a method-appendix**, fix-7.)* All scoped; **none needs a fresh 115k run.**

---

## 🛠️ §4 — What changes (all **eval-only** — the training scheduler is untouched)

```text
   TRAIN scheduler (iter18_poc_ngpu.py)          run_eval.sh
   run_train.sh / m09*                            │
   ┌─────────────────────────────┐                ├─ our 16 arms (2B)   → already trained → SCORE on subset_10k
   │   NOT TOUCHED — no new       │   ─── feeds ──▶├─ external WMs       → NEW: load + SCORE, NO training  ◀── add here ONLY
   │   training, no new arms      │                └─ (1B = an EVAL-ONLY row; 9-arm retrain = method-appendix, fix-7)
   └─────────────────────────────┘
```

### 🔌 4a — Wire external open-weight WMs into `scripts/run_eval.sh` **ONLY** (no training)
0. **🔬 DE-RISK FIRST (DONE, fix-10):** before building the full loader, smoke-load **one non-V-JEPA external** end-to-end. ✅ **Proven this session** — `VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')` → a clean `(1, 768)` frozen feature in **7 s** on this box, so the headline cross-model board is **feasible, not a hope**. *(Pin `qkv_bias`/config in `load_external_wm` — HF 5.5.4 flagged a few attention-bias params as newly-initialized.)* The cheapest externals to load are HF-`transformers` video encoders (**VideoMAE · VideoMAEv2 · InternVideo2**); Cosmos/WHAM/GR00T need custom loaders → vet next.
1. **`configs/arm_registry.yaml`** — add each external model as `kind: external`, **`scheduler: false`** (so it is **NOT** in the train DAG — run_eval-only), plus `hf_repo` + `loader` + `tiers: [E]` or `[E, P]`. *(Registry stays the single source — no model-ids hardcoded in `run_eval.sh`.)*
2. **`src/utils/load_external_wm.py`** — returns a frozen **encoder** (Tier E, always) and, where the weights ship one, an open **predictor** (Tier P). A *declared* missing predictor → the 7 predictor metrics emit a documented **`N/A`** (FAIL-LOUD only on *unexpected* missing, never on declared-N/A).
3. **`scripts/run_eval.sh`** — when `kind == external`, load via that loader instead of a trained `student_encoder.pt`, then run the **same 15-metric stages**; `N/A` the predictor tier when absent.
4. **UNCHANGED:** `run_train.sh`, `iter18_poc_ngpu.py`, every `m09*` trainer — **zero training touch.** ✅ matches *"run_eval.sh only, no training."*
5. **`src/m13`** — render `order` as a **reported-not-ranked** column + an **N/A-aware** applicability matrix panel (so an 8-only row shows blanks as *N/A*, not as losses).

### 🍴 4b — The 1B backbone — ⛔ CUT from the benchmark (method-appendix only, fix-7)
> 🧬 The benchmark's "2nd backbone" = **V-JEPA 2.0 1B as an EVAL-ONLY row** (§1), *no retrain*. The 9-arm retrain below (2 OURS + 7 competitors) proves *"surgery ≫ FT on a 2nd backbone"* = a **method** result — the **most expensive task for the least benchmark credit** → **method-paper appendix only.** **Step 1 (CI-tightening) still helps the benchmark; Steps 2–3 (the retrain) are appendix.**

**Step 1 — tighten the error bars first.** Today every arm shows `n_test = 1825` (only the **20 %** test slice of the 75/5/20 split) → the close OURS arms have **overlapping** CIs, so you *can't* cleanly pick 2.

> 🧨 **Honest catch:** a plain re-run on `subset_10k` re-splits it 75/5/20 and tests on **~2 k again → SAME wide bars** (this doc's own n_test caveat). To actually shrink them you must test on the **FULL 10 k**:

```text
  USE the ARMs ALREADY trained on 75% of the 10k — let's NOT re-burn money          ┐
  just because we found 1 more dataset;  just TEST them on ALL of fresh subset_10k ──┘→ n_test ≈ 10 k
  5× more test clips  →  CI half-width shrinks ~√5 ≈ 2.2×  →  close arms can SEPARATE
```
> ✅ **This is NOT leakage** — heads trained on `eval_10k`, tested on **disjoint** `subset_10k`. The disjointness is **verified, not assumed** (§4c) — that audit *is* what makes the tighter CIs clean.

**Step 2 — pick the 2 OURS with *non-intersecting* CIs** (after the tighter-CI eval; selection is **data-driven, not pre-chosen**). Today's `eval_metrics.json` ordering by future-frame MSE (↓ better) is a **3-way tie** — exactly why we tighten first:
```text
intervene 0.4955 │ tccaux 0.4956 │ diheavy 0.4959   ← CI-tie (±0.001) at n=1825 → NObody separates yet
3stage_DI 0.4976 │ replay25 0.4975 │ noDI 0.4984
wiseft_f30 0.5078 │ f50 0.5194 │ f70 0.5327          (Frozen 0.5574 — every OURS beats it)
```
> 💡 **Tiebreak principle (only if ≥2 still tie after tightening):** prefer **one predictor champion + one trade-off arm** so the 1B replication covers **both** OURS claims, not two near-identical arms. **Final pick = yours, from the tightened CIs.** *(If a `wiseft` arm is picked, its merge base `intervene` trains too — the merge itself is post-hoc, free.)*

**Step 3 — the 1B training roster = the 2 separated OURS + ALL 7 competitor FT techniques** — so OURS-1B is judged against competitors-**1B**, never against competitors-2B. **External WMs are NEVER retrained on 1B** — they keep their own native open weights (eval-only).
```text
  TRAIN on 1B (METHOD-APPENDIX only, fix-7):   2 separated OURS   +   7 competitors
     COMPETITORS (7) = pretrain_encoder · surgical_autorgn_encoder · surgery_raw_encoder
                       · full_ft_encoder · lpft_encoder · peft_lora_encoder · peft_dora_encoder
     OURS (pick 2 of 9) = surgery_3stage_DI · surgery_noDI · …_replay25 · …_diheavy · …_tccaux
                       · …_intervene · …_intervene_wiseft_f30 · _f50 · _f70   (from §1b)
  NOT trained on 1B:  the other 7 OURS · the *_head arms · cassle / ewc · EVERY external WM
```

### 🔒 4c — The leakage audit that makes the tighter CIs clean (→ appendix evidence)
The reuse-heads / full-`subset_10k` eval is clean **only if** the two sets are truly disjoint — but **both were carved from the same 115k**, so the silent failure mode is **near-duplicate / re-sampled clips**. Verify it, don't assume it — a **two-level audit** (field standard):
1. **Exact:** `set(eval_10k clip-IDs) ∩ set(subset_10k clip-IDs) == ∅` → report **"0 shared clip IDs."** *(a `src/utils/audit_disjoint.py` reading both `*_local/*.json` manifests; **FAIL-LOUD** if non-empty.)*
2. **Near-dup (metadata, NOT embeddings — fix-4):** flag pairs sharing the **same source-video ID within ±30 s** (the project's hard-mode rule). ⚠️ Do **not** use frame-embedding cosine ≥ 0.90 — consecutive WalkIndia clips (same street/camera) are *naturally* > 0.90 → it floods false "duplicates". *(An embedding pass, if kept, runs cross-source-only at a high threshold with a stated rationale.)*

> 🧷 This assertion **IS** the clean-CI claim's evidence — it pre-empts the exact reviewer question. Put the two numbers in the **appendix**.

### ▶️ Run mechanics
```bash
# 1) fetch the held-out 10k (network only, no GPU) + the ONE light prep
python -u src/utils/hf_outputs.py download-data data/subset_10k_local \
  2>&1 | tee logs/download_subset_10k_$(date +%Y%m%d_%H%M%S).log
python -u src/m04d_motion_features.py --POC --subset data/subset_10k_local/subset_10k.json \
  --local-data data/subset_10k_local --cache-policy 1

# 2) our 16 reference arms (2B) → EVAL-ONLY. REUSE the heads already trained on 75% of the 10k
#    (NO retraining), TEST on ALL of subset_10k → n_test≈10k → ~2.2× tighter CIs. Multi-GPU tagged eval.
#    (needs the cross-set test mode from 4b: reuse-heads + full-set test, instead of the 75/5/20 re-split)
ITER18_EVAL_TAG=benchmark10k ITER18_BACKBONE=vjepa_2_1_vitG PT_H_MEMO=1 \
  python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 \
  2>&1 | tee logs/iter19_benchmark_eval_$(date +%Y%m%d_%H%M%S).log

# 3) each EXTERNAL open-weight WM → run_eval.sh ONLY (after 4a wires the external branch); fan 1 per free GPU
#    CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh <external_arm> --local-data data/subset_10k_local ...

# 4) 1B (ViT-g) — METHOD-APPENDIX ONLY (fix-7; benchmark's 1B = an eval-only row): 2 OURS + 7 competitors
#    (run_train.sh <2 OURS> <7 competitors> with ITER18_BACKBONE=vjepa_2_0 ; then the same tagged eval)
```
> 📂 Lands under `outputs/benchmark10k/…` (the `ITER18_EVAL_TAG` patch keeps the iter18 numbers intact).

---

## 🌍 §5 — Cross-domain panel: show the protocol **TRAVELS** (kills the "one domain" rejection)

> WalkIndia alone = **one domain** → the "useful but narrow" taste-rejection. Cheapest mitigation — **no data to own:**

- Run the **8-metric encoder board** on **Walking Tours** (`shawshankvkt/Walking_Tours`, HF) — **same modality (urban walking), different cities / continents, public, unlabeled.** *(10 long egocentric city-walk videos; Amsterdam, Bangkok, Venice, … Istanbul held out.)*
- Unlabeled ⇒ the **label-free subset** applies cleanly: `mcos` + the 5 encoder-temporal metrics (`aot` `tov` `pace` `tcc_tau` `tcc_cycle`). `act` / `tax` need labels → **skip or pseudo-label, stated.**
- **One** non-WalkIndia panel proves *"the protocol generalizes across domains,"* not just *"our clips."* *(Ego4D = an optional 2nd egocentric source, license-gated.)*

---

## ⚠️ Honest caveats (carry into the paper as stated limits)

- 🧱 **Two boards, not one.** The **encoder board (8)** is the broad cross-model benchmark; the **predictor board (7)** is **V-JEPA-family + our arms only** (DINO-WM restrained to encoder-only — its predictor adapter is a validity landmine) → present it as a **within-family deep-dive**, never a cross-model leaderboard. Pixel/token generators (Cosmos, WHAM, …) are **8-only**, scored **N/A** on the 7 (never blank). **Document the probe protocol** — *attentive* probing ≫ *linear* for a fair cross-architecture comparison.
- 🧪 **Construct validity is owed (item #1).** The 15 metrics are computed, not yet **shown** to measure their named capability → ship the **convergent/discriminant heatmap + ≥1 criterion correlation** (§2b), or the suite is dismissible as "15 numbers of unknown meaning."
- 🔒 **Disjointness is verified, not assumed.** `eval_10k` and `subset_10k` share a 115k parent → run the **exact + metadata audit** (§4c) and report **"0 shared clip IDs · 0 same-source-video within ±30 s"** (metadata, *not* embeddings — fix-5) in the appendix. The tighter CIs are clean **because** of this audit.
- 🧮 **1B is eval-only for the benchmark (fix-7).** The benchmark's 1B coverage = **V-JEPA 2.0 1B as a scored row** (no training). The 9-arm 1B *retrain* (2 OURS + 7 competitors) proves a **method** claim → **appendix only**, deliberately cut from the benchmark's critical path.
- 🌍 **Domain scope.** All clips are WalkIndia → a **stated domain limit**; the protocol generalizes, the *numbers* are this domain.
- 🎯 **Tighter CIs = reuse-heads cross-set eval (the chosen path).** Don't re-split `subset_10k` 75/5/20 (re-tests ~2k → same wide bars). **Reuse the heads already trained on 75 % of `eval_10k`, test on ALL of `subset_10k`** → n_test ≈ 10k → CIs ~2.2× tighter (§4b). Needs a small **cross-set test mode** in the eval + the **§4c audit** to be clean. (Also *is* the generalization-to-fresh-clips check.)
- 🧮 **Multiple-comparison: correct over VALID cells.** **Benjamini-Hochberg FDR per board** over the **non-N/A (model × metric)** cells — encoder and predictor boards are **separate families.** Naive 15×N counts N/A cells, over-corrects, and **buries real effects.**
- 🆚 **Novelty vs prior benchmarks.** VBench / VBench-2.0, WorldScore, WorldModelBench, FVD / FVMD all judge **generation quality** (pixels a generator emits). This suite is **generation-free** — it probes the **encoder's representation + the predictor's latent dynamics**. That gap is the contribution; TimeBlind (video-LLM temporal minimal-pairs) is the closest in spirit.

---

## ✅ Bottom line
**Decision = Path B, eval-only.** External open-weight WMs plug into **`run_eval.sh` ONLY** (`kind: external`, `scheduler: false`) — the training scheduler and every `m09*` trainer stay untouched. Our **16 reference arms (2B)** score eval-only on the fresh 10k; for the benchmark there is **no training at all** — the **1B backbone is just another eval-only row** (V-JEPA 2.0 1B). *(The 1B 9-arm retrain is a method-paper appendix, fix-7 — not benchmark credit.)* **Headline = the broad 8-metric encoder board** (V-JEPA family · DINO-WM · Cosmos · WHAM · GR00T · NWM · video encoders + our arms); the **7-metric predictor board is a within-family deep-dive** (V-JEPA-family + our arms). Ship **all 15** — `order` reported-not-ranked — with the **model × tier applicability matrix** as the spine. **Still owed, in priority order:** ① **construct-validity study** (§2b — #1), ② **FDR per board**, ③ **leakage audit** (§4c), ④ **cross-domain panel** (Walking Tours, §5) + the **`load_external_wm` loader** (de-risked — VideoMAE smoke-loads in 7 s, fix-10). *(The 1B retrain is **appendix, not owed** — fix-7/fix-9.)* **None needs a fresh 115k run.**

---

### 📚 Sources (model pool + prior benchmarks)
- V-JEPA 2 / 2.1 — [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) · [HF collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) · [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- NVIDIA Cosmos 3 — [Newsroom launch](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) · [Cosmos platform](https://www.nvidia.com/en-us/ai/cosmos/) · [github.com/nvidia-cosmos](https://github.com/nvidia-cosmos) · [Cosmos WFM paper (2501.03575)](https://arxiv.org/abs/2501.03575)
- DINO-WM — [project](https://dino-wm.github.io/) · [github.com/gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) · [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- Prior WM benchmarks (generation-quality) — WorldScore, WorldModelBench, [VBench/FVD survey](https://www.emergentmind.com/topics/fvd-and-vbench-metrics), [WorldArena (2602.08971)](https://arxiv.org/pdf/2602.08971), [TimeBlind (2602.00288)](https://arxiv.org/html/2602.00288v3)
- New 8-board models — [microsoft/wham (Muse/WHAM)](https://huggingface.co/microsoft/wham) · [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0) · GR00T N1 · [Navigation World Models (2412.03572)](https://arxiv.org/pdf/2412.03572)
- **Construct validity** (§2b) — [Construct Validity for Evaluating ML (2510.23191)](https://arxiv.org/abs/2510.23191) · [Measuring what Matters (2511.04703)](https://arxiv.org/abs/2511.04703) · [Nomological Networks (2603.15121)](https://arxiv.org/abs/2603.15121)
- **Leakage / dedup audit** (§4c) — [BigCode near-dedup](https://huggingface.co/blog/dedup) · [cross-dataset deduplication](https://www.emergentmind.com/topics/cross-dataset-deduplication) · SSCD / FAISS ANN copy-detection
- **Cross-domain panel** (§5) — [Walking Tours (HF)](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) · [Ego4D](https://ego4d-data.org/)
- **Probe protocol + FDR** — attentive vs linear probing ([UNMUTE THE PATCH TOKENS](https://openreview.net/pdf/115e78f4e68bfeecb9fa1d1a5d8a67537935ba2d.pdf)) · [Benjamini-Hochberg FDR per family](https://stats.libretexts.org/Bookshelves/Applied_Statistics/Biological_Statistics_(McDonald)/06:_Multiple_Tests/6.01:_Multiple_Comparisons)
