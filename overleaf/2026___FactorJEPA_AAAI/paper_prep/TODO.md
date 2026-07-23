# ✅ TODO — AAAI submission task tracker (`0_main_AAAI.tex`)

> **Status 2026-07-22.** The **abstract is submission-ready** (all 26 figures recomputed from source
> and matching) and **contains no identity leak**. The **body is not ready**, and the **code release
> would break blind review as it stands**.
>
> Evidence: `results_tables/` (this folder) · `../AUDIT_delivered_vs_claimed.md`

**Progress: 14 / 40 done** · abstract SUBMITTED to AAAI 2027 (#43837); full paper due in 7 days

| tier | what | open | done |
|---|---|---|---|
| 🚨 **P0-ANON** | breaks double-blind review | 0 | 10 |
| 🔴 **P0** | blocks submission | 3 | 3 |
| 🟠 **P1** | factual corrections | 8 | 0 |
| 🟡 **P2** | ablations: losses + missing controls | 9 | 0 |
| 🟢 **P3** | hygiene | 4 | 1 |

**Closed this session:** anonymized repo built, scrubbed (0 identity hits) and pushed private with
an anonymous commit author; HF dataset renamed `walkindia-200k` → `denseworld-115k`; abstract
purged of the dead NEPHOS link; code shipped as the OpenReview *Code and Data Supplement* zip
(17 MB) because AAAI **forbids** linking anonymized repos.

---

## 🚨 P0-ANON — breaks double-blind review

Audited over `src/`, `scripts/`, `README.md` (the payload proposed for anonymous.4open.science)
and over every `.tex` the AAAI build inputs.

### Already clean — the PAPER

- [x] No real names / emails / GitHub / project-page URLs in any built `.tex`
- [x] Every paper URL is a third-party tool ref (`ffmpeg`, `scenedetect`, SCRFD, ByteTrack,
      ultralytics, opencv) plus the anonymized `anonymous.4open.science` link
- [x] `[submission]` renders the author block as "Anonymous submission" (`aaai2027.sty:131`)

### Blocking — the CODE payload

- [ ] **`README.md` de-anonymizes 5 authors + 4 affiliations + the GitHub account — do NOT upload as-is**
  - `:5`, `:11` project-page badge/link → `kapilw25.github.io/factorjepa`
  - `:34` `git clone https://github.com/kapilw25/factorjepa.git`
  - `:130` full author list · `:132` `Canva Research · Google · Apple · Pragya Lab, BITS Pilani Goa`
  - `:141` BibTeX with all five names
- [ ] **Real name embedded in shipping code** — `src/utils/hf_finetuned_push.py:398`
      `author = {{Wanaskar, Kapil and others}}`, auto-written into every HF model card
- [ ] **`src/CLAUDE.md:111`** — "prompts for Kapil/Gaytri account"
- [ ] **Do not ship `src/CLAUDE.md` (35 KB) or `src/MEMORY.md` (34 KB)** — internal agent notes,
      iteration history, incident logs, account details. Not part of the method.
- [ ] **`anonymousML123/…` appears 28× across 4 HF repos** — one searchable handle tying the whole
      payload to a single account. Scrub, or confirm the org has no profile/link back.
- [x] ~~HF dataset renamed~~ `walkindia-200k` → **`anonymousML123/denseworld-115k`** (2026-07-22).
      Card rewritten, old-name mentions stripped, all 13 code refs updated across 8 files.
      ⚠️ HF keeps redirects from both old names — the string is gone from *our* payload, not from HF.
- [ ] **`walkindia` still appears in ~49 files** as local paths (`data/walkindia_drive`,
      `venv_walkindia`, output dirs). The HF repo id is clean; these local names are not.
- [ ] Re-run the leak scan after scrubbing (command in §How-to below) and require **0 hits**

### Verified NOT a leak — do not waste time

`Canva` (18× `canvas`, 2× `CanvasAgg`) · `BITS` (17× `bitsandbytes`, 6× `bits`) ·
`Apple`/`Google` (SALT and TCC citations). Only `README.md:132` is a real affiliation.

---

## 🔴 P0 — blocks submission

> **Progress 2026-07-22:** 3 of 5 done. `2_factor_jepa.tex` went **1789 → 663 lines**;
> backups at `2_factor_jepa.tex.bak`, `*.bak2`.

- [x] ~~**Abstract and body described DIFFERENT methods.**~~ §`sec:factorjepa` rewritten as
      *"Predictor Surgery over Factorized Views"*: automatic factor-view construction
      (Grounding DINO + SAM 3 → `D_L`/`D_A`/`D_I`), the three-stage curriculum with progressive
      unfreezing (≤ 8/48 and 6/40 blocks), and the **real** objective
      `L_JEPA + β·L_probe + γ·L_motion + λ·L_drift`. Verified removed: `CAᵀ` 0, `L_cov` 0,
      "visibility gate" 0. `fig:frozen_forest`, `fig:eval_scorecard`, `fig:scale_replication`
      preserved in place.
- [x] ~~**Latent-to-RGB section**~~ deleted from `2_factor_jepa.tex` (§`sec:latent_to_rgb`,
      `tab:latent_rgb` with its 20 `--` cells, `fig:latent_to_rgb_pipeline`). "Cosmos" 0 hits.
- [x] ~~**Undefined `\ref` → `??`**~~ all **9** fixed, 0 remaining:
      `app:adaptation_details`→`app:baseline_configuration`,
      `app:annotation_protocol`→`app:factor_targets`,
      `app:taxonomy_mapping`→`app:shared_taxonomy`; and the 3 never-authored floats
      (`tab:executed_attribution_results` ×3, `tab:five_axis_comparison`,
      `fig:executed_attribution_forest`) de-referenced with the prose kept.
- [ ] **`8_conclusion.tex` still carries 3 dropped claims** — `Cosmos` ×1, `1,000 hours` ×1,
      `intervention-sensitive` ×1. Must match the abstract (276 hours, no decoder, causal
      future-block).
- [ ] **Causal L1 is not an intervention metric.** Experiments section still defines
      `mean‖Δ̂_a − Δ*_a‖₁` over annotated interventions; `utils/pt_causal.py` masks the future
      temporal half and predicts it from the past half. 9 "intervention" hits remain in
      `2_factor_jepa.tex`. Rewrite as *"causal future-block L1"*.
- [ ] **The human audit split `Ψ_human` does not exist** (1 hit remaining). No code produces or
      consumes human annotations → drop the equation and the `tab:evaluator_provenance` row.

---

## 🟠 P1 — factual corrections

- [ ] "approximately **1,000 hours**" (×4 places) → **275.8 hours** (`docs/index.html:1090`)
- [ ] "partnered with **two professional video-collection companies**" → **714 YouTube videos at
      480p** (`src/m01_download.py:2`)
- [ ] "fixed **DINOv2**-based pipeline", `F_DINOv2`, cites `oquab2024dinov2` → actually
      **Grounding DINO** + **SAM 3** (`m10_sam_segment.py`, `surgery_base.yaml:170,183`)
- [ ] "**80/10/10** split" → **86,831 / 909 / 23,106 = 75.06 / 0.79 / 19.97**, residual
      **4,841 (4.18%)** lacked factor targets
- [ ] "frames 1:12 context, 13:16 target, horizon 1.0 s" → V-JEPA **random spacetime multiblock**
      mask, 8 blocks @0.15 + 2 @0.70 (`base_optimization.yaml:72-80`)
- [ ] Panel A "16 attention heads, MLP ratio 4" → ViT-G is **26 heads, MLP ratio 4.92**
      (`configs/model/vjepa2_1.yaml:13`)
- [ ] "Motion cosine = cosine alignment of **linearly decoded** descriptors" → intra-minus-inter
      class cosine **separation** on pooled features (`m12b_motion_cos.py`)
- [ ] `0_main_neurips.tex` says **500 hours** — reconcile with 276 or delete that main

---

## 🟡 P2 — ablations: 4 of 13 axes are in the abstract

Covered: model scale · data scale · 6 FT techniques · frozen cross-family.
Full verdicts in `results_tables/`. These need an **ablation section** + a **limitations paragraph**.

### Losses that must be disclosed

- [ ] **Factor targets ON vs OFF** (flagship vs `surgery_raw`) — **LOSE**: 2B `W0 L2 T2`, 1B
      `W0 L4 T0`. `surgery_raw` is the identical schedule with **no factor targets** and it wins.
      The defining ingredient is not what produces the gain — predictor surgery is. Also explains why
      `surgery_raw` won 4 of the 8 headline forest values.
- [ ] **Interaction stage ON vs OFF** (`delta_7_DI_head_vs_noDI_head`) — **FAIL** `0/1/3`.
      A *declared* hypothesis in `configs/eval/paired_deltas.yaml`.
- [ ] **WiSE-FT weight merge** f30/f50/f70 — **LOSE BADLY**: f50 vs its own base `W0 L4`,
      fut −17.9×, mcos −20.0×
- [ ] **V-JEPA 2.0 base** — **METHOD FAILS**: controlled pair (same arch/recipe/data, only the
      checkpoint differs) → *"surgery < frozen on 5 metrics"*. A real precondition.
      → `iter/iter17_ablations_model/result_outputs/why_surgery_loses_on_vjepa2_0.md`

### Declared hypotheses never evaluated

- [ ] **`delta_3_surgical_vs_pretrain_2X`** — the **compute-matched control**, answering *"is this
      just more training?"*. `probe_encoders.yaml` documents it as isolating *"the surgery method,
      not just extra compute"*. Declared in a committed config, **never run**. ⚠️ most exposed gap.
- [ ] `delta_4_pretrain_vs_pretrain_head` — `pretrain_head` never evaluated
- [ ] `delta_6_surgical_head_vs_pretrain_head` — same
- [ ] CaSSLe / EWC continual-learning baselines — trained, never evaluated

### Wins currently unused

- [ ] Surface **encoder-update vs head-only** (`delta_5`): `W3 L1` — fut **+26.8×**, causal
      **+14.4×**, maskratio **+7.9×**. Justifies touching the encoder at all.
      (Also `delta_1` `4/0/0`, `delta_2` `3/1/0`; improvement arms diheavy `W2`, tccaux `W2`,
      intervene `W1L1`, replay25 `W0L0T4`.)

---

## 🟢 P3 — hygiene

- [ ] `\clearpage` ×2 + `\newpage` ×2 active (l.345, 346, 363, 364) — template's own disallowed list
- [ ] Author block is the AAAI placeholder — fine at submission, **must be filled for camera-ready**
- [ ] `0_main.tex` still `\input`s `10_india.tex` (a different, older "IndianSuburb-5K / Shampoo /
      K-FAC" paper) and `12_new_plan.tex` (literal `\textbf{:contentReference[oaicite:0]{index=0}}`)
- [ ] **"We release the dataset"** — 714 third-party YouTube videos. If only clip IDs + the
      extraction pipeline can ship, narrow the claim.
- [ ] Body says **"clips"**, abstract says **"videos"** for the same objects. Mitigated by the
      abstract's "each 6 to 13 seconds" clause; align fully if desired.

---

## 🛠️ How-to

### A. HF dataset rename — `walkindia-200k` → ?

Three names and two counts are currently in play for one artifact:

```text
  paper            DENSEWORLD 1.0 · 115,687 videos
  HF dataset       anonymousML123/walkindia-200k
  README citation  "DenseWorld-200K"
  code             6 hard refs to anonymousML123/walkindia-200k
```

- [ ] **Verify the actual clip count in the HF repo first.** If it holds ~200k (a pre-filter
      superset), naming it `-116k` would be wrong; prefer an unnumbered `denseworld` and state the
      paper's 115,687 as the *filtered* corpus.
- [ ] Rename to match the paper, then update the 6 code refs + `README` citation in one pass
- [ ] ⚠️ HF keeps a redirect from the old name — the rename does **not** erase `walkindia`.
      For blind review, what matters is scrubbing the string from the 4open.science payload.

### B. Publishing only `src/`, `scripts/`, `README` to anonymous.4open.science

Anonymous GitHub **has no file-exclusion option**. It takes one GitHub repo and anonymizes owner,
org, repo name, file/directory names and file contents; a *terms list* you supply is replaced with
`XXXX` ([docs](https://github.com/tdurieux/anonymous_github), [service](https://anonymous.4open.science/anonymize)).
So the subsetting has to happen **before** the service sees it.

- [ ] Create a **separate, PRIVATE** GitHub repo (private so GitHub code-search cannot be used to
      find the original by grepping a distinctive string). 4open.science can proxy private repos
      via its GitHub OAuth.
- [ ] Copy in **only** `src/`, `scripts/`, and an **anonymized** `README_anon.md`. Exclude
      `src/CLAUDE.md`, `src/MEMORY.md`, `src/legacy/`, `scripts/legacy/`.
- [ ] **Scrub contents before pushing** — do not rely on the terms list alone; it replaces words,
      so a URL like `kapilw25.github.io` becomes `XXXX.github.io`, which still narrows the search.
- [ ] Add the terms list anyway as defense in depth:
      `kapil, kapilw25, Wanaskar, gaytri, Jena, Vinija, Chadha, Amitava, Canva, Pragya,
      anonymousML123, walkindia, factorjepa`
- [ ] ⚠️ Anonymizing `factorjepa` renames files/dirs too — check the served repo still reads
      sensibly before submitting the link.

### C. Leak-scan commands (re-run until 0 hits)

```bash
# 1) the PAPER (what reviewers read)
cd overleaf/2026___FactorJEPA_AAAI
grep -rinE "kapil|gaytri|wanaskar|jena|vinija|chadha|amitava|kapilw25|@gmail|walkindia" \
     0_main_AAAI.tex 1_introduction.tex 2_data.tex 2_factor_jepa.tex 8_conclusion.tex 11_appendix.tex
grep -ohn "https\?://[^ }]*" 0_main_AAAI.tex 2_*.tex 8_*.tex 11_*.tex | sed 's/[},].*//' | sort -u

# 2) the CODE payload (what 4open.science serves)
grep -rinE "kapil|gaytri|wanaskar|jena|vinija|chadha|amitava|kapilw25|@gmail|github\.com/kapil|walkindia|anonymousML123" \
     src/ scripts/ README_anon.md

# 3) affiliations, word-boundary only (avoids canvas/bitsandbytes false positives)
grep -rinwE "Canva|BITS|Pragya" src/ scripts/ README_anon.md
```

### D. Is the ABSTRACT clean? — already verified

Scan #1 above returns **0 hits** on the abstract and on every `.tex` the AAAI build inputs.
The only URL in the abstract is the anonymized `anonymous.4open.science` link.

---

## 📁 This folder

```text
paper_prep/
  TODO.md                     <- this file
  results_tables/
    README.md                 provenance, sign convention, roster, caveats
    pairwise_all.csv          3,317 rows — every arm pair x metric x scale
    ours_vs_technique.csv     1,374 rows — the OURS x COMPETITOR subset
    arm_vs_frozen.csv           548 rows — every arm vs the frozen backbone
    per_technique_tables.txt  rendered ASCII, one block per (OURS arm x scale)
    per_technique_tables.tex  booktabs LaTeX, ready to \input
```
