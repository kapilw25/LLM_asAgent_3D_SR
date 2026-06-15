# 🎯 Validate the 2B WiSE-FT story on a **fresh 10k** — our FULL-scale run

> 💡 **Scope (honest + compute-bounded).** We are **NOT** running 115k clips.
> Our declared **FINAL / "FULL" scale = 10k**: train on **75 % of 10k + 5 % val**, evaluate on a **fresh, disjoint 10k**.
> An individual researcher's budget is a legitimate, openly-stated scope — *"we collected 115k"* ≠ *"we must burn money on 115k."*
> 📄 In the paper, **"FULL" = this 10k-scale result.**

---

## 🔬 QQ0 — what an AAAI reviewer bites (and what THIS run fixes)

| 🐛 Gap in the experiment | 🗣️ Reviewer's line | ✅ Fixed by this run? |
|---|---|---|
| 🎲 **Single seed** — the 95 % bands are eval-bootstrap over test clips, *not* training-seed variance | *"Report ≥ 3 seeds — your bands don't capture training stochasticity."* | ❌ needs **re-train × seeds** *(biggest one)* |
| 📏 **Scale = 10k, not 115k** | *"Why only 10k?"* | ✅ **declared scope** (compute-bounded, stated honestly) |
| 🧮 **14 metrics × ~18 arms, NO multiple-comparison correction** | *"Some 'wins' are false positives at α = 0.05."* | ❌ free stats fix — **just do it** (Bonferroni / FDR) |
| 🌍 **In-domain only** (all WalkIndia) | *"Does it hold on fresh, unseen clips?"* | ✅ **this run is exactly that test** (disjoint 10k) |
| 📉 **WiSE-FT wins overlap baselines** | *"Not statistically significant."* | 〰️ partially — see caveat ⚠️ |

> 🎯 **Still owed before submission:** ① ≥ 3 **seeds** · ② **multiple-comparison correction**. This run buys the **generalization** check — not those two.

---

## 📥 Step 1 — download the fresh 10k  · 🌐 network only, **no GPU**
```bash
python -u src/utils/hf_outputs.py download-data data/subset_10k_local \
  2>&1 | tee logs/download_subset_10k_$(date +%Y%m%d_%H%M%S).log
```
> 📦 Raw on HF: `subset-0000{0..9}.tar` (the clips) + `manifest.json` + `subset_10k.json` + `tags.json` — **no labels / features yet.**

## ⚙️ Step 2 — prep = **ONE** light GPU step · 🟢 1 GPU, ~30–60 min
```bash
python -u src/m04d_motion_features.py --POC \
  --subset data/subset_10k_local/subset_10k.json \
  --local-data data/subset_10k_local --cache-policy 1
```
> ✅ The **only** prep. 🏷️ action + taxonomy labels are auto-derived by `run_eval` **Stage 1** (m04e/m04f) from `tags.json`.
> 🚫 `m10` / `m11` / split-files are **training-only** → the eval never reads them *(verified in m12c / m12e / m12f)*.

## 💾 Step 3 — free disk **FIRST**  · you have **206 G**, the fresh frame-cache ≈ **423 G**
```bash
rm -rf data/eval_10k_local/m12_frame_cache      # regenerable + its 2B eval is DONE → frees 423 G
```
> 🗑️ Also droppable after the HF backup: `outputs/poc/**/*.pt` (~242 G).

## 🖥️ Step 4 — eval via the **SCHEDULER** (⚡ 4-GPU fan-out) — patch `scripts/iter18_poc_ngpu.py`
> ❌ Running `run_eval.sh` by hand serializes **one encoder at a time** — wasteful on a 4× node.
> ✅ Teach the scheduler to do a **tagged, eval-only** pass → it fans the encoders across **all 4 GPUs**.

### 🔧 Code change — `scripts/iter18_poc_ngpu.py` (add an `ITER18_EVAL_TAG` env)
When `ITER18_EVAL_TAG` is set, the **eval** jobs (`E:` `P:` `F:` + the §3 finale):

| # | change | why |
|---|---|---|
| 1️⃣ | prepend `LOCAL_DATA=data/subset_10k_local EVAL_SUBSET=…/subset_10k.json` to each `run_eval` cmd | read the **fresh** data |
| 2️⃣ | prepend `OUTPUT_ACTION=outputs/<tag>/probe_action … OUTPUT_ENCTEMP=outputs/<tag>/encoder_temporal` | 🛡️ **don't clobber** the eval_10k 2B numbers |
| 3️⃣ | point the resume done-markers (`aggregate_*.json` checks) at `outputs/<tag>/…` | eval runs **fresh** under the tag |
| 4️⃣ | *(no change needed)* train jobs auto-skip via `--cache 1` + existing 2B `student_encoder.pt` | ✅ **eval-only** |

### ▶️ Run — fans 6 encoders across 4 GPUs
```bash
ITER18_EVAL_TAG=subset10k ITER18_BACKBONE=vjepa_2_1_vitG PT_H_MEMO=1 \
  python -u scripts/iter18_poc_ngpu.py --mode POC --gpus 4 --cache 1 \
  --skip-arms cassle_encoder ewc_encoder surgery_3stage_DI_head surgery_noDI_head \
  2>&1 | tee logs/iter18_eval_subset10k_$(date +%Y%m%d_%H%M%S).log
```
> 📂 New numbers land under `outputs/subset10k/…` → refresh figures with the same tag, the eval_10k results stay intact.

---

## ⚠️ Honest caveats (don't skip)

- 🚨 **n_test ≈ 2k, NOT 10k.** Stage 1 (`m04e`) re-splits `subset_10k` **75 / 5 / 20** (the 75 % trains the probe heads). So the test set is ~2k ≈ today's **1,825** → the **confidence bands barely shrink.** This run proves *the wins hold on fresh clips* (**generalization**) — it does **NOT** settle the overlap *significance* question. *(To force n_test = 10k you'd need a cross-set eval — heads trained on the train set, tested on ALL of the fresh 10k — a bigger code change; optional.)*
- 🧊 The temporal-metric ties with **Frozen** (Arrow-of-Time accuracy, TCC Kendall τ, …) are **equal by construction** — WiSE-FT f70 *is* 70 % Frozen — they won't separate at any N. The real claim is the **trade-off win**: Frozen-level temporal structure **AND** better future-frame prediction than Frozen.

---

## ✅ Bottom line
Run it for the **generalization** check (cheap: 1 download + 1 motion-feature pass + a 4-GPU eval). It pre-empts the *"does it hold off your training clips?"* reviewer bite. It does **not** replace the still-owed **seeds** + **multiple-comparison correction**.
