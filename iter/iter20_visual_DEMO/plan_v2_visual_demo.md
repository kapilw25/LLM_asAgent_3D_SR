# 🎬 iter20 — VISUAL DEMO · v12 (🔒 CLOSED: visible OURS>FROZEN demo NOT achievable · edge is IN-DOMAIN ONLY — see §F)

| 🎯 GOAL | 2-3 sample demo videos for the research page, `demo_cosmos`/`demo_vJPEA` format: clip + QUESTION + GROUND TRUTH → 🧊 FROZEN answer ❌ · 🥇 OURS answer ✅ — the answers from **OUR model** (a tiny probe on FROZEN vs OURS V-JEPA features), on the 3060. |
|---|---|

## ✅ §D — THE DECISION (2026-07-19)

- 🅱️ **API (ChatGPT/Claude): ❌ no** — it's like hiring a *stranger* to take the test; it uses *its own* eyes, never our model's, so it literally **can't** show FROZEN-twin vs OURS-twin (you'd get "GPT vs GPT"). ✅ ok only to *word* the questions nicely.
- 🅰️ **VLM+LLM on 96 GB: ✅ BUILT + RAN (2026-07-19) → gate INCONCLUSIVE** — both arms landed at chance (projector starvation), so it tested our *training budget*, not the encoder. **Superseded by the encoder-level OOD probe (§F), which answered the question decisively: OURS LOSES out-of-domain.**
- 🚫 **Cheap probe demo: EXHAUSTED** — §E0 proved OURS wins the motion probe IN-DOMAIN (**+15.7pp**), but every *eye-verifiable* question failed (§2: viewpoint→FROZEN wins · magnitude→VM30 backwards · retrieval→motion invisible on WalkIndia radial flow). No layman-verifiable card exists → escalated to the VLM.

---

## 🔒 §F — CLOSING FINDING (2026-07-19): the visible demo is NOT achievable · OURS's edge is IN-DOMAIN ONLY

Four independent closures. The intersection of **"eye-verifiable"** and **"OURS wins"** is **EMPTY** — now proven at the
**encoder** level, not merely at the label/probe level.

| # | test | domain | n | result | verdict |
|---|---|---|---|---|---|
| 1 | scene / taxonomy questions | in-domain | — | OURS loses **0/15** | ❌ |
| 2 | magnitude / direction labels | in-domain | 1878 | OURS wins **+8.7…+15.7pp** | ⚠️ wins but **INVISIBLE** (VM30 radial flow) |
| 3 | WalkIndia straight-vs-turn | in-domain | 440 | no bimodal boundary; only ~1-3/12 read as turns | ❌ neither derivable nor visible |
| 4 | **ghat POV ego-yaw** | **OOD** | **167** | **OURS LOSES −6.0…−10.8pp** | ❌ **decisive** |

### 🧪 The OOD transfer result (new, decisive)

Encoder-level probe — **no LLM**, so it is immune to the projector-starvation that invalidated the VLM gate.
Leave-one-of-5-**temporal**-blocks-out (adjacent windows of one continuous drive are autocorrelated; a random
split would leak and fabricate a win). Source: 18-min ghat-road POV driving, 8 s windows, flow-derived labels.

| head | FROZEN | OURS | Δ(O−F) |
|---|---|---|---|
| LINEAR | 0.737 | **0.629** | **−10.8pp** |
| MLP | 0.731 | **0.671** | **−6.0pp** |
| majority baseline | 0.503 | — | both arms ≫ baseline → **the test had power** |

Replicates **Diving48 (−1.1pp × 2)**. A cab-POV confirmation (n=55) was **underpowered and mixed**
(LINEAR +1.8pp, MLP −3.6pp, SE ±5.8pp) → neither confirms nor contradicts; the ghat probe is the load-bearing evidence.

> 🧬 **Mechanism:** surgery **specialises** the encoder to WalkIndia's motion statistics (radial walking flow), and that
> specialisation **costs** it on a different motion regime (large ego-yaw). This is a **publishable, quantified
> limitation** across 3 OOD experiments — a characterisation of what the method does, not a failed run.

### ⛔ The VLM early gate did NOT test this — do not cite it as OOD evidence

FROZEN **0.446** [0.405,0.486] vs OURS **0.444** [0.404,0.486] → −0.2pp. **But both arms sat at chance:**
yes/no 0.503 / 0.474 (coin flip on a balanced set) and MC 0.361 / 0.400 against a 0.374 majority baseline.
A from-scratch projector on **2 885 samples / 1 epoch cannot align video→language** (LLaVA stage-1 uses 558K, ~200×).

| the run is… | about |
|---|---|
| ✅ **conclusive** | the **training budget** — the cheap skip-align shortcut does not work |
| ❌ **inconclusive** | the **encoder** — the hypothesis was never actually exercised |

> ⚠️ An earlier chain printed **"✅ PASS +29.8pp"**. That was a **metric artefact**, not a result: FROZEN was scored with a
> letter-only extractor that auto-failed all 340 yes/no rows (60% of the set) while OURS used the fixed parser.
> Re-scored on the matched metric the gap is **−0.2pp**. Artefacts quarantined as `*.BROKEN_METRIC.json` / `*.INVALID.json`.

### 📦 What remains shippable

| artefact | status |
|---|---|
| 🥇 **forest plots** | ✅ rigorous in-domain wins, non-overlapping CIs — the honest headline evidence |
| 🎬 `outputs/demo/mcq/demo_mcq.mp4` | ✅ real +13.7pp retrieval, overlay-free, honestly captioned as **metric-verified, not eye-verified** |
| 🧬 **OOD limitation section** | ✅ NEW — 3 quantified experiments; strengthens the paper rather than weakening it |
| 🎥 visible `demo_cosmos`-style video | ⛔ **not achievable** — stop hunting; 4 independent closures |

---

# 🛠️ EXECUTION — the cheap probe demo (engineering steps)

## ✅ §1 — PRE-CHECK (done, 3060)

OURS beats FROZEN on a trained motion probe, reusing cached features — the mechanism works in-domain:
- §E0 understanding (full-clip): magnitude **+15.7pp (MLP)**, action-14class +9.6pp.
- §E0-B anticipation (causal `pred_future`): magnitude **+10.3pp (MLP)**, action +8.7pp.
- ⛔ but Diving48 **out-of-domain** = −1pp (App B) → **demo IN-DOMAIN on WalkIndia only.**

## 🎯 §2 — pick the demo QUESTION (in-domain WalkIndia · OURS-wins **AND** eye-verifiable)

| candidate question | OURS-win (probe) | a layman can verify? | verdict |
|---|---|---|---|
| **"drive / walk / drone?"** (viewpoint, from the 4 category tags) | ⏳ **TEST next** | ✅ obvious (POV vs aerial) | 🥇 best-if-it-wins; risk = too easy (both near-ceiling → tiny gap) |
| "how much motion?" (magnitude) | ✅ **+15.7pp** | ❌ anti-correlates with vision (VM30) | strong win but **looks backwards** — avoid |
| "what motion class?" (action-14) | ✅ +9.6pp | ⚠️ compound (mag×dir), not intuitive | fallback only |

> ⛔ **RESULT (2026-07-19) — cheap-probe path EXHAUSTED:** viewpoint → FROZEN *wins* (3-way OURS 0.777 vs FROZEN 0.814,
> −3.7pp; ground-vs-aerial −2.4pp) — an appearance cue, FROZEN's turf. The motion-similarity RETRIEVAL card
> (OURS 38.6% vs FROZEN 24.9% same-motion, **+13.7pp**) IS real + demo_cosmos-format (`outputs/demo/mcq/demo_mcq.mp4`),
> but WalkIndia's radial flow has ~0 *net* motion → the difference is NOT eye-verifiable. **No visible cheap card
> exists → the VLM (App A) is the only path to a visible OURS>FROZEN demo.**

## 🎬 §3 — build the demo (reuse `m17_vqa_demo.py`)

1. From the chosen question's OOF probe, dump per-clip `(clip, GT, frozen_ans, ours_ans)` → select **OURS-right / FROZEN-wrong** heroes (like `hero_extract.py`).
2. Render `m17` cards: question + options + GROUND TRUTH + 🧊 FROZEN ❌ + 🥇 OURS ✅ + honest aggregate footer.
3. (optional) use the API to *word* the MCQ / distractor options nicely — language only, never the answer.
4. Produce **2-3 curated** cards.

## 🕵️ §4 — audit + sign-off (the honesty gate)

- visual-audit **C10** (gate-backed: a passing probe number must exist) + **C8-LAYMAN** blind test.
- footer must carry the aggregate ("OURS X% vs FROZEN Y% over N clips") + "answer = probe on frozen features".
- If the layman blind test fails (GT not eye-verifiable) → do NOT ship that question; try the next candidate.

## 📋 §5 — build queue

> legend: ✅ done · ⏳ next · ⛔ blocked · 🚧 gate

| # | step | box | status |
|---|---|---|---|
| 0 | §E0 pre-check (understanding + anticipation) | 3060 | ✅ **done** (+15.7pp) |
| 1 | §2 — viewpoint probe + motion-similarity retrieval card (cheap-probe path) | 3060 | ✅ **done → EXHAUSTED** (not eye-verifiable) |
| 2 | App A — build VLM stack (`vlm.yaml` + `vlm_model` + `m18_vlm_{data,train,eval}`) | 3060 | ✅ **BUILT + verified** (751 LoC) |
| 3 | §E1 step 0-1 — data + **EARLY GATE** (OURS vs FROZEN on TempCompass subset) | 96 GB | ⏳ **next (user runs)** |
| 4 | §E1 step 2-4 — full 2-stage → full gate → render (ONLY if early gate PASSED) | 96 GB | 🚧 gate |

---

# 📚 APPENDIX

## App A — VLM-head (✅ BUILT 2026-07-19 · 3060-verified · awaiting the 96 GB early gate)

The `demo_vJPEA`-style talking VLM — the **only path to a VISIBLE OURS>FROZEN demo** (the cheap probe path is exhausted, §2).
**Code BUILT + 3060-verified** (751 LoC): `configs/vlm.yaml` + `src/utils/vlm_model.py` + `src/m18_vlm_{data,train,eval}.py`.
Full spec `plan_E1_vlm_build.md`; commands `runbook.md` §E1; state [[project_iter20_vlm_built]].

| piece | value |
|---|---|
| 🧊🥇 **two VLMs, only the encoder differs** | V-JEPA 2.1 ViT-G (frozen vs ours, native loader) → spatial-pool 24²→8² (512 tok) → MLP 5632→4096 GELU → **Qwen3-8B** (+LoRA r32, **non-thinking**), LLaVA 2-stage, MOTION QA only. LLM swap = 1 config line (projector out-dim is dynamic) |
| 📦 **data (HF-reachable only)** | TempCompass (gate: 4033 temporal MC-QA ✅) + LLaVA-Video-178K (align+instruct). SSv2/EK100 gated → excluded |
| 🚦 **EARLY GATE first (cheap de-risk)** | instruct-only capped 3000 + eval 60-video TempCompass subset. PASS iff OURS−FROZEN ≥2pp → full run; ⛔ FAIL → STOP (days saved) → forest plots |
| ⚠️ **the CRUX** | TempCompass = general video = OOD → the early gate **IS** the OOD-transfer test (Diving48 App B warned the edge fades OOD). Real risk OURS≈FROZEN |
| ✅ **3060-verified** | encoder→pool→projector both arms (512×3584 finite) · dataset/collator · bootstrap/gate logic. Full LLM train+gate → 96 GB box |

> **Next:** on the 96 GB box run `runbook.md` §E1 step 0→1 (data → early gate). The early-gate result decides go/stop.

## App B — Diving48 OUT-of-domain probe result (2026-07-19, 3060)

657 Diving48-v2 clips (fine-grained dive class, maximally OOD vs WalkIndia). `scratchpad/ood_diving_e0.py`:

| task | head | FROZEN | OURS | Δ(O−F) |
|---|---|---|---|---|
| dive-48way | LINEAR | 0.097 | 0.108 | +1.1pp |
| dive-48way | MLP | 0.113 | 0.102 | **−1.1pp** |
| takeoff-type | MLP | 0.907 | 0.896 | −1.1pp |

→ **OURS's edge does NOT transfer out-of-domain.** So the demo stays IN-domain (WalkIndia); the flashy kitchen/diving
content is off the table unless a VLM (App A) is built and re-gated.

## App C — references & background

**The two references:** 📄 `demo_cosmos.mp4` = the FORMAT (MCQ card, zero-shot ❌ vs LoRA ✅ → map to FROZEN vs OURS).
🤿 `demo_vJPEA.mp4` (Meta) = CONTENT (motion caption + action anticipation) via **V-JEPA 2 + language modeling** (a VLM).
**Synthesis:** `demo_cosmos` asks *scene* Qs → OURS loses **0/15** ([[project_iter20_demo_cosmos_impossible]]); `demo_vJPEA`
asks *motion* Qs → OURS's strength. So FORMAT from cosmos + CONTENT from vJPEA.

**Null-risk (tested & refuted in-domain):** OURS's wins are linear-probe wins; a trained MLP *could* recover FROZEN's
info → null. §E0 tested it: the MLP does NOT close the gap (holds +9.6pp, grows to +15.7pp). (OOD still fails — App B.)

**WalkIndia-200k** (`logs/non_log/stats.png` · [dataset](https://huggingface.co/datasets/anonymousML123/walkindia-200k)):
4 categories, **walk ≫ drive > drone ≈ rain** (rain tier-2 only). drive/walk/drone = camera-motion (OURS's turf);
rain = weather/appearance (FROZEN's turf, taxonomy 0.890 v 0.899 → exclude).

**websearch (2026-07-15):** V-JEPA 2 → LLM via non-tokenized early fusion, TempCompass 76.9 ([paper](https://arxiv.org/html/2506.09985v1)) ·
LLaVA 2-stage ([LLaVA](https://github.com/haotian-liu/LLaVA)) · encoder-swap shifts VQA ±5-6% *cross-family* ([OpenVision](https://arxiv.org/pdf/2505.04601)) ·
HF [transformers vjepa2](https://huggingface.co/docs/transformers/model_doc/vjepa2).

## App D — Prof.'s PIXEL-generation track (separate; 96 GB; longer-term)

Prof. Amitava green-lit (Slack 2026-07-14): *"Tuning on pretrained SDXL/Cosmos/VLLaVa will produce better visuals."*
Canonical directive in **`plan_v0_pixel_generation.md`** (RAW). Pipeline: clip → 🧊 frozen V-JEPA (enc+predictor) →
predicted future latent → 🔧 trainable projector → 🎨 LoRA-adapted SDXL→Cosmos → 🖼️ future frame(s).

```mermaid
flowchart LR
    CLIP["🎞️ clip"] --> VJ["🧊 FROZEN V-JEPA<br>enc + predictor"] --> LAT["🔮 predicted<br>future latent"]
    LAT --> PROJ["🔧 projector"] --> DIFF["🎨 LoRA SDXL → Cosmos"] --> PX["🖼️ future frame(s)"]
    style VJ fill:#5e35b1,color:#fff,font-weight:bold
    style PROJ fill:#ef6c00,color:#fff,font-weight:bold
    style DIFF fill:#ef6c00,color:#fff,font-weight:bold
    style PX fill:#2e7d32,color:#fff,font-weight:bold
```

- 🥇 immediate goal: PoC that V-JEPA predicted latents render to realistic future frames (SDXL, +1 s → multi-horizon → Cosmos video).
- ⚠️ OURS-vs-FROZEN here is **sub-perceptual** (VM29): if used to show OURS>FROZEN, feed the SAME decoder both latents + ablate the prior; the m15 tiny-MLP → fog is the *evidence* a pretrained prior is needed.
- 🗄️ **superseded/parked:** v4 DINOv2-vs-V-JEPA (off-goal + too static) · v7 probe-VQA on magnitude/direction/pace (anti-correlate VM30 or marginal) · v8/v9 VLM-head → now App A (on hold).

| 🔗 refs | `plan_v0_pixel_generation.md` · `plan_E1_vlm_build.md` · `future_prediction_questions.md` · `project_iter20_demo_cosmos_impossible.md` · `visual_mistakes.md` VM23–30 · [V-JEPA 2](https://arxiv.org/html/2506.09985v1) · [LLaVA](https://github.com/haotian-liu/LLaVA) · [diffusers SDXL](https://github.com/huggingface/diffusers) · [Cosmos](https://github.com/NVIDIA/Cosmos) |
|---|---|
