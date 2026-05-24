# 📄 ECLIPTICA + CITA — ACL ARR 2026 March Review Dossier

> **Paper**: ECLIPTICA — A Framework for Switchable LLM Alignment via CITA (Contrastive Instruction-Tuned Alignment)
>
> **Authors**: Kapil Wanaskar · Gaytri Jena · Aman Chadha · Vinija Jain · Amitava Das
>
> **Submission #500** · 📅 13 Mar 2026 (modified 20 Apr 2026)
>
> **Venue**: 🏛️ ACL ARR 2026 March · 🎯 Preferred venue: **EMNLP** · 📜 License: CC BY 4.0
>
> **Preprint**: https://arxiv.org/abs/2601.06157
>
> **Keywords**: Safety & Alignment in LLMs · Resources & Evaluation · Interpretability for NLP
>
> **TL;DR**: 💡 ECLIPTICA reframes LLM alignment as instruction-driven runtime policy switching; CITA achieves **86.7%** instruction-alignment efficiency via contrastive preference learning with a mandatory KL anchor.

---

## 🎯 Headline scores

```
┌──────────────────────────┬──────┬───────┬─────────┬─────────────────────────────┐
│ Reviewer                  │ Conf │ Snd   │ Excite  │ Overall                       │
├──────────────────────────┼──────┼───────┼─────────┼─────────────────────────────┤
│ 🏛️ Meta · AC 3Z1z         │  —   │  —    │   —     │ 🔴 2 = Resubmit next cycle   │
│ 👤 Reviewer KpHA          │  4   │ 4     │  3.5    │ 🟡 3.5 = Borderline Conference│
│ 👤 Reviewer shuX          │  3   │ 2.5   │  3      │ 🔴 2 = Resubmit next cycle   │
│ 👤 Reviewer gMno          │  4   │ 2     │  2      │ 🔴 2 = Resubmit next cycle   │
└──────────────────────────┴──────┴───────┴─────────┴─────────────────────────────┘
```

```
┌────────────────────────────┬────────────────────────────────────────────────┐
│ Cross-cutting verdict       │ 3/3 reviewers + AC say RESUBMIT NEXT CYCLE     │
│ Single dissent              │ KpHA gave 3.5 (borderline accept)              │
│ Primary blockers (consensus)│ ① one-model evaluation (Llama-3.1-8B only)     │
│                            │ ② missing baselines (SimPO / KTO / safety SOTA)│
│                            │ ③ no capability-preservation eval (MMLU/etc.)  │
│                            │ ④ instruction set too small (10 types)         │
│                            │ ⑤ proofreading / repeated refs                 │
└────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 📝 Abstract

Alignment in large language models (LLMs) remains largely static: frozen after training. Methods such as DPO and GRPO typically imprint a single behavioral policy into the model weights, leaving no room for run-time policy control beyond prompt hacks or costly re-alignment cycles. We introduce **ECLIPTICA**, which reframes alignment as instruction-driven and runtime-controllable. Models are trained to accept natural-language alignment instructions as an explicit behavioral contract — covering epistemic stance, refusal boundary, and verbosity — that modulates behavior on the fly, enabling policy updates under evolving safety requirements, user roles, and governance constraints.

We introduce **CITA** (Contrastive Instruction-Tuned Alignment), which combines supervised fine-tuning with contrastive preference optimization under an explicit, mandatory geometric anchor to a reference model. The resulting objective yields a stable Riemannian chart that keeps instruction updates within a shared manifold neighborhood of the reference policy, ensuring that behavioral regimes remain nearby and traversable. This enables stable switching across alignment instructions without superficial variation or catastrophic drift.

To separate policy switching from standard instruction following, we introduce the **ECLIPTICA benchmark**, consisting of **3,000 controlled test cases** (300 prompts × 10 instruction types) in which the user request is held fixed and only the alignment instruction varies. On **Llama-3.1-8B** across five evaluation suites (ECLIPTICA, TruthfulQA, Conditional Safety, Length Control, and LITMUS), CITA achieves **86.7% instruction-alignment efficiency**, outperforming **DPO (56.1%)**, **GRPO (36.1%)**, and **PPO (20.4%)**. Together, ECLIPTICA and CITA move alignment beyond one-policy-per-checkpoint toward switchable, instruction-governed behavior aligned with modern deployment and agentic settings. 🔓 Code and dataset publicly released.

---

═══════════════════════════════════════════════════════════════════════════════
## 🏛️ META REVIEW — Area Chair 3Z1z
═══════════════════════════════════════════════════════════════════════════════

📅 19 May 2026 (modified 22 May 2026) · Overall: 🔴 **2 = Resubmit next cycle**

### 📋 Summary

The submission presents an interesting, timely approach to LLM alignment by moving away from static parameters toward switchable runtime policy execution. By creating the CITA algorithm and introducing a prompt-held-constant diagnostic benchmark (ECLIPTICA), the paper formalizes a key distinction between **task instruction-following** and **policy instruction-alignment**. The core method demonstrates notable mathematical validation via separated Fisher trust geometry, yielding compelling evaluation scores over conventional DPO and PPO baselines, particularly on truthfulness calibration and semantic cluster separation.

### ✅ Reasons to publish

- 🧮 **Decoupled knobs**: Unlike DPO (where β implicitly couples preference sharpness with reference-model regularization), CITA splits them into two independent knobs (β and λ). This architectural shift forces a stable Riemannian trust region that preserves behavioral diversity, allowing multi-regime policy switching without collapsing into a single implicit stance.
- 🎨 **ECLIPTICA dataset**: Addresses evaluation confounding by holding user queries entirely fixed while counterfactually modifying only the policy instruction.
- 📊 **Calibration win**: CITA shows **54× stronger calibration adaptation** on TruthfulQA than DPO (+0.054 vs +0.001) and reaches a superior **AQI of 55.0** on LITMUS, proving the instruction channel penetrates deep into semantic decision boundaries rather than performing surface token matching.

### ⚠️ Suggested revisions

- 🔴 **Narrow experimental evaluation + no scale/cross-arch generalizability.** The empirical validation is severely constrained, leaving major claims regarding structural adaptability unverified. By restricting all core experiments exclusively to a single model architecture at a single scale (**Llama-3.1-8B**), the paper fails to prove CITA transfers reliably across structurally distinct, open-source model families (**Mistral, Gemma, Qwen**).
- 🔴 **Limited behavioral diversity + dataset artifacts + potential eval bias.** The scope of the instruction-conditioned alignment is surprisingly narrow and relies on a highly restricted taxonomy (**only 10 distinct behavioral types**) — too small to reflect the highly nuanced, open-ended operational demands of real-world dynamic deployment.
- 🔴 **Insufficient comparative baseline depth + missing trade-off analyses.** The paper overstates its algorithmic novelty by presenting CITA in isolation from contemporary alignment paradigms that natively handle implicit policy regularization and multi-criteria optimization. The experimental suite lacks critical comparisons against SOTA reference-free or token-level objectives such as **SimPO** and **KTO**, which explicitly govern the trade-offs between optimization and policy drift.

### ✂️ Additional formatting / technical note for the authors

The manuscript requires significant and thorough proofreading prior to resubmission. It currently suffers from multiple academic text defects, including **repeated reference entries** (e.g., TruthfulQA and RLHF citations appearing redundantly).

### 🧾 Compliance flags

- Reported issues: **No**
- Publication-ethics policy compliance: privacy-preserving tool used for PEC-approved purposes only (language edits).

---

═══════════════════════════════════════════════════════════════════════════════
## 👤 REVIEW #1 — Reviewer KpHA
═══════════════════════════════════════════════════════════════════════════════

📅 23 Apr 2026 (modified 03 May 2026)

```
┌─────────────────────┬─────────────────────────────────────────────┐
│ Confidence          │ 4 = Quite sure                              │
│ Soundness           │ 4 = Strong                                  │
│ Excitement          │ 3.5                                         │
│ Overall             │ 🟡 3.5 = Borderline Conference              │
│ Reproducibility     │ 4 = Mostly reproducible                     │
│ Datasets            │ 4 = Useful                                  │
│ Software            │ 3 = Potentially useful                      │
│ Ethical concerns    │ None                                        │
│ Knowledge of paper  │ None (no outside info)                      │
└─────────────────────┴─────────────────────────────────────────────┘
```

### 📋 Paper summary

This paper tackles a genuinely practical problem: LLM alignment is usually baked into the weights at training time and then frozen, which is increasingly at odds with real-world deployments where the same base model needs to serve different roles (customer support, compliance, creative writing, etc.) with different behavioral postures. The authors propose reframing alignment as an instruction-driven, runtime-switchable control mechanism. Concrete contributions: (1) **CITA**, a training algorithm that conditions preference optimization on explicit alignment instructions and uses a mandatory KL anchor to prevent mode collapse; (2) **ECLIPTICA**, a diagnostic benchmark with 3,000 test cases designed to isolate the causal effect of alignment instructions by holding the user prompt fixed while varying only the instruction.

### ✅ Strengths

- 🎯 **Well-motivated, increasingly relevant problem.** As LLMs get deployed as agentic backbones behind multiple products, the "one model, many behavioral contracts" scenario is not hypothetical — it is happening right now. The paper articulates this deployment pain point clearly.
- 🧠 **CITA's objective is theoretically grounded.** Making the preference loss conditional on the alignment instruction I (i.e., learning π(Y|I,X)) rather than learning a single implicit preference regime π(Y|X) is a principled departure from standard DPO. The mandatory KL anchor is structurally important — without it, gradients from competing instructions would interfere or collapse into a single dominant mode. The self-quenching and directional purity analysis in the gradient derivation adds depth.
- 🎨 **ECLIPTICA benchmark is carefully designed.** Holding the user prompt X constant while varying only I is the right way to isolate instruction-conditioned alignment from superficial instruction following. Instruction-synthesis pipeline (5 judge models → BERTScore filtering → human quality gate) is more rigorous than hand-writing a few tone keywords.
- 📈 **Evaluation is broad.** Five different benchmarks (ECLIPTICA, TruthfulQA, Conditional Safety, Length Control, LITMUS) cover truthfulness, safety, verbosity control, and general alignment quality — not just one narrow dimension.

### ⚠️ Weaknesses

- 🌫️ **Core distinction from standard instruction following is murky.** Authors repeatedly claim "instruction-conditioned alignment" rather than "prompt hacks," but at inference time CITA still concatenates the alignment instruction with the user prompt (Table 1 essentially shows system-prompt-like instructions prepended to the user query). While the training objective is indeed conditional, the runtime switching mechanism is textually indistinguishable from asking a model "Respond concisely and professionally." Paper needs to more clearly articulate what CITA learns that a well-tuned model cannot already do via careful system prompt engineering. The bad cases shown for DPO/PPO/GRPO in Figure 1 look more like alignment failures than inherent limitations of static alignment.
- 💰 **No discussion of training cost.** CITA requires preference quadruples (I, X, Y⁺, Y⁻) across multiple instruction conditions. How much more data and compute vs standard DPO? Fisher metric computation and KL anchoring add overhead — negligible or prohibitive at scale? Without this, hard to judge whether switchability gains are worth the cost.
- 📏 **Only one model size tested.** All experiments on Llama-3.1-8B. For a method whose main selling point is deployment flexibility, the absence of results on larger models (e.g., 70B) or smaller ones (e.g., 3B) is a significant gap. Does switchability degrade or improve with scale?
- 🎯 **Benchmark coverage is narrow in domain.** The 300 prompts are all drawn from HH-RLHF, which skews toward safety/helpfulness scenarios. Creative writing, coding assistance, and educational tutoring — domains where style switching is arguably even more valuable — are underrepresented. Also, 10 instruction types may not capture richness of real deployment needs.
- 🔄 **No analysis of interference across many instructions.** Paper shows model can switch among 10 instructions, but what happens at 50 or 100? Does CITA's KL anchor still suffice, or do instructions start bleeding into each other? Scalability of the switchable policy space is a crucial question for deployment and is not addressed.

### ✏️ Comments / suggestions / typos

1. Table 1 is an effective teaser. Clarify in caption whether all responses come from a single trained checkpoint or different ones.
2. The gradient derivation (self-quenching, directional purity) is relegated to the appendix, but its key intuitions deserve a sentence or two in the main text for readers who skip math.
3. Figure 5 shows CITA_Instruct and CITA_NoInstruct have a clear gap, but DPO_Instruct and DPO_NoInstruct barely differ. If this means DPO effectively ignores the instruction signal, that is a strong finding — call it out explicitly.
4. The "shared manifold neighborhood" / "Riemannian chart" language is elegant but could use a concrete intuition for non-geometry readers.
5. Minor: "shared manifold neighborhood" is used before being formally defined; Figure 3 caption has some rendering issues with Δθ.

---

### 💬 Author response to Reviewer KpHA

> **Title**: "Not Just Prompting: Causal and Empirical Evidence for Instruction-Conditioned Alignment"
> 📅 03 May 2026 13:04

🙏 We thank the reviewer for the thoughtful and constructive feedback.

**🧭 Clarification — instruction-conditioned alignment vs. prompting**

We agree this distinction must be clearer. While CITA uses textual instructions at inference time, its effect is **not equivalent to system prompting**. CITA learns an **instruction-conditioned policy family π(y|I,X)**, whereas prompting only steers a fixed policy π(y|X).

Empirically, if prompting were sufficient, DPO with the same instruction concatenation should match CITA. Instead:

```
┌──────────────────────┬──────────────────┬─────────────┐
│ Benchmark             │ DPO + prompting  │ CITA         │
├──────────────────────┼──────────────────┼─────────────┤
│ TruthfulQA Δ          │ ≈ 0.001          │ ≈ 0.054 (54×)│
│ ECLIPTICA             │ ≈ 0.25           │ ≈ 0.37 (~50%)│
└──────────────────────┴──────────────────┴─────────────┘
```

Thus, prompt concatenation alone does **not** induce policy switching; instruction-conditioned preference learning is required.

This is also identifiable by construction: under ECLIPTICA's paired setup (same X, different I), any instruction-invariant policy π(y|X) incurs unavoidable loss. The only degree of freedom that explains behavior is I, making alignment instructions a **causal control variable**, not a prompt feature.

🔬 **Query-scrambled diagnostic** (added to directly test "prompting vs learned control"):

```
┌─────────────────────────────────┬──────────────────────┐
│ Condition                        │ Alignment fidelity    │
├─────────────────────────────────┼──────────────────────┤
│ Original (X, I)                  │ 0.560                 │
│ Mismatched query (X′, I)         │ 0.565  (stable)       │
│ Instruction-only (no X)          │ 0.571  (high)         │
│ Query-only (no I)                │ 0.545  (lower)        │
└─────────────────────────────────┴──────────────────────┘
```

Alignment instructions control policy selection; X governs grounding and correctness. The learned factorization **π(y|I,X) ≈ π_policy(I) ∘ π_content(X)** is precisely the intended behavior.

Across all 10 instruction modes, CITA yields consistent semantic shift (**~0.37–0.43**, 95% bootstrap CIs), versus SFT at **~0.11–0.14**, showing the effect is broad rather than driven by a single instruction type.

**🧱 Role of KL (structural, not a standard regularizer)**

KL appears in prior work, but its role here is **structural**. CITA optimizes instruction-indexed optima under a shared trust region; removing KL collapses regimes (**0.37 → 0.22, ≈68% drop**) and destabilizes training. KL enables **coexistence** of multiple nearby policies rather than stabilizing a single one.

**💰 Training cost**

CITA has **comparable cost to DPO**: 120 vs 103 minutes on A100-40GB, same memory, no reward model or online sampling. We will surface this in the main text.

**📏 Model scale**

We agree this is important. Revision will include **3B and 70B results** to study scaling of switchability. We will also add capability-preservation checks (**MMLU / GSM8K / HumanEval**) to ensure switchability does not come at the cost of general utility. Plus paired bootstrap CIs and multi-seed KL sweeps so the KL effect is quantified statistically rather than only through single-run trends.

**🎨 Benchmark scope**

ECLIPTICA is designed for **causal isolation**, not domain coverage. We will expand evaluation to **creative writing, coding, and tutoring** domains without retraining.

**🔄 Instruction scalability**

We agree scaling beyond 10 instructions is important and will include experiments with larger instruction sets to analyze interference and stability.

**🎁 Presentation improvements**

We will: clarify Table 1 (single checkpoint), surface gradient intuition in main text, explicitly highlight that DPO ignores instruction signals, and provide a more intuitive explanation of the "shared manifold neighborhood."

We also clarify Fig. 1 is **not** intended to show that DPO/PPO/GRPO cannot ever be made safe, but that standard alignment objectives do not learn the same instruction-indexed switching behavior under a fixed X. The relevant comparison is therefore not "can a system prompt improve a model?" but "does the trained model reliably bind different alignment contracts to the same user request?" Our paired design + DPO+prompting baseline directly test this.

On data construction, the 10 instruction types are **not** hand-written style tags: they are derived from preference evidence using **5 independent judge models**, semantic-agreement filtering, and human quality validation. The 300 prompts are **balanced across 12 categories**, and the evaluation further includes TruthfulQA, Conditional Safety, Length Control, and LITMUS — ECLIPTICA is a controlled diagnostic, not the only evidence source.

For **interference**, current per-instruction results show stable semantic shift in every mode, suggesting no single instruction dominates. We will add a per-instruction heatmap + larger-instruction-set study to quantify bleed-through as the policy space grows.

---

═══════════════════════════════════════════════════════════════════════════════
## 👤 REVIEW #2 — Reviewer shuX
═══════════════════════════════════════════════════════════════════════════════

📅 20 Apr 2026 (modified 03 May 2026)

```
┌──────────────────────────┬─────────────────────────────────────────────┐
│ Confidence                │ 3 = Pretty sure                              │
│ Soundness                 │ 2.5                                          │
│ Excitement                │ 3 = Interesting                              │
│ Overall                   │ 🔴 2 = Resubmit next cycle                   │
│ Reproducibility           │ 3 = Some difficulty                          │
│ Datasets                  │ 4 = Useful                                   │
│ Software                  │ 4 = Useful                                   │
│ Ethical concerns          │ None                                         │
│ Needs ethics review       │ No                                           │
│ Knowledge of authors      │ Yes (preprint not anonymized)                │
│ Knowledge of paper        │ After review process started                 │
│ Knowledge source          │ Another reviewer posted a link to the paper  │
│ Impact of knowledge       │ Not at all                                   │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### 📋 Paper summary

ECLIPTICA, a framework for treating alignment as an instruction-conditioned, runtime-controllable interface; CITA, a preference-optimization method incorporating instruction conditioning with an explicit KL anchor. Key idea: enable a single model to switch between multiple behavioral policies (safety, tone, verbosity) using natural-language instructions. Also introduces a prompt-held-constant benchmark. Experiments on Llama-3.1-8B show improved instruction sensitivity over DPO, PPO, GRPO across several benchmarks.

### ✅ Strengths

- 🎯 **Timely and relevant problem.** Treating alignment as a runtime control interface is compelling, particularly for agentic systems and multi-policy deployment settings where a single model must adapt to different behavioral contracts.
- 📊 **Empirical improvements.** Consistent gains in instruction sensitivity across multiple benchmarks, particularly on calibration and alignment-quality metrics.
- 🎨 **Benchmark contribution.** Prompt-held-constant benchmark to isolate instruction-conditioned behavior is meaningful. Dataset construction pipeline is interesting; if validated carefully (e.g., ruling out shortcut learning), it has the potential to be a useful community resource.

### ⚠️ Weaknesses

- 📏 **Limited evaluation scope.** Results restricted to a single backbone (Llama-3.1-8B), limiting claims of generality across architectures and scales.
- 🧪 **Missing capability evaluation.** Paper does not assess standard capabilities (MMLU, GSM8K, HumanEval). Given known trade-offs in alignment tuning, important to verify base capabilities are preserved.
- 🌫️ **Unclear novelty over prior work.** Method is closely related to DPO with instruction conditioning and explicit KL regularization. Paper does not clearly demonstrate behavior beyond existing approaches.
- 🧱 **KL contribution not fully substantiated.** While paper emphasizes a "mandatory KL" anchor, KL-based stabilization is standard in prior work. Current experiments do not convincingly establish this component is uniquely necessary.
- 📐 **Theoretical overstatement.** Gradient analysis largely follows standard DPO formulations; claimed properties do not appear specific to the proposed method.
- 📊 **Lack of statistical rigor.** Key ablations are single-seed and do not report variance.
- 🎨 **Benchmark construction concerns.** Instruction types are derived from query-specific preference pairs and then clustered into fixed categories. This raises the possibility that instruction classes may encode latent biases from their source distribution (e.g., certain instructions correlating with particular query types), which weakens the claim of clean causal control.
- 🐍 **Potential shortcut learning.** Because both training and evaluation vary instructions while holding the query fixed, the setup does not fully rule out degenerate solutions where the model maps instructions directly to response patterns rather than learning genuine interaction between instruction and query.
- ✂️ **Presentation.** Repeated high-level framing obscures the core technical contribution.

### ✏️ Comments / suggestions / typos

1. ✅ Evaluate **capability preservation** (MMLU, GSM8K, HumanEval).
2. 🧪 **Isolate the contribution of KL**: controlled ablation along with a joint (β, λ) sweep and multi-seed results.
3. 🔬 **Test instruction–query interaction explicitly**: diagnostics like query-scrambled evaluation (replace X while keeping I fixed) or instruction-only evaluation to verify model depends jointly on (I, X) rather than instruction shortcuts.
4. 📊 **Analyze instruction distribution**: statistics showing instruction types are balanced across query categories and semantics are not biased by the underlying data distribution.
5. 🔄 **Evaluate multi-regime stability**: per-instruction performance, loss curves, or interference analysis.
6. 📏 **Evaluate on more models** — variations in backbone and size.
7. 🧑‍⚖️ Report **inter-annotator agreement** for LLM judges as well as humans, even for the rejected samples.
8. ✂️ **Tighten presentation.**

### 📝 Limitations and societal impact

The societal/broad impact is not mentioned. However the authors do a good job of highlighting the limitations. One cannot fix all of them in a single paper, but clearly stating them helps the community.

---

### 💬 Author response to Reviewer shuX

> **Title**: "Beyond Prompting: Evidence for Learned Instruction-Conditioned Alignment"
> 📅 03 May 2026 13:19

🙏 We thank the reviewer for the detailed and constructive feedback.

**🆕 Novelty beyond DPO + prompting**

CITA is **not** "DPO + instructions" — it learns an instruction-conditioned policy family π(y|I,X), rather than a single π(y|X). If prompting were sufficient, DPO+prompting should match CITA. Instead:

```
┌──────────────┬──────────────────┬─────────────┐
│ Benchmark     │ DPO + prompting  │ CITA         │
├──────────────┼──────────────────┼─────────────┤
│ TruthfulQA Δ  │ ≈ 0.001          │ ≈ 0.054 (54×)│
│ ECLIPTICA     │ ≈ 0.25           │ ≈ 0.37 (~50%)│
└──────────────┴──────────────────┴─────────────┘
```

Instruction-conditioned preference learning — not prompt concatenation — drives switching.

**🐍 Identifiability and shortcut learning**

We agree shortcut learning must be ruled out. Under ECLIPTICA's paired construction (same X, different I), any instruction-invariant π(y|X) incurs unavoidable loss. Behavior must depend on I.

Following the reviewer's suggestion, we performed additional controlled diagnostics:

1. 🔬 **Query-scrambled evaluation.** Replacing X with mismatched query X′ yields negligible drop in alignment fidelity (**0.560 → 0.565**), while instruction-only input achieves **0.571** and query-only input **0.545**. Instructions control policy selection; X governs grounding — consistent with π(y|I,X) rather than a trivial I → response mapping.
2. 📊 **DPO vs CITA instruction sensitivity.** Near-zero difference between DPO_Instruct vs DPO_NoInstruct, whereas CITA_Instruct shows a clear gain — standard preference optimization fails to bind behavior to instructions.
3. 🔄 **Per-instruction stability.** Across all 10 instruction types, CITA maintains consistent semantic shift (**~0.37–0.43**, 95% bootstrap CIs), with low variance across modes. No instruction dominates — stable multi-regime learning.
4. 🧩 **Instruction–query disentanglement.** Evidence supports a factorized model **π(y|I,X) ≈ π_policy(I) ∘ π_content(X)**: I selects behavioral regime, X determines semantic grounding. External benchmarks (TruthfulQA, Conditional Safety) require dependence on X, ruling out trivial instruction-only mappings.

**🧱 KL is structurally necessary**

KL is common in prior work, but its role here is different: it maintains **multiple instruction-conditioned optima in a shared trust region**. Extended ablations show a clear inverted-U:

```
┌──────────────────┬──────────────────────┐
│ λ                │ ECLIPTICA score      │
├──────────────────┼──────────────────────┤
│ 0   (no KL)      │ ≈ 0.22 (collapse 68%↓)│
│ 2.3e−4 (optimal) │ ≈ 0.37                │
│ Higher           │ ↓ (over-constrains)   │
└──────────────────┴──────────────────────┘
```

KL enables **multi-regime coexistence**, not just stabilization.

**📏 Evaluation scope and capability preservation**

We will add **MMLU, GSM8K, HumanEval** to verify no capability regression, along with **3B and 70B models** to study scaling. Current results already span five benchmarks (ECLIPTICA, TruthfulQA, Conditional Safety, Length Control, LITMUS), showing consistent gains across alignment axes.

**📊 Statistical rigor**

Multi-seed results (**≥3 seeds**), paired bootstrap CIs, and joint (β, λ) sweeps to isolate KL effects rigorously.

**🎨 Benchmark construction and bias**

Instruction types are **not arbitrary clusters**: derived via 5 independent judge models, semantic-agreement filtering, and human validation, with prompts **balanced across 12 categories**. We will additionally report instruction × category distributions and per-instruction breakdowns. We will also report **inter-annotator agreement** for both LLM-judge agreement and human validation, **including rejected samples**, to make the filtering process auditable rather than only describing retained cases.

**🔄 Multi-regime stability and scalability**

Current results show stable behavior across all modes. We will include per-instruction curves, interference analysis, and **larger instruction sets (≥50)** to quantify regime scalability.

**💰 Training cost and reproducibility**

CITA remains practical relative to DPO: **120 vs 103 minutes** on the same A100-40GB, same memory footprint, no reward model, no online sampling. KL term is implemented as standard forward KL to the frozen reference, **not explicit Fisher computation**, so the geometric anchor adds minimal overhead. We will move these compute details from appendix to main text and release scripts/configs for CITA, DPO+prompting, KL-ablation, and query-scrambled settings.

**🎁 Presentation**

Reduce repeated framing, clarify the technical contribution, better separate intuition from formal derivation.

🌟 **Overall**, we believe the key contribution is a shift from static alignment to a **learned, instruction-conditioned control interface**, supported by both causal benchmark design and new empirical diagnostics that directly address shortcut learning, KL necessity, and instruction–query interaction.

---

═══════════════════════════════════════════════════════════════════════════════
## 👤 REVIEW #3 — Reviewer gMno
═══════════════════════════════════════════════════════════════════════════════

📅 19 Apr 2026 (modified 03 May 2026)

```
┌──────────────────────────┬─────────────────────────────────────────────┐
│ Confidence                │ 4 = Quite sure                               │
│ Soundness                 │ 2 = Poor                                     │
│ Excitement                │ 2 = Potentially Interesting                  │
│ Overall                   │ 🔴 2 = Resubmit next cycle                   │
│ Reproducibility           │ 2 = Hard pressed                             │
│ Datasets                  │ 3 = Potentially useful                       │
│ Software                  │ 3 = Potentially useful                       │
│ Ethical concerns          │ None                                         │
│ Needs ethics review       │ No                                           │
│ Knowledge of authors      │ No                                           │
│ Knowledge of paper        │ N/A                                          │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### 📋 Paper summary

This paper studies instruction-conditioned alignment to enable switchable alignment policies for LLMs.

### ✅ Strengths

- 🎯 Studies an important problem that fits within the scope of *ACL.
- 📊 Empirical studies are applied to demonstrate the usefulness of CITA.

### ⚠️ Weaknesses

- 🌫️ **Paper overstates its novelty.** The paper could benefit from comparison with:
  - **SimPO, KTO** — use regularization to trade-off policies
  - **Multi-objective decoding** baselines
- 📏 **Instruction set has 10 types**, which is very limited.
- 🌐 **Multilingual analysis is missing.**
- 📐 **Empirical studies use Llama-3.11-8B** (sic). More models should be used for evaluations to demonstrate generalizability.
- ⚖️ **Trade-off analysis on helpfulness, safety when using switchable policy should be more detailed.**
- 🛡️ Paper could benefit from comparison with **other safety baselines**.
- 🥷 CITA should be tested against **SOTA jailbreak attacks**.
- ✂️ Paper needs significant and careful proofreading. Contains **repeated references** (TruthfulQA, RLHF) and a **hallucinated reference (OPT-IML)**.

### ✏️ Comments / suggestions / typos

- The paper manipulates the space between main body text and figure/table captions, which makes the paper harder to read than it should.

---

### 💬 Author response to Reviewer gMno

> **Title**: "Bounded Switching, Baselines, and Robustness"
> 📅 03 May 2026 13:40

🙏 We thank the reviewer for the constructive feedback.

**📚 Novelty and baselines**

We agree the paper should better position CITA against SimPO, KTO, multi-objective decoding, and safety baselines. We will add these comparisons.

CITA is **not** intended as another single-policy preference optimizer; it learns an instruction-conditioned policy family π(y|I,X), where the same X must realize different behavioral contracts under different I. This differs from SimPO/KTO-style trade-offs and decoding-time control, which optimize or steer a **single** policy at inference time but do not directly train prompt-held-constant, instruction-indexed switching.

```
┌──────────────┬──────────────────┬─────────────┐
│ Benchmark     │ DPO + prompting  │ CITA         │
├──────────────┼──────────────────┼─────────────┤
│ ECLIPTICA     │ ≈ 0.25           │ ≈ 0.37       │
│ TruthfulQA Δ  │ ≈ 0.001          │ ≈ 0.054      │
└──────────────┴──────────────────┴─────────────┘
```

The gain is **not** merely from adding instructions to the prompt.

**📏 Scope: instructions, multilinguality, and model scale**

The 10 instruction types are a **controlled diagnostic set**, not a claim of deployment completeness; goal is causal isolation: hold X fixed and vary only I. We already analyze per-instruction behavior: CITA maintains ≈ **0.37–0.43** semantic shift with 95% bootstrap CIs, while SFT stays near ≈ **0.11–0.14** — gain not driven by one easy mode.

For the single-backbone concern, we added cross-backbone checks on **Qwen and GPT-OSS** variants. The same trend holds: CITA improves instruction-alignment efficiency over DPO-style baselines, suggesting the effect is **not Llama-specific**. We will report these results and expand multilingual coverage.

**⚖️ Helpfulness / safety trade-offs**

We will strengthen this analysis by reporting per-instruction helpfulness, refusal, safety, and utility trade-offs, especially for strict vs permissive policies. Importantly, **permissive ≠ unsafe**: target behavior is safer assistance with different refusal boundaries, not removal of safety constraints. Explicit trade-off tables showing when CITA increases safety conservatism, when it preserves helpfulness, and when it refuses.

**🧪 Capability preservation**

We will add **MMLU, GSM8K, HumanEval** to verify alignment switching does not degrade general utility. Directly addresses the concern that stronger alignment control may come at the cost of base capabilities.

**🛡️ Safety baselines and jailbreak attacks**

We agree switchability must remain bounded. Jailbreak robustness is a broader open problem: safety-tuned LLMs remain vulnerable to optimized attacks (e.g., Zou et al., 2023; Carlini et al., 2023). Our claim is not that CITA solves adversarial robustness, but that it enables **bounded instruction-conditioned switching**.

**🔬 Additional diagnostics**

Across all 10 instruction modes, CITA shows stable semantic shift of ≈ **0.37–0.43**, while SFT remains ≈ **0.11–0.14**. Query-scrambled evaluation shows fidelity remains stable when X is replaced by X′ (**0.560 → 0.565**), instruction-only remains high (**0.571**), and query-only is lower (**0.545**). I controls the behavioral contract; X governs grounding/correctness. Per-instruction plots and confidence intervals will be included.

**🧱 KL / interference analysis**

We will also add a clearer λ sweep and interference view. Preliminary ablations show an inverted-U: **λ = 0 collapses to ≈ 0.22**, **moderate λ ≈ 2.3e−4 gives ≈ 0.37**, and larger λ over-constrains switching. Supports our claim that KL is not merely a generic regularizer but a **trust-region mechanism** for keeping multiple instruction-conditioned regimes co-located. We will report per-instruction loss/shift curves to detect bleed-through between regimes.

**🔁 Reproducibility and cost**

CITA is comparable to DPO in cost: **120 vs 103 minutes** on the same A100-40GB setup, same memory footprint, no reward model, no online sampling. We will release data, scripts, configs, evaluation prompts, and move key implementation/compute details into the main text. We will also report exact train/eval splits, random seeds, decoding settings, and judge prompts to make the protocol fully auditable.

**🎁 Presentation and formatting**

We will proofread carefully.

🌟 **Overall**, we will revise the paper to make the contribution more precise and better bounded: CITA studies **learned instruction-conditioned alignment switching**, and the added baselines, scaling, multilingual, trade-off, capability, and attack evaluations will clearly delimit its scope.

---

═══════════════════════════════════════════════════════════════════════════════
## 📋 Submission metadata
═══════════════════════════════════════════════════════════════════════════════

```
┌──────────────────────────────────┬──────────────────────────────────────────┐
│ Paper Type                        │ Long                                      │
│ Research Area                     │ Language Modeling                         │
│ Research-area keywords            │ Safety & Alignment in LLMs · Preference   │
│                                  │ Optimization · Instruction Following ·    │
│                                  │ RLHF · Evaluation Benchmarks              │
│ Contribution types                │ Model analysis & interpretability · NLP   │
│                                  │ engineering experiment · Publicly         │
│                                  │ available software / pre-trained models · │
│                                  │ Data resources                            │
│ Languages studied                 │ English                                   │
│ Reassignment request (AC)         │ ✅ Yes — wants different AC               │
│ Reassignment request (Reviewers)  │ ✅ Yes — wants different reviewer set     │
│ Justification                     │ Previous submission desk-rejected without │
│                                  │ review; no reviewers or AC were assigned. │
│ Preprint                          │ ✅ Yes (non-anonymous, arXiv 2601.06157)  │
│ Consent to share data             │ ✅ Yes                                    │
│ Submission #                      │ 500                                       │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

### ✅ Responsible-NLP checklist

```
┌─────────────────────────────────────────┬──────────────────────────────────┐
│ A1 Limitations section                   │ ✅ Yes                            │
│ A2 Potential risks                       │ ✅ Yes (§ 6.3 Safety & Ethics)    │
│ B  Use or create scientific artifacts    │ ✅ Yes                            │
│ B4 PII / offensive content               │ ❌ No (synthetically generated)   │
│ B6 Statistics for data                   │ ✅ Yes (Appendix D)               │
│ C  Computational experiments             │ ✅ Yes                            │
│ C2 Setup & hyperparameters               │ ✅ Yes (Appendix C — HW, HPs,    │
│                                          │ Optuna HPO search)                │
│ C3 Descriptive statistics                │ ✅ Yes (§ 4, Appendix E & F)      │
│ D  Human subjects / annotators           │ ❌ No                             │
│ E  AI assistants in research/writing     │ ✅ Yes (code debugging + LaTeX    │
│                                          │ formatting — acknowledged)        │
└─────────────────────────────────────────┴──────────────────────────────────┘
```

---

> 🏛️ OpenReview · ACL ARR 2026 March · Submission #500
> 🔗 Previous URL: `/forum?id=9wcgjNvw75`
> © 2026 OpenReview · CC BY 4.0
