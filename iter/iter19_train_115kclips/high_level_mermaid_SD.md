# 🧪 iter15 — Head-only vs Encoder-update surgery (paper §3 Method figures)
> ## 🎯 Paper goal:  `vjepa_surgery` [X_epochs(surgery) +X_epochs(pretrain)] ≫ `vjepa_pretrain` [2X epochs] ≫ `vjepa_frozen` on motion / temporal features
> 🎯 Claim: `head-only surgery` ≈ `encoder-update surgery` on motion features ⇒ 1/40× GPU.
> Diagrams only — paper-figure aesthetic. One concept per diagram.
> 🎨 Style: Deep Purple `#5e35b1` (Aggregate/Collect) · white bold 28px text · per `.claude/mermaid.md`.

---

## § 1 — 🔬 Research question

```mermaid
flowchart LR
    Q["❓ Does factor surgery still help<br>🧊 when the backbone is frozen?"]
    Q --> H1["🧪 Δ1: 🏋️ continual SSL &gt; 🧊 frozen<br>✅ proven (iter13)"]
    Q --> H2["🧪 Δ2 / Δ3: 🔧 surgery &gt; 🏋️ pretrain · 🔁 pretrain_2X<br>✅ proven (iter14 recipe-v3)"]
    Q --> H3["⭐ Δ5 / Δ6 / Δ7: 🧠 head-only ≈ 🔓 encoder-update<br>🎯 iter15 headline"]
    style Q fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style H1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style H2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style H3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 2 — 🧬 Encoder zoo (8 arms compared at eval · iter15 Phase 8)

```mermaid
flowchart LR
    M["🦣 V-JEPA 2.1 ViT-G<br>🧮 1.84 B · 🏛️ Meta init"]
    M --> A0["0️⃣ 🧊 frozen<br>🎯 zero-shot baseline"]
    M --> A1["1️⃣ 🏋️ pretrain · 2 ep<br>🔄 continual SSL"]
    M --> A2["2️⃣ 🔁 pretrain_2X · 4 ep<br>⚖️ Δ3 compute control"]
    A1 --> A3["3️⃣ 🔧 surg_3stage_DI · 2 ep<br>🧩 D_L → D_A → D_I"]
    A1 --> A4["4️⃣ ✂️ surg_noDI · 2 ep<br>🧩 D_L → D_A"]
    M --> A5["5️⃣ 🧠 pretrain_head · 2 ep<br>🧊 enc frozen · 🧠 head trains"]
    A1 --> A6["6️⃣ 💡 surg_3stage_DI_head · 2 ep<br>🧊 enc frozen · 🧠 head trains"]
    A1 --> A7["7️⃣ 🪒 surg_noDI_head · 2 ep<br>🧊 enc frozen · 🧠 head trains"]
    style M fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A0 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A4 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A5 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A6 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A7 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 3 — 🔗 Sequential composition + paired-Δ tests

```mermaid
flowchart LR
    F["🧊 frozen"]
    P["🏋️ pretrain"]
    P2["🔁 pretrain_2X"]
    S["🔧 surgery"]
    F -.->|"🧪 Δ1"| P
    P -.->|"🧪 Δ2"| S
    P2 -.->|"⭐ Δ3 · 🎯 causal"| S
    style F fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style P fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style P2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style S fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 4 — 🧬 Recipe-v3 winning recipe (iter14 R1 · 🏆 top-1 = 0.8456)

```mermaid
flowchart LR
    subgraph Teacher["🧊 Teacher"]
        direction TB
        T["🧊 SALT · frozen pretrain encoder<br>🚫 no EMA · 🚫 no updates"]
    end
    subgraph Student["🔓 Student"]
        direction TB
        H["🧠 LP-FT Stage 0 · head-only warmup"]
        L["✂️ Surgical subset · 4 / 8 / 8 blocks"]
        W["🔥 Single warmup over total budget"]
        O["🛡️ SPD optimizer · selective projection decay"]
        H ~~~ L ~~~ W ~~~ O
    end
    subgraph Data["📥 Data"]
        direction TB
        FV["🧩 factor views · D_L · D_A · D_I"]
        R["🔁 50% raw mp4 replay · CLEAR"]
        FV ~~~ R
    end
    subgraph Loss["🎯 Loss"]
        direction TB
        J["🎯 saliency-weighted JEPA"]
        AA["🧠 motion_aux head · CE + MSE"]
        J ~~~ AA
    end
    FV --> Student
    R --> Student
    T -->|"📡 target latents"| J
    Student -->|"📡 predicted latents"| J
    Student --> AA
    style T fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style H fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style L fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style W fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style O fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style FV fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style R fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style J fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style AA fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Teacher fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Student fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Data fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Loss fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 5 — 🧪 iter15 7-cell paired-Δ matrix (iter15 Phase 8 + pret2X compute control)

```mermaid
flowchart LR
    subgraph EU["🔓 encoder-update · ViT-G backward · ⏱️ ~25 min/cell at POC"]
        direction TB
        P1["🦣 pretrain_enc · 2 ep<br>📦 m09a1"]
        P2X["🐘 pretrain_2X_enc · 4 ep<br>📦 m09a1 · ⚖️ Δ3 control"]
        D["🅳 🏋️ pretrain_encoder<br>📦 m09a1 (= P1 above)"]
        E["🅴 🔧 surg_3stage_DI_enc<br>📦 m09c1 · 2+2=4 ep total"]
        Ff["🅵 ✂️ surg_noDI_enc<br>📦 m09c1 · 2+2=4 ep total"]
        P1 ~~~ P2X ~~~ E ~~~ Ff
    end
    subgraph HO["🧊 head-only · encoder frozen · ⚡ ~25 min / cell"]
        direction TB
        A["🅰️ 🧠 pretrain_head<br>📦 m09a2 · Meta init"]
        B["🅱️ 💡 surg_3stage_DI_head<br>📦 m09c2 · P1 init"]
        C["🅲 🪒 surg_noDI_head<br>📦 m09c2 · P1 init"]
        A ~~~ B ~~~ C
    end
    P2X -.->|"⭐ Δ3 compute-matched"| E
    P2X -.->|"⭐ Δ3 compute-matched"| Ff
    P1 -.->|"🧪 Δ6"| A
    E -.->|"🧪 Δ5"| B
    Ff -.->|"🧪 Δ7"| C
    style P1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style P2X fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style E fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Ff fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style B fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style C fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style EU fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style HO fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 6 — 📊 Paired-Δ tests · POC compute-matched verdicts (paper §4 Results)

```mermaid
flowchart LR
    subgraph Tests["📐 paired BCa 95% CI · 🎲 10K resample · POC N=220"]
        direction TB
        D3["⭐ Δ3 · compute-matched 4ep<br>🔧🔓 surg_enc − 🐘 pretrain_2X"]
        D5["🧪 Δ5 · 🔧🔓 surg_DI_enc − 💡 surg_DI_head"]
        D6["🧪 Δ6 · 🦣 pretrain_enc − 🧠 pretrain_head"]
        D7["🧪 Δ7 · 🔧 surg_DI_head − 🪒 surg_noDI_head"]
        D3 ~~~ D5 ~~~ D6 ~~~ D7
    end
    subgraph Verdicts3["⭐ Δ3 verdict per metric (POC)"]
        direction TB
        V1["🎯 top1 = −4.55 pp · 🟠 marginal · pretrain_2X wins"]
        V2["🧭 m_cos = +0.005 to +0.009 · 🟡 noise"]
        V3["🔮 future_mse = −0.040 · 🟢 surgery WINS · escapes CI"]
        V1 ~~~ V2 ~~~ V3
    end
    D3 --> Verdicts3
    style D3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style D5 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style D6 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style D7 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style V1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style V2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style V3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Tests fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Verdicts3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 7 — 🔭 Probe-trio evaluation protocol

```mermaid
flowchart LR
    enc["🧬 encoder<br>(any arm)"]
    enc -->|"🎞️ N = 1000 val clips"| pool["🧮 pooled features"]
    pool --> p1["🎯 probe_top1<br>🧪 LOOCV kNN · 🏷️ 14 motion cls"]
    pool --> p2["🧭 motion_cos<br>📐 cos(feat, RAFT flow)"]
    pool --> p3["⏭️ future_l1<br>📏 next-frame latent L1"]
    p1 --> headline["⭐ 🏆 paper headline metric"]
    style enc fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style pool fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style p1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style p2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style p3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style headline fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 8 — 🧠 motion_aux head (auxiliary supervision)

```mermaid
flowchart LR
    raw["🎬 raw mp4 clip"] -->|"🌊 m04d · RAFT optical flow"| tgt["🎯 motion target<br>🏷️ K-cls label + 📏 D-vec"]
    enc["🧬 encoder pooled feats"] --> mh["🧠 motion_aux head"]
    mh --> ce["🏷️ CE loss"]
    mh --> mse["📏 MSE loss"]
    tgt --> ce
    tgt --> mse
    ce --> L["⚖️ α · CE + β · MSE"]
    mse --> L
    style raw fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style tgt fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style enc fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style mh fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style ce fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style mse fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style L fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 9 — 🏗️ Training-loop schematic (m09a / m09c × encoder / head)

```mermaid
flowchart LR
    subgraph A1["📦 m09a1 · 🏋️🔓 pretrain_encoder"]
        direction TB
        a1a["🦣 ViT-G · 🔓 full backward"] --> a1b["🧠 motion_aux head"]
    end
    subgraph A2["📦 m09a2 · 🏋️🧊 pretrain_head"]
        direction TB
        a2a["🦣 ViT-G · 🧊 frozen"] --> a2b["🧠 motion_aux head ✅ trains"]
    end
    subgraph C1["📦 m09c1 · 🔧🔓 surgery_encoder"]
        direction TB
        c1a["🦣 ViT-G · ✂️ stage-gated backward<br>(subset 4 / 8 / 8)"] --> c1b["🧠 motion_aux head"]
        c1a -->|"🎯 saliency-weighted JEPA"| c1t["🧊 SALT teacher"]
    end
    subgraph C2["📦 m09c2 · 🔧🧊 surgery_head"]
        direction TB
        c2a["🦣 ViT-G · 🧊 frozen"] --> c2b["🧠 motion_aux head ✅ trains"]
    end
    A1 -->|"⚓ init pretrain encoder"| C1
    A1 -.->|"🧪 Δ6 paired test"| A2
    C1 -.->|"⭐ Δ5 headline · paired test"| C2
    style a1a fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style a1b fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style a2a fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style a2b fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style c1a fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style c1b fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style c1t fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style c2a fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style c2b fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style C1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style C2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 10 — 🎲 Surgery per-step data mixing (CLEAR replay × stage mode_mixture)

```mermaid
flowchart LR
    subgraph Src["📥 Per-item sources"]
        direction TB
        RAW["🎬 raw mp4<br>pretrain pool"]
        MP4M["🎬 raw mp4 + 🎭 m10 SAM mask"]
        RAW ~~~ MP4M
    end
    subgraph Fac["⚙️ stream_factor() — on-the-fly"]
        direction TB
        F["🧩 D_L · D_A · D_I<br>m11_factor_datasets"]
    end
    subgraph Coin["🎲 per-clip Bernoulli · p=0.5"]
        direction TB
        C{{"🎲 rng &lt; 0.5 ?"}}
    end
    subgraph SM["🎚️ stage mode_mixture (factor branch only)"]
        direction TB
        S1["Stage 1 · 40% steps · 🌫️ D_L heavy"]
        S2["Stage 2 · 30% steps · L:0.30 · A:0.70 · I:0.00"]
        S3["Stage 3 · 30% steps · 🔗 D_I phase"]
        S1 ~~~ S2 ~~~ S3
    end
    subgraph Out["🦣 Encoder input"]
        direction TB
        B["🦣 ViT-G batch<br>(every batch is mixed,<br>NOT first-half-then-second)"]
    end
    MP4M --> F
    RAW --> C
    F --> C
    C -->|"✅ raw · 50%"| B
    C -->|"🧩 factor · 50%"| SM
    SM --> B
    style RAW fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style MP4M fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style F fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style C fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style S1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style S2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style S3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style B fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Src fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Fac fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Coin fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style SM fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Out fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```
