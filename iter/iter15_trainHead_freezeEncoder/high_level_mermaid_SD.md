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

## § 2 — 🧬 Encoder zoo (5 arms compared at eval)

```mermaid
flowchart LR
    M["🦣 V-JEPA 2.1 ViT-G<br>🧮 1.84 B · 🏛️ Meta init"]
    M --> A0["0️⃣ 🧊 frozen<br>🎯 zero-shot baseline"]
    M --> A1["1️⃣ 🏋️ pretrain · 5 ep<br>🔄 continual SSL"]
    M --> A2["2️⃣ 🔁 pretrain_2X · 10 ep<br>⚖️ compute control"]
    A1 --> A3["3️⃣ 🔧 surgery_3stage_DI · 5 ep<br>🧩 D_L → D_A → D_I"]
    A1 --> A4["4️⃣ ✂️ surgery_noDI · 5 ep<br>🧩 D_L → D_A"]
    style M fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A0 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A1 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A2 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A3 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A4 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
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

## § 5 — 🧪 iter15 6-cell paired-Δ matrix

```mermaid
flowchart LR
    subgraph EU["🔓 encoder-update · ViT-G backward · ⏱️ ~80 min / cell"]
        direction TB
        D["🅳 🏋️ pretrain_encoder<br>📦 m09a1"]
        E["🅴 🔧 surg_3stage_DI_enc<br>📦 m09c1"]
        Ff["🅵 ✂️ surg_noDI_enc<br>📦 m09c1"]
        D ~~~ E ~~~ Ff
    end
    subgraph HO["🧊 head-only · encoder frozen · ⚡ ~9 min / cell"]
        direction TB
        A["🅰️ 🏋️ pretrain_head<br>📦 m09a2"]
        B["🅱️ 🔧 surg_3stage_DI_head<br>📦 m09c2"]
        C["🅲 ✂️ surg_noDI_head<br>📦 m09c2"]
        A ~~~ B ~~~ C
    end
    D -.->|"🧪 Δ6"| A
    E -.->|"⭐ Δ5 headline"| B
    Ff -.->|"🧪 Δ7"| C
    style D fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style E fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Ff fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style A fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style B fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style C fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style EU fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style HO fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
```

---

## § 6 — 📊 Paired-Δ tests (paper §4 Results)

```mermaid
flowchart LR
    subgraph Tests["📐 paired BCa 95% CI · 🎲 10K resample"]
        direction TB
        D5["⭐ Δ5 = 🔧🔓 surg_DI_enc − 🔧🧊 surg_DI_head"]
        D6["🧪 Δ6 = 🏋️🔓 pretrain_enc − 🏋️🧊 pretrain_head"]
        D7["🧪 Δ7 = ✂️🔓 surg_noDI_enc − ✂️🧊 surg_noDI_head"]
        D5 ~~~ D6 ~~~ D7
    end
    D5 --> O5{"🔍 sign of Δ5"}
    O5 -->|"🟰 ≈ 0 · CI ∋ 0"| W["🟢 🧠 head-only WINS<br>⚡ 1/40× GPU"]
    O5 -->|"➕ Δ5 &gt; 0"| Enc["🔵 🔓 encoder margin"]
    O5 -->|"➖ Δ5 &lt; 0"| Hd["🔴 🧠 head outperforms"]
    style D5 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style D6 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style D7 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style O5 fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style W fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Enc fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Hd fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style Tests fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
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
