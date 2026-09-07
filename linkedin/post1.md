# Launch Posts #1 — FactorJEPA (arXiv 2608.01049)

> LinkedIn: **Option A** (links inline, simplest) or **Option B** (links in first comment, better reach — recommended).
> Twitter/X: **Option C** (thread) — A/B don't fit X's 280-char tweets, so X gets its own format.

> **Formatting note:** LinkedIn has no native bold/italic. The **bold** and *italic* below are Unicode
> glyphs baked into the text, so they copy-paste straight in and render on LinkedIn. Trade-off: screen
> readers can't read them and they aren't searchable, so I kept them sparse (numbers, the 3 channels, the
> two names, one italic). If they look like boxes in your editor, that's just your editor's font; LinkedIn is fine.

## 🎬 Video to upload

**Primary (recommended):** `linkedin/linkedin_two_encoders.mp4`
> Stitched two-card clip, 9.4 s, 854×480. Turn-direction first (the hook), motion-speed second (the numbers). Each reveal holds ~1.2 s. LinkedIn autoplays it muted and loops it, and all the text is on-screen, so no audio needed.

**Single-clip alternative:** `linkedin/card_turn_direction.mp4` (3.5 s) — use this if you'd rather post just the one turn-direction example.

---

## ✅ OPTION A — links inside the post (simplest)

Copy everything in the box below, paste into LinkedIn, then attach the video.

```text
Same question, two video encoders: "Which way is this vehicle turning?"
𝗢𝗻𝗲 𝗴𝗲𝘁𝘀 𝗶𝘁 𝗿𝗶𝗴𝗵𝘁, 𝗼𝗻𝗲 𝗱𝗼𝗲𝘀𝗻'𝘁. The only thing that changed is the backbone.

Frozen video foundation models (V-JEPA 2.1) turn out to be surprisingly 𝘮𝘰𝘵𝘪𝘰𝘯-𝘣𝘭𝘪𝘯𝘥 on crowded, chaotic Global South streets, exactly the setting where world models will have to work. Across 1,825 held-out clips, a frozen encoder scores 𝟲𝟬.𝟵% on a simple motion-speed question. Ours scores 𝟲𝟵.𝟴%, using an 𝘪𝘥𝘦𝘯𝘵𝘪𝘤𝘢𝘭 probe head. Only the backbone changed.

Excited to share 𝗙𝗮𝗰𝘁𝗼𝗿𝗝𝗘𝗣𝗔, now on arXiv.

The idea: instead of predicting the future as one entangled latent, FactorJEPA splits it into three explicit channels, 𝗹𝗮𝘆𝗼𝘂𝘁, 𝗮𝗴𝗲𝗻𝘁𝘀, and 𝗶𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝗼𝗻𝘀, with a visibility gate for occluded agents, and adapts a frozen V-JEPA 2.1 through a staged "factor-curriculum" surgery.

At the full 115k-clip scale it beats the strongest fine-tuning rival on all four predictive diagnostics (in 95%-CI units): mask-ratio 𝟰𝟯.𝟯𝘅, future-frame 𝟯𝟯.𝟮𝘅, motion-cosine 𝟮𝟬.𝟬𝘅, causal 𝟭𝟯.𝟵𝘅.

We also release 𝗗𝗘𝗡𝗦𝗘𝗪𝗢𝗥𝗟𝗗: 115k video-clips of drive, walk, and aerial video across 22 Indian cities.

Paper: https://arxiv.org/abs/2608.01049
Project page + demos: https://kapilw25.github.io/factorjepa/
Dataset: https://huggingface.co/datasets/anonymousML123/denseworld-115k
Code: https://github.com/kapilw25/factorjepa

Huge thanks to my co-authors @Gaytri Jena, @Aman Chadha, @Vinija Jain, @Vasu Sharma, @Amitava Das. Questions and feedback very welcome.

#MachineLearning #ComputerVision #WorldModels #JEPA #AIResearch
```

---

## ✅ OPTION B — links in the FIRST COMMENT (better reach)

LinkedIn tends to throttle posts that carry outbound links. To maximize reach, post the body **without** the links, then drop them in the first comment right after posting.

### Post body (paste, then attach the video)

```text
Same question, two video encoders: "Which way is this vehicle turning?"
𝗢𝗻𝗲 𝗴𝗲𝘁𝘀 𝗶𝘁 𝗿𝗶𝗴𝗵𝘁, 𝗼𝗻𝗲 𝗱𝗼𝗲𝘀𝗻'𝘁. The only thing that changed is the backbone.

Frozen video foundation models (V-JEPA 2.1) turn out to be surprisingly 𝘮𝘰𝘵𝘪𝘰𝘯-𝘣𝘭𝘪𝘯𝘥 on crowded, chaotic Global South streets, exactly the setting where world models will have to work. Across 1,825 held-out clips, a frozen encoder scores 𝟲𝟬.𝟵% on a simple motion-speed question. Ours scores 𝟲𝟵.𝟴%, using an 𝘪𝘥𝘦𝘯𝘵𝘪𝘤𝘢𝘭 probe head. Only the backbone changed.

Excited to share 𝗙𝗮𝗰𝘁𝗼𝗿𝗝𝗘𝗣𝗔, now on arXiv.

The idea: instead of predicting the future as one entangled latent, FactorJEPA splits it into three explicit channels, 𝗹𝗮𝘆𝗼𝘂𝘁, 𝗮𝗴𝗲𝗻𝘁𝘀, and 𝗶𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝗼𝗻𝘀, with a visibility gate for occluded agents, and adapts a frozen V-JEPA 2.1 through a staged "factor-curriculum" surgery.

At the full 115k-clip scale it beats the strongest fine-tuning rival on all four predictive diagnostics (in 95%-CI units): mask-ratio 𝟰𝟯.𝟯𝘅, future-frame 𝟯𝟯.𝟮𝘅, motion-cosine 𝟮𝟬.𝟬𝘅, causal 𝟭𝟯.𝟵𝘅.

We also release 𝗗𝗘𝗡𝗦𝗘𝗪𝗢𝗥𝗟𝗗: 115k video-clips of drive, walk, and aerial video across 22 Indian cities.

Links in the first comment.

Huge thanks to my co-authors @Gaytri Jena, @Aman Chadha, @Vinija Jain, @Vasu Sharma, @Amitava Das. Questions and feedback very welcome.

#MachineLearning #ComputerVision #WorldModels #JEPA #AIResearch
```

### First comment (paste as the first reply)

```text
Links:
Paper : 
https://arxiv.org/abs/2608.01049
Project page + Demos: 
https://kapilw25.github.io/factorjepa/
Dataset : 
https://huggingface.co/datasets/anonymousML123/denseworld-115k
Surgery checkpoints : 
https://huggingface.co/datasets/anonymousML123/factorjepa-outputs
Code : 
https://github.com/kapilw25/factorjepa
```

---

## 🐦 OPTION C — Twitter / X thread

X isn't LinkedIn: 280 chars per tweet, threads beat walls of text, Unicode-bold reads as spammy, and X down-ranks outbound links. So this is a plain-text 6-tweet thread with the links in a later tweet. Attach the video to Tweet 1. Every tweet is under 280.

**Tweet 1** — attach `linkedin_two_encoders.mp4`

```text
Same question, two video encoders: "which way is this vehicle turning?"

One gets it right, one doesn't. The only thing that changed is the backbone.

A thread on FactorJEPA, our new paper 👇
```

**Tweet 2**

```text
Frozen video foundation models (V-JEPA 2.1) are surprisingly motion-blind on the crowded, chaotic Global South streets where world models must work.

Across 1,825 held-out clips: frozen 60.9% vs ours 69.8% on a motion probe. Same probe head, only the backbone changed.
```

**Tweet 3**

```text
The idea: stop predicting the future as one entangled latent.

FactorJEPA splits it into 3 explicit channels, layout, agents, and interactions, with a visibility gate for occluded agents, and adapts a frozen V-JEPA 2.1 via a staged "factor-curriculum" surgery.
```

**Tweet 4**

```text
At the full 115k-clip scale, FactorJEPA beats the strongest fine-tuning rival on all 4 predictive diagnostics (in 95%-CI units):

• mask-ratio 43.3x
• future-frame 33.2x
• motion-cosine 20.0x
• causal 13.9x
```

**Tweet 5**

```text
We also release DENSEWORLD: 115k video-clips of drive, walk, and aerial video across 22 Indian cities.

📄 Paper: arxiv.org/abs/2608.01049
🌐 Project: kapilw25.github.io/factorjepa
🤗 Data: huggingface.co/datasets/anonymousML123/denseworld-115k
💻 Code: github.com/kapilw25/factorjepa
```

**Tweet 6**

```text
Work with my co-authors @Gaytri @Aman @Vinija @Vasu @Amitava (swap in real @handles).

Questions and feedback very welcome 🙏

#MachineLearning #ComputerVision #WorldModels
```

**Single-tweet TL;DR** (skip the thread; Premium fits it in one, otherwise use it as Tweet 1 + a link reply)

```text
FactorJEPA: frozen video encoders are motion-blind on crowded Global South streets. We factorize future prediction into layout / agent / interaction channels and beat every fine-tuning rival on 4 predictive metrics.

📄 arxiv.org/abs/2608.01049  🌐 kapilw25.github.io/factorjepa
```

---

## 📋 Before you hit "Post"

- [ ] **Tag the co-authors for real** — the `@Gaytri Jena` etc. above are plain text. In LinkedIn's editor, delete each name and re-type `@`, then pick the person from the dropdown so they get notified (drives first-hour reach).
- [ ] **Attach the video** (`linkedin_two_encoders.mp4`) as the post media.
- [ ] **`anonymousML123` handle** — the Hugging Face links sit under an "anonymous"-looking username on a de-anonymized post. Rename the HF namespace first if you want it to match your name, or leave it.
- [ ] **Accessibility trade-off** — the Unicode bold/italic is invisible to screen readers. If inclusive reach matters more than visual pop, tell me and I'll ship a plain-text version.
- [ ] **First hour matters** — reply to early comments quickly; it compounds reach.
- [ ] Video is 480p (source resolution) and has no audio (by design).

## 📊 Every number here is from the paper / figures
- 60.9% vs 69.8%, 1,825 held-out clips — motion-speed probe (project-page demo).
- 43.3x / 33.2x / 20.0x / 13.9x at full 115k — forest plot (FactorJEPA vs strongest competitor, 95%-CI units).
- 115,687 clips (115k), 22 cities — DENSEWORLD.
