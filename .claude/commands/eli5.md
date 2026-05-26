---
description: Re-explain the most recent concept / code / decision the way you'd explain it to a curious 5-year-old — zero jargon, vivid analogy, real-world grounding
argument-hint: [optional concept name; default = most recent technical thing discussed]
---

# /eli5 — explain like I'm 5

Target: $ARGUMENTS (if blank, take the most recent technical concept, code construct, decision, or piece of jargon from the conversation).

## Constraints (hard)

1. **Zero jargon.** No domain terms unless you immediately define them with a non-domain analogy. "Encoder" → "the camera that takes a photo of an idea". "POC" → "a tiny test before we cook the whole meal".
2. **One concrete analogy from everyday life.** Cooking, sports, building blocks, animals, weather, school, kitchen, post office — pick the one that maps cleanest. Use it for the whole explanation; don't switch analogies mid-paragraph.
3. **Three paragraphs maximum.** Brevity matters more than completeness — kid attention span.
4. **Stop before nuance.** Edge cases, exceptions, caveats are NOT eli5 material. Caveat: if a misunderstanding would be actively dangerous to act on, add ONE sentence at the end starting with "Grown-up footnote:".

## Output structure

Paragraph 1 — **The thing in one sentence using the analogy.**
"<thing> is like <analogy>: when you <action>, the <analogy-element> does <behavior>."

Paragraph 2 — **Why we care.**
"We care because <consequence in everyday terms>. Without it, <bad outcome>."

Paragraph 3 — **How we know it worked.**
"You can tell it's working when <observable signal in analogy terms>."

Optional final line: `Grown-up footnote: <single nuance only if a 5-year-old's mental model would lead to real harm>.`

## Self-check before sending

Re-read your draft. Cross out any word that requires a textbook to define. If you crossed out 3+ words, rewrite.

**Distinction from sibling commands**: this is translation (complex → simple), not critique. Don't add `/brutal`-style failure lists or `/missing`-style gap enumeration. Pure pedagogy.
