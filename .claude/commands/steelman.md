---
description: Devil's advocate — argue the strongest possible OPPOSITE of the user's most recent claim/decision, the way a true believer in the counter-position would
argument-hint: [optional explicit claim to steelman; default = user's most recent position or decision]
---

# /steelman — argue the opposite, as the smartest opponent would

Target: $ARGUMENTS (if blank, find the user's most recent claim, decision, or stated preference in this conversation — the position they currently hold).

## Three-step protocol (do not skip step 1)

### Step 1 — State the target position precisely
Write ONE sentence: "Your current position is: <X>."
If you can't restate it without weakening it, ask for clarification BEFORE proceeding — a strawman steelman is worthless.

### Step 2 — Build the strongest case for ¬X
Argue for the OPPOSITE as someone who genuinely believes it. Rules:

- **First-principles**, not just "experts disagree". Show the actual mechanism / evidence / cost-benefit.
- **No strawmen of the user's position.** The steelman wins on the merits of its own case, not by misrepresenting X.
- **Address the user's strongest reason for X** — and explain why someone informed could still pick ¬X.
- **Concrete domain knowledge**, not generic skepticism. ("Have you considered risks?" is not a steelman.)

### Step 3 — Verdict line
End with exactly one of these three lines:

- `If any of the above lands, your X needs revision because: <pinpoint>.`
- `If none of the above lands, your X is more robust than I can challenge — record this as a stress-tested decision.`
- `My steelman is weak here because: <reason> — your X is likely sound on this axis.`

Output format:

```text
TARGET POSITION
  Your current position is: ...

THE OPPOSITE CASE  (in order of strength)

[1] <strongest counter-argument>
    Mechanism / evidence: ...
    Where X's logic breaks: ...

[2] <second counter-argument>
    ...

[3] <third counter-argument>
    ...

VERDICT
  <one of the three verdict lines above>
```

**Distinction from sibling commands**: `/brutal` critiques X's quality. `/steelman` argues ¬X is correct. `/missing` lists absences in X. `/10x` improves X. If the user wants critique not opposition, suggest `/brutal` and stop.
