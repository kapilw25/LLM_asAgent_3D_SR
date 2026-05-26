---
description: Brutally critique the most recent assistant artifact — no sugar-coating, no balance, just enumerated failures
argument-hint: [optional file path or topic to target; default = most recent significant assistant output]
---

# /brutal — no-mercy review of what I just produced

Target: $ARGUMENTS (if blank, take the most recent significant artifact from this conversation — a file edit, plan, claim, decision, or code block).

Posture rules (read all before responding):

1. **No sugar-coating.** Do not open with "this is great BUT". Do not close with "overall solid". The artifact already exists — the only useful service now is finding what's wrong with it.
2. **Apply the "fired-for-shipping" standard.** Imagine a senior reviewer who would reject this and email the team about it. What would they say?
3. **Concrete failures only.** Each item must point to a specific line, claim, decision, or omission — not "could be clearer" generalities.
4. **No fabrication.** If you can't find 5 failures, list 2. Padding the list with weak items contaminates the signal.
5. **Rank by severity.** Top of list = the failure that most damages the artifact's purpose.

Output format (ASCII table, wrapped in ```text fence per project memory rule):

```text
┌─────┬──────────┬──────────────────────────────────────────┬──────────────────────────────────────┐
│  #  │ Severity │ The failure                               │ Why it's bad / cheapest fix          │
├─────┼──────────┼──────────────────────────────────────────┼──────────────────────────────────────┤
│  1  │ CRITICAL │ <specific failure with file:line if avail>│ <consequence + concrete fix>         │
│  2  │ HIGH     │ ...                                       │ ...                                  │
└─────┴──────────┴──────────────────────────────────────────┴──────────────────────────────────────┘
```

Severity scale: CRITICAL (ship-blocker / factually wrong) · HIGH (would be flagged in review) · MEDIUM (sloppy but survivable) · LOW (taste-level nit — only include if asked).

After the table: ONE sentence — the single most damaging issue, restated. Nothing else. No softener.

**Distinction from sibling commands**: this is critique of what IS — not the absence (`/missing`), not the opposite position (`/steelman`), not the growth path (`/10x`). If the user's intent is one of those, suggest the right command and stop.
