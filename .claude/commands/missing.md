---
description: Systematically find what's ABSENT from the most recent plan / analysis / decision — edge cases not enumerated, stakeholders not consulted, assumptions not surfaced
argument-hint: [optional artifact path or topic; default = most recent plan or analysis in conversation]
---

# /missing — what am I missing?

Target: $ARGUMENTS (if blank, take the most recent plan, analysis, decision, or design document from the conversation).

## What this is NOT

- Not `/brutal` — don't criticize the quality of what IS written. Only enumerate what ISN'T.
- Not `/steelman` — don't argue for the opposite position. Find absences in the current one.
- Not "general feedback" — every item must be a specific named omission.

## Blind-spot checklist (apply ALL — skip a category only if the artifact genuinely doesn't apply)

```text
┌────┬───────────────────────────────┬────────────────────────────────────────────────────────┐
│  # │ Category                       │ What to scan for                                         │
├────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│  1 │ Failure modes not considered   │ What happens when X breaks, returns null, times out,    │
│    │                                │ races, runs out of memory, gets a malformed input?       │
│  2 │ Stakeholders not consulted     │ Who else is affected and hasn't been factored in?       │
│  3 │ Time horizons not modeled      │ Day 1 vs week 1 vs month 6 vs year 2 behavior?          │
│  4 │ Edge cases not enumerated      │ Zero / one / many / unicode / negative / very-large?    │
│  5 │ Hidden assumptions             │ What's implicitly true that, if false, breaks the plan? │
│  6 │ Reversal scenarios             │ How do you UNDO this if it turns out wrong?              │
│  7 │ Second-order effects           │ What does this enable / disable / incentivize next?      │
│  8 │ Comparable prior attempts      │ Has someone tried X before? What happened?              │
│  9 │ Measurement / definition of    │ Will you know if the plan worked? How? When?            │
│    │ success                        │                                                          │
│ 10 │ Cost not modeled               │ $ / time / cognitive load / opportunity / political?    │
└────┴───────────────────────────────┴────────────────────────────────────────────────────────┘
```

## Output format

For each category above where you find a real absence, emit one row:

```text
┌──────┬──────────────────────────────┬──────────────────────────────────────────┬──────────────────────────────┐
│ Cat  │ The missing thing             │ Why it matters                            │ Cheapest way to address       │
├──────┼──────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────┤
│ 1    │ <specific named omission>     │ <consequence if left unaddressed>         │ <smallest move to plug gap>   │
│ ...  │ ...                          │ ...                                        │ ...                           │
└──────┴──────────────────────────────┴──────────────────────────────────────────┴──────────────────────────────┘
```

## Rules

- **One row per real omission.** If category N has nothing missing, omit the row — don't pad.
- **No quality critique.** "Section 3 is poorly worded" is NOT missing — that's `/brutal`. Skip it.
- **Be specific to THIS artifact**, not generic to "all plans of this type". A reviewer's blind-spot list for THIS plan, not a textbook checklist.
- **Close with a one-line summary**: "The most load-bearing absence is: <which row>" — so the user knows where to focus.

**Distinction from sibling commands**: critique-of-what's-there = `/brutal` · counter-position = `/steelman` · forward-growth = `/10x` · simplification = `/eli5`. This skill is uniquely about absence.
