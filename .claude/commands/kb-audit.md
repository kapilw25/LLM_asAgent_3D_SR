# /kb-audit — auto-audit the mistake knowledge base before giving output

Audit the mistake KBs against the CURRENT artifacts, so no logged mistake ships twice. Run this
BEFORE presenting any figure/video/plot output to the user (the demo-loop skill calls it as step 0).

## Steps

1. Read `.claude/memory/visual_mistakes.md` — every `VM<n>` entry.
2. Read `.claude/plotting.md` — the layout contract + cell-annotated rules.
3. Read `.claude/memory/bug_log.md` headers — bug classes that could affect the artifact pipeline.
4. For the artifact(s) about to be presented (the ones produced this session): Read each rendered
   PNG / contact sheet at full resolution and check it against EVERY entry from steps 1–2.
5. Output one ASCII box table: `entry · applies? · PASS/FAIL · evidence`. Any FAIL → fix before
   presenting; if the failure is a NEW class, append it to `visual_mistakes.md` (never renumber
   existing entries).

## Contract

- Silence is not a pass — every KB entry gets an explicit verdict row (or `n/a` with a reason).
- This command is read-only except for APPENDING new `VM<n>` entries.
