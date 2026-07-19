---
name: feedback-tee-logs-on-terminal
description: Every operator-facing long-running command ends with `2>&1 | tee logs/<process_name>_$(date +%Y%m%d_%H%M%S).log` — tee shows logs LIVE on the user's terminal while also writing the file
type: feedback
---

Every runbook / operator command I hand the user MUST end with:

```bash
2>&1 | tee logs/<process_name>_$(date +%Y%m%d_%H%M%S).log
```

**Why:** the user wants to WATCH logs scroll on the terminal while a timestamped file lands in
`logs/` (user request 2026-07-14, iter20). The user's shorthand was "2&>1 | logs/<name>.log" — that
exact syntax is broken (malformed redirect, no `tee`, would send logs to a non-executable path and
show NOTHING on screen); `2>&1 | tee` is the working form of what they asked for.

**How to apply:**
- `2>&1` BEFORE the pipe (merge stderr so crashes reach both terminal and file).
- `tee` (never a bare `>` redirect) — bare redirects hide the live output the user wants.
- timestamped name per src/CLAUDE.md ("Timestamped logs" rule) — re-runs never overwrite.
- pair with `set -o pipefail` in the setup block — without it `python | tee` reports tee's exit
  code and a mid-run crash looks like success (bug_log #D2, 2026-07-13).
- long jobs the user may detach from: prefix `nohup` + suffix `&`, and give a separate
  `tail -f logs/<name>.log` line for re-attaching.
