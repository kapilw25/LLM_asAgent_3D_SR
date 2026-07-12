#!/bin/bash
# demo-audit-gate.sh — PostToolUse:Bash hook (iter19 task2.2a, 2026-07-12).
# Fires after any Bash call that ran the visual demo (m14_metric_demo) or regenerated figures
# (m13_eval_plot --cross-plots). Injects a standing reminder that the artifact MUST pass the
# visual-audit agent BEFORE being presented to the user — the loop's non-skippable gate.
# Exit 0 always: this hook REMINDS, the agent verdict is the actual gate.
input=$(cat)
cmd=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)
case "$cmd" in
  *m14_metric_demo*|*"--cross-plots"*)
    echo "AUDIT GATE: this command produced visual artifacts. Before presenting ANY figure/video to the user:" >&2
    echo "  1. Read .claude/memory/visual_mistakes.md (KB auto-audit — every VM entry is a checklist item)" >&2
    echo "  2. Spawn the visual-audit agent (.claude/agents/visual-audit.md) on the rendered PNG/contact sheet" >&2
    echo "  3. AUDIT: FAIL -> fix, append NEW mistake classes to the KB, re-render, re-audit. Loop until PASS." >&2
    ;;
esac
exit 0
