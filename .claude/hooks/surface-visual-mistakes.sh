#!/usr/bin/env python3
"""
PreToolUse (Edit|Write) hook — progressive-disclosure retrieval for the visual-mistakes KB.

When Claude is about to edit a figure/plot/.tex (or a plotting script or a demo), this injects
the MATCHING logged mistakes from .claude/memory/visual_mistakes/ (one file per mistake) plus the
universal VM38 render-and-read GATE, so a fixed mistake cannot silently recur. On any non-figure
edit it exits silently (no context added). Uses only the stdlib — no jq / external deps.

Contract (PreToolUse): read tool_input on stdin, print
  {"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "..."}}
to stdout with exit 0. Claude Code wraps it in a system-reminder for the next model call.
"""
import sys, json, os, re

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

ti = data.get("tool_input", {}) or {}
fp = str(ti.get("file_path", "") or "")
content = " ".join(str(ti.get(k, "") or "") for k in ("new_string", "content", "old_string", "instruction"))
blob = (fp + " " + content).lower()

# Fire only on figure / plot / demo edits — otherwise stay silent.
if not re.search(r"\.tex|includegraphics|figure\*?}|figsize|suptitle|savefig|matplotlib|\\caption|scenepanel|fig_|_plot|demo|vqa|\.gif", blob):
    sys.exit(0)

proj = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
kb = os.path.join(proj, ".claude", "memory", "visual_mistakes")
idx = os.path.join(kb, "INDEX.md")
if not os.path.exists(idx):
    sys.exit(0)

cats = []
if re.search(r"\.tex|includegraphics|figure\*?}|\\caption|\\textwidth|scenepanel", blob):
    cats.append("latex-figure-placement")
if re.search(r"figsize|suptitle|savefig|matplotlib|dpi|xtick|ytick|ax\.|plt\.|\.png|\.pdf", blob):
    cats.append("plot-authoring")
if re.search(r"demo|vqa|\.gif|overlay|\bframe\b|\bclip\b", blob):
    cats.append("demo-readability")

ids = []
for line in open(idx, encoding="utf-8"):
    m = re.match(r"\|\s*(VM\d+)\s*\|\s*([a-z-]+)\s*\|", line)
    if m and (not cats or m.group(2) in cats):
        ids.append(m.group(1))
if "VM38" not in ids:
    ids.append("VM38")  # always attach the render-and-read gate

def _key(v):
    return int(v[2:])
ids = sorted(set(ids), key=_key)

ctx = (
    "[visual-mistakes KB — PreToolUse gate] You are editing a figure/plot. Apply the VM38 GATE "
    "before presenting: rebuild, render EVERY changed page at >=110 dpi, READ the axis labels/ticks/"
    "legend for legibility, ensure connective text between consecutive figures and no figure before "
    "the abstract, and prefer any *_bold/_text_enlarge figure variant. Relevant logged mistakes: "
    + " ".join(ids)
    + ". Open .claude/memory/visual_mistakes/<id>.md for the full rule only if it matches this edit."
)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": ctx}}))
sys.exit(0)
