---
name: memory-graph
description: DERIVED graph view of the .claude/memory knowledge base — mermaid visualization + pointer to the machine-readable memory_graph.jsonl (MCP memory-server format). MEMORY.md stays the source of truth; regenerate this pair whenever a memory file is added/retired.
type: project
---

# 🕸️ Memory graph (derived view — source of truth = `MEMORY.md`)

> 🤖 Machine-readable twin: [`memory_graph.jsonl`](memory_graph.jsonl) — one JSON record per line in the
> **official MCP memory-server schema** (`{"type":"entity"|"relation", ...}`), so it can be dropped straight
> into `@modelcontextprotocol/server-memory` via `MEMORY_FILE_PATH` if graph *queries* are ever wanted.
> ⚠️ **Derived, not loaded**: Claude Code auto-loads `MEMORY.md` each session; nothing auto-loads this graph.
> Update discipline: edit `MEMORY.md` + the memory file FIRST, then mirror the node/edge here — never the
> reverse. If they ever disagree, `MEMORY.md` wins.

```mermaid
flowchart LR
    %% derived from MEMORY.md 2026-07-12 — node = memory/contract file, edge = active-voice relation
    IDX["📇 MEMORY.md<br>session-onboarding index<br>(auto-loaded each session)"]

    subgraph PROJ["🎯 project state"]
        PP["project_pulse"]
        HW["hardware_split<br>(+ venv_walkindia box)"]
        NA["next_actions"]
    end
    subgraph ARCH["🏗️ architecture"]
        PL["pipeline_layout"]
        CI["codebase_inventory"]
        CS["config_schema"]
        MA["iter14_motion_aux_pivot"]
        L13["legacy/iter13_multi_task"]
    end
    subgraph OPS["🛡️ ops knowledge bases"]
        BL["bug_log"]
        VM["visual_mistakes<br>VM1-VM6 append-only"]
        CV["conventions"]
        FB["feedback_no_hardcoded_defaults"]
    end
    subgraph GOV["📜 contracts + enforcers (outside memory/)"]
        VA["🕵️ agents/visual-audit.md"]
        PM["plotting.md<br>layout contract"]
        MM["mermaid.md"]
        PF["skills/preflight"]
        CM["CLAUDE.md"]
        PFOR["src/m13_eval_plot.py<br>plot_forest (ref impl)"]
    end

    IDX --> PP & HW & NA & PL & CI & CS & MA & L13 & BL & VM & CV & FB
    NA -->|depends on| HW
    NA -->|depends on| PP
    PP -->|tracks| MA
    MA -->|supersedes| L13
    CI -->|details| PL
    CS -->|complements| PL
    CV -->|condenses| CM
    FB -->|refines| CM
    BL -->|feeds guards into| PF
    VA -->|re-reads every audit +<br>appends new mistakes| VM
    VA -->|enforces| PM
    VA -->|audits output of| PFOR
    PM -->|cites ref impl| PFOR
    VM -->|prevents regressions in| PFOR
    MM -->|complements| PM

    style IDX fill:#5e35b1,color:#fff,font-weight:bold
    style VA fill:#2e7d32,color:#fff,font-weight:bold
    style VM fill:#2e7d32,color:#fff,font-weight:bold
    style PM fill:#2e7d32,color:#fff,font-weight:bold
    style PFOR fill:#2e7d32,color:#fff,font-weight:bold
```

## 🔁 The self-correcting loop encoded above (green nodes)

| step | node | action |
|---|---|---|
| 1 | `plot_forest` | renders a figure |
| 2 | `visual-audit` agent | loads `plotting.md` + `visual_mistakes.md`, Reads the PNG, verdicts |
| 3 | `visual_mistakes.md` | any NEW failure class appended as VM<n> |
| 4 | next audit | VM<n> is now a checklist item → same mistake can never ship twice |
