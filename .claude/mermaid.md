# Mermaid Rules for This Project

## Line Breaks
- **NEVER use `\n`** — renders as literal text
- **ALWAYS use `<br>`** for line breaks inside node labels
- Example: `A["Step 1:<br>Load Data"]`

## Node Style Template
```
style X fill:#HEX,color:#fff,font-weight:bold,font-size:28px
```

## Color Palette (white-text safe)

| Color | Hex | Usage |
|-------|-----|-------|
| Blue | `#1e88e5` | Input / Load data |
| Purple | `#8e24aa` | Loop / Iteration |
| Teal | `#00897b` | Compute / Process |
| Red | `#e53935` | Measure / Evaluate |
| Deep Orange | `#f4511e` | Decision / Branch |
| Deep Purple | `#5e35b1` | Aggregate / Collect |
| Cyan | `#00acc1` | Rank / Sort |
| Green | `#43a047` | Select / Pick |
| Pink | `#d81b60` | Final Output / Result |
| Dark Red | `#b71c1c` | Reject / Exclude |
| Blue Grey | `#546e7a` | Optional / Fallback |
| Brown | `#6d4c41` | External / Reference |

## Supported Types
- USE: `flowchart`, `graph`, `sequenceDiagram`, `classDiagram`, `stateDiagram`, `gantt`, `mindmap`
- AVOID: `quadrantChart`, `sankey`, `xychart` (require newer mermaid versions)

## VS Code Extension
- KEEP: `bierner.markdown-mermaid` only
- UNINSTALL: `mermaidchart.vscode-mermaid-chart`, `vstirbu.vscode-mermaid-preview` (conflict)

## Direction
- **Outer flowchart: always `flowchart LR`** (left-to-right at the high level)
- **Inside every `subgraph`: always declare `direction TB`** (top-to-bottom for the subgraph's internals)
- This applies uniformly — no per-diagram judgement about "pipeline vs hierarchy". Outer = LR, inner = TB.

### Gotcha — `direction TB` is a no-op without internal edges
Mermaid only honors a subgraph's `direction` directive when there are edges to lay out. If a subgraph contains multiple atoms with **no internal edges** between them, mermaid arranges them side-by-side regardless of `direction TB`.

**Solution:** add an invisible chain (`~~~`) between sibling atoms inside any multi-atom subgraph. This gives mermaid the edges it needs to honor TB without drawing visible arrows.

```
flowchart LR
    subgraph S1["group A"]
        direction TB
        a1["leaf 1"]
        a2["leaf 2"]
        a3["leaf 3"]
        a1 ~~~ a2 ~~~ a3            %% invisible chain → forces TB stacking
    end
    subgraph S2["group B"]
        direction TB
        b1 --> b2                   %% real edge → TB already honored, no chain needed
    end
    S1 --> S2
```

**Rule:** any subgraph with ≥ 2 atoms AND zero internal `-->` / `-.->` edges MUST add an invisible `~~~` chain across all its atoms.

### Gotcha — orphan top-level atoms break the LR / TB split
Per the outer-LR / inner-TB rule, every atomic node should live inside a subgraph. A bare atom at the top level (not wrapped in any subgraph) gets laid out by the outer LR engine, which can produce ambiguous flow when mixed with sibling subgraphs.

**Solution:** wrap every atom in a subgraph (even single-atom subgraphs are fine — the subgraph acts as the labeled container). Top-level edges then connect subgraph IDs to subgraph IDs, never to bare atoms.

## Compile-and-view loop (paper figures)

1. ALWAYS compile + view PNG before committing — grep audit misses dimension blowouts.
2. Target: ≤1400×800 px for 2-col half-page · ≤700×500 px for 1-col half-page.
3. If too tall: flip leaf subgraphs TB→LR, shorten labels, switch to ELK renderer.
4. Multi-row layouts: prepend `%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%` for clean grids (no zig-zag).
5. Keep recompiling until `file diag.png` shows W×H within target — never ship a render that overflows page.
6. ELK clips last char of every subgraph header → pad with trailing ` ······` so dots take the clip, not your text (e.g. `subgraph X["📚 SOTA continual-FT ······"]`).
7. Compile with `-w 1400 --scale 2` for readable text at paper scale; render <2× small with default canvas can hide truncation bugs.

```bash
awk '/^```mermaid$/{f=1;next}/^```$/{f=0}f' file.md > /tmp/diag.mmd
npx mmdc -i /tmp/diag.mmd -o /tmp/diag.png -p /tmp/puppeteer-config.json
file /tmp/diag.png   # check W×H; Read tool to view
```
