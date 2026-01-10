# POC Plan: Hybrid Judge for 3D Navigation

> **Goal**: Build a Judge Model that evaluates how well AI agents navigate in 3D environments
> **Approach**: Algorithms for ground truth + LLM for explanations
> **Status**: Phase 0 - In Progress
> **Created**: Jan 2026

---

## Hybrid Judge Architecture

**Key Insight**: Use classical shortest path algorithms for ground truth, LLM only where it adds value.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID JUDGE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ALGORITHMIC JUDGE (Primary)       LLM JUDGE (Secondary)        │
│  ┌─────────────────────────┐      ┌─────────────────────────┐  │
│  │ • Dijkstra's Algorithm  │      │ • Natural language      │  │
│  │ • A* Search             │      │   explanations          │  │
│  │ • BFS for unweighted    │      │ • VLM image reasoning   │  │
│  │                         │      │ • Subjective criteria   │  │
│  │ OUTPUT:                 │      │ • Learning to Rank      │  │
│  │ • Optimal path length   │      │                         │  │
│  │ • Efficiency score      │      │ OUTPUT:                 │  │
│  │ • Collision detection   │      │ • Why path is bad       │  │
│  │                         │      │ • Pair/List ranking     │  │
│  └─────────────────────────┘      └─────────────────────────┘  │
│           │                                │                    │
│           └────────────┬───────────────────┘                    │
│                        ▼                                        │
│               ┌─────────────────┐                               │
│               │  FINAL METRICS  │                               │
│               │  (Combined)     │                               │
│               └─────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Use What

| Metric | Algorithm | LLM | Why |
|--------|-----------|-----|-----|
| Optimal path length | Dijkstra/A* | ❌ | Mathematically provable |
| Efficiency score | `len(agent)/len(optimal)` | ❌ | Simple ratio |
| Collision detection | Grid check | ❌ | Binary yes/no |
| Path comparison | ❌ | ✅ | Explain WHY one is worse |
| VLM image reasoning | ❌ | ✅ | No graph available |
| Subjective criteria | ❌ | ✅ | "Prefer scenic route" |
| Learning to Rank | ❌ | ✅ | Pairwise/Listwise ranking |

---

## POC Phases

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ PHASE 0 │ ──► │ PHASE 1 │ ──► │ PHASE 2 │ ──► │ PHASE 3 │
│  Algo   │     │  MVP    │     │ Hybrid  │     │  Full   │
│ Ground  │     │  Test   │     │  Judge  │     │ Pipeline│
│  Truth  │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
  Current
```

---

## Phase 0: Algorithmic Ground Truth Generator (Current)

**Goal**: Build shortest path calculator with obstacles & weighted edges

### Algorithms
| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| BFS | Unweighted grids (all moves cost 1) | O(V+E) |
| Dijkstra | Weighted graphs (stairs=2, walk=1) | O(E log V) |
| A* | Heuristic-guided (faster for large grids) | O(E log V) |

### Files to Create
```
src/
├── m01_shortest_path.py   # BFS, Dijkstra, A* algorithms
├── m02_hybrid_judge.py    # Combined algo + LLM judge
├── utils/
│   ├── __init__.py
│   └── grid.py            # Grid creation, obstacles, weights
└── legacy/
    └── m01_llm_judge_validation.py  # Original LLM-only (archived)
```

### Run Commands
```bash
python src/m01_shortest_path.py --test     # Test algorithms
python src/m02_hybrid_judge.py --compare   # Compare algo vs LLM
```

---

## Phase 1: MVP Test

**Goal**: VLM navigates, Algorithm judges

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Room    │────►│   VLM    │────►│ Algorithm│
│  Image   │     │  Agent   │     │  Judge   │
└──────────┘     └──────────┘     └──────────┘
                      │                │
                 Agent Path      Optimal Path
                      │                │
                      └───────┬────────┘
                              ▼
                    Efficiency = agent/optimal
```

### Tasks
| # | Task | Output |
|---|------|--------|
| 1.1 | Get room images (AI2-THOR/Matterport3D) | Dataset |
| 1.2 | Define navigation tasks | Task list |
| 1.3 | VLM generates path from image | Agent path |
| 1.4 | Algorithm computes optimal + scores | Efficiency % |

---

## Phase 2: Hybrid Judge

**Goal**: Add LLM for explanations + ranking

| Task | Method |
|------|--------|
| Compute optimal | Dijkstra/A* |
| Score efficiency | `optimal_len / agent_len * 10` |
| Detect collisions | Grid boundary check |
| Explain errors | LLM ("Path went South unnecessarily") |
| Rank multiple paths | LLM (pairwise comparison) |

---

## Phase 3: Full Pipeline

**Goal**: Complete end-to-end system with 3DGS + Red-Blue agents

```
┌──────────────────────────────────────────────────────────────┐
│                    FULL SYSTEM PIPELINE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  3D     │    │ Nav     │    │ Hybrid  │    │ Metrics │  │
│  │  Env    │───►│ Agent   │───►│ Judge   │───►│ Output  │  │
│  │ (3DGS)  │    │ (VLM)   │    │(Algo+LLM│    │         │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                      │                                       │
│                      ▼                                       │
│               ┌─────────────┐                                │
│               │  Red-Blue   │                                │
│               │ Interaction │                                │
│               └─────────────┘                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Resources

### Datasets
| Dataset | Use Case | Link |
|---------|----------|------|
| AI2-THOR | Indoor navigation | ai2thor.allenai.org |
| Matterport3D | Real indoor scans | matterport.com |
| HM3D | Habitat environments | aihabitat.org |

### Models
| Model | Purpose |
|-------|---------|
| GPT-4V | Navigation agent + LLM explanations |
| Claude | Alternative LLM judge |

---

## References

- [LLM-as-a-Judge Guide](https://labelyourdata.com/articles/llm-as-a-judge)
- [Agent-as-a-Judge Framework](https://arxiv.org/html/2508.02994v1)
- [Plan Verification for Embodied AI](https://arxiv.org/html/2509.02761v2)

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 0: Algorithmic Ground Truth | **Complete** | 100% |
| Phase 1: MVP Test (VLM + Algo Judge) | Not Started | 0% |
| Phase 2: Hybrid Judge (Algo + LLM) | Not Started | 0% |
| Phase 3: Full Pipeline (3DGS + Red-Blue) | Not Started | 0% |

### Phase 0 Checklist
- [x] `src/m01_shortest_path.py` - BFS, Dijkstra, A* algorithms
- [x] `src/utils/grid.py` - Grid/graph utilities
- [x] `src/m02_hybrid_judge.py` - Combined algo + LLM judge
- [x] Move `m01_llm_judge_validation.py` to `src/legacy/`

---

*Last Updated: Jan 2026*
