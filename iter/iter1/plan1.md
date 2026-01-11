# POC Plan: Hybrid Judge for 3D Navigation

> **Goal**: Build a Judge Model that evaluates how well AI agents navigate in 3D environments
> **Approach**: Algorithms for ground truth + LLM for explanations
> **Status**: Phase 1 - Starting (VLM + AI2-THOR)
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
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PHASE 0   │ ──► │   PHASE 1   │ ──► │   PHASE 2   │
│ Algo + LLM  │     │ VLM + Image │     │    Full     │
│ Ground Truth│     │  AI2-THOR   │     │  Pipeline   │
│             │     │             │     │  (3DGS)     │
└─────────────┘     └─────────────┘     └─────────────┘
      ✓               Current
```

---

## Phase 0: Algorithmic Ground Truth + LLM Explanations (Complete)

**Goal**: Build shortest path calculator + LLM explanations for path evaluation

### Algorithms
| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| BFS | Unweighted grids (all moves cost 1) | O(V+E) |
| Dijkstra | Weighted graphs (stairs=2, walk=1) | O(E log V) |
| A* | Heuristic-guided (faster for large grids) | O(E log V) |

### LLM Integration
| Feature | Description |
|---------|-------------|
| `load_prompts()` | Load templates from `prompts.json` |
| `--llm` flag | Enable LLM explanations on demand |
| `_generate_explanation()` | Use templates for LLM calls |

### Files Created
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
python src/m01_shortest_path.py --test      # Test algorithms
python src/m02_hybrid_judge.py --demo       # Demo (algo only)
python src/m02_hybrid_judge.py --demo --llm # Demo with LLM explanations
```

---

## Phase 1: VLM + AI2-THOR Images (Current)

> **Environment**: AI2-THOR on Remote GPU Server, GPT-4V for VLM
> **View Types**: Top-down grid + First-person views

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1 PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GPU SERVER                          M1 MAC (API Calls)            │
│  ┌─────────────┐                    ┌─────────────────────────┐    │
│  │  AI2-THOR   │                    │      GPT-4V (VLM)       │    │
│  │  ┌───────┐  │   Save Images      │  "Navigate from bed     │    │
│  │  │ Room  │──┼──────────────────► │   to lamp..."           │    │
│  │  │ Scene │  │                    │         │               │    │
│  │  └───────┘  │                    │         ▼               │    │
│  │      │      │                    │  [North, East, North]   │    │
│  │      ▼      │                    └─────────────────────────┘    │
│  │  ┌───────┐  │                              │                    │
│  │  │Top-down│ │                              ▼                    │
│  │  │ + FPV │  │                    ┌─────────────────────────┐    │
│  │  └───────┘  │                    │   Algorithmic Judge     │    │
│  └─────────────┘                    │   + LLM Explanation     │    │
│                                     └─────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 1.1: Setup AI2-THOR on GPU Server

**File**: `src/m03_ai2thor_capture.py`

**Commands (GPU Server):**
```bash
python src/m03_ai2thor_capture.py --metadata-only   # 120 metadata.json (fast)
python src/m03_ai2thor_capture.py --scenes 5        # 5 images (POC)
```

**Workflow:**
```
GPU Server                          M1 Mac
-----------                         ------
--metadata-only (120 JSON)  ──scp──►  Develop with metadata
--scenes 5 (5 images)       ──scp──►  Test VLM pipeline
```

| Task | Description |
|------|-------------|
| Install AI2-THOR | `./setup_env.sh --gpu` on GPU server |
| Capture **120 metadata** | All iTHOR scenes (metadata only, fast) |
| Capture **5 images** | POC scenes with top_down + first_person |
| Transfer to M1 Mac | `scp -r gpu:data/images/ai2thor/ ./data/images/` |

**Output:**
```
data/images/ai2thor/
├── FloorPlan1/
│   ├── top_down.png       # Only for 5 POC scenes
│   ├── first_person.png   # Only for 5 POC scenes
│   └── metadata.json      # All 120 scenes
├── FloorPlan2/
│   └── metadata.json
...
└── FloorPlan430/
    └── metadata.json
```

#### Step 1.2: Create VLM Agent

**File**: `src/m04_vlm_agent.py`

| Task | Description |
|------|-------------|
| Load image | Read PNG from `data/images/ai2thor/` |
| Call GPT-4V | Send image + navigation prompt |
| Parse response | Extract path as list of directions |
| Return path | `["North", "East", "North", ...]` |

#### Step 1.3: Integrate with Hybrid Judge

**File**: `src/m02_hybrid_judge.py`

Add `--vlm` flag for VLM-based evaluation:
```bash
python src/m02_hybrid_judge.py --vlm --image data/images/ai2thor/FloorPlan1/
```

#### Step 1.4: End-to-End Pipeline

**File**: `src/m05_vlm_pipeline.py`

```bash
python src/m05_vlm_pipeline.py --scene FloorPlan1 --task "bed to lamp" --llm
```

---

### Files to Create (Phase 1)

| File | Purpose | Location |
|------|---------|----------|
| `m03_ai2thor_capture.py` | Capture images from AI2-THOR | GPU Server |
| `m04_vlm_agent.py` | GPT-4V agent for path generation | M1 Mac |
| `m05_vlm_pipeline.py` | End-to-end pipeline | M1 Mac |
| `data/images/ai2thor/` | Stored room images | Both |

### Dependencies (Phase 1)

| Package | Purpose | Where |
|---------|---------|-------|
| `ai2thor` | 3D simulation | GPU Server only |
| `openai` | GPT-4V API | M1 Mac |
| `Pillow` | Image handling | Both |

---

### Available Prompts (from prompts.json)

| Prompt Key | Use Case | Phase |
|------------|----------|-------|
| `point_based_judge` | Score single path | 0 |
| `pairwise_judge` | Compare A vs B | 0 |
| `llm_judge_v2_with_gt` | Full evaluation with ground truth | 0 |
| `vlm_judge_v1` | Image-based evaluation | 1 |

---

## Phase 2: Full Pipeline (3DGS + Ranking + Multi-Agent)

**Goal**: Complete end-to-end system with 3DGS, ranking, and Red-Blue agents

### Learning to Rank

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNING TO RANK                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  POINT-BASED          PAIR-BASED          LIST-BASED        │
│  ┌───────────┐       ┌───────────┐       ┌───────────┐      │
│  │ Score each│       │ Compare   │       │ Rank all  │      │
│  │ path 1-10 │       │ A vs B    │       │ paths     │      │
│  └───────────┘       └───────────┘       └───────────┘      │
│       ↓                   ↓                   ↓              │
│   [8, 5, 3]          A > B > C           [A, B, C]          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Task | Method |
|------|--------|
| Compute optimal | Dijkstra/A* |
| Score efficiency | `optimal_len / agent_len * 10` |
| Detect collisions | Grid boundary check |
| Explain errors | LLM ("Path went South unnecessarily") |
| Pairwise ranking | LLM (`pairwise_judge` prompt) |
| Listwise ranking | LLM (rank N paths at once) |

### Full System Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   3D Env    │    │  Nav Agent  │    │Hybrid Judge │    │  Metrics  │  │
│  │   AI2-THOR  │───►│   GPT-4V    │───►│  Algo + LLM │───►│  Output   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘  │
│        │                  │                   │                  │        │
│        ▼                  ▼                   ▼                  ▼        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │     m03_    │    │    m04_     │    │ m01_ + m02_ │    │   m05_    │  │
│  │   ai2thor   │    │  vlm_agent  │    │shortest_path│    │   vlm     │  │
│  │  capture.py │    │     .py     │    │hybrid_judge │    │pipeline.py│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘  │
│                           │                                               │
│                           ▼                                               │
│                    ┌─────────────┐                                        │
│                    │  Red-Blue   │                                        │
│                    │ Interaction │                                        │
│                    └─────────────┘                                        │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

| Script | Role | Input | Output |
|--------|------|-------|--------|
| `m01_shortest_path.py` | Compute optimal | Grid + start/goal | Optimal path (BFS/Dijkstra/A*) |
| `m02_hybrid_judge.py` | Evaluate + explain | Agent path + optimal | Efficiency score + LLM explanation |
| `m03_ai2thor_capture.py` | Capture 3D scenes | Scene name | Images + metadata (GPU) |
| `m04_vlm_agent.py` | Generate nav path | Image + task | Path directions `["N","E",...]` |
| `m05_vlm_pipeline.py` | **Orchestrator** | Scene + task | Full evaluation results |

### Files to Create
```
src/
├── m05_ranking.py         # Learning to Rank module
├── m06_3dgs_env.py        # 3DGS environment integration
data/
└── step0/
    └── prompts.json       # Already has pairwise_judge prompt
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
| Phase 0: Algo + LLM Ground Truth | **Complete** | 100% |
| Phase 1: VLM + AI2-THOR Images | **In Progress** | 50% |
| Phase 2: Full Pipeline (3DGS + Ranking) | Not Started | 0% |

### Phase 0 Checklist (Algo + LLM)
- [x] `src/m01_shortest_path.py` - BFS, Dijkstra, A* algorithms
- [x] `src/utils/grid.py` - Grid/graph utilities
- [x] `src/m02_hybrid_judge.py` - Combined algo + LLM judge
- [x] `data/step0/test_cases.json` - Test cases for path comparison
- [x] `data/step0/prompts.json` - Prompt templates for LLM judge
- [x] Move `m01_llm_judge_validation.py` to `src/legacy/`
- [x] Add `load_prompts()` function
- [x] Add `--llm` CLI flag
- [x] Test with `python src/m02_hybrid_judge.py --demo --llm`

### Phase 1 Checklist (VLM + AI2-THOR)
- [ ] Install AI2-THOR on GPU server
- [x] Create `m03_ai2thor_capture.py` for image capture
- [ ] Capture **5 sample scenes** only (POC, not full dataset)
- [x] Create `m04_vlm_agent.py` for GPT-4V calls
- [x] Create `m05_vlm_pipeline.py` for end-to-end
- [x] Add `--vlm` flag to `m02_hybrid_judge.py`
- [ ] Test with `python src/m05_vlm_pipeline.py --scene FloorPlan1`
- [ ] Compare top-down vs first-person VLM accuracy

### Phase 2 Checklist (Full Pipeline)
- [ ] Pairwise ranking with LLM
- [ ] Listwise ranking for multiple paths
- [ ] 3DGS environment integration
- [ ] Red-Blue agent interactions

---

*Last Updated: Jan 9, 2026*
