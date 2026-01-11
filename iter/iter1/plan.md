# POC Plan: Hybrid Judge for 3D Navigation

> **Goal**: Build a Judge Model that evaluates how well AI agents navigate in 3D environments
> **Status**: Phase 1 - VLM + A* Hybrid (Current)
> **Last Updated**: Jan 11, 2026

---

## Project Scope

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MACRO: README Vision (Multi-Agent City Sandbox)                            │
│  "Agents cooperate, compete, collide in NYC streets"                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  MESO: Phase 2 System Pipeline                                        │  │
│  │  AI2-THOR → VLM Agent → Hybrid Judge → Metrics → Red-Blue             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  MICRO: Current Work (VLM + A* Hybrid)                          │  │  │
│  │  │  VLM sees "bed top-left" → parse to grid(8,0) → A* path         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Scale | Focus | Status |
|-------|-------|--------|
| Micro | VLM + A* Hybrid precision | **Current** |
| Meso | Full evaluation pipeline | Phase 2 |
| Macro | Multi-agent city simulation | Vision |

---

## POC Phases

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PHASE 0   │ ──► │   PHASE 1   │ ──► │   PHASE 2   │
│ Algo + LLM  │     │ VLM + A*    │     │    Full     │
│ Ground Truth│     │   Hybrid    │     │  Pipeline   │
└─────────────┘     └─────────────┘     └─────────────┘
      ✓               Current
```

---

## Environment Split

| Module | Purpose | Environment | Status |
|--------|---------|-------------|--------|
| `m01_shortest_path.py` | BFS, Dijkstra, A* | M1 Mac | ✓ |
| `m02_hybrid_judge.py` | Algo + LLM evaluation | M1 Mac | ✓ |
| `m03_ai2thor_capture.py` | Capture top-down images | **GPU Server** | ✓ (120 scenes) |
| `m04_vlm_agent.py` | VLM object detection | M1 Mac (API) | ✓ |
| `m05_vlm_pipeline.py` | End-to-end pipeline | M1 Mac (API) | ✓ |
| `m06_position_parser.py` | VLM positions → grid | M1 Mac | **TODO** |
| `utils/grid.py` | Grid utilities | M1 Mac | ✓ |

---

## Phase 1: Current Work

### The Problem

VLM outputs **high-level directions**, evaluation expects **precise grid steps**.

```
VLM Path: ['South', 'East']     # 2 abstract directions
Optimal:  19 grid steps          # A* precise path
```

### Solution: VLM + A* Hybrid

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: VLM Object Detection (m04) ✓                       │
│  Output: "bed is at top-left, desk is at bottom-right"      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Position Parsing (m06) ← TODO                      │
│  Convert: "top-left" → grid(8,0)                            │
│                                                             │
│  Options:                                                   │
│    A) Rule-based: "top-left" maps to quadrant               │
│    B) VLM-based: Ask GPT-4V for exact grid cell             │
│    C) Hybrid: VLM gives region, algorithm refines           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: A* Pathfinding (m01) ✓                             │
│  Output: ["S","S","S","S","S","S","S","S","E","E",...] (19)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Evaluation (m02) ✓                                 │
│  Compare VLM-guided path vs pure A* optimal                 │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1 Checklist

- [x] VLM identifies objects + positions (m04)
- [x] End-to-end pipeline structure (m05)
- [ ] **Position Parser** - Convert VLM positions → grid (m06)
- [ ] **Hybrid Integration** - VLM detection + A* pathfinding
- [ ] Test with obstacles

---

## Phase 2: Full Pipeline (Future)

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   3D Env    │    │  Nav Agent  │    │Hybrid Judge │    │  Metrics  │  │
│  │   AI2-THOR  │───►│   VLM + A*  │───►│  Algo + LLM │───►│  Output   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘  │
│                           │                                               │
│                           ▼                                               │
│                    ┌─────────────┐                                        │
│                    │  Red-Blue   │                                        │
│                    │ Interaction │                                        │
│                    └─────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Phase 2 Checklist

- [ ] Learning to Rank (Point/Pair/List)
- [ ] 3DGS environment integration
- [ ] Red-Blue agent interactions
- [ ] Multi-room navigation

---

## Quick Commands

```bash
# M1 Mac (API-based)
python src/m01_shortest_path.py --demo
python src/m02_hybrid_judge.py --demo --llm
python src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan1/ --task "refrigerator to stove"
python src/m05_vlm_pipeline.py --scene FloorPlan301 --task "bed to desk" --llm

# GPU Server (AI2-THOR)
python src/m03_ai2thor_capture.py --scenes 50
python src/m03_ai2thor_capture.py --grid-only  # M1 Mac: add grid to existing images
```

---

## References

- [CVPR 2025: VLMs as Path Planning Evaluators](https://openaccess.thecvf.com/content/CVPR2025/papers/Aghzal_Evaluating_Vision-Language_Models_as_Evaluators_in_Path_Planning_CVPR_2025_paper.pdf)
- [MapGPT: Map-Guided Prompting (ACL 2024)](https://arxiv.org/abs/2401.07314)
