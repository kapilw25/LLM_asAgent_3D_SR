# Plan: VLM + A* Hybrid Navigation

## Problem Statement

VLM outputs **high-level directions** but evaluation expects **precise grid steps**.

```
$ python src/m05_vlm_pipeline.py --scene FloorPlan301 --task "bed to desk" --llm

Result:
  VLM Path: ['South', 'East']     # 2 abstract directions
  Length: 2 (optimal: 19.0)        # A* expects 19 grid steps
  Efficiency: 5.0/10
  Reached Goal: False
```

**Root Cause:** VLM gives semantic directions ("go south, then east"), not grid-cell-by-cell paths.

---

## Research Finding

> CVPR 2025: VLMs have a "vision bottleneck" - they struggle with precise spatial counting and path perception.

---

## Solution Options

| # | Approach | How it works | Verdict |
|---|----------|--------------|---------|
| 1 | Grid Overlay | Draw grid lines on image, VLM counts cells | VLM still miscounts |
| 2 | **Hybrid** | VLM identifies positions → A* calculates path | **Recommended** |
| 3 | VLMaps | Fuse VLM features with 3D reconstruction | Complex, needs depth |

---

## Recommended: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Top-down image + "Navigate from bed to desk"        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: VLM Object Detection (m04)                         │
│  Output: "bed is at top-left, desk is at bottom-right"      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Position Parsing (NEW)                             │
│  Convert: "top-left" → grid(8,0), "bottom-right" → grid(0,11)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: A* Pathfinding (m01)                               │
│  Input: start=(8,0), goal=(0,11), obstacles                 │
│  Output: ["S","S","S","S","S","S","S","S","E","E",...] (19)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Evaluation (m02)                                   │
│  Compare VLM-guided path vs pure A* optimal                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

| Step | Module | Status | Description |
|------|--------|--------|-------------|
| 1 | m04 | Done | VLM identifies objects + positions |
| 2 | NEW | TODO | Parse VLM positions → grid coordinates |
| 3 | m01 | Done | A* calculates precise path |
| 4 | m02 | Done | Evaluate path efficiency |
| 5 | m05 | TODO | Integrate hybrid pipeline |

---

## Key Insight

**VLM strength:** Visual understanding, object identification, spatial relationships
**VLM weakness:** Precise counting, grid-level navigation

**A* strength:** Precise pathfinding, obstacle avoidance
**A* weakness:** Needs explicit coordinates, no visual understanding

**Hybrid:** Use each for what it's good at.

---

## References

- [CVPR 2025: VLMs as Path Planning Evaluators](https://openaccess.thecvf.com/content/CVPR2025/papers/Aghzal_Evaluating_Vision-Language_Models_as_Evaluators_in_Path_Planning_CVPR_2025_paper.pdf)
- [MapGPT: Map-Guided Prompting (ACL 2024)](https://arxiv.org/abs/2401.07314)
- [VLMaps for Robot Navigation](https://vlmaps.github.io/)
