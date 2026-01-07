# Date: Jan 06, 2026
# Discussion Summary: Judge Model Framework for Path Evaluation

## Overview

We discussed building a **Judge Model** - a system that evaluates how well an AI agent navigates through 3D environments.

---

## System Design (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                    JUDGE MODEL FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│   │   INPUT      │      │   JUDGE      │      │   OUTPUT     │ │
│   │              │ ──── │   MODEL      │ ──── │              │ │
│   │ Predicted    │      │              │      │  Score /     │ │
│   │ Path from    │      │ Evaluates    │      │  Ranking     │ │
│   │ LLM Agent    │      │ Quality      │      │              │ │
│   └──────────────┘      └──────────────┘      └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Learning to Rank Techniques

The Judge Model uses **Learning to Rank** methods to score paths:

| Technique | How it Works (Layman Terms) |
|-----------|----------------------------|
| **Point-based** | Scores each path independently (like grading exams - each paper gets its own score) |
| **Pair-based** | Compares two paths at a time (like a tournament - which path is better?) |
| **List-based** | Ranks all paths together (like sorting candidates from best to worst) |

---

## Key Evaluation Metrics

```
┌─────────────────────────────────────────────────────────┐
│                  PATH QUALITY METRICS                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. START/END ACCURACY                                  │
│     ├── Did the agent start at the right place?        │
│     └── Did it reach the correct destination?          │
│                                                         │
│  2. DETOUR DETECTION                                    │
│     └── Did the agent take unnecessary long routes?    │
│                                                         │
│  3. COLLISION COUNT                                     │
│     └── Did the agent bump into walls/objects?         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3D Environment Generation Pipeline

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Text Prompts  │───▶│  3D Generator  │───▶│  3D Environments│
│  (4,000 total) │    │                │    │  (3,800 usable) │
└────────────────┘    └────────────────┘    └────────────────┘
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │  3DGS to Mesh  │
                                           │  Conversion    │
                                           └────────────────┘
```

- **Generated**: 4,000 prompt-based 3D environments
- **Usable**: 3,800 (after removing ~200 with errors)
- **Resources shared**: Tools to convert 3D Gaussian Splatting (3DGS) to meshes

---

## Team Contributions

| Member | Contribution |
|--------|--------------|
| **Sushant** | Judge model design, evaluation metrics definition |
| **Alam** | 3D environment generation (4,000 prompts) |
| **Kapil** | 3DGS-to-mesh resources, judge model references | 