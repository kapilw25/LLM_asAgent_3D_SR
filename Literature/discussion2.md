# Date: Jan 06, 2026

---

## Original Notes

> **Goal**: Development of a judge model framework for evaluating predicted paths using Learning to Rank techniques: point-based, list-based, and pair-based.
>
> - Use [LLM/VLM] judge models, focusing on metrics (path start and end points, detours, and collisions).
> - Generating prompt-based 3D environments.
> - Generate 3D meshes from 3DGS (3D Gaussian Splatting) and judge models.
>
> **Note 1**: Everything we have to build from scratch.
> **Note 2**: No external dependencies - self-contained project.

---

# Discussion Summary: Judge Model Framework for Path Evaluation

## Overview

Build a **Judge Model** from scratch - a system that evaluates how well an AI agent navigates through 3D environments.

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
│   │ Predicted    │      │ (LLM/VLM)    │      │  Score /     │ │
│   │ Path from    │      │              │      │  Ranking     │ │
│   │ Agent        │      │ Evaluates    │      │              │ │
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

## 3D Environment Generation Pipeline (Build from Scratch)

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Text Prompts  │───▶│  3D Generator  │───▶│  3D Environments│
│                │    │  (to build)    │    │                 │
└────────────────┘    └────────────────┘    └────────────────┘
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │  3DGS to Mesh  │
                                           │  Conversion    │
                                           └────────────────┘
```

---

## What is 3DGS (3D Gaussian Splatting)?

```
┌─────────────────────────────────────────────────────────────────┐
│  3DGS = 3D GAUSSIAN SPLATTING                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A technique to create 3D scenes from 2D images/videos         │
│                                                                 │
│  INPUT           PROCESS              OUTPUT                    │
│  ┌─────┐        ┌─────────┐          ┌─────────┐               │
│  │ 2D  │  ───►  │ Gaussian│   ───►   │ 3D Scene│               │
│  │Images│       │ Splats  │          │ (Mesh)  │               │
│  └─────┘        └─────────┘          └─────────┘               │
│                                                                 │
│  Application in this project:                                   │
│  • Convert real videos/images → 3D navigable environments      │
│  • Create mesh for collision detection                         │
│  • Enable agent to "walk through" the scene                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components to Build (From Scratch)

| Component | Status | Description |
|-----------|--------|-------------|
| 3D Environment Generator | To Build | Text/Video → 3D scene |
| 3DGS to Mesh Converter | To Build | Gaussian splats → navigable mesh |
| Navigation Agent | To Build | LLM/VLM that outputs paths |
| Judge Model | To Build | Evaluates path quality |
| Metrics Calculator | To Build | Start/end, detours, collisions |
