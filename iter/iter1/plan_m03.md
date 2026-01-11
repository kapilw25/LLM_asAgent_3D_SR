# m03_ai2thor_capture.py

## I/O

| Mode | Input | Output |
|------|-------|--------|
| `--metadata-only` | All 120 iTHOR scenes | `metadata.json` per scene (no images) |
| `--scenes 5` | 5 POC scenes | `top_down.png`, `first_person.png`, `metadata.json` |
| `--scene X` | Single scene | Full capture for one scene |

## Can it run on M1 Mac?

**No.** AI2-THOR requires:
- Linux (Ubuntu 14.04+)
- Nvidia GPU with OpenGL
- X server with GLX

## Commands (GPU Server)

```bash
python src/m03_ai2thor_capture.py --metadata-only   # 120 metadata.json (fast)
python src/m03_ai2thor_capture.py --scenes 5        # 5 images (POC)
```

## Workflow

```
GPU Server                          M1 Mac
-----------                         ------
--metadata-only (120 JSON)  ──scp──►  Develop with metadata
--scenes 5 (5 images)       ──scp──►  Test VLM pipeline
```

```bash
# Step 1: On GPU Server
python src/m03_ai2thor_capture.py --metadata-only   # 120 metadata.json
python src/m03_ai2thor_capture.py --scenes 5        # 5 POC images

# Step 2: Transfer to M1 Mac
scp -r gpu_server:~/project/data/images/ai2thor/ ./data/images/

# Step 3: On M1 Mac - develop/test
python src/m04_vlm_agent.py --test
python src/m05_vlm_pipeline.py --demo
```

## iTHOR Coverage (120 scenes)

| Category | Scenes | Count |
|----------|--------|-------|
| Kitchens | FloorPlan1-30 | 30 |
| Living Rooms | FloorPlan201-230 | 30 |
| Bedrooms | FloorPlan301-330 | 30 |
| Bathrooms | FloorPlan401-430 | 30 |

## GPU Requirement

**A10-24GB sufficient** (OpenGL rendering only)
