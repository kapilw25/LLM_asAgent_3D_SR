# m03_ai2thor_capture.py

## I/O

| Mode | Input | Output |
|------|-------|--------|
| `--metadata-only` | All 120 iTHOR scenes | `metadata.json` per scene (no images) |
| `--scenes 5` | 5 POC scenes | `top_down.png`, `metadata.json` |
| `--scene X` | Single scene | Full capture for one scene |

> **Note**: First-person views removed. Top-down only for VLM navigation (full layout visible, better spatial reasoning).

## Can it run on M1 Mac?

**No.** AI2-THOR requires Linux + Nvidia GPU + X server.

## Commands (GPU Server)

```bash
python -u src/m03_ai2thor_capture.py --metadata-only | tee logs/log1.log
python -u src/m03_ai2thor_capture.py --scenes 5 | tee logs/log2.log
```

## iTHOR Coverage (120 scenes)

| Category | Scenes | Count |
|----------|--------|-------|
| Kitchens | FloorPlan1-30 | 30 |
| Living Rooms | FloorPlan201-230 | 30 |
| Bedrooms | FloorPlan301-330 | 30 |
| Bathrooms | FloorPlan401-430 | 30 |
