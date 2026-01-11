# m02_hybrid_judge.py

## I/O

| | Description |
|---|---|
| **Input** | Agent path `["N","E","S"]`, start/goal positions, grid config |
| **Output** | Efficiency score (0-10), optimal path, LLM explanation (optional) |

## Can it run on M1 Mac?

**Yes.** API-based (OpenAI), no GPU required.

## Commands

```bash
# M1 Mac
python src/m02_hybrid_judge.py --demo           # Algo only
python src/m02_hybrid_judge.py --demo --llm     # With LLM explanation
python src/m02_hybrid_judge.py --vlm --image data/images/ai2thor/FloorPlan1/
```
