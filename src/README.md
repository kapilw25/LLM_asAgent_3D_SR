# src/

> See [iter/iter1/plan1.md](../iter/iter1/plan1.md) for POC plan

## Setup
```bash
pip install -r requirements.txt
```

## Run Step 0
```bash
python m01_llm_judge_validation.py             # Default (gpt-4o-mini)
python m01_llm_judge_validation.py --with-gt   # Include ground truth
python m01_llm_judge_validation.py --model gpt-4o  # Stronger model
python m01_llm_judge_validation.py --dry-run   # Show prompts only
python m01_llm_judge_validation.py --save      # Save results
```

## Files
- `data/step0/test_cases.json` - Test cases with ground truth
- `data/step0/prompts.json` - Prompt templates
- `outputs/step0/` - Results (auto-created on --save)
