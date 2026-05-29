# iter16 — Runbook

```bash
# RUN 1 · m12e only
SKIP_STAGES="8c,9c" CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC 2>&1 | tee logs/iter16_poc_eval_m12e_only_$(date +%Y%m%d_%H%M%S).log
ls outputs/poc/probe_plot/eval/{head,predictor}/*.png outputs/poc/probe_plot/eval/m13_hero_*.{png,csv}
```

```bash
# RUN 2 · m12f only
SKIP_STAGES="8b,9b" CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC 2>&1 | tee logs/iter16_poc_eval_m12f_only_$(date +%Y%m%d_%H%M%S).log
ls outputs/poc/probe_plot/eval/{head,predictor,encoder}/*.png outputs/poc/probe_plot/eval/m13_hero_*.{png,csv}
```
