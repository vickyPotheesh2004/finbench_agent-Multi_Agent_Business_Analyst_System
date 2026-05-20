# FinBench Evaluation Report

- **Dataset:**       sample
- **Total:**         3
- **Seed:**          42
- **Model:**         gemma4:e4b
- **Timestamp:**     2026-04-24T17:14:24

## Headline Metrics

| Metric | Value |
|---|---|
| Accuracy | **0.0%** (0/3) |
| Numeric accuracy | 0.0% |
| Errors | 0/3 |
| Mean confidence | 0.000 |
| Mean latency | 435.54s |
| p50 latency | 411.33s |
| p95 latency | 411.33s |
| SniperRAG hit rate | 0.0% |

## Statistical Significance

- Baseline: **60.0%**
- Chi-square: 0.5079
- p-value: 0.476033
- Significant at α=0.05: ❌ NO

## Milestone Gates

| Gate | Threshold | Status | Actual |
|---|---|---|---|
| M1 — CI Gate | >= 50% on 10+ questions | ❌ NOT MET | 0.0% on 3 questions |
| M7 — Launch Gate | >= 82% on full FinanceBench | ❌ NOT MET | 0.0% on 3 questions |
| M8 — Public Release Gate | >= 82% AND Chi-Square p < 0.05 | ❌ NOT MET | 0.0% · p=0.476033 |

## Per-Question Breakdown

| # | ID | Document | Match | Conf | Time | Pod | Predicted |
|---|---|---|---|---|---|---|---|
| 1 | sample_01 | AAPL_FY2023_10-K.htm | ❌ | 0.00 | 520.4s | — | N/A |
| 2 | sample_02 | AAPL_FY2023_10-K.htm | ❌ | 0.00 | 411.3s | — | Analysis could not be completed — RETRIEVAL_MISS [complete a |
| 3 | sample_03 | AAPL_FY2023_10-K.htm | ❌ | 0.00 | 374.8s | — | Analysis could not be completed — RETRIEVAL_MISS [complete a |

## Reproducibility

```bash
python eval/run_eval.py --seed 42 --dataset sample
```