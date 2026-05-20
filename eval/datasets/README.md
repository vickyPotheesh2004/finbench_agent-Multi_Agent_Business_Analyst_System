# FinBench Custom Eval Datasets

## Purpose

Hand-crafted ground-truth question/answer pairs for testing the FinBench 
pipeline against real SEC filings. Separate from the official Patronus AI 
FinanceBench benchmark.

## Format

JSONL — one question per line:
```json
{"qid":"AAPL-001","company":"Apple Inc.","doc_type":"10-K","fiscal_year":"FY2023","question":"...","answer":"...","unit":"million","category":"numerical","difficulty":"easy"}
```

## Field Reference

| Field | Required | Example | Notes |
|-------|----------|---------|-------|
| qid | yes | "AAPL-001" | Unique ID, prefix=ticker |
| company | yes | "Apple Inc." | Must match document metadata |
| doc_type | yes | "10-K" | |
| fiscal_year | yes | "FY2023" | |
| question | yes | "What was..." | Natural language |
| answer | yes | "383285" | Number only (no commas, no $) |
| unit | optional | "million" | million/billion/%/USD per share |
| category | optional | "numerical" | numerical/narrative/calculation |
| difficulty | optional | "easy" | easy/medium/hard |

## Files

| File | Company | Doc | # Qs |
|------|---------|-----|------|
| apple_fy2023.jsonl | Apple Inc. | AAPL_FY2023_10-K.html | 20 |
| microsoft_fy2023.jsonl | Microsoft | MSFT_FY2023_10-K.html | 20 |
| nvidia_fy2024.jsonl | NVIDIA | NVDA_FY2024_10-K.html | 20 |
| amazon_fy2023.jsonl | Amazon | AMZN_FY2023_10-K.html | 20 |
| alphabet_fy2023.jsonl | Alphabet | GOOGL_FY2023_10-K.html | 20 |
| meta_fy2023.jsonl | Meta Platforms | META_FY2023_10-K.html | 20 |
| tesla_fy2023.jsonl | Tesla | TSLA_FY2023_10-K.html | 20 |

**Total: 140 questions across 7 megacap 10-K filings.**

## Question Categories

1. **Income statement** — revenue, net income, gross profit, operating expenses, R&D, SG&A
2. **Balance sheet** — total assets, liabilities, equity, cash, inventory, debt, A/R, A/P
3. **Cash flow** — operating CF, CapEx, free cash flow (OCF − CapEx)
4. **Per-share** — basic EPS, diluted EPS
5. **Margins/ratios** — gross margin, operating margin, effective tax rate
6. **Segment** — AWS, Google Services, Family of Apps, automotive revenue

## Data Sources

Numerical answers verified from the official SEC filings stored in 
`documents/sec_filings/`. Each company's 10-K was the primary source.

## Running The Eval

```bash
# All 140 questions across 7 companies
python eval/run_custom_eval.py

# Single company
python eval/run_custom_eval.py --company apple
python eval/run_custom_eval.py --company nvidia

# Smoke test (first 5 questions)
python eval/run_custom_eval.py --limit 5
```

Results saved to `eval/results/custom_<timestamp>.json` with markdown summary.

## Note On Answer Verification

Answers were sourced from the SEC filings using both manual verification 
and Apple's iXBRL ground truth (which scored 100% accuracy in our 
benchmark). Some numbers (especially calculated metrics like free cash 
flow, gross margin, effective tax rate, and segment revenues) should 
be cross-checked against the actual filings if precise to the dollar.

## Why Custom + FinanceBench Both Matter

- **Custom dataset (this folder):** Fast regression tests, our 7 known 
  filings, validates that our pipeline works correctly.
- **FinanceBench (Patronus AI, 150 Qs, 84 PDFs):** Public benchmark, 
  used for comparison against published baselines like GPT-4, Claude, 
  Gemini, BloombergGPT, FinGPT.