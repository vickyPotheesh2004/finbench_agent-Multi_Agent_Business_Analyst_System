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

| File | Company | FY | # Qs | Doc Source |
|------|---------|----|----|------------|
| apple_fy2023.jsonl | Apple Inc. | FY2023 | 20 | documents/sec_filings/AAPL_FY2023_10-K.html |
| microsoft_fy2023.jsonl | Microsoft | FY2023 | 20 | documents/sec_filings/MSFT_FY2023_10-K.html |
| nvidia_fy2024.jsonl | NVIDIA | FY2024 | 20 | documents/sec_filings/NVDA_FY2024_10-K.html |
| amazon_fy2023.jsonl | Amazon | FY2023 | 20 | documents/sec_filings/AMZN_FY2023_10-K.html |
| alphabet_fy2023.jsonl | Alphabet | FY2023 | 20 | documents/sec_filings/GOOG_FY2023_10-K.html |
| meta_fy2023.jsonl | Meta | FY2023 | 20 | documents/sec_filings/META_FY2023_10-K.html |
| tesla_fy2023.jsonl | Tesla | FY2023 | 20 | documents/sec_filings/TSLA_FY2023_10-K.html |

## Question Categories

1. **Income statement** — revenue, net income, gross profit, operating expenses, R&D, SG&A
2. **Balance sheet** — total assets, total liabilities, equity, cash, inventory, debt
3. **Cash flow** — operating CF, investing CF, financing CF, CapEx
4. **Derived/calculated** — free cash flow (OCF - CapEx), effective tax rate
5. **Per-share** — basic EPS, diluted EPS

## Adding A New Company

1. Save 10-K HTML to documents/sec_filings/<TICKER>_<FY>_10-K.html
2. Create eval/datasets/<ticker>_<fy>.jsonl
3. Copy template from existing file, fill answers from the filing
4. Add to README table above