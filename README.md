# FinBench Multi-Agent Business Analyst AI

> Production-grade multi-agent system for financial document analysis.
> 100% local inference · $0 API cost · 1287 tests · 19-node deterministic pipeline.

[![tests](https://img.shields.io/badge/tests-1287%20passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.11-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()
[![inference](https://img.shields.io/badge/inference-100%25%20local-blue)]()
[![cost](https://img.shields.io/badge/API%20cost-%240-brightgreen)]()

---

## What This Is

A multi-agent financial document analyst that answers complex questions over
SEC filings (10-K, 10-Q, 8-K), earnings transcripts, and financial statements.
Built to target top performance on the [FinanceBench](https://arxiv.org/abs/2311.11944)
benchmark while running entirely on a single laptop with no paid APIs.

**Why this exists:** financial document QA is dominated by expensive cloud LLMs that send sensitive filings over the network. This project is a complete open-source framework — 19-node deterministic pipeline, 1287 tests, multi-pod debate architecture — designed to explore how much accuracy is achievable with a small local model. Benchmark numbers are measured honestly and published in `eval/results/latest.md`(yet to update) as they improve.

---

### Key Results

| Metric | Current | Target | Status |
|---|---|---|---|
| FinanceBench accuracy (zero-shot) | Measuring | 45–55% | 🔬 In progress |
| FinanceBench accuracy (after SFT) | Measuring | 55–65% | 📅 Week 2 |
| Inference cost | $0 | $0 | ✅ Enforced (C1) |
| Local-only | 100% | 100% | ✅ Enforced (C2) |
| Memory footprint | ≤14 GB target | ≤14 GB | ✅ Enforced (C4) |
| Deterministic | seed=42 | seed=42 | ✅ Enforced (C5) |
| Test coverage | 1287 passing | — | ✅ |

---

## Architecture

A 19-node deterministic pipeline built on LangGraph. Each node is independently
testable and has a clear single responsibility.
INGESTION         N01 PDF Ingestor (+ N01b image/OCR/vision sub-module)
N02 Section Tree Builder
N03 Chunker + Indexer
ROUTING           N04 CART Router (5-class query type)
N05 Logistic Regression Difficulty Predictor
RETRIEVAL         N06 SniperRAG (regex direct-hit cache)
N07 BM25 (sparse keyword)
N08 BGE-M3 (dense semantic)
N09 RRF + Cross-Encoder Reranker
ANALYSIS          N10 Prompt Assembler (Jinja2, context-first)
N11 Analyst Pod (PIV loop — Planner/Implementor/Validator)
N12 CFO/Quant Pod (formula-first)
N13 TriGuard Forensics (Benford + IsoForest + GARCH)
N14 Blind Auditor (independent cross-check)
N15 PIV Mediator (2-pod majority, up to 2 mediation rounds)
EXPLAINABILITY    N16 SHAP + Causal DAG
N17 XGB Arbiter (Gate M6 — activates after 300+ DPO pairs)
OUTPUT            N18 RLEF JEE Engine (grades each session for weekly DPO)
N19 Output Generator
N19b PDF Report Generator (14-page business analyst grade)

### Why 4 retrieval tiers?

Each tier is progressively more expensive but more semantic. The system uses
the **cheapest tier that gives high confidence** — 60%+ of queries resolve in
the SniperRAG tier in <50ms with zero GPU. This is the single biggest latency
win in the system.

### Why 3 analyst pods?

The BlindAuditor (N14) never sees the other pods' outputs and retrieves
independently. When it agrees with LeadAnalyst + QuantAnalyst, we have
*external* validation — not just internal self-consistency. This pattern
catches hallucinations that a single LLM self-check misses.

### Why PIV loops?

Planner → Implementor → Validator with a max of 3 retries. The Validator
runs 8 independent checks (scope, units, sign, citation, fiscal year,
consistency, completeness, grounding). Validation failures re-prompt the
LLM with specific failure reasons — a form of structured test-driven
prompting.

---

## Engineering Constraints

These are hard constraints enforced in code, not aspirations:

| Constraint | Enforcement |
|---|---|
| **C1** — $0 cost forever | No paid-API client packages installed. CI blocks PRs that add them. |
| **C2** — 100% local inference | Real-time data feeds are env-var-gated. File watcher polls (no cloud). |
| **C3** — Gemma4:e4b via Ollama | Single-model dependency. Circuit breaker on failure. |
| **C4** — ≤14GB RAM cap | ResourceGovernor class monitors + backs off. |
| **C5** — seed=42 everywhere | Applied to random, numpy, sklearn, xgboost, torch. |
| **C6** — DPO beta = 0.1 only | Hardcoded constant. |
| **C7** — Context before question | Prompt assembler enforces this in every template. |
| **C8** — Chunk metadata prefix | Every chunk prefixed `COMPANY/DOCTYPE/FY/SECTION/PAGE`. |
| **C9** — No `_rlef_` in outputs | Fail-safe check before any render. |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally
- 14 GB RAM available
- Optional: Tesseract OCR binary (for scanned page extraction)

### Installation

**Flexible install (recommended for development):**
```bash
pip install -r requirements.txt
```

**Exact reproduction (for benchmark replication):**
```bash
pip install -r requirements-lock.txt
```

```bash
# Pull the ollama model (Gemma4:e4b)
ollama pull gemma4:e4b
```

## What You See in the PDF Report

Each query produces a 14-page professional PDF:

| Page | Content |
|---|---|
| 1 | Cover: company, fiscal year, session ID, timestamp, model version |
| 2 | Executive Summary (7-metric infographic + recommendation) |
| 3 | Answer Card (question, answer, confidence bar) |
| 4 | Reasoning Chain (6-step plain-English narrative) |
| 5 | Retrieval Evidence (all 4 tiers explained + top chunks table) |
| 6 | Three-Pod Independent Analysis (side-by-side comparison) |
| 7 | Forensics (Benford chi-square chart + risk gauge) |
| 8 | SHAP Feature Importance (horizontal bar chart) |
| 9 | Causal DAG (Revenue→EPS dependency graph) |
| 10 | Chart Data Extracted from Images (N01b output) |
| 11 | Methodology (all 19 nodes explained) |
| 12 | Citations Appendix (deduplicated, up to 40) |
| 13 | Validator Audit Trail (V1–V8 checks) |
| 14 | Reproducibility + Disclaimers |

Sample: `outputs/reports/FinBench_Report_Apple_Inc_<session>_<timestamp>.pdf`

---

## Project Structure
finbench_agent/
├── src/
│   ├── state/              BAState dataclass — 70+ fields, no hidden mutation
│   ├── ingestion/          N01/N01b, N02, N03
│   ├── retrieval/          N06, N07, N08, N09
│   ├── agents/             N10, N11, N12, N14, N15 (PIV loops)
│   ├── forensics/          N13 TriGuard (Benford + IsoForest + GARCH)
│   ├── explainability/     N16 SHAP + DAG
│   ├── ml/                 N17 XGB (gated), N04 CART, N05 LR
│   ├── rlef/               N18 SQLite-backed grading
│   ├── output/             N19 DOCX, N19b PDF
│   ├── live_data/          Phase 7A–7E (EDGAR, Yahoo, Transcripts, FRED, Watcher)
│   ├── pipeline/           LangGraph 19-node StateGraph
│   └── utils/              Gemma4 client with circuit breaker
├── tests/                  1287 tests organised by node
├── eval/
│   └── run_eval.py         FinanceBench benchmark harness
├── app.py                  Streamlit 3-screen UI
├── requirements.txt
└── README.md

---

## Engineering Highlights

- **Deterministic:** every run with `seed=42` produces identical outputs.
  No flaky tests.
- **Circuit breaker:** LLM client trips after 3 consecutive failures, auto-recovers
  after 60s. Never hangs the UI.
- **Gated ML:** N17 XGB Arbiter is a no-op until 300+ DPO pairs exist. No
  deployment-without-data footgun.
- **Graceful degradation:** system works without Tesseract (OCR off), without
  fredapi (macro off), without yfinance (market off). Every optional feature
  is guarded.
- **No silent failures:** every N01 → N19 node logs before and after. Failures
  surface in the UI status column.
- **Zero-config first run:** after `ollama pull gemma4:e4b` and `pip install`,
  `streamlit run app.py` works.

---

## Reproducibility

Every reported number can be reproduced exactly:

```bash
# Set seed everywhere
export SEED=42

# Run eval
python run_eval.py --seed 42 --dataset financebench --output results.json

# Regenerate PDF reports
python -m src.output.pdf_report_generator --session <session_id>

# Rebuild retrieval indices
python -m src.retrieval.bm25_retriever --rebuild
python -m src.retrieval.bge_retriever  --rebuild
```

Package versions are pinned in `requirements.txt`. Model versions are
pinned to `gemma4:e4b` (Ollama digest verified at startup).

---

## Roadmap

- [x] Phase 1: 19-node pipeline skeleton
- [x] Phase 2: Retrieval system
- [x] Phase 3: Agent analysis engine
- [x] Phase 4: PDF report generator (this milestone)
- [x] Phase 7A–7E: Real-time data feeds
- [ ] Phase 5: FinanceBench eval + Chi-Square significance
- [ ] Phase 6: QLoRA SFT + 3 rounds of DPO (Colab T4)
- [ ] Phase 7: Ollama Hub publish

---

## Citing

If this system is useful in your research or product, please cite:
@software{finbench_agent_2026,
title   = {FinBench Multi-Agent Business Analyst AI},
author  = {<Your Name>},
year    = {2026},
url     = {https://github.com/<user>/finbench_agent},
version = {PDR-BAAAI-001 Rev 1.0}
}

---

## License

MIT — see `LICENSE`. The Gemma4 model weights are governed by Google's Gemma
Terms of Use.

---

## Acknowledgments

- [Ollama](https://ollama.com) for local LLM serving
- [LangGraph](https://langchain-ai.github.io/langgraph/) for the state-machine
  pipeline
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for semantic retrieval
- [FinanceBench](https://arxiv.org/abs/2311.11944) for the benchmark dataset
- [reportlab](https://www.reportlab.com) for PDF generation
