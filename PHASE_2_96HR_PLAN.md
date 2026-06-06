# PHASE_2_96HR_PLAN.md
# FinBench — 96-Hour Sprint to HuggingFace Upload
# Created: 2026-06-06
# HONEST plan. No fake numbers. No inflated claims.

═══════════════════════════════════════════════════════════════════════════
                       HONEST GOAL FOR 96 HOURS
═══════════════════════════════════════════════════════════════════════════

PRIMARY GOAL:
  Upload a LEGITIMATE HuggingFace model card with REAL confirmed
  benchmark numbers (honest 25-55%, not fake 75%+).

NOT THE GOAL (be honest with yourself):
  - We will NOT match GPT-4o (54%) in 96 hours
  - We will NOT match Claude/Gemini — they have $billions of compute
  - We will NOT fine-tune properly (needs 1-2 weeks)
  - We will NOT have "perfect hard question" answers

WHAT WE WILL HAVE:
  - A real reproducible system on HuggingFace
  - An honest model card with seed=42 commands
  - The N20 Composite Resolver as our unique selling point
  - The 8-lib architecture (genuine novel contribution)
  - A confirmed accuracy number (whatever it is)
  - Live data fetch capability documented (free APIs only)


═══════════════════════════════════════════════════════════════════════════
                  HOUR-BY-HOUR 96-HOUR BREAKDOWN
═══════════════════════════════════════════════════════════════════════════

────────────────────────────── DAY 1 (Hours 0-24) ──────────────────────────
GOAL: Get ONE legitimate benchmark number on FinanceBench

H 0-2:   Install question_lib + run all self-tests
         pip install -e D:\projects\fina_question_lib
         python -m question_lib.intent_classifier
         python -m question_lib.decomposer
         python -m question_lib.plan_executor

H 2-4:   Verify pipeline runs without --sniper-only (Ollama warm-up)
         ollama serve
         ollama pull llama3.1:8b
         ollama run llama3.1:8b "test"    # confirm working
         python eval\run_financebench.py --limit 5    # NO sniper-only

H 4-6:   If 5-Q hits >= 2/5, run FULL FinanceBench (150 Qs)
         python eval\run_financebench.py
         (takes 2-3 hours; let it run overnight if needed)

H 6-12:  Sleep / let eval run

H 12-18: Analyze results — categorize misses
         What % were:
           - Retrieval failures (top-10 didn't have answer)?
           - Format mismatches (right number, wrong format)?
           - Genuine reasoning failures (LLM got it wrong)?
           - Library bugs (extract_lib picked wrong cell)?

H 18-24: Targeted fixes for top-3 miss categories
         (Don't try to fix everything — focus on biggest bucket)


────────────────────────────── DAY 2 (Hours 24-48) ─────────────────────────
GOAL: Build tool-calling agent (LLM-as-orchestrator-using-libs)

H 24-32: Build src/analysis/tool_calling_agent.py
         (~500 lines: system prompt + JSON parser + tool dispatcher + loop)

         Tools exposed to LLM:
           - extract_metric(metric_id, period)
           - compute_formula(formula_id, **inputs)
           - classify_decision(rule_id, **inputs)
           - search_text(phrase)
           - verify_value(metric, value)
           - final_answer(text)

H 32-36: Wire into pipeline.py as new mode --hybrid

H 36-44: Test on 5-Q sample, debug any tool-call failures

H 44-48: Run full 150-Q with --hybrid
         Compare vs Day 1 number


────────────────────────────── DAY 3 (Hours 48-72) ─────────────────────────
GOAL: Live data fetch + extend benchmarks

H 48-54: Live data fetch (HONESTLY documented)
         What ACTUALLY works:
           - yfinance (Yahoo Finance, free, rate-limited)
           - sec-edgar-api (SEC EDGAR, free, public)
           - alpha_vantage_api (free tier 5 calls/min, 500/day)

         What DOESN'T work / cannot promise:
           - Real-time prices (need paid subscriptions)
           - Bloomberg Terminal data (paid only)
           - Refinitiv (paid only)

         Create: src/live_data/fetchers.py with:
           - get_latest_filing(ticker)  → SEC EDGAR
           - get_historical_prices(ticker, days) → yfinance
           - get_company_overview(ticker) → AV free tier

H 54-60: Run on 2 SECONDARY benchmarks (smaller, faster)
         - FinQA (50 random questions): python eval\run_finqa.py --limit 50
         - TAT-QA (50 questions):         python eval\run_tatqa.py --limit 50

H 60-66: Analyze results, write findings to results/ folder

H 66-72: Sleep + light fixes


────────────────────────────── DAY 4 (Hours 72-96) ─────────────────────────
GOAL: HuggingFace upload + documentation

H 72-78: Write the HF model card (README.md for HF repo)
         (See template below)

H 78-82: Clean up repo for upload
         - Remove cache/__pycache__/.pyc
         - Add LICENSE (MIT recommended)
         - Add .gitattributes for git-lfs (model files)
         - Add CITATION.cff

H 82-88: Create HuggingFace repo + upload
         huggingface-cli login
         huggingface-cli repo create finbench-analyst-v1 --type model
         git lfs install
         git lfs track "*.gguf" "*.bin" "*.pt"
         git add . && git commit -m "Initial release"
         git push

H 88-92: HuggingFace Space (free demo)
         Create gradio app: hf_space/app.py
         Push to: https://huggingface.co/spaces/<you>/finbench-demo

H 92-96: Test demo end-to-end + announce
         - Test from a clean machine
         - Tweet/Reddit post (factual, no exaggeration)
         - LinkedIn post about architecture (not benchmark claims)


═══════════════════════════════════════════════════════════════════════════
              REQUIREMENTS TO HAVE NUMBERS PRESENTABLE TO
              TOP AI COMPANIES (NVIDIA / OPENAI / GOOGLE / META)
═══════════════════════════════════════════════════════════════════════════

To get OFFICIAL real numbers that would impress AI recruiters:

1. REPRODUCIBILITY
   - Every number must be reproducible with: python <script> --seed 42
   - Commit hash must be tagged
   - All dependencies in requirements.txt with exact versions

2. MINIMUM BENCHMARKS (need at least 3)
   - FinanceBench full 150 (REQUIRED — most cited)
   - FinQA (REQUIRED — academic standard)
   - TAT-QA OR ConvFinQA (recommended)

3. HONEST METHODOLOGY DOCUMENT
   - Eval framework version
   - Random seed value
   - Number of runs (1 is OK, but say so)
   - Compute used (CPU/GPU, RAM)
   - Wall-clock time per question

4. COMPARISON TABLE (must be HONEST, no fake numbers)
   |  System         | FinanceBench | Cost | Local | Reference        |
   | GPT-4o + RAG    | 54%          | $$$  | No    | Patronus 2024    |
   | Llama 4 Maverick | 39%         | -    | -     | HF leaderboard   |
   | OURS (Llama 3.1 8B + 8 libs) | XX%  | $0   | Yes   | This repo (seed=42) |

5. FOR NVIDIA / TOP AI COMPANY APPLICATIONS:
   - A blog post explaining the architecture innovation
   - A 5-minute demo video on YouTube
   - GitHub repo with clean code
   - HuggingFace Space anyone can try
   - Honest LinkedIn post (no inflation)

   Application story to recruiter:
     "I built an open-source financial-QA system that achieves XX% on
      FinanceBench at $0 inference cost, fully local, using a novel
      19-node pipeline + 8 specialized libs + N20 composite resolver
      (deterministic JEE side-word method). It runs on a single laptop
      and beats GPT-4o + RAG at $0 cost / Llama 4 Maverick by Npp.
      Reproducible with seed=42. Code: github.com/<you>"

   This is what gets the interview. NOT inflated benchmark numbers.


═══════════════════════════════════════════════════════════════════════════
                          HARD CONSTRAINTS — DO NOT VIOLATE
═══════════════════════════════════════════════════════════════════════════

DO NOT under any circumstance:
  ❌ Inflate benchmark numbers
  ❌ Cherry-pick easy questions for the "official" run
  ❌ Run eval with --limit (then claim full-benchmark)
  ❌ Use --sniper-only (then claim LLM-grade accuracy)
  ❌ Train on FinanceBench test set
  ❌ Make claims you can't reproduce
  ❌ Claim "beats GPT-4" without showing methodology

These mistakes destroy reputation FOREVER. Companies blacklist names.

DO:
  ✅ Report real numbers, even if low
  ✅ Document methodology honestly
  ✅ Make every number reproducible
  ✅ Acknowledge limitations explicitly
  ✅ Show competitive context (vs other open-source)
  ✅ Position based on COST / SPEED / LOCALITY, not raw accuracy


═══════════════════════════════════════════════════════════════════════════
              HUGGINGFACE MODEL CARD TEMPLATE (for D4)
═══════════════════════════════════════════════════════════════════════════

---
license: mit
language: en
tags:
  - financial-qa
  - rag
  - retrieval-augmented-generation
  - llama-3.1
  - finance
  - financebench
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
pipeline_tag: question-answering
---

# FinBench Analyst v1 — Open-Source Financial QA System

## Quick Summary

A 19-node deterministic+LLM hybrid system for financial document QA.
Achieves **XX% on FinanceBench** at **$0 inference cost**, **100% local**.

## Reproducibility

```bash
git clone https://huggingface.co/<you>/finbench-analyst-v1
cd finbench-analyst-v1
pip install -r requirements.txt
python eval/run_financebench.py --seed 42
```

## Benchmarks (confirmed, seed=42)

| Benchmark        | Our Score | GPT-4o+RAG | Llama-4 Maverick |
| FinanceBench 150 | XX%       | 54%        | 39%              |
| FinQA            | XX%       | -          | -                |
| TAT-QA           | XX%       | -          | -                |

(Replace XX% with REAL confirmed numbers — never projections.)

## Architecture

19-node pipeline + 8 specialised libraries:
1. maths_lib (200+ deterministic formulas)
2. extract_lib (scored cell extraction)
3. pattern_lib (question classification)
4. format_lib (deterministic formatting)
5. logic_lib (195 rules across 6 layers)
6. algo_lib (advanced finance algorithms)
7. verify_lib (answer verification)
8. question_lib (JEE side-word decomposer — novel)

## Hardware

- CPU: any x86_64
- RAM: 14 GB minimum
- GPU: optional (CPU-only inference works)
- Storage: 6 GB (model + libs)

## Limitations

- Narrative questions ("what drove X") accuracy lower than numeric
- Sparse PDFs (pre-2018 filings) have lower cell extraction
- Currently English-only
- Trained for SEC 10-K / 10-Q filings

## License

MIT (model weights via Llama 3.1 Community License)

## Citation

```bibtex
@software{finbench2026,
  author = {<your name>},
  title  = {FinBench Analyst: Multi-Agent Financial QA},
  year   = {2026},
  url    = {https://huggingface.co/<you>/finbench-analyst-v1}
}
```


═══════════════════════════════════════════════════════════════════════════
              IMMEDIATE COMMANDS TO START THE 96-HOUR SPRINT
═══════════════════════════════════════════════════════════════════════════

Run these IN ORDER (PowerShell, in D:\projects\finbench_agent with venv active):

# Step 1: Install question_lib
pip install -e D:\projects\fina_question_lib

# Step 2: Self-tests of question_lib (catches bugs fast)
python -m question_lib.intent_classifier
python -m question_lib.subject_extractor
python -m question_lib.period_extractor
python -m question_lib.modifier_extractor
python -m question_lib.operation_detector
python -m question_lib.formula_matcher
python -m question_lib.dependency_graph
python -m question_lib.decomposer
python -m question_lib.plan_executor

# Step 3: Start Ollama (separate terminal)
ollama serve

# Step 4: Confirm Llama 3.1 8B available
ollama list

# Step 5: Clear stale cache
Remove-Item -Recurse -Force D:\projects\finbench_agent\cache

# Step 6: THE REAL TEST — 5-Q eval with LLM enabled
python eval\run_financebench.py --limit 5

# Step 7: Paste output to me. Then we decide:
#   - If 2-3 of 5 → run full 150 immediately
#   - If 0-1 of 5 → fix critical bug first, then full 150
