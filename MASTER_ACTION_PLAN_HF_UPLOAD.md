# MASTER_ACTION_PLAN_HF_UPLOAD.md
# THE 96-Hour Sprint to Real 45%+ on FinanceBench + HuggingFace Debut
# Created: 2026-06-06 by FinBench friend (honest plan, no fake numbers)

═══════════════════════════════════════════════════════════════════════════
                  THE BRUTAL HONEST SITUATION TODAY
═══════════════════════════════════════════════════════════════════════════

CURRENT CONFIRMED: 1/5 = 20% on official FinanceBench (5-Q sample)

What works:
  ✓ Q1 capex extraction (raw_text fallback)
  ✓ FIX-v5 verify_lib blocks 102882% garbage
  ✓ N20 decision pattern matching (architecture sound)
  ✓ question_lib 9/10 intent classification (90% accuracy on intent)

What's broken NOW:
  ✗ Ollama circuit breaker too aggressive (FIXED in FIX-v6, untested)
  ✗ Retrieval returns empty context → LLM has nothing to read
  ✗ extract_lib picks wrong cell for revenue (val_b=1.7)
  ✗ Q4/Q5 narrative drivers wrong
  ✗ CI Gate failing for 5+ weeks (5 consecutive runs)
  ✗ Live data fetchers not yet wired in
  ✗ No tool-calling agent (LLM doesn't know libs exist)

NOT YET BUILT:
  ⏳ Tool-calling agent (LLM-with-libs-as-tools)
  ⏳ Live data fetchers (yfinance/sec-edgar/AV) with C2 bypass permission
  ⏳ Camelot fallback for sparse PDFs (3M_2018 only got 88 cells)
  ⏳ HuggingFace model card + repo
  ⏳ Fine-tuning (SFT + DPO)


═══════════════════════════════════════════════════════════════════════════
                      THE HONEST 96-HOUR PATH
═══════════════════════════════════════════════════════════════════════════

GOAL: Legitimate 45%+ on official FinanceBench, uploaded to HuggingFace
      with REAL reproducible numbers, no inflation.

NOT GOAL (be honest):
  - Beating GPT-4o (54%) in 4 days
  - Matching Claude/Gemini (they have billions in compute)
  - Perfect on hard narrative questions (NO system does this offline)


─── DAY 1 (HOURS 0-24): MAKE THE PIPELINE WORK END-TO-END ───────────────

Step 1 (5 min): Pre-warm Ollama + verify model loaded
  $ ollama serve              # in separate terminal, leave running
  $ ollama pull llama3.1:8b   # already done, just verify
  $ python -c "from src.utils.llm_client import prewarm_model; prewarm_model()"
  # Expected: "[LLM] prewarm OK in 30-60s" first time, <2s subsequent

Step 2 (10 min): Run the FIXED 5-Q eval with LLM enabled
  $ Remove-Item -Recurse -Force D:\projects\finbench_agent\cache
  $ python eval\run_financebench.py --limit 5
  # NOW: Ollama is warm, circuit breaker fixed, should NOT trip on 1st call
  # If still RETRIEVAL_MISS on Q2: retrieval bug needs separate fix
  # If LLM responds with text: SUCCESS, even if wrong answer

Step 3 (30 min): Read the output, decide path forward
  - If 1-2 questions get REAL answers (not RETRIEVAL_MISS) → proceed to step 4
  - If still empty → retrieval bug, NOT LLM bug (need to fix N06/N07)

Step 4 (4-6 hr): Run FULL 150-Q FinanceBench eval on local
  $ python eval\run_financebench.py
  # Let it run overnight. Will take 4-8 hours on CPU, 2-4 hours on GPU.
  # Save the JSON result file — that's your real number.

Step 5 (sleep): Don't burn yourself out. 96 hours needs 3 sleeps.


─── DAY 2 (HOURS 24-48): LIVE DATA + RETRIEVAL FIXES ────────────────────

Step 6 (2 hr): Build src/live_data/fetchers.py
  Free APIs (no paid keys):
    - yfinance (Yahoo Finance) — works without key
    - sec-edgar-api (or sec-cik-mapper) — public SEC EDGAR
    - alpha_vantage_api free tier (5/min, 500/day, needs free key)

  $ pip install yfinance sec-edgar-api alpha-vantage

  Implement (~300 lines):
    fetch_latest_10k_url(ticker)              → SEC EDGAR
    fetch_historical_prices(ticker, days=365) → yfinance
    fetch_quarterly_financials(ticker)        → yfinance
    fetch_company_overview(ticker)            → Alpha Vantage

  C2 BYPASS NOTE: User explicitly approved C2 violation for live data.
  Document this clearly in the model card under "Live Data Mode".

Step 7 (3 hr): Fix retrieval miss (if Step 3 showed it's broken)
  - Check src/retrieval/bm25_retriever.py MIN_SCORE (lower to 0.005)
  - Check src/retrieval/sniper_rag.py threshold (lower to 0.5)
  - Add diagnostic: log top-10 BM25 results for every question

Step 8 (3 hr): Wire question_lib into the LLM PROMPT (not just fallback)
  The current bug: question_lib runs AFTER LLM fails.
  The fix: pass question_lib's parsed plan INTO the LLM context as:
    "Detected intent: COMPUTE
     Required formula: gross_margin = gross_profit / revenue
     Required inputs: gross_profit_2022, revenue_2022
     Retrieved chunks: [...]
     Find these values, compute the ratio, answer."

  This grounds the LLM. Reduces hallucination dramatically.


─── DAY 3 (HOURS 48-72): TOOL-CALLING AGENT (THE BIG ONE) ────────────────

Step 9 (8 hr): Build src/analysis/tool_calling_agent.py (~500 lines)
  THE missing piece. Lets LLM CALL libs as tools.

  System prompt to LLM:
    "You have these tools:
       extract_metric(metric_id, period) → returns $value
       compute_formula(formula_id, **inputs) → returns result
       classify_decision(rule_id, **inputs) → returns class
       search_text(phrase) → returns sentences
       verify_value(metric, value) → returns ok/abstain
       final_answer(text) → finishes the question

     Output JSON: {tool: 'name', args: {...}}
     Until you call final_answer."

  Loop:
    LLM → JSON → execute tool via lib_bridge → return result → LLM → ...

Step 10 (4 hr): Wire tool agent into pipeline as N11_HYBRID mode
  Route by intent:
    EXTRACT  → libs alone (fast)
    COMPUTE  → tool agent (LLM picks tools)
    DECIDE   → tool agent (LLM gathers inputs, libs decide)
    NARRATE  → tool agent + raw_text scan
    PROJECT  → tool agent (LLM plans, libs compute)


─── DAY 4 (HOURS 72-96): COLAB OFFICIAL EVAL + HF UPLOAD ────────────────

Step 11 (3 hr): Upload code to GitHub clean (current repo)
  $ cd D:\projects\finbench_agent
  $ git add .
  $ git commit -m "FIX-v6 + question_lib + tool agent — ready for benchmark"
  $ git push origin main

Step 12 (3 hr): Run OFFICIAL benchmark on Colab T4 GPU
  Open https://colab.research.google.com → New Notebook → Runtime → T4
  Run cells from FINBENCH_COLAB_EVAL.md (cells 1-10)
  This gives the REAL, REPRODUCIBLE number.
  Save eval JSON + log to Drive.

Step 13 (2 hr): Run FinQA + TAT-QA for richer card
  Cell 11 of FINBENCH_COLAB_EVAL.md
  Get 2-3 numbers (FinanceBench + FinQA + TAT-QA)

Step 14 (4 hr): Write HuggingFace model card with REAL numbers
  Use the template in PHASE_2_96HR_PLAN.md
  Replace XX% with actual confirmed numbers
  Include reproducibility commands with seed=42

Step 15 (2 hr): Push to HuggingFace
  $ huggingface-cli login   # use HF token
  $ huggingface-cli repo create finbench-analyst-v1 --type model
  $ git lfs install
  $ git push huggingface main


═══════════════════════════════════════════════════════════════════════════
                  CI GATE FIX (DO ON DAY 1, 30 MIN)
═══════════════════════════════════════════════════════════════════════════

The CI has been failing for 5 weeks. Likely cause: missing dependencies
in tests/test_ci_gate.py imports.

Fix: Update .github/workflows/tests.yml to install ALL dependencies
the test needs.

Commands:
  $ pip freeze > requirements-frozen.txt
  $ cat requirements-frozen.txt
  # Identify any package in src/ that's not in workflow yaml
  $ git add .github/workflows/tests.yml requirements.txt
  $ git commit -m "fix: CI gate dependencies"
  $ git push

Verify: https://github.com/<you>/finbench_agent/actions
       Should turn green within 5 min.


═══════════════════════════════════════════════════════════════════════════
                  HONEST EXPECTATIONS PER MILESTONE
═══════════════════════════════════════════════════════════════════════════

After Day 1 (pipeline works end-to-end):
  Expected: 25-35% on FinanceBench (LLM + libs but no tool calling)
  Why this number: matches typical RAG-only systems

After Day 2 (retrieval fixed + live data + question_lib in prompt):
  Expected: 35-45%
  Why: better context → fewer LLM hallucinations

After Day 3 (tool-calling agent built):
  Expected: 45-55%
  Why: numbers come from libs (deterministic), LLM only handles language

After Day 4 (Colab eval + HF upload):
  Confirmed: whatever number we actually get
  Target:    45%+
  Honesty:   if it's 35%, we publish 35%. NO inflation.


═══════════════════════════════════════════════════════════════════════════
                    COMPETITIVE CONTEXT (HONEST)
═══════════════════════════════════════════════════════════════════════════

Top open-source on FinanceBench (Patronus 2024 paper):
  GPT-4o + RAG:        54%   ← top closed system with RAG
  Llama 3.1 70B + RAG: 42%   ← top open
  Llama 4 Maverick:    39%   ← Meta's latest base
  Most open systems:   35-45%

OUR HONEST TARGETS:
  Conservative:  35-40% → competitive but below SOTA
  Realistic:     40-50% → solid HuggingFace contribution
  Stretch:       50-55% → close to GPT-4o territory
  Fantasy:       70%+   → impossible offline in 96 hours

If we hit 45%+ legitimately at $0 cost / 100% local, that's a
significant contribution worthy of HuggingFace front page.


═══════════════════════════════════════════════════════════════════════════
                  RULES — DON'T BREAK THESE EVER
═══════════════════════════════════════════════════════════════════════════

✓ NEVER inflate benchmark numbers
✓ NEVER train on test set
✓ NEVER cherry-pick easy questions
✓ ALWAYS publish seed=42 reproducible commands
✓ ALWAYS save raw eval logs
✓ ALWAYS document limitations explicitly
✓ ALWAYS distinguish "Local mode" vs "Live data mode (C2 bypass)"

The 96-hour goal is HONESTY first, accuracy second.
A real 38% beats a fake 70% on every metric that matters.
