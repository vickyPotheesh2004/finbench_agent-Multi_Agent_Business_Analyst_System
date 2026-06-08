# THE_HONEST_TRUTH_2026-06-07.md
# After 2 hours of wasted CPU time, here's what's REALLY going on.

═══════════════════════════════════════════════════════════════════════════
                  THE HARD TRUTH FROM YOUR 2-HOUR RUN
═══════════════════════════════════════════════════════════════════════════

YOUR MACHINE:
  - 15.7 GiB RAM total
  - 6.7 GiB available to Ollama
  - NO GPU (CPU-only inference)
  - Each LLM call: 2-4 MINUTES (confirmed from Ollama log)

THE EVAL LOOP:
  - PIV (Planner + Implementor + Validator) makes ~3 LLM calls per pod
  - 3 pods (Analyst, CFO/Quant, Auditor) = 9 LLM calls per question
  - 9 calls × 3 min = 27 minutes PER QUESTION
  - 150-question eval = 67 hours (~3 days)

THIS IS UNPLAYABLE LOCALLY. PERIOD.

That's why you saw:
  - Q1 stuck at 1833 seconds (30 min) BEFORE you killed it
  - Multiple Ollama 500 errors (4-min timeouts)
  - "Validation failed: empty" because retrieval finished BEFORE LLM responded


═══════════════════════════════════════════════════════════════════════════
                  THE 3 FIXES I APPLIED THIS SESSION
═══════════════════════════════════════════════════════════════════════════

FIX-v7-A: N20 fast-path runs in FULL mode (not just sniper-only)
  File: src/pipeline/pipeline.py
  Effect: Q1-style questions answered in 1-2s by libs, skip 27min LLM work

FIX-v7-B: EXTRACT_LIB fast-path runs in FULL mode too
  File: src/pipeline/pipeline.py
  Effect: Direct value lookups (revenue, capex) skip LLM entirely

FIX-v7-C: SKIP_LLM=1 env var bails out if neither fast-path hits
  File: src/pipeline/pipeline.py
  Effect: User can force fast eval (lower accuracy, but completes)

FIX-v7-D: plan_executor stops over-blocking EXTRACT answers
  File: D:\projects\fina_question_lib\src\question_lib\plan_executor.py
  Effect: verify_lib won't reject simple value lookups


═══════════════════════════════════════════════════════════════════════════
                  THE TWO PATHS FROM HERE
═══════════════════════════════════════════════════════════════════════════

────────── PATH A: CPU LOCAL (FAST, LOWER ACCURACY) ──────────
For: Quick iteration on your laptop
Mode: SKIP_LLM=1 + N20 + EXTRACT fast-paths only
Speed: ~30-60 sec per question (mostly PDF ingest)
Expected score: 15-25% on FinanceBench
Use for: Debugging pipeline bugs, testing N20 patterns

Commands:
  $env:SKIP_LLM="1"
  Remove-Item -Recurse -Force D:\projects\finbench_agent\cache
  python eval\run_financebench.py --limit 5

────────── PATH B: COLAB T4 GPU (SLOW SETUP, HIGH ACCURACY) ──────────
For: REAL benchmark numbers for HuggingFace
Mode: Full LLM pipeline on T4 GPU
Speed: ~30-60 sec per question (T4 vs your CPU = 10x faster)
Expected score: 40-55% on FinanceBench (REAL number)
Use for: The OFFICIAL number that goes on HuggingFace

Steps:
  1. Push code to GitHub (instructions below)
  2. Open https://colab.research.google.com → New notebook
  3. Runtime → Change runtime type → T4 GPU
  4. Run cells from FINBENCH_COLAB_EVAL.md (already written)


═══════════════════════════════════════════════════════════════════════════
                  MY HONEST RECOMMENDATION
═══════════════════════════════════════════════════════════════════════════

DO TODAY (2 hours):

  Step 1 (15 min): Test FIX-v7 fast-path locally
    $env:SKIP_LLM="1"
    Remove-Item -Recurse -Force D:\projects\finbench_agent\cache
    python eval\run_financebench.py --limit 10
    # Should complete in 5-10 minutes (vs hours)
    # Should now show N20_DECISION / N20_NARRATIVE / EXTRACT_LIB_FAST_PATH
    # for ~30-40% of questions
    # PASTE OUTPUT TO ME

  Step 2 (30 min): Push current code to GitHub
    cd D:\projects\finbench_agent
    git add .
    git commit -m "FIX-v7: N20 fast-path + SKIP_LLM + verify_lib calibration"
    git push origin main

  Step 3 (1 hour): Set up Colab T4
    - Open Colab
    - Runtime → T4 GPU
    - Mount Drive (for persistent results)
    - Run setup cells

DO TOMORROW (4-6 hours):

  Step 4: Run full 150-Q FinanceBench on Colab T4
    - All 12 cells of FINBENCH_COLAB_EVAL.md
    - Expected duration: 4-6 hours
    - Output: REAL accuracy number on the full benchmark

  Step 5: Based on Step 4 result:
    - If ≥ 40%: Upload to HuggingFace (we have a debut number)
    - If 25-40%: One more iteration on retrieval, then upload
    - If < 25%: Fix critical bugs found in Step 4 logs


═══════════════════════════════════════════════════════════════════════════
                  COWORK ANSWER (your earlier question)
═══════════════════════════════════════════════════════════════════════════

How to connect with Cowork:
  Cowork is Anthropic's desktop tool currently in beta.
  Access via: Claude desktop app → Settings → Beta features → Cowork

For NOW the filesystem MCP I'm using gives equivalent read/write access
to your D:\projects\ directory. That's why I can edit your code directly.


═══════════════════════════════════════════════════════════════════════════
                  CI GATE FAILURE (5+ weeks)
═══════════════════════════════════════════════════════════════════════════

Likely cause: tests/test_ci_gate.py imports from src/state/ba_state.py and
src/utils/seed_manager.py and src/utils/resource_governor.py.

The workflow only installs minimal deps. If any of those modules import
something else (e.g. pydantic, langgraph), tests fail with ImportError.

QUICK CHECK locally before push:
  pytest tests/test_ci_gate.py -v --tb=short

If green locally but red on GitHub Actions → add the missing dep to
.github/workflows/tests.yml in the "Install dependencies" step.


═══════════════════════════════════════════════════════════════════════════
                  WHY 96 HOURS IS TIGHT (HONEST)
═══════════════════════════════════════════════════════════════════════════

To upload to HuggingFace WITH official benchmark, we need:

  ✓ Code in GitHub  (you have this)
  ✓ Reproducible eval (we have, seed=42)
  ✗ ONE confirmed FinanceBench number  (we don't have this yet)
  ✗ Working live data fetcher (not built)
  ✗ Clean model card (not written yet)

The bottleneck is #3 — the confirmed number. On Colab T4 it takes ~4-6 hours.
With FIX-v7 fast-paths it might be even less (questions that hit N20 are 10x faster).

So realistic path:
  Day 1 (today): Local SKIP_LLM test + push to GitHub + start Colab setup
  Day 2:         Colab full eval (4-6 hrs running) + analyze results
  Day 3:         Build live data fetcher if score is good + write model card
  Day 4:         Push to HuggingFace + announce

That's 96 hours used WELL if we don't get sidetracked.
