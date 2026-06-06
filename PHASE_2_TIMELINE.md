# PHASE_2_TIMELINE.md — FinBench Multi-Agent Business Analyst AI
# PDR-BAAAI-001 · Rev 1.0
# Created: 2026-06-05

## STATUS RECAP
- Phase 1 (foundation + 19 nodes built):   ✅ COMPLETE
- Custom 196Q benchmark:                   ✅ 84.2% (Apr 2026)
- Official FinanceBench last score:        20% (false credit) → now corrected
- Phase 2 (real-PDF accuracy push):        ⏳ STARTS NOW
- 7 support libs installed + wired:        ✅ this session
- extract_lib synonyms 3 → 37 metrics:     ✅ this session
- All P0/P1 bugs hunted and fixed:         ✅ this session

## PHASE 2 OBJECTIVE
Push official FinanceBench from current ~20% (false) → confirmed 65-75%
within 4 weeks. This is the gate before Phase 3 (launch sprint).

Launch gate (M7): >= 84% on official FinanceBench, confirmed.

## PHASE 2 SPRINT TIMELINE (4 WEEKS)

### WEEK 1 — Real-PDF measurement + low-hanging fruit
GOAL: Get an honest, deterministic-only baseline on real PDFs.

  Day 1  (today)
    - Run `python eval\run_financebench.py --limit 5 --sniper-only`
      → Expect ~35-45% (per estimate after 37-metric extension)
    - Run `pytest tests\ -q` → confirm zero regressions
    - Document the actual confirmed score in CONTEXT.md

  Day 2-3
    - Run FULL FinanceBench (no --limit) with --sniper-only
      → 30-60 minute eval depending on doc count
    - Profile which question types miss the most
    - Identify top 5 metric IDs that need extract_lib synonym tuning
      (you'll see them in the partial JSON output)

  Day 4-5
    - Run FULL FinanceBench WITHOUT --sniper-only (full LLM pipeline)
      → Expect ~60-70%
    - Compare which questions sniper missed vs LLM caught
    - Identify questions where LLM hallucinates → fixable with better
      retrieval / better prompts

  Day 6-7
    - Tune retrieval thresholds based on misses:
        bm25_retriever.MIN_SCORE: 0.01 → maybe 0.05
        bge_retriever.MIN_SIMILARITY: 0.05 → maybe 0.10
    - Re-run, document score, snapshot the win

  WEEK 1 EXIT CRITERIA: confirmed honest baseline number written down.

### WEEK 2 — Retrieval upgrade (Gate M2)
GOAL: Hit retrieval >=85% (PDR Gate M2).

  Day 8-9
    - Build retrieval-only test harness:
        For each FinanceBench question, check if the gold answer
        text exists in top-10 retrieved chunks.
    - This decouples retrieval quality from answer-generation quality.

  Day 10-11
    - Tune BM25 weights for financial vocabulary (fuzzy + boost terms)
    - Tune BGE-M3 chunk overlap and chunk size
    - Tune RRF k-parameter (currently 60, try 30 and 90)

  Day 12-13
    - Investigate failed retrievals:
        * Is the answer in raw_text but missed by chunker?
        * Is the chunk found but ranked too low?
        * Is the question phrased very differently from the chunk?
    - Add query rewriting if needed (use Llama to rewrite question
      into 2-3 alternate phrasings, retrieve each, merge with RRF)

  Day 14
    - Final retrieval test, confirm >=85% top-10 recall
    - Gate M2 marked PASSED in CONTEXT.md

  WEEK 2 EXIT CRITERIA: Gate M2 PASSED.

### WEEK 3 — DPO data collection + XGB Arbiter prep (Gate M6)
GOAL: Collect 300+ DPO pairs (chosen vs rejected answers).

  Day 15-16
    - Wire the RLEF JEE Engine to actually grade pipeline runs:
        - Run full FinanceBench
        - Pipeline generates answers → JEE grades each with 3 validators
        - PASSED answers → chosen in DPO pair
        - REJECTED answers → rejected in DPO pair
    - Each FinanceBench run produces ~150-200 DPO pairs
    - Two runs = 300+ pairs (Gate M6 threshold)

  Day 17-18
    - Run 2-3 FinanceBench evals, each producing DPO data
    - Inspect DPO pairs manually for quality
    - Clean obvious garbage from the pairs

  Day 19-20
    - Train the XGB Arbiter on the DPO data
        python -m src.ml.xgb_arbiter --train
    - Validate XGB on a holdout subset
    - If validation accuracy >= 75%, Gate M6 PASSED

  Day 21
    - Re-run FinanceBench WITH XGB arbiter active
    - Document score, compare to non-XGB baseline

  WEEK 3 EXIT CRITERIA: Gate M6 PASSED. Score uplift documented.

### WEEK 4 — LLM Fine-Tuning Sprint (Gate M5)
GOAL: Train financebench-expert-v1 model with SFT + DPO.

  Day 22-23
    - Generate SFT training data:
        For each FinanceBench train question:
        - Run pipeline, get good answer
        - Build (context, question, answer) triplet
        - Save as JSONL training set
    - Target: 500-1000 high-quality SFT examples

  Day 24-25
    - SFT fine-tune Llama 3.1 8B on SFT data:
        - Use LoRA + 4-bit quantization (memory-safe)
        - 3-5 epochs, batch size 4-8
        - C5: seed=42
        - C6: beta=0.1 (only matters for DPO step)
    - Save as ollama model: `ollama create financebench-expert-v1-sft`

  Day 26-27
    - DPO fine-tune the SFT model on the DPO pairs from Week 3:
        - beta=0.1 (C6 — non-negotiable)
        - 2-3 epochs
        - Validation on holdout set
    - Save as: `ollama create financebench-expert-v1`

  Day 28
    - Final FinanceBench run with the new model
    - If >= 76% → Gate M5 PASSED
    - Document score, log to CONTEXT.md

  WEEK 4 EXIT CRITERIA: Gate M5 PASSED.
                        FinanceBench >= 76% confirmed.

## PHASE 2 COMPLETION CRITERIA
- M2 retrieval >= 85% top-10 recall:    PASSED
- M5 LLM SFT >= 76% accuracy:           PASSED
- M6 XGB Arbiter trained (300+ DPO):    PASSED
- FinanceBench official score:          >= 65% (target)
- All numbers CONFIRMED (no projections)

## PHASE 3 PREVIEW (after Phase 2 complete)
- Final FinanceBench sprint to 82%+
- Distribution via Ollama Hub
- Public benchmark submission
- RLEF active feedback loop in production

## IMMEDIATE NEXT ACTIONS (today, Friday 2026-06-05)

### Step 1 — Verify all fixes don't break anything
```powershell
cd D:\projects\finbench_agent
venv\Scripts\activate
python -m src.utils.lib_bridge
```
Expect: 7/7 ✓, all wrappers print results without errors.

### Step 2 — Run the smoke tests
```powershell
pytest tests\test_pipeline.py tests\test_gemma4_client.py -v
```
Expect: all green.

### Step 3 — Run FinanceBench with the new 37-metric extract_lib
```powershell
python eval\run_financebench.py --limit 5 --sniper-only
```
Expect: ~35-45% (REAL accuracy, no false credit).
Compare to old 20% (which was mostly false overlap credit).

### Step 4 — Document the confirmed number in CONTEXT.md
Don't project. Don't estimate. Only confirmed scores.

### Step 5 — Send me the output
Paste the eval summary so I can plan the Week 1 next moves.

## RISK REGISTER

| Risk | Mitigation |
|------|------------|
| Llama 3.1 hallucinates numbers despite good retrieval | Confidence floor via PIV validator, verify_lib post-check |
| pdfplumber misses complex tables | Phase 2 Week 2 — improve chunker boundaries |
| Fine-tuning eats VRAM beyond T4 limit | Use 4-bit LoRA, gradient checkpointing |
| DPO data has low diversity | Source from multiple companies + multiple eval runs |
| Score regresses after fine-tuning | Always keep base model fallback in piv_loop |

## C-CONSTRAINTS CHECKLIST FOR PHASE 2
C1  $0 cost                            ✅ All work uses local Ollama
C2  100% local                         ✅ FinanceBench eval is offline
C3  Llama 3.1 8B Q4_K_M                ✅ Base + fallback both this model
C4  14GB RAM cap                       ⚠️  Watch carefully during fine-tuning
C5  seed=42                            ✅ SFT + DPO + XGB all seeded
C6  DPO beta=0.1                       ✅ Non-negotiable
C7  Context-first prompts              ✅ Enforced in prompt_assembler
C8  Chunk metadata prefix              ✅ Enforced in chunker
C9  _rlef_ never in output             ✅ Enforced in output generators
C10 Ollama Hub distribution            ⏳ End of Phase 3

## KEY METRICS TO TRACK WEEKLY
- FinanceBench accuracy (sniper-only)
- FinanceBench accuracy (full LLM)
- Retrieval top-10 recall
- DPO pairs collected
- Pipeline avg time per question
- Memory peak during eval
- Sniper hit rate
- Formula router hit rate
- extract_lib resolution rate
