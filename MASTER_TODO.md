# MASTER_TODO.md — FinBench Multi-Agent Business Analyst AI
# Created: 2026-06-05 (after FIX-v5 — honest after-eval reality check)
# Owner: friend + AI mentor
#
# READ THIS BEFORE EVERY SESSION.
# Honest tier goals. No projections. Real calculated truth only.

═══════════════════════════════════════════════════════════════════════════
                          HONEST GOAL TIERS
═══════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────┐
│ TIER 1 (PHASE 2 EXIT GATE) — Official + Recognised Benchmarks        │
│ ─────────────────────────────────────────────────────────────────── │
│   Target:    60% – 70% accuracy                                       │
│   Where:     FinanceBench, FinQA, ConvFinQA, TAT-QA,                  │
│              FinanceReasoning, HuggingFace open finance leaderboards  │
│   Status:    Currently 0–20% on official FinanceBench (sniper-only)   │
│   Mode:      With LLM enabled (no --sniper-only)                      │
│   ETA:       4 weeks                                                  │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ TIER 2 (FINAL POST-ML EXIT GATE) — Same Benchmarks, Fine-Tuned       │
│ ─────────────────────────────────────────────────────────────────── │
│   Target:    75% – 85% accuracy                                       │
│   Where:     Same as Tier 1 + HuggingFace public leaderboards         │
│   Requires:  SFT + DPO fine-tune + XGB arbiter + better extraction    │
│   Status:    Not started                                              │
│   ETA:       +4 weeks after Tier 1 hits                               │
└───────────────────────────────────────────────────────────────────────┘

NO HALLUCINATED PROMISES.  No "90%+ in 2 weeks" — that was a fantasy.
Top open-source systems on FinanceBench hit 54% (GPT-4o + RAG).
We're aiming to MATCH or BEAT them at $0 cost and 100% local.


═══════════════════════════════════════════════════════════════════════════
                        WHAT WAS DONE (2026-06-05)
═══════════════════════════════════════════════════════════════════════════

✅ PHASE 0 (Unblock):  pipeline runs clean end-to-end
✅ PHASE 1 (Wire libs): all 7 libs (maths, extract, pattern, format,
                        logic, algo, verify) installed + bridged
✅ PHASE 2 (N20 — partial):
   ✅ decision_engine.py    (Q3 "Is X capital-intensive?" pattern)
   ✅ narrative_extractor.py (Q4 "what drove X" + Q5 "which segment")
   ✅ composite_resolver.py  (N20 orchestrator wired into pipeline)
   ❌ side-word NER (spaCy)               — NOT YET
   ❌ formula dependency graph            — NOT YET
   ❌ single-value anchoring              — NOT YET
   ❌ 3-path self-consistency verification — NOT YET

✅ FIX-v5 (2026-06-05) — post-eval bug hunt:
   ✅ Number regex \d{1,3} → \d+ (fixed Q1 regression)
   ✅ Decision engine sanity bounds (reject 102882% garbage)
   ✅ Composite resolver BLOCKS verify_lib abstain (not warn-and-return)
   ✅ Segment extractor company-aware (3M no longer picks "Services")
   ✅ Narrative scan: ALWAYS use raw_text (not only fallback)

⏳ PHASE 3 (Universal Extraction):    Camelot for sparse PDFs — NOT BUILT
⏳ PHASE 4 (Fine-tuning):              SFT + DPO + BGE-M3 — NOT STARTED
⏳ PHASE 5 (Full benchmark sweep):     all 6 benchmarks — NOT RUN
⏳ PHASE 6 (Output 3-tier reports):    not built
⏳ PHASE 7 (Ship + career):            HF Space, Ollama Hub, paper, demo


═══════════════════════════════════════════════════════════════════════════
                  PHASE 2 — DETAILED TO-DO (TIER 1 PATH)
═══════════════════════════════════════════════════════════════════════════

WEEK 1 — MEASURE + LOW-HANGING FRUIT (Days 1-7)
───────────────────────────────────────────────────────────────────────────
□  W1.1  Verify FIX-v5 in eval:
         Remove-Item -Recurse -Force D:\projects\finbench_agent\cache
         python eval\run_financebench.py --limit 5 --sniper-only
         EXPECTED: Q1 capex back to PASS (was regressed to FAIL by FIX-v4)
                   Q3 capital-intensive ABSTAINS instead of 102882%
                   Q5 segment maybe still misses (test data dependent)

□  W1.2  Run FULL FinanceBench (no --limit) sniper-only baseline
         python eval\run_financebench.py --sniper-only
         EXPECTED: 25-40% — true deterministic ceiling

□  W1.3  Run FULL FinanceBench WITHOUT sniper-only (full LLM)
         python eval\run_financebench.py
         EXPECTED: 45-55% — full pipeline including LLM
         NOTE: Will take 1-2 HOURS. Need Ollama warm + Llama 3.1 8B loaded

□  W1.4  Camelot fallback for sparse PDFs (P3.1)
         Modify src/ingestion/pdf_ingestor.py:
         - If pdfplumber returns < 200 cells AND PDF has > 50 pages
         - Try Camelot lattice + Camelot stream
         - Merge cells, dedupe by (page, row, col)
         pip install camelot-py[cv]
         pip install ghostscript  (Windows: install gsdll64.dll)
         EXPECTED IMPACT: +5-10pp on Activision/3M_2018 style sparse PDFs

□  W1.5  Add diagnostic logging to extract_lib resolver
         Currently silently fails. Add per-metric "tried X, picked Y" logs.

□  W1.6  Tune retrieval thresholds:
         src/retrieval/bm25_retriever.py MIN_SCORE 0.01 → 0.05
         src/retrieval/bge_retriever.py MIN_SIMILARITY 0.05 → 0.10
         Re-run, compare


WEEK 2 — RETRIEVAL UPGRADE — GATE M2 (Days 8-14)
───────────────────────────────────────────────────────────────────────────
□  W2.1  Build retrieval-only test harness
         eval/retrieval_test.py
         For each Q: check if gold answer text is in top-10 chunks
         Target: top-10 recall >= 85% (PDR Gate M2)

□  W2.2  Query rewriting:
         Use Llama to generate 2-3 alternate phrasings per question
         Run BM25 on each, merge via RRF
         Test impact on recall

□  W2.3  BGE-M3 fine-tune preparation:
         Generate ~5000 triplets (query, positive_chunk, hard_negative)
         from custom 196Q eval results
         Target: MRR@10 ≥ 0.90

□  W2.4  Cell-cluster builder (HDBSCAN):
         Group neighbouring cells by header similarity
         60x speedup for table search

□  W2.5  GATE M2 CONFIRMATION


WEEK 3 — DPO DATA + XGB ARBITER — GATE M6 (Days 15-21)
───────────────────────────────────────────────────────────────────────────
□  W3.1  Wire RLEF JEE engine to actually grade pipeline runs
         Each FinanceBench run produces 150-200 (chosen, rejected) DPO pairs

□  W3.2  Run 2-3 evals to collect 300+ DPO pairs (Gate M6 threshold)

□  W3.3  Manual quality review of DPO pairs (remove obvious garbage)

□  W3.4  Train XGB arbiter on DPO data
         python -m src.ml.xgb_arbiter --train
         Holdout accuracy ≥ 75% = Gate M6 PASS

□  W3.5  Re-eval with XGB active. Document uplift.

□  W3.6  GATE M6 CONFIRMATION


WEEK 4 — LLM FINE-TUNING — GATE M5 (Days 22-28)
───────────────────────────────────────────────────────────────────────────
□  W4.1  Generate SFT training data (1500-2000 examples)
         (context, question, answer) triplets from passing runs

□  W4.2  SFT LoRA fine-tune Llama 3.1 8B on free Colab T4
         3-5 epochs, batch=4-8, LoRA r=16, alpha=32
         C5: seed=42, C6: not yet (DPO step)
         Save as: ollama create financebench-expert-v1-sft

□  W4.3  DPO fine-tune with beta=0.1 (C6 non-negotiable)
         2-3 epochs, validation holdout
         Save as: ollama create financebench-expert-v1

□  W4.4  Final FinanceBench eval with tuned model
         Target: ≥ 76% (Gate M5 PASS)
         TIER 1 GOAL ACHIEVED if this hits 60-70%

□  W4.5  GATE M5 CONFIRMATION + PHASE 2 EXIT GATE
         Confirmed score documented in CONTEXT.md
         No projections. Only real numbers.


═══════════════════════════════════════════════════════════════════════════
              POST-PHASE 2 — TIER 2 PATH (75-85% goal)
═══════════════════════════════════════════════════════════════════════════

PHASE 3 — UNIVERSAL EXTRACTION
───────────────────────────────────────────────────────────────────────────
□  P3.1  Camelot fallback (started in W1.4 — finish + test)
□  P3.2  PaddleOCR for scanned PDFs
□  P3.3  python-docx for .docx
□  P3.4  openpyxl for .xlsx + chardet for text
□  P3.5  Universal extraction router by file type
□  P3.6  Test on ALL 84 FinanceBench PDFs — target zero cells=0

PHASE 4 — ADVANCED N20 (the missing JEE pieces)
───────────────────────────────────────────────────────────────────────────
□  P4.1  spaCy NER + dependency parser for side-words
□  P4.2  Formula dependency graph builder
□  P4.3  Single-value anchoring (Sniper picks the seed)
□  P4.4  Topological sort + matrix solve
□  P4.5  3-path self-consistency verification

PHASE 5 — FULL BENCHMARK SWEEP — TIER 2 CONFIRMATION
───────────────────────────────────────────────────────────────────────────
□  P5.1  FinanceBench full 150  (TIER 2 target: 75-85%)
□  P5.2  FinQA full              (TIER 2 target: 65-75%)
□  P5.3  ConvFinQA              (TIER 2 target: 60-70%)
□  P5.4  TAT-QA                 (TIER 2 target: 75-85%)
□  P5.5  FinanceReasoning hard  (TIER 2 target: 60-75%)
□  P5.6  Custom 175-Q          (target 95%+)
□  P5.7  HuggingFace public benchmarks (FinBen, etc.)
□  P5.8  All numbers seed=42 reproducible

PHASE 6 — OUTPUT GENERATION (the showcase)
───────────────────────────────────────────────────────────────────────────
□  P6.1  Tier 1: Single-Query Answer Card (PDF + JSON + PNG)
□  P6.2  Tier 2: Multi-Query Dashboard (interactive HTML)
□  P6.3  Tier 3: Investment Thesis (25-40 page McKinsey-style PDF)
         Cover, TOC, Executive Summary, Business Overview,
         Financial Performance, Segment Analysis, Industry Position,
         Valuation (DCF), Risk Analysis, Bull/Bear/Base, Recommendation
□  P6.4  Charts: revenue trend, margin waterfall, DCF sensitivity, peer compare
□  P6.5  Brand polish: navy/gold, Inter typography, citations footer

PHASE 7 — SHIP + CAREER
───────────────────────────────────────────────────────────────────────────
□  P7.1  GitHub README polish (real numbers, architecture diagram)
□  P7.2  HuggingFace Space demo (free hosting)
□  P7.3  Ollama Hub model push (C10 requirement)
□  P7.4  Demo video (5-10 min YouTube)
□  P7.5  Blog post / Reddit r/LocalLLaMA
□  P7.6  Paper draft (EMNLP/ACL — N20 Composite Resolver as novelty)
□  P7.7  LinkedIn outreach to NVIDIA + AI/ML roles


═══════════════════════════════════════════════════════════════════════════
                        IMMEDIATE NEXT ACTIONS
═══════════════════════════════════════════════════════════════════════════

DO THIS NOW, IN ORDER (PowerShell):

1. Clear cache so FIX-v5 takes effect on fresh ingest:
   Remove-Item -Recurse -Force D:\projects\finbench_agent\cache

2. Verify FIX-v5 self-tests pass:
   python -m src.analysis.decision_engine
   python -m src.analysis.narrative_extractor
   python -m src.analysis.composite_resolver

3. Re-run the 5-Q sniper-only eval:
   python eval\run_financebench.py --limit 5 --sniper-only

   EXPECTED RESULTS (honest):
   - Q1 capex      → PASS (raw_text fallback with fixed regex finds 1,577)
   - Q2 PPE        → maybe PASS (whitespace-tolerant scan + fixed regex)
   - Q3 capital-intensive → ABSTAIN (no false 102882% answer)
   - Q4 what drove → maybe PASS (if MD&A text mentions litigation/PFAS)
   - Q5 segment    → maybe PASS (now uses 3M-scoped segments)

   ANY abstention is OK — that's HONEST output, not garbage.

4. Paste me the full output. We diagnose any remaining miss with REAL data.


═══════════════════════════════════════════════════════════════════════════
                       NON-NEGOTIABLE RULES
═══════════════════════════════════════════════════════════════════════════

C1   $0 cost                            ✅ enforced
C2   100% local (no network at inference)  ✅ enforced
C3   Llama 3.1 8B Q4_K_M via Ollama     ✅ enforced
C4   14GB RAM cap                       ⚠️  watch during fine-tuning
C5   seed=42                            ✅ enforced everywhere
C6   DPO beta=0.1                       ✅ locked
C7   Context-first prompts              ✅ enforced
C8   Chunk metadata prefix              ✅ enforced
C9   _rlef_ never in any output         ✅ enforced
C10  Ollama Hub distribution            ⏳ Phase 7
