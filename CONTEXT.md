# CONTEXT.md — FinBench Multi-Agent Business Analyst AI
# PDR-BAAAI-001 · Rev 1.0 · Session State File
# PASTE THIS ENTIRE FILE AT THE START OF EVERY NEW AI SESSION

## BUILD_STEP
Week 5 Day 1 — Starting N08 BGE-M3 Semantic Retrieval

## PHASE
Phase 2 — VectorlessFirst Retrieval System

## PROJECT_GOAL
FinanceBench >= 82% at launch | 91-93% full stack | $0 cost | 100% local
Evaluated by: python run_eval.py --seed 42
Never post projections. Only confirmed scores.

## LAST_GATE
M1 PASSED — Week 1 (544/544 tests passing)

## THIS_SESSION_TASK
Build N08 BGE-M3 Semantic Retriever
(sentence-transformers, ChromaDB, domain-aware embeddings)

## OVERALL_PROGRESS
Nodes complete  : 3 of 19 (16%)
Tests passing   : 544
Gates passed    : 1 of 9
Overall         : ~20%

## NODE_STATUS
INGESTION
  N01 PDF Ingestor          ⏳ PENDING — Week 3 spec, real impl needed
  N02 Section Tree Builder  ⏳ PENDING — Week 3 spec, real impl needed
  N03 Chunker + Indexer     ✅ COMPLETE — used in test_bm25.py

ROUTING
  N04 CART Router           ⏳ PENDING — Week 7
  N05 LR Difficulty         ⏳ PENDING — Week 7

RETRIEVAL
  N06 SniperRAG             ✅ COMPLETE — 93 tests, .hit() + .run(state)
  N07 BM25 Retriever        ✅ COMPLETE — 24 tests, LangChain wrapper
  N08 BGE-M3 Semantic       ⏳ NEXT — Week 5 (THIS SESSION)
  N09 RRF + Reranker        ⏳ PENDING — Week 7

ANALYSIS
  N10 Prompt Assembler      ⏳ PENDING — Week 8
  N11 Analyst Pod (PIV)     ⏳ PENDING — Week 9
  N12 CFO/Quant Pod         ⏳ PENDING — Week 10
  N13 TriGuard Forensics    ⏳ PENDING — Week 10
  N14 Auditor Pod (Blind)   ⏳ PENDING — Week 9
  N15 PIV Mediator          ⏳ PENDING — Week 9

POST-ANALYSIS
  N16 SHAP + Causal DAG     ⏳ PENDING — Week 11
  N17 XGB Arbiter           ⏳ PENDING — Week 14 (Gate M6 required)
  N18 RLEF JEE Engine       ⏳ PENDING — Week 13
  N19 Output Generator      ⏳ PENDING — Week 15

## PDR_SECTION_7_STATUS
7.2  Tier 1 SniperRAG  N06   ✅ COMPLETE
7.3  Tier 2 BM25       N07   ✅ COMPLETE
7.4  Tier 3 BGE-M3     N08   ⏳ THIS SESSION
7.5  Tier 4 RRF+Rerank N09   ⏳ Week 7
7.6  Cascade wiring          ⏳ Week 7 (at N09)

## PDR_SECTION_5B_STATUS
Part A  14 Document Types         📋 Spec — N01 implements
Part B  Ingestion N01-N03 stack   ⚠️  N03 done, N01+N02 pending
Part C  Routing+Retrieval N04-N09 ⚠️  N06+N07 done, rest pending
Part D  Analysis N10-N15 PIV      ⏳ Not started (Week 8+)
Part E  Post-Analysis N16-N19     ⏳ Not started (Week 11+)
Part F  LangChain+LangGraph map   ⏳ Wired at N10+N11

## LIVE_DATA_STATUS
Real PDF reading (pdfplumber)     ⏳ Needs N01
Real SEC filings connected        ⏳ Needs N01
FinanceBench PDFs loaded          ⏳ run_eval.py stub only
Ollama / Llama 8B connected       ⏳ Needs N10
ChromaDB with real embeddings     ⏳ Needs N08 (THIS SESSION)
BM25 index from real docs         ✅ N03+N07 wired

## GATE_STATUS
M1  Schema+Eval+CI/CD   ✅ PASSED  Week 1  (544 tests)
M2  Retrieval >=85%     ⏳ PENDING Week 7
M3  BGE-M3 MRR>=0.85    ⏳ PENDING Week 6
M4  Full Pipeline 100%  ⏳ PENDING Week 9
M5  LLM SFT >=76%       ⏳ PENDING Week 12
M6  XGB >=300 DPO pairs ⏳ PENDING Week 14
M7  FB >=84% confirmed  ⏳ PENDING Week 15
M8  LAUNCH FB>=82%      ⏳ PENDING Week 17-18
M9  RLEF Active         ⏳ PENDING Post-launch

## FILES_WRITTEN
src/state/ba_state.py
src/utils/seed_manager.py
src/utils/resource_governor.py
src/ingestion/chunker.py
src/retrieval/__init__.py
src/retrieval/sniper_rag.py
src/retrieval/bm25_retriever.py
eval/run_eval.py
tests/test_unit.py (475 tests)
tests/test_n06_sniper_rag.py (93 tests)
tests/test_bm25.py (24 tests)
.github/workflows/tests.yml

## BA_STATE_KEY_FIELDS
session_id, document_path, company_name, doc_type, fiscal_year
seed=42 (always, C5)
raw_text, table_cells, heading_positions
section_tree, chunk_count, bm25_index_path, chromadb_collection
query, query_type, query_difficulty, context_window_size
sniper_hit, sniper_result, sniper_confidence
bm25_results, bm25_confidence
retrieval_stage_1, retrieval_stage_2
assembled_prompt, analyst_output, quant_result
auditor_output, contradiction_flags
piv_candidates, piv_round, iteration_count (hard cap=5)
confidence_score, low_confidence, final_answer
forensic_flags, risk_score
_rlef_grade, _rlef_chosen (PRIVATE — never in output)

## NON_NEGOTIABLE_RULES
C1  $0 cost — no paid APIs ever
C2  100% local — zero network calls during inference
C3  Llama 3.1 8B Q4_K_M via Ollama at localhost:11434
C4  14GB RAM cap — ResourceGovernor enforces halt
C5  seed=42 everywhere — SeedManager wraps all calls
C6  DPO beta=0.1 — never change
C7  Context BEFORE question in 100% of prompts
C8  Every chunk prefix: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE
C9  _rlef_ fields never in any output, UI, or log
C10 Distribution via: ollama pull USERNAME/financebench-expert

## ARCHITECTURE_SUMMARY
N01  PDF Ingestor        — pdfplumber tables + PyMuPDF headings
N02  Section Tree        — hierarchical JSON, Llama summaries
N03  Chunker+Indexer     — section-boundary chunks, bm25s + ChromaDB
N04  CART Router         — sklearn DecisionTree, 5 query classes
N05  LR Difficulty       — sklearn LogisticRegression, 3 levels
N06  SniperRAG           — 20+ regex, table_index, confidence>=0.95
N07  BM25 Retriever      — bm25s, mmap, top-10, LangChain wrapper
N08  BGE-M3 Retriever    — sentence-transformers, ChromaDB, top-10
N09  RRF+Reranker        — 8-line RRF + bge-reranker-base, top-3
N10  Prompt Assembler    — Jinja2, 5 templates, context always first
N11  Analyst Pod         — PIV loop, Llama 8B, Candidate 1
N12  CFO/Quant Pod       — PIV + Monte Carlo + VaR + GARCH
N13  TriGuard Forensics  — Benford + IsolationForest + GARCH + RF
N14  Auditor Pod (BLIND) — independent retrieval, Candidate 3
N15  PIV Mediator        — 3-pod debate, 2-agree wins
N16  SHAP + Causal DAG   — TreeExplainer + networkx + matplotlib
N17  XGB Arbiter         — xgboost ranking (Gate M6, Week 14+)
N18  RLEF JEE Engine     — sqlite3, 3-validator grader, DPO pairs
N19  Output Generator    — python-docx DOCX, Plotly charts

## KEY_CLASSES
TableCell         — single cell with metadata_key (C8)
TableIndex        — from_raw_cells() factory
SniperRAG         — .hit(query) + .run(state) for LangGraph
SniperResult      — sniper_hit, answer, confidence, citation
BM25Retriever     — .run(state), .search_direct(), .as_langchain_retriever()

## CONFIDENCE_THRESHOLDS (N06)
_CONF_EXACT      = 0.98
_CONF_PREFIX     = 0.92
_CONF_CONTAINS   = 0.85
_CONF_UNIT_BONUS = 0.02
_HIT_THRESHOLD   = 0.95

## KNOWN_ISSUES
None active. All 544 tests passing.

## NEXT_SESSION
Week 5 Day 1 — N08 BGE-M3 Semantic Retriever
Library: sentence-transformers + chromadb
Base model: BAAI/bge-m3
Gate M3: MRR@10 >= 0.85 required before deploy



## SESSION_8  ·  2026-04-30  ·  Bugs #1 + #1.1 + #1.5 FIXED
- src/retrieval/bm25_retriever.py: rewrote _search() with single-chunk
  bypass and term-overlap fallback for bm25s library quirks
- src/ingestion/chunker.py: _chunk_by_paragraphs no longer consolidates;
  DISABLE_CHROMADB env var made permanent
- tests/test_n07_bm25_regression.py: 18 new regression tests
- Test totals: 1287 + 18 = 1305 passing
- HIDDEN: bm25s.retrieve() still throws "list index out of range" but
  fallback path returns useful results (lower quality than ideal BM25
  but non-zero). Tracked as Bug #1.2, deferred.
- Next session: Bug #2 — HTML ingestor _ingest_html() produces 0 headings
  and 0 table_cells, breaking section_tree and chunk quality on real
  10-K HTML files. File: src/ingestion/pdf_ingestor.py method _ingest_html()

  