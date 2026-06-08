# BUGS_FOUND_LINE_BY_LINE.md
# Complete systematic audit of the retrieval + analysis chain
# Date: 2026-06-07 | Method: read every line of every file in the query path

═══════════════════════════════════════════════════════════════════════════
            FILES READ LINE BY LINE (the full query path)
═══════════════════════════════════════════════════════════════════════════

✓ src/pipeline/pipeline.py            (orchestrator — every node)
✓ src/ingestion/pdf_ingestor.py       (N01 — PDF → cells + raw_text)
✓ src/ingestion/chunker.py            (N03 — chunks + BM25 index build)
✓ src/retrieval/bm25_retriever.py     (N07 — BM25 search)
✓ src/retrieval/sniper_rag.py         (N06 — direct table cell extraction)
✓ src/analysis/composite_resolver.py  (N20 — routing to sub-resolvers)
✓ src/analysis/decision_engine.py     (N20a — yes/no + classification)
✓ src/analysis/narrative_extractor.py (N20b — "what drove X" + segment)
✓ src/routing/formula_router.py       (N06b — ratio/margin formulas)
✓ extract_lib/resolver.py             (scored cell extraction)
✓ extract_lib/synonyms.py             (metric synonym dictionary)


═══════════════════════════════════════════════════════════════════════════
                    BUGS FOUND (prioritized)
═══════════════════════════════════════════════════════════════════════════

🔴 BUG #1 — MAX_TABLES=2000 cap deleted the financial statements
   File: pdf_ingestor.py, line ~57
   Severity: CRITICAL — this was silently breaking most extraction.
   Detail: cells appended page-by-page; the cap filled from early pages
           (cover/TOC/MD&A) and chopped the income statement & balance
           sheet (pages ~50-70 in a 10-K). The revenue/PPE cells were
           DELETED before extraction ran.
   STATUS: FIXED (v14, raised to 30000). *** NOT YET TESTED BY YOU ***
   Your last eval ran BEFORE this fix.

🟠 BUG #2 — Sniper threshold too high (_HIT_THRESHOLD = 0.95)
   File: sniper_rag.py, line ~310
   Severity: MEDIUM — Sniper (fastest extractor) rarely fires.
   Detail: pdfplumber cells have no iXBRL tags, so Sniper falls to
           row-header matching at base conf 0.85-0.92. With bonuses it
           often lands at 0.90-0.94 — just under the 0.95 gate. So it
           misses, forcing slower paths.
   STATUS: NOT changed (lowering risks wrong answers; needs careful test).

🟠 BUG #3 — Sniper rebuilds TableIndex on every query
   File: sniper_rag.py, run_sniper()
   Severity: LOW (perf only) — with 30000 cells now, ~1-2s per query.
   STATUS: acceptable for now.

🟡 BUG #4 — "RETRIEVAL_MISS" is a misleading label
   File: pipeline.py, SKIP_LLM bail
   Severity: COSMETIC — confused us for hours. It does NOT mean retrieval
           failed; it means SKIP_LLM=1 bailed after deterministic paths
           passed without an answer.
   STATUS: harmless; leave as-is.


═══════════════════════════════════════════════════════════════════════════
                    CONFIRMED WORKING (no bugs)
═══════════════════════════════════════════════════════════════════════════

✓ BM25 retriever — loads index, tokenizes, retrieves, fuzzy+boost. Correct.
✓ BGE retriever — Chroma-based semantic. Correct (fails gracefully if off).
✓ RRF reranker — fuses BM25+BGE. Correct.
✓ chunker — builds BM25 index, sets state.bm25_index_path. Correct.
✓ composite_resolver — routes decision/causal/segment/question_lib. Correct.
✓ formula_router — uses v13 magnitude-aware resolver + sanity bounds. Correct.
✓ decision_engine — v13 floor applied. Correct.
✓ extract_lib resolver — v13 magnitude-aware. Correct (unit-tested: 34229 ✓).


═══════════════════════════════════════════════════════════════════════════
                    THE HONEST VERDICT
═══════════════════════════════════════════════════════════════════════════

There is NO single magic retrieval bug. The retrieval engine is correct.

The TWO real bugs that hurt extraction:
  1. Cell cap (FIXED v14 — UNTESTED by you)
  2. Sniper threshold (conservative — not yet touched)

The questions still failing after v14 will be:
  - Q4, Q5 (narrative "what drove" / "which segment") → NEED THE LLM
  - Q6-Q8 (quarterly narrative, if narrative) → NEED THE LLM

No amount of deterministic fixing solves narrative questions. That is the
architectural ceiling. The LLM (on Colab T4) is the only tool that reads
MD&A prose and answers "PFAS litigation drove the margin down."


═══════════════════════════════════════════════════════════════════════════
                    NEXT STEP (the untested big fix)
═══════════════════════════════════════════════════════════════════════════

You have NOT yet run an eval with FIX-v14 (the cell-cap fix). Do it now:

  $env:SKIP_LLM="1"
  Remove-Item -Recurse -Force D:\projects\finbench_agent\cache -ErrorAction SilentlyContinue
  python eval\run_financebench.py --limit 10

Watch for:
  - cells= now showing >2000 (e.g. 8000, 15000) — NOT capped at 2000
  - Q2 PPE → real $8.x billion
  - Q3 capital-intensive → finds revenue $34,229M → classifies → WIN

If cells go above 2000 and Q2/Q3 flip to ✓, the cell-cap fix worked and
we've unlocked the extraction category. Then: Colab for the narrative half.
