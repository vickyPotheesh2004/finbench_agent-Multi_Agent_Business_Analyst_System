from __future__ import annotations

import logging
import os
from typing import Any

from src.state.ba_state import BAState

from src.utils.runtime_cache import RuntimeCache
from src.utils.parallel_executor import ParallelExecutor
from src.analysis.answer_validator import AnswerValidator
from src.utils.memory_guard import cleanup_memory
from src.utils.query_classifier import classify_query

# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.ingestion.section_tree_builder import run_section_tree_builder
from src.ingestion.chunker import run_chunker

# ──────────────────────────────────────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────────────────────────────────────

from src.routing.cart_router import run_cart_router
from src.routing.lr_difficulty import run_lr_difficulty
from src.routing.formula_router import run_formula_router

# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

from src.retrieval.sniper_rag import (
    run_sniper,
    SniperResult,
)

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.bge_retriever import run_bge
from src.retrieval.rrf_reranker import run_rrf_reranker

# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

from src.analysis.prompt_assembler import (
    run_prompt_assembler,
)

from src.analysis.piv_loop import (
    run_analyst_pod,
)

from src.analysis.cfo_quant_pod import (
    run_cfo_quant_pod,
)

from src.analysis.triguard import (
    run_triguard,
)

from src.analysis.auditor_pod import (
    run_auditor_pod,
)

from src.analysis.piv_mediator import (
    run_piv_mediator,
)

# ──────────────────────────────────────────────────────────────────────────────
# Explainability + ML
# ──────────────────────────────────────────────────────────────────────────────

from src.analysis.shap_dag import run_shap_dag

try:
    from src.ml.xgb_arbiter import run_xgb_arbiter
except ImportError:
    run_xgb_arbiter = None

# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

from src.rlef.jee_engine import run_rlef_engine
from src.output.docx_generator import (
    run_output_generator,
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional LLM Client
# ──────────────────────────────────────────────────────────────────────────────

try:
    from src.utils.llm_client import (
        get_llm_client,
    )
except ImportError:
    get_llm_client = None

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Query Type Mapping
# ──────────────────────────────────────────────────────────────────────────────
#
# 2026-06-05: query_classifier.QueryType constants now match BAState's
# canonical names directly, so QUERY_TYPE_MAP is empty (kept as identity
# map for legacy callers that pass non-canonical names like "numeric").

QUERY_TYPE_MAP = {
    # legacy → canonical (kept for backward compatibility)
    "numeric":     "numerical",
    "narrative":   "text",
    "qualitative": "text",
    "comparison":  "multi_doc",
}

VALID_QUERY_TYPES = {
    "numerical",
    "ratio",
    "multi_doc",
    "text",
    "forensic",
}

# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Wrappers
# ──────────────────────────────────────────────────────────────────────────────


def _run_sniper_node(state):

    query = getattr(state, "query", "") or ""

    cells = getattr(
        state,
        "table_cells",
        [],
    ) or []

    if not query or not cells:

        state.sniper_hit = False
        state.sniper_confidence = 0.0
        state.sniper_answer = ""
        state.sniper_result = None

        return state

    try:

        result: SniperResult = run_sniper(
            query,
            cells,
        )

        state.sniper_hit = bool(
            result.sniper_hit
        )

        state.sniper_confidence = float(
            result.confidence
        )

        state.sniper_answer = str(
            result.answer or ""
        )

        state.sniper_result = (
            result.answer
            if result.sniper_hit
            else None
        )

    except Exception as exc:

        logger.exception(
            "[SNIPER] Failed"
        )

        state.sniper_hit = False
        state.sniper_confidence = 0.0
        state.sniper_answer = ""
        state.sniper_result = None

    return state


def _run_bm25_node(state):

    try:

        retriever = BM25Retriever()

        return retriever.run(state)

    except Exception:

        logger.exception(
            "[BM25] Failed"
        )

        state.bm25_results = []

        return state


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class FinBenchPipeline:

    def __init__(self):

        # Windows + Ollama + BGE stability
        self.parallel = ParallelExecutor(
            max_workers=1
        )

        self.llm_client = None

        if get_llm_client is not None:

            try:

                self.llm_client = (
                    get_llm_client()
                )

            except Exception:

                logger.exception(
                    "[PIPELINE] LLM unavailable"
                )

        logger.info(
            "[PIPELINE] Ready"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Ingest
    # ──────────────────────────────────────────────────────────────────────────

    def ingest(
        self,
        document_path: str,
        session_id: str = "",
        company_name: str = "",
        doc_type: str = "",
        fiscal_year: str = "",
        enable_images: bool = False,
        **_unused,
    ) -> BAState:

        cache_key = (
            f"{document_path}:"
            f"{company_name}:"
            f"{fiscal_year}"
        )

        if RuntimeCache.exists(
            "ingest",
            cache_key,
        ):

            logger.info(
                "[PIPELINE] Using cached ingest"
            )

            return RuntimeCache.load(
                "ingest",
                cache_key,
            )

        state = BAState(
            session_id=session_id or "session",
            document_path=document_path,
            company_name=company_name,
            doc_type=doc_type,
            fiscal_year=fiscal_year,
        )

        logger.info(
            "[PIPELINE] Ingest: %s",
            document_path,
        )

        state = run_pdf_ingestor(
            state,
            enable_images=enable_images,
            llm_client=self.llm_client,
        )

        state = run_section_tree_builder(
            state,
            llm_client=self.llm_client,
        )

        state = run_chunker(state)

        RuntimeCache.save(
            "ingest",
            cache_key,
            state,
        )

        # FIX-v3 (2026-06-05): diagnostic logging — surface what was extracted
        # FIX-v4 (2026-06-05): use warning() so it shows under root WARNING level
        # (the eval script sets root logger to WARNING, so INFO is hidden)
        _table_cells = getattr(state, "table_cells", None) or []
        _raw_text    = getattr(state, "raw_text",    "") or ""
        logger.warning(
            "[PIPELINE] Ingest complete | chunks=%d | cells=%d | raw_text=%dKB",
            getattr(state, "chunk_count", 0),
            len(_table_cells),
            len(_raw_text) // 1024,
        )

        cleanup_memory()

        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        state: BAState,
        question: str,
    ) -> BAState:

        if state is None:

            raise ValueError(
                "state is None"
            )

        state.query = question

        # Safe defaults

        state.bm25_results = getattr(
            state,
            "bm25_results",
            [],
        ) or []

        state.bge_results = getattr(
            state,
            "bge_results",
            [],
        ) or []

        state.retrieval_stage_1 = getattr(
            state,
            "retrieval_stage_1",
            [],
        ) or []

        state.reranked_chunks = getattr(
            state,
            "reranked_chunks",
            [],
        ) or []

        # Query type normalization

        query_type = classify_query(
            question
        )

        query_type = QUERY_TYPE_MAP.get(
            query_type,
            query_type,
        )

        if query_type not in VALID_QUERY_TYPES:

            logger.warning(
                "[PIPELINE] Invalid query type: %s",
                query_type,
            )

            query_type = "text"

        state.query_type = query_type

        logger.info(
            "[PIPELINE] Query: %s",
            question[:120],
        )

        # ──────────────────────────────────────────────────────────────────────
        # Routing
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N04 CART",
            run_cart_router,
            state,
        )

        state = _safe_run(
            "N05 LR",
            run_lr_difficulty,
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Sniper
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N06 Sniper",
            _run_sniper_node,
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # N06b Formula Router (maths_lib deterministic path)
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N06b Formula",
            run_formula_router,
            state,
        )

        if getattr(state, "formula_hit", False):

            logger.info(
                "[PIPELINE] Formula router answered: %s",
                getattr(state, "formula_answer", ""),
            )

            state = _safe_run(
                "N18 RLEF",
                run_rlef_engine,
                state,
            )

            state = _safe_run(
                "N19 Output",
                run_output_generator,
                state,
            )

            cleanup_memory()

            return state

        # ──────────────────────────────────────────────────────────────────────
        # Sniper Early Exit
        # ──────────────────────────────────────────────────────────────────────

        _sniper_only = os.environ.get("SNIPER_ONLY") == "1"
        _sniper_hit = getattr(state, "sniper_hit", False)
        _sniper_conf = getattr(state, "sniper_confidence", 0.0)
        _sniper_ans = getattr(state, "sniper_answer", "") or ""

        # FIX-v15 (2026-06-08): guard Sniper's early-exit so it can't jump the
        # queue with garbage. After FIX-v14 raised cells to ~30000, Sniper
        # started matching noise cells (e.g. "Recognition" from the revenue-
        # recognition accounting policy) at conf>=0.85 and exiting in 0.6s
        # BEFORE N20 (which had the right answer) could run. Two guards:
        #   1. The sniper answer must contain an actual digit (a financial
        #      value), not a bare word like "Recognition".
        #   2. Narrative / decision / segment questions must NEVER early-exit
        #      on Sniper — they belong to N20. Let them fall through.
        _ans_has_number = bool(_re.search(r"\d", _sniper_ans))
        _q_lower = (getattr(state, "query", "") or "").lower()
        _is_n20_question = any(
            kw in _q_lower
            for kw in (
                "what drove", "why did", "which segment", "capital-intensive",
                "capital intensive", "explain", "what caused", "reason",
                "driver", "is it", "does the",
            )
        )
        _sniper_exit_ok = (
            _sniper_hit
            and _ans_has_number
            and not _is_n20_question
            and (_sniper_only or _sniper_conf >= 0.90)
        )

        if _sniper_exit_ok:

            logger.info(
                "[PIPELINE] Sniper early exit"
            )

            state.final_answer = (
                state.sniper_answer
            )

            state.final_answer_pre_xgb = (
                state.sniper_answer
            )

            state.confidence_score = (
                state.sniper_confidence
            )

            state.winning_pod = (
                "SNIPER_EARLY_EXIT"
            )

            state = _safe_run(
                "N18 RLEF",
                run_rlef_engine,
                state,
            )

            state = _safe_run(
                "N19 Output",
                run_output_generator,
                state,
            )

            cleanup_memory()

            return state

        # ──────────────────────────────────────────────────────────────────────
        # Retrieval
        # ──────────────────────────────────────────────────────────────────────

        def _bm25():
            return _safe_run(
                "N07 BM25",
                _run_bm25_node,
                state,
            )

        def _bge():
            return _safe_run(
                "N08 BGE",
                run_bge,
                state,
            )

        try:

            bm25_state, bge_state = (
                self.parallel.run(
                    _bm25,
                    _bge,
                )
            )

        except Exception:

            logger.exception(
                "[PIPELINE] Parallel retrieval failed"
            )

            bm25_state = state
            bge_state = state

        state.bm25_results = getattr(
            bm25_state,
            "bm25_results",
            [],
        ) or []

        state.bge_results = getattr(
            bge_state,
            "bge_results",
            [],
        ) or []

        if not isinstance(
            state.bm25_results,
            list,
        ):
            state.bm25_results = []

        if not isinstance(
            state.bge_results,
            list,
        ):
            state.bge_results = []

        # ──────────────────────────────────────────────────────────────────────
        # RRF
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N09 RRF",
            run_rrf_reranker,
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Prompt Assembly
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N10 Prompt",
            run_prompt_assembler,
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # FIX-v7 (2026-06-07): N20 Composite Resolver runs in FULL mode too,
        # BEFORE the slow LLM pods. If N20 hits with high confidence we skip
        # the LLM entirely — saving 5-30 MINUTES per question on CPU systems.
        # ──────────────────────────────────────────────────────────────────────
        try:
            from src.analysis.composite_resolver import run_composite_resolver
            _comp_ans = run_composite_resolver(state)
            if _comp_ans.answered and _comp_ans.confidence >= 0.70:
                state.final_answer         = _comp_ans.final_answer
                state.final_answer_pre_xgb = _comp_ans.final_answer
                state.confidence_score     = _comp_ans.confidence
                state.winning_pod          = _comp_ans.winning_pod
                state = _safe_run("N18 RLEF", run_rlef_engine, state)
                state = _safe_run("N19 Output", run_output_generator, state)
                cleanup_memory()
                logger.warning(
                    "[PIPELINE] N20 fast-path HIT (full mode) | pod=%s | conf=%.2f | %s",
                    _comp_ans.winning_pod, _comp_ans.confidence,
                    _comp_ans.final_answer[:120],
                )
                return state
        except Exception:
            logger.exception("[PIPELINE] N20 fast-path failed")

        # FIX-v7: also try the deterministic extract_lib fallback BEFORE LLM
        # if the question is a simple value extraction. This solves Q1-style
        # questions in 1-2 seconds instead of 30 minutes of LLM work.
        _alt = _sniper_only_extract_fallback(state)
        if _alt:
            state.final_answer         = _alt
            state.final_answer_pre_xgb = _alt
            state.confidence_score     = 0.75
            state.winning_pod          = "EXTRACT_LIB_FAST_PATH"
            state = _safe_run("N18 RLEF", run_rlef_engine, state)
            state = _safe_run("N19 Output", run_output_generator, state)
            cleanup_memory()
            logger.warning(
                "[PIPELINE] EXTRACT_LIB fast-path HIT (full mode) | %s",
                _alt[:120],
            )
            return state

        # FIX-v7: CPU users can set SKIP_LLM=1 to bail out before the slow
        # LLM pods if neither N20 nor extract_lib could answer.
        if os.environ.get("SKIP_LLM") == "1":
            state.final_answer         = "RETRIEVAL_MISS"
            state.final_answer_pre_xgb = "RETRIEVAL_MISS"
            state.confidence_score     = 0.0
            state.winning_pod          = "SKIP_LLM_MISS"
            state = _safe_run("N18 RLEF", run_rlef_engine, state)
            state = _safe_run("N19 Output", run_output_generator, state)
            cleanup_memory()
            return state

        # ──────────────────────────────────────────────────────────────────────
        # SNIPER-ONLY LLM BYPASS (P0)
        # ──────────────────────────────────────────────────────────────────────
        #
        # FIX-B (2026-06-04): Removed the buggy fallback that placed the
        # first BM25 chunk text into final_answer. That caused the
        # FinanceBench eval to return the document's TABLE OF CONTENTS as
        # the answer to capex / PPNE questions — scoring 0% true accuracy
        # but getting ~6-7% accidental word-overlap credit.
        #
        # New behaviour: if Sniper missed, try extract_lib direct resolution
        # on table_cells using the matched metric. If that also misses,
        # return a clean RETRIEVAL_MISS — never pollute final_answer with
        # arbitrary retrieval text.
        # ──────────────────────────────────────────────────────────────────────
        if os.environ.get("SNIPER_ONLY") == "1":
            # ───────────────────────────────────────────────────────────────────────────
            # N20 Composite Resolver — deterministic offline answer for
            # complex (decision / causal / segment) questions BEFORE we
            # give up and emit RETRIEVAL_MISS.
            # ───────────────────────────────────────────────────────────────────────────
            try:
                from src.analysis.composite_resolver import run_composite_resolver
                _comp_ans = run_composite_resolver(state)
                if _comp_ans.answered:
                    # N20 found a deterministic answer — use it.
                    state.final_answer         = _comp_ans.final_answer
                    state.final_answer_pre_xgb = _comp_ans.final_answer
                    state.confidence_score     = _comp_ans.confidence
                    state.winning_pod          = _comp_ans.winning_pod
                    state = _safe_run("N18 RLEF", run_rlef_engine, state)
                    state = _safe_run("N19 Output", run_output_generator, state)
                    cleanup_memory()
                    logger.warning(
                        "[PIPELINE] N20 hit | pod=%s | conf=%.2f | %s",
                        _comp_ans.winning_pod,
                        _comp_ans.confidence,
                        _comp_ans.final_answer[:120],
                    )
                    return state
            except Exception:
                logger.exception("[PIPELINE] N20 composite resolver failed")

            if getattr(state, "sniper_hit", False) and getattr(state, "sniper_answer", ""):
                state.final_answer = state.sniper_answer
                state.final_answer_pre_xgb = state.sniper_answer
                state.confidence_score = getattr(
                    state, "sniper_confidence", 0.5
                )
                state.winning_pod = "SNIPER_ONLY"
            else:
                # FIX-B: Try extract_lib as second deterministic pass before giving up
                _alt = _sniper_only_extract_fallback(state)
                if _alt:
                    state.final_answer = _alt
                    state.final_answer_pre_xgb = _alt
                    state.confidence_score = 0.75
                    state.winning_pod = "EXTRACT_LIB_FALLBACK"
                else:
                    state.final_answer = "RETRIEVAL_MISS"
                    state.final_answer_pre_xgb = "RETRIEVAL_MISS"
                    state.confidence_score = 0.0
                    state.winning_pod = "SNIPER_ONLY_MISS"

            state = _safe_run("N18 RLEF", run_rlef_engine, state)
            state = _safe_run("N19 Output", run_output_generator, state)
            cleanup_memory()
            return state

        # ──────────────────────────────────────────────────────────────────────
        # Pods
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N11 Analyst",
            lambda s: run_analyst_pod(
                s,
                llm_client=self.llm_client,
            ),
            state,
        )

        state = _safe_run(
            "N12 CFO",
            lambda s: run_cfo_quant_pod(
                s,
                llm_client=self.llm_client,
            ),
            state,
        )

        state = _safe_run(
            "N13 TriGuard",
            run_triguard,
            state,
        )

        state = _safe_run(
            "N14 Auditor",
            lambda s: run_auditor_pod(
                s,
                llm_client=self.llm_client,
            ),
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Mediation
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N15 Mediator",
            lambda s: run_piv_mediator(
                s,
                llm_client=self.llm_client,
            ),
            state,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Explainability
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N16 SHAP",
            run_shap_dag,
            state,
        )

        if run_xgb_arbiter:

            state = _safe_run(
                "N17 XGB",
                run_xgb_arbiter,
                state,
            )

        # ──────────────────────────────────────────────────────────────────────
        # Validation
        # ──────────────────────────────────────────────────────────────────────

        valid, reason = (
            AnswerValidator.validate(
                getattr(
                    state,
                    "final_answer",
                    "",
                )
            )
        )

        if not valid:

            logger.warning(
                "[PIPELINE] Validation failed: %s",
                reason,
            )

            state.final_answer = (
                f"RETRIEVAL_MISS: {reason}"
            )

        # ───────────────────────────────────────────────────────────────────────
        # FIX-D (2026-06-04, v2): Optional verify_lib post-check
        # ───────────────────────────────────────────────────────────────────────
        # If verify_lib is installed, run a final deterministic verification
        # on the answer. If verify_lib is not installed this is a no-op.
        # The verifier is informational only — we log warnings but never
        # block the answer at this stage.
        #
        # v2: Uses lib_bridge's verify_final_answer_safe which adapts the
        # answer-string to the (metric, value) signature expected by
        # verify_lib.verify_answer. Returns (ok, abstain, reasons[]).
        try:
            from src.utils.lib_bridge import verify_final_answer_safe

            # Choose a metric hint based on the resolved query type
            _metric_hint = "answer"
            _qt = getattr(state, "query_type", "") or ""
            if _qt in ("numerical", "ratio"):
                _metric_hint = "value"

            v_ok, v_abstain, v_reasons = verify_final_answer_safe(
                getattr(state, "final_answer", "") or "",
                metric=_metric_hint,
                confidence=float(
                    getattr(state, "confidence_score", 1.0) or 1.0
                ),
                company=getattr(state, "company_name", "") or "",
                fiscal_year=getattr(state, "fiscal_year", "") or "",
                doc_type=getattr(state, "doc_type", "") or "",
            )
            if not v_ok or v_abstain:
                logger.warning(
                    "[PIPELINE] verify_lib flagged answer | ok=%s abstain=%s | %s",
                    v_ok,
                    v_abstain,
                    "; ".join(v_reasons)[:240],
                )
        except Exception:
            logger.exception("[PIPELINE] verify_lib post-check failed")

        # ──────────────────────────────────────────────────────────────────────
        # Final Output
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            "N18 RLEF",
            run_rlef_engine,
            state,
        )

        state = _safe_run(
            "N19 Output",
            run_output_generator,
            state,
        )

        cleanup_memory()

        logger.info(
            "[PIPELINE] Complete"
        )

        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        document_path: str,
        question: str,
        **kwargs,
    ) -> BAState:

        state = self.ingest(
            document_path=document_path,
            **kwargs,
        )

        return self.query(
            state,
            question,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Safe Runner
# ──────────────────────────────────────────────────────────────────────────────


def _sniper_only_extract_fallback(state) -> str:
    """
    FIX-B fallback: when Sniper misses in SNIPER_ONLY mode, attempt one
    deterministic resolution using extract_lib + pattern_lib before giving up.

    Returns a formatted answer string on success, or empty string on miss.
    Never raises.
    """
    try:
        cells = getattr(state, "table_cells", None) or []
        # FIX-v3 (2026-06-05): proceed even when cells empty — path 2/3 try raw_text/chunks

        query = getattr(state, "query", "") or ""
        if not query:
            return ""

        # Lazy import — these libs are optional
        try:
            from extract_lib.resolver import resolve_metric
        except Exception:
            return ""

        # Map common question phrasings to extract_lib metric IDs.
        # NOTE: extract_lib.synonyms currently defines only 3 metrics:
        #   revenue, cogs, net_income
        # The other aliases below are present so we route the QUESTION TYPE
        # correctly; resolve_metric() will return invalid for unsupported
        # metric IDs and we'll cleanly return "" (no garbage answer).
        # When extract_lib adds more METRIC_SYNONYMS entries, these mappings
        # immediately start working with no code change.
        q = query.lower()
        metric_aliases = {
            # ── Currently supported by extract_lib.synonyms ───────────────
            "net revenue":           "revenue",
            "net sales":             "revenue",
            "total revenue":         "revenue",
            "total net sales":       "revenue",
            "net income":            "net_income",
            "net earnings":          "net_income",
            "cost of revenue":       "cogs",
            "cost of sales":         "cogs",
            "cost of goods sold":    "cogs",
            # ── Routable, will resolve once extract_lib adds these ────────────
            "capital expenditure":   "capex",
            "capex":                 "capex",
            "operating income":      "operating_income",
            "gross profit":          "gross_profit",
            "total assets":          "total_assets",
            "total liabilities":     "total_liabilities",
            "shareholders equity":   "shareholders_equity",
            "stockholders equity":   "shareholders_equity",
            "long-term debt":        "long_term_debt",
            "long term debt":        "long_term_debt",
            "diluted eps":           "eps_diluted",
            "basic eps":             "eps_basic",
            "earnings per share":    "eps_diluted",
            "cash and cash equivalents": "cash",
            "operating cash flow":   "operating_cash_flow",
            "free cash flow":        "free_cash_flow",
            "r&d":                   "r_and_d",
            "research and development": "r_and_d",
            "sg&a":                  "sg_and_a",
            "interest expense":      "interest_expense",
            "income tax":            "income_tax",
            "effective tax rate":    "effective_tax_rate",
        }

        metric_id = ""
        # Prefer longer matches first
        for alias, mid in sorted(metric_aliases.items(), key=lambda x: -len(x[0])):
            if alias in q:
                metric_id = mid
                break
        if not metric_id:
            return ""

        # Period hint from state
        period = getattr(state, "fiscal_year", "") or ""

        # Path 1: extract_lib on table_cells (only if cells exist)
        if cells:
            result = resolve_metric(metric_id, cells, period or "")
            if result and result.valid and result.value is not None:
                raw  = getattr(result, "raw_value", "") or str(result.value)
                row  = getattr(result, "row_header", "") or metric_id
                page = getattr(result, "page", 0) or 0
                company  = getattr(state, "company_name", "") or ""
                doc_type = getattr(state, "doc_type",     "") or ""
                citation = f"{company}/{doc_type}/{period}/{row}/{page}"
                return f"{raw} [{citation}]"

        # Path 2: raw_text scan via extract_lib synonyms (when cells sparse)
        raw_text = getattr(state, "raw_text", "") or ""
        if raw_text and len(raw_text) > 100:
            hit = _scan_raw_text_for_metric(raw_text, metric_id)
            if hit:
                value_str, idx = hit
                company  = getattr(state, "company_name", "") or ""
                doc_type = getattr(state, "doc_type",     "") or ""
                citation = f"{company}/{doc_type}/{period}/{metric_id}/raw@{idx}"
                return f"{value_str} [{citation}]"

        # Path 3: BM25 chunk scan (last resort)
        chunks = getattr(state, "bm25_results", None) or []
        if chunks:
            hit = _scan_chunks_for_metric(chunks, metric_id)
            if hit:
                value_str, page = hit
                company  = getattr(state, "company_name", "") or ""
                doc_type = getattr(state, "doc_type",     "") or ""
                citation = f"{company}/{doc_type}/{period}/{metric_id}/{page}"
                return f"{value_str} [{citation}]"

        return ""

    except Exception:
        logger.exception("[PIPELINE] _sniper_only_extract_fallback failed")
        return ""


def _safe_run(
    label: str,
    fn,
    state,
):

    try:

        result = fn(state)

        if result is None:

            logger.warning(
                "[%s] Returned None",
                label,
            )

            return state

        return result

    except Exception:

        logger.exception(
            "[%s] Failed",
            label,
        )

        return state


# ──────────────────────────────────────────────────────────────────────────────
# Convenience API
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    document_path: str,
    question: str,
    **kwargs,
) -> Any:

    return FinBenchPipeline().run(
        document_path,
        question,
        **kwargs,
    )


# ───────────────────────────────────────────────────────────────────────────────
# FIX-v3 (2026-06-05): Raw-text and chunk fallback helpers
# These let SNIPER_ONLY mode score points even when pdfplumber table
# extraction is sparse (which is common on complex 10-K PDFs).
# ───────────────────────────────────────────────────────────────────────────────

import re as _re

_NUMBER_RE = _re.compile(
    # FIX-v5 (2026-06-05): use \d+ instead of [\d]{1,3} so plain digit
    # sequences (e.g. "1577" after comma-stripping in _norm()) match
    # FULLY. Previously only first 3 digits were captured, truncating
    # "1,577" → "157".  This single bug regressed Q1 from PASS to FAIL.
    r"\$?\s*\(?\-?\s*\d+(?:,\d{3})*(?:\.\d+)?\)?"
    r"(?:\s*(?:million|billion|thousand|bn|mn|m|b))?",
    _re.IGNORECASE,
)


def _scan_raw_text_for_metric(raw_text: str, metric_id: str):
    """
    Scan raw_text for any positive synonym of metric_id and extract the
    nearest plausible number. Returns (value_str, char_index) or None.

    FIX-v4 (2026-06-05): whitespace-tolerant matching. pdfplumber output
    often has "Property, Plant\n  and Equipment" instead of
    "Property, Plant and Equipment". We normalise BOTH the synonym and
    a search window of the raw_text by collapsing all whitespace and
    stripping punctuation, then look for the normalised synonym.
    """
    try:
        from extract_lib.synonyms import METRIC_SYNONYMS
    except Exception:
        return None

    syn = METRIC_SYNONYMS.get(metric_id, {})
    positives = syn.get("positive", [])
    if not positives:
        return None

    def _norm(s: str) -> str:
        # Collapse all whitespace runs to single space, strip commas/colons,
        # lowercase. Keeps & and digits for things like "r&d".
        s = s.lower()
        s = _re.sub(r"[,;:]", "", s)
        s = _re.sub(r"\s+", " ", s)
        return s.strip()

    # Pre-normalise the raw_text ONCE — expensive on 200KB text but the
    # whole eval has only a few queries per ingest so this is fine.
    raw_norm = _norm(raw_text)
    # Build an index of original positions for each char in raw_norm.
    # This lets us map a hit-position back to the original text for citation.
    # For simplicity, citation index becomes the position in raw_norm.

    for synonym in sorted(positives, key=len, reverse=True):
        s_norm = _norm(synonym)
        idx = raw_norm.find(s_norm)
        if idx < 0:
            continue

        # Look in the next 300 chars (of normalised text) for a number
        window_start = idx + len(s_norm)
        window_end   = min(window_start + 300, len(raw_norm))
        window       = raw_norm[window_start:window_end]

        # FIX-v8 (2026-06-07): scan ALL candidate numbers in the window,
        # not just the first one. Skip 4-digit year-like values (1990-2030)
        # and tiny values (<$1M) for big balance-sheet metrics. This fixes
        # the Q2 PPE-extracts-"2018" bug and Q10 revenue-extracts-"18.00".
        for m in _NUMBER_RE.finditer(window):
            number_str = m.group(0).strip()
            digits_only = _re.sub(r"[^\d]", "", number_str)
            if len(digits_only) < 2:
                continue
            # Parse to float for sanity checks
            _raw_no_sign = number_str.replace("(", "").replace(")", "")
            _raw_no_sign = _raw_no_sign.replace("$", "").replace(",", "").strip()
            _raw_no_sign = _re.sub(r"\s*(million|billion|thousand|bn|mn|m|b)\s*$",
                                    "", _raw_no_sign, flags=_re.IGNORECASE).strip()
            try:
                val = float(_raw_no_sign)
            except ValueError:
                continue
            av = abs(val)
            # Skip year-like values
            if 1990 <= av <= 2030 and av == int(av):
                continue
            # Skip tiny values for big balance-sheet metrics
            _big_metrics = {
                "revenue", "cogs", "gross_profit", "operating_income",
                "net_income", "total_assets", "total_liabilities",
                "shareholders_equity", "long_term_debt", "cash",
                "current_assets", "current_liabilities", "inventory",
                "capex", "ppe", "goodwill", "intangible_assets",
                "operating_cash_flow", "free_cash_flow",
                "dividends_paid", "depreciation_amortization",
                "interest_expense", "ebitda", "sg_and_a", "r_and_d",
            }
            if metric_id in _big_metrics and av < 1.0:
                continue
            return (number_str, idx)

    return None


def _scan_chunks_for_metric(chunks, metric_id: str):
    """
    Scan top BM25 chunks for any positive synonym of metric_id and extract
    the nearest plausible number. Returns (value_str, page) or None.

    FIX-v4 (2026-06-05): whitespace-tolerant matching (see
    _scan_raw_text_for_metric for rationale).
    """
    try:
        from extract_lib.synonyms import METRIC_SYNONYMS
    except Exception:
        return None

    syn = METRIC_SYNONYMS.get(metric_id, {})
    positives = syn.get("positive", [])
    if not positives:
        return None

    def _norm(s: str) -> str:
        s = s.lower()
        s = _re.sub(r"[,;:]", "", s)
        s = _re.sub(r"\s+", " ", s)
        return s.strip()

    for chunk in chunks[:6]:
        text = (chunk.get("text") or chunk.get("page_content") or "")
        if not text or len(text) < 30:
            continue

        text_norm = _norm(text)
        page = chunk.get("page", 0) or 0

        for synonym in sorted(positives, key=len, reverse=True):
            s_norm = _norm(synonym)
            idx = text_norm.find(s_norm)
            if idx < 0:
                continue

            window_start = idx + len(s_norm)
            window_end   = min(window_start + 250, len(text_norm))
            window       = text_norm[window_start:window_end]

            m = _NUMBER_RE.search(window)
            if not m:
                continue

            number_str = m.group(0).strip()
            digits_only = _re.sub(r"[^\d]", "", number_str)
            if len(digits_only) < 2:
                continue

            return (number_str, page)

    return None