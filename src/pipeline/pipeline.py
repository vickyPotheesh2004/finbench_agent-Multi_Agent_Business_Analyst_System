from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
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

        logger.warning(
            "[SNIPER] Failed: %s",
            exc,
        )

        state.sniper_hit = False
        state.sniper_confidence = 0.0
        state.sniper_answer = ""

    return state


def _run_bm25_node(state):

    try:

        retriever = BM25Retriever()

        return retriever.run(state)

    except Exception as exc:

        logger.warning(
            "[BM25] Failed: %s",
            exc,
        )

        if not hasattr(
            state,
            "bm25_results",
        ):
            state.bm25_results = []

        return state


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class FinBenchPipeline:

    def __init__(self):

        self.parallel = ParallelExecutor(
            max_workers=2
        )

        self.llm_client = None

        if get_llm_client is not None:

            try:

                self.llm_client = (
                    get_llm_client()
                )

            except Exception as exc:

                logger.warning(
                    "[PIPELINE] LLM unavailable: %s",
                    exc,
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

        logger.info(
            "[PIPELINE] Ingest complete | chunks=%d",
            getattr(
                state,
                "chunk_count",
                0,
            ),
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

        state.query_type = classify_query(
            question
        )

        logger.info(
            "[PIPELINE] Query: %s",
            question[:100],
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
        # Sniper Early Exit
        # ──────────────────────────────────────────────────────────────────────

        if (
            getattr(
                state,
                "sniper_hit",
                False,
            )
            and getattr(
                state,
                "sniper_confidence",
                0.0,
            ) >= 0.97
        ):

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
        # Parallel Retrieval
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

        bm25_state, bge_state = (
            self.parallel.run(
                _bm25,
                _bge,
            )
        )

        state.bm25_results = getattr(
            bm25_state,
            "bm25_results",
            [],
        )

        state.bge_results = getattr(
            bge_state,
            "bge_results",
            [],
        )

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


def _safe_run(
    label: str,
    fn,
    state,
):

    try:

        result = fn(state)

        return (
            result
            if result is not None
            else state
        )

    except Exception as exc:

        logger.error(
            "[%s] Failed: %s",
            label,
            exc,
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