"""
src/retrieval/rrf_reranker.py

Production-Grade Reciprocal Rank Fusion Reranker

Optimized for:
- FinanceBench
- Hybrid retrieval
- BM25 + BGE fusion
- Stable ranking
- Deduplication
- Retrieval robustness
- Windows
- Colab
- Low memory
"""

from __future__ import annotations

import logging
import math

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TOP_K = 12

RRF_K = 60

MIN_RRF_SCORE = 0.0001

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def normalize_text(
    text: str,
) -> str:

    return (
        text or ""
    ).strip().lower()


def safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        if isinstance(
            value,
            bool,
        ):
            return default

        if math.isnan(
            float(value)
        ):
            return default

        return float(value)

    except Exception:

        return default


def deduplicate_chunks(
    chunks: List[Dict],
) -> List[Dict]:

    seen = set()

    output = []

    for chunk in chunks:

        text = normalize_text(
            chunk.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        key = hash(text)

        if key in seen:
            continue

        seen.add(key)

        output.append(chunk)

    return output

# ─────────────────────────────────────────────────────────────────────────────
# RRF Reranker
# ─────────────────────────────────────────────────────────────────────────────


class RRFReranker:

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = RRF_K,
    ):

        self.top_k = max(
            1,
            int(top_k),
        )

        self.rrf_k = max(
            1,
            int(rrf_k),
        )

    # ─────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        state,
    ):

        bm25_results = getattr(
            state,
            "bm25_results",
            [],
        ) or []

        bge_results = getattr(
            state,
            "bge_results",
            [],
        ) or []

        reranked = self.rerank(
            bm25_results=bm25_results,
            bge_results=bge_results,
        )

        state.reranked_chunks = (
            reranked
        )

        state.retrieval_stage_1 = (
            reranked
        )

        logger.info(
            "[RRF] reranked=%d "
            "(bm25=%d bge=%d)",
            len(reranked),
            len(bm25_results),
            len(bge_results),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────
    # Core Fusion
    # ─────────────────────────────────────────────────────────────────────

    def rerank(
        self,
        bm25_results: List[Dict],
        bge_results: List[Dict],
    ) -> List[Dict]:

        bm25_results = (
            deduplicate_chunks(
                bm25_results
            )
        )

        bge_results = (
            deduplicate_chunks(
                bge_results
            )
        )

        if (
            not bm25_results
            and not bge_results
        ):

            return []

        # Single-source fallback

        if (
            bm25_results
            and not bge_results
        ):

            return self._finalize(
                bm25_results,
                source="bm25_only",
            )

        if (
            bge_results
            and not bm25_results
        ):

            return self._finalize(
                bge_results,
                source="bge_only",
            )

        fused_scores = defaultdict(
            float
        )

        chunk_lookup = {}

        # BM25

        for rank, chunk in enumerate(
            bm25_results,
            start=1,
        ):

            key = self._chunk_key(
                chunk
            )

            fused_scores[key] += (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

            chunk_lookup[
                key
            ] = chunk

        # BGE

        for rank, chunk in enumerate(
            bge_results,
            start=1,
        ):

            key = self._chunk_key(
                chunk
            )

            fused_scores[key] += (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

            if key not in chunk_lookup:

                chunk_lookup[
                    key
                ] = chunk

        fused = []

        for (
            key,
            score,
        ) in fused_scores.items():

            if score < MIN_RRF_SCORE:
                continue

            base = dict(
                chunk_lookup[key]
            )

            base[
                "rrf_score"
            ] = round(
                score,
                8,
            )

            base[
                "fusion_sources"
            ] = self._detect_sources(
                key,
                bm25_results,
                bge_results,
            )

            fused.append(base)

        fused.sort(
            key=lambda x: (
                safe_float(
                    x.get(
                        "rrf_score",
                        0.0,
                    )
                ),
                safe_float(
                    x.get(
                        "bm25_score",
                        0.0,
                    )
                ),
                safe_float(
                    x.get(
                        "bge_score",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

        return self._finalize(
            fused,
            source="hybrid_rrf",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────────

    def _finalize(
        self,
        chunks: List[Dict],
        source: str,
    ) -> List[Dict]:

        output = []

        for rank, chunk in enumerate(
            chunks[
                : self.top_k
            ],
            start=1,
        ):

            item = dict(chunk)

            item[
                "final_rank"
            ] = rank

            item[
                "reranker"
            ] = "rrf"

            item[
                "retrieval_source"
            ] = source

            item[
                "retrieval_confidence"
            ] = self._confidence(
                item
            )

            output.append(item)

        return output

    # ─────────────────────────────────────────────────────────────────────
    # Confidence
    # ─────────────────────────────────────────────────────────────────────

    def _confidence(
        self,
        chunk: Dict,
    ) -> float:

        bm25 = safe_float(
            chunk.get(
                "bm25_score_norm",
                chunk.get(
                    "bm25_score",
                    0.0,
                ),
            )
        )

        bge = safe_float(
            chunk.get(
                "bge_score",
                0.0,
            )
        )

        rrf = safe_float(
            chunk.get(
                "rrf_score",
                0.0,
            )
        )

        confidence = (
            bm25 * 0.35
            + bge * 0.45
            + rrf * 0.20
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return round(
            confidence,
            6,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_key(
        chunk: Dict,
    ) -> str:

        chunk_id = str(
            chunk.get(
                "chunk_id",
                "",
            )
        )

        if chunk_id:
            return chunk_id

        text = normalize_text(
            chunk.get(
                "text",
                "",
            )
        )

        return str(
            hash(text)
        )

    def _detect_sources(
        self,
        key: str,
        bm25_results: List[Dict],
        bge_results: List[Dict],
    ) -> List[str]:

        sources = []

        if any(
            self._chunk_key(c)
            == key
            for c in bm25_results
        ):

            sources.append(
                "bm25"
            )

        if any(
            self._chunk_key(c)
            == key
            for c in bge_results
        ):

            sources.append(
                "bge"
            )

        return sources

    # ─────────────────────────────────────────────────────────────────────
    # Direct API
    # ─────────────────────────────────────────────────────────────────────

    def rerank_direct(
        self,
        bm25_results: List[Dict],
        bge_results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if top_k is not None:

            self.top_k = max(
                1,
                int(top_k),
            )

        return self.rerank(
            bm25_results=bm25_results,
            bge_results=bge_results,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_rrf_reranker(
    state,
):

    reranker = RRFReranker(
        top_k=DEFAULT_TOP_K,
        rrf_k=RRF_K,
    )

    return reranker.run(
        state
    )