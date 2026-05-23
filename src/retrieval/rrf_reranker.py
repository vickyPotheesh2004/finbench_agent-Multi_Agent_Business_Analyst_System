"""
src/retrieval/rrf_reranker.py

Production-Grade RRF + Cross-Encoder Reranker
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. Reciprocal Rank Fusion (RRF)
2. Cross-encoder reranking
3. Early-exit optimization
4. Graceful degradation
5. LangChain compatibility
6. CPU-safe execution
7. Retrieval deduplication
8. Financial-priority boosting
9. Stable deterministic ranking
10. Memory-safe candidate handling
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_RRF_K = 60

_DEFAULT_TOP_K = 3

_RERANK_POOL_MULTIPLIER = 4

_RERANKER_MODEL = "BAAI/bge-reranker-base"

_RETRIEVER_LABEL = "rrf_reranker"

_HIGH_CONFIDENCE_THRESHOLD = 0.85

# Singleton
_RERANKER_INSTANCE = None

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Loader
# ─────────────────────────────────────────────────────────────────────────────


def get_reranker():

    global _RERANKER_INSTANCE

    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE

    try:

        from FlagEmbedding import (
            FlagReranker,
        )

        _RERANKER_INSTANCE = (
            FlagReranker(
                _RERANKER_MODEL,
                use_fp16=False,
            )
        )

        logger.info(
            "[RRF] Loaded reranker"
        )

        return _RERANKER_INSTANCE

    except Exception as exc:

        logger.warning(
            "[RRF] Failed loading reranker: %s",
            exc,
        )

        return None

# ─────────────────────────────────────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = _RRF_K,
) -> List[Tuple[str, float]]:

    scores: Dict[str, float] = {}

    for ranked in ranked_lists:

        for rank, doc_id in enumerate(
            ranked,
            start=1,
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:

    return (
        text or ""
    ).strip().lower()


def deduplicate_results(
    results: List[Dict],
) -> List[Dict]:

    seen = set()

    unique = []

    for item in results:

        text = normalize_text(
            item.get("text", "")
        )

        if not text:
            continue

        key = hash(text)

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

    return unique


def financial_boost(
    chunk: Dict,
) -> float:

    text = normalize_text(
        chunk.get("text", "")
    )

    boost = 0.0

    important_terms = [
        "revenue",
        "net income",
        "operating income",
        "cash flow",
        "eps",
        "gross profit",
    ]

    for term in important_terms:

        if term in text:
            boost += 0.03

    return boost

# ─────────────────────────────────────────────────────────────────────────────
# RRFReranker
# ─────────────────────────────────────────────────────────────────────────────


class RRFReranker:

    def __init__(
        self,
        final_top_k: int = _DEFAULT_TOP_K,
        rrf_k: int = _RRF_K,
        use_reranker: bool = True,
    ):

        self.final_top_k = final_top_k

        self.rrf_k = rrf_k

        self.use_reranker = use_reranker

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state):

        query = getattr(
            state,
            "query",
            "",
        ) or ""

        bm25_results = getattr(
            state,
            "bm25_results",
            [],
        ) or []

        bge_results = getattr(
            state,
            "bge_results",
            [],
        ) or getattr(
            state,
            "retrieval_stage_1",
            [],
        ) or []

        if not query:

            logger.warning(
                "[RRF] Empty query"
            )

            state.retrieval_stage_2 = []

            return state

        final_results = self.rerank(
            query=query,
            bm25_results=bm25_results,
            bge_results=bge_results,
        )

        state.retrieval_stage_2 = (
            final_results
        )

        logger.info(
            "[RRF] bm25=%d bge=%d final=%d",
            len(bm25_results),
            len(bge_results),
            len(final_results),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Core
    # ─────────────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        bm25_results: List[Dict],
        bge_results: List[Dict],
    ) -> List[Dict]:

        bm25_results = deduplicate_results(
            bm25_results
        )

        bge_results = deduplicate_results(
            bge_results
        )

        if (
            not bm25_results
            and not bge_results
        ):

            logger.warning(
                "[RRF] Empty retrieval"
            )

            return []

        chunk_lookup = {}

        for item in (
            bm25_results
            + bge_results
        ):

            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or str(hash(
                    item.get(
                        "text",
                        "",
                    )
                ))
            )

            if chunk_id not in chunk_lookup:

                item["chunk_id"] = chunk_id

                chunk_lookup[
                    chunk_id
                ] = item

        bm25_ids = [
            r["chunk_id"]
            for r in bm25_results
            if r.get("chunk_id")
        ]

        bge_ids = [
            r["chunk_id"]
            for r in bge_results
            if r.get("chunk_id")
        ]

        ranked_lists = []

        if bm25_ids:
            ranked_lists.append(
                bm25_ids
            )

        if bge_ids:
            ranked_lists.append(
                bge_ids
            )

        rrf_results = (
            reciprocal_rank_fusion(
                ranked_lists,
                self.rrf_k,
            )
        )

        if not rrf_results:
            return []

        candidates = []

        pool_size = max(
            self.final_top_k
            * _RERANK_POOL_MULTIPLIER,
            10,
        )

        for (
            chunk_id,
            rrf_score,
        ) in rrf_results[:pool_size]:

            chunk = chunk_lookup.get(
                chunk_id
            )

            if not chunk:
                continue

            merged = dict(chunk)

            merged["rrf_score"] = round(
                float(rrf_score),
                6,
            )

            merged["score"] = (
                merged["rrf_score"]
                + financial_boost(
                    merged
                )
            )

            merged["retriever"] = (
                _RETRIEVER_LABEL
            )

            candidates.append(
                merged
            )

        if not candidates:
            return []

        candidates.sort(
            key=lambda x: x.get(
                "score",
                0.0,
            ),
            reverse=True,
        )

        # Early Exit Optimization

        top_score = candidates[0].get(
            "score",
            0.0,
        )

        if (
            top_score
            >= _HIGH_CONFIDENCE_THRESHOLD
        ):

            logger.info(
                "[RRF] Early skip reranker"
            )

            final = candidates[
                : self.final_top_k
            ]

            for idx, item in enumerate(
                final,
                start=1,
            ):

                item["rank"] = idx

            return final

        # Cross Encoder

        if (
            self.use_reranker
            and len(candidates) > 1
        ):

            candidates = (
                self.apply_cross_encoder(
                    query,
                    candidates,
                )
            )

        final = candidates[
            : self.final_top_k
        ]

        for idx, item in enumerate(
            final,
            start=1,
        ):

            item["rank"] = idx

            item["retriever"] = (
                _RETRIEVER_LABEL
            )

        return final

    # ─────────────────────────────────────────────────────────────────────────
    # Cross Encoder
    # ─────────────────────────────────────────────────────────────────────────

    def apply_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
    ) -> List[Dict]:

        reranker = get_reranker()

        if reranker is None:

            logger.warning(
                "[RRF] Cross-encoder unavailable"
            )

            return candidates

        try:

            pairs = []

            for c in candidates:

                text = (
                    c.get("text")
                    or c.get(
                        "page_content",
                        "",
                    )
                )

                pairs.append(
                    [query, text]
                )

            scores = (
                reranker.compute_score(
                    pairs,
                    normalize=True,
                )
            )

            if isinstance(
                scores,
                float,
            ):
                scores = [scores]

            for (
                candidate,
                score,
            ) in zip(
                candidates,
                scores,
            ):

                candidate[
                    "reranker_score"
                ] = round(
                    float(score),
                    6,
                )

                candidate["score"] = (
                    candidate[
                        "reranker_score"
                    ]
                    + financial_boost(
                        candidate
                    )
                )

            candidates.sort(
                key=lambda x: x.get(
                    "score",
                    0.0,
                ),
                reverse=True,
            )

            logger.info(
                "[RRF] Cross-encoder reranked=%d",
                len(candidates),
            )

            return candidates

        except Exception as exc:

            logger.warning(
                "[RRF] Cross-encoder failed: %s",
                exc,
            )

            return candidates

# ─────────────────────────────────────────────────────────────────────────────
# LangChain Compatibility
# ─────────────────────────────────────────────────────────────────────────────


class RRFEnsembleRetriever:

    def __init__(
        self,
        retrievers: List[Any],
        rrf_reranker: RRFReranker,
    ):

        self.retrievers = retrievers

        self.rrf_reranker = (
            rrf_reranker
        )

    def invoke(
        self,
        query: str,
    ):

        try:

            from langchain_core.documents import (
                Document,
            )

        except Exception:

            logger.warning(
                "[RRF] LangChain unavailable"
            )

            return []

        all_results = []

        for retriever in self.retrievers:

            try:

                docs = retriever.invoke(
                    query
                )

                converted = []

                for rank, doc in enumerate(
                    docs,
                    start=1,
                ):

                    meta = getattr(
                        doc,
                        "metadata",
                        {},
                    )

                    converted.append(
                        {
                            "chunk_id": meta.get(
                                "chunk_id",
                                f"doc_{rank}",
                            ),
                            "text": doc.page_content,
                            "rank": rank,
                            "metadata": meta,
                        }
                    )

                all_results.append(
                    converted
                )

            except Exception as exc:

                logger.warning(
                    "[RRF] Retriever failed: %s",
                    exc,
                )

                all_results.append([])

        bm25 = (
            all_results[0]
            if len(all_results) > 0
            else []
        )

        bge = (
            all_results[1]
            if len(all_results) > 1
            else []
        )

        final = (
            self.rrf_reranker.rerank(
                query=query,
                bm25_results=bm25,
                bge_results=bge,
            )
        )

        return [
            Document(
                page_content=r.get(
                    "text",
                    "",
                ),
                metadata={
                    k: v
                    for k, v in r.items()
                    if k != "text"
                },
            )
            for r in final
        ]

    def get_relevant_documents(
        self,
        query: str,
    ):

        return self.invoke(query)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_rrf_reranker(
    state,
    use_reranker: bool = True,
):

    node = RRFReranker(
        final_top_k=3,
        rrf_k=_RRF_K,
        use_reranker=use_reranker,
    )

    return node.run(state)