"""
src/retrieval/bm25_retriever.py

Production-Grade BM25 Retriever
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. BM25 keyword retrieval
2. bm25s compatibility fixes
3. Tiny-corpus safe retrieval
4. Fallback lexical scorer
5. Financial term boosting
6. Query normalization
7. Deduplication
8. LangChain compatibility
9. Stable ranking
10. Resource-safe loading
11. Graceful degradation
12. Production logging
13. Retrieval confidence
14. Manual overlap fallback
15. Direct search mode
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOP_K = 10

RETRIEVER_LABEL = "bm25"

MIN_SCORE = 0.01

IMPORTANT_TERMS = [
    "revenue",
    "net income",
    "operating income",
    "gross profit",
    "cash flow",
    "eps",
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def normalize_query(
    query: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        query.strip().lower(),
    )


def deduplicate_results(
    results: List[Dict],
) -> List[Dict]:

    seen = set()

    unique = []

    for item in results:

        text = (
            item.get("text", "")
            .strip()
            .lower()
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
    text: str,
) -> float:

    t = text.lower()

    boost = 0.0

    for term in IMPORTANT_TERMS:

        if term in t:
            boost += 0.03

    return boost

# ─────────────────────────────────────────────────────────────────────────────
# BM25Retriever
# ─────────────────────────────────────────────────────────────────────────────


class BM25Retriever:

    def __init__(
        self,
        top_k: int = TOP_K,
    ):

        self.top_k = top_k

        self._retriever = None

        self._chunks: List[
            Dict[str, Any]
        ] = []

        self._lc_retriever = None

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state):

        index_path = getattr(
            state,
            "bm25_index_path",
            "",
        )

        query = getattr(
            state,
            "query",
            "",
        )

        if not index_path:

            logger.warning(
                "[BM25] Missing index path"
            )

            state.bm25_results = []

            state.bm25_confidence = 0.0

            return state

        if not query:

            logger.warning(
                "[BM25] Empty query"
            )

            state.bm25_results = []

            state.bm25_confidence = 0.0

            return state

        self._load_index(
            index_path
        )

        if not self._chunks:

            logger.warning(
                "[BM25] Empty chunks"
            )

            state.bm25_results = []

            state.bm25_confidence = 0.0

            return state

        results = self._search(
            query
        )

        results = (
            deduplicate_results(
                results
            )
        )

        state.bm25_results = (
            results
        )

        state.bm25_confidence = (
            float(
                results[0][
                    "bm25_score_norm"
                ]
            )
            if results
            else 0.0
        )

        logger.info(
            "[BM25] query='%s' results=%d",
            query[:50],
            len(results),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Load Index
    # ─────────────────────────────────────────────────────────────────────────

    def _load_index(
        self,
        index_path: str,
    ) -> None:

        import bm25s

        index_dir = Path(
            index_path
        )

        meta_path = (
            index_dir
            / "chunks_meta.json"
        )

        if not index_dir.exists():

            logger.warning(
                "[BM25] Index missing: %s",
                index_dir,
            )

            self._chunks = []

            self._retriever = None

            return

        if not meta_path.exists():

            logger.warning(
                "[BM25] chunks_meta.json missing"
            )

            self._chunks = []

            self._retriever = None

            return

        try:

            with open(
                meta_path,
                "r",
                encoding="utf-8",
            ) as f:

                self._chunks = json.load(
                    f
                )

        except Exception as exc:

            logger.error(
                "[BM25] Meta load failed: %s",
                exc,
            )

            self._chunks = []

            self._retriever = None

            return

        try:

            self._retriever = (
                bm25s.BM25.load(
                    str(index_dir),
                    load_corpus=True,
                )
            )

        except Exception as exc:

            logger.error(
                "[BM25] BM25 load failed: %s",
                exc,
            )

            self._retriever = None

            return

        logger.info(
            "[BM25] Loaded index chunks=%d",
            len(self._chunks),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────

    def _search(
        self,
        query: str,
    ) -> List[Dict]:

        import bm25s

        if (
            self._retriever is None
            or not self._chunks
        ):
            return []

        query = normalize_query(
            query
        )

        if not query:
            return []

        n_chunks = len(
            self._chunks
        )

        # Single chunk shortcut

        if n_chunks == 1:

            chunk = self._chunks[0]

            return [
                {
                    **chunk,
                    "bm25_score": 1.0,
                    "bm25_score_norm": 1.0,
                    "rank": 1,
                    "retriever": RETRIEVER_LABEL,
                }
            ]

        # Tokenize

        try:

            query_tokens = (
                bm25s.tokenize(
                    [query],
                    stopwords="en",
                )
            )

        except Exception as exc:

            logger.error(
                "[BM25] Tokenize failed: %s",
                exc,
            )

            return []

        safe_k = min(
            self.top_k,
            n_chunks,
        )

        if safe_k < 2:
            safe_k = 2

        # Main retrieve

        try:

            results, scores = (
                self._retriever.retrieve(
                    query_tokens,
                    k=safe_k,
                )
            )

        except Exception as exc:

            logger.debug(
                "[BM25] retrieve failed -> fallback: %s",
                exc,
            )

            return self._fallback_score_all(
                query
            )

        try:

            top_results = list(
                results[0]
            )

            top_scores = list(
                scores[0]
            )

        except Exception:

            top_results = list(
                results
            )

            top_scores = list(
                scores
            )

        if not top_results:
            return []

        max_score = max(
            (
                float(s)
                for s in top_scores
            ),
            default=1.0,
        )

        if max_score <= 0:
            max_score = 1.0

        output = []

        for rank, (
            item,
            score,
        ) in enumerate(
            zip(
                top_results,
                top_scores,
            ),
            start=1,
        ):

            chunk = self._resolve_chunk(
                item,
                rank - 1,
            )

            if chunk is None:
                continue

            score_f = float(score)

            text = chunk.get(
                "text",
                "",
            )

            fuzzy = (
                fuzz.partial_ratio(
                    query.lower(),
                    text.lower(),
                ) / 100.0
            )

            final_score = (
                score_f
                + fuzzy
                + financial_boost(
                    text
                )
            )

            if final_score < MIN_SCORE:
                continue

            output.append(
                {
                    **chunk,
                    "bm25_score": round(
                        final_score,
                        6,
                    ),
                    "bm25_score_norm": round(
                        final_score
                        / max_score,
                        6,
                    ),
                    "rank": rank,
                    "retriever": RETRIEVER_LABEL,
                }
            )

        output.sort(
            key=lambda x: x[
                "bm25_score"
            ],
            reverse=True,
        )

        for i, r in enumerate(
            output,
            start=1,
        ):
            r["rank"] = i

        return output[
            : self.top_k
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_score_all(
        self,
        query: str,
    ) -> List[Dict]:

        if not self._chunks:
            return []

        q_terms = set(
            query.lower().split()
        )

        if not q_terms:
            return []

        scored = []

        for idx, chunk in enumerate(
            self._chunks
        ):

            text = (
                chunk.get(
                    "text",
                    "",
                )
                .lower()
            )

            t_terms = set(
                text.split()
            )

            overlap = len(
                q_terms & t_terms
            )

            fuzzy = (
                fuzz.partial_ratio(
                    query.lower(),
                    text,
                ) / 100.0
            )

            score = (
                overlap
                + fuzzy
                + financial_boost(
                    text
                )
            )

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    idx,
                    chunk,
                )
            )

        if not scored:
            return []

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        max_score = float(
            scored[0][0]
        )

        if max_score <= 0:
            max_score = 1.0

        output = []

        for rank, (
            score,
            _idx,
            chunk,
        ) in enumerate(
            scored[
                : self.top_k
            ],
            start=1,
        ):

            output.append(
                {
                    **chunk,
                    "bm25_score": round(
                        float(score),
                        6,
                    ),
                    "bm25_score_norm": round(
                        score
                        / max_score,
                        6,
                    ),
                    "rank": rank,
                    "retriever": "bm25_fallback",
                }
            )

        return output

    # ─────────────────────────────────────────────────────────────────────────
    # Resolve
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_chunk(
        self,
        item: Any,
        rank: int,
    ) -> Optional[Dict]:

        n_chunks = len(
            self._chunks
        )

        if isinstance(
            item,
            dict,
        ):

            chunk_id = item.get(
                "id",
                "",
            )

            chunk = (
                self._find_chunk_meta(
                    chunk_id
                )
                if chunk_id
                else None
            )

            if chunk is not None:
                return chunk

            if rank < n_chunks:
                return self._chunks[
                    rank
                ]

            return None

        try:

            idx = int(item)

            if (
                0 <= idx < n_chunks
            ):
                return self._chunks[
                    idx
                ]

        except Exception:
            pass

        chunk_id = str(item)

        chunk = self._find_chunk_meta(
            chunk_id
        )

        if chunk is not None:
            return chunk

        if rank < n_chunks:
            return self._chunks[
                rank
            ]

        return None

    def _find_chunk_meta(
        self,
        chunk_id: str,
    ) -> Optional[Dict]:

        if not chunk_id:
            return None

        for chunk in self._chunks:

            if (
                chunk.get(
                    "chunk_id"
                )
                == chunk_id
            ):
                return chunk

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # LangChain
    # ─────────────────────────────────────────────────────────────────────────

    def as_langchain_retriever(
        self,
        index_path: str,
    ):

        from langchain_core.documents import (
            Document,
        )

        from langchain_community.retrievers import (
            BM25Retriever as LCBM25,
        )

        if not self._chunks:

            self._load_index(
                index_path
            )

        if not self._chunks:
            return None

        docs = []

        for chunk in self._chunks:

            docs.append(
                Document(
                    page_content=chunk.get(
                        "text",
                        "",
                    ),
                    metadata={
                        "chunk_id": chunk.get(
                            "chunk_id",
                            "",
                        ),
                        "company": chunk.get(
                            "company",
                            "",
                        ),
                        "doc_type": chunk.get(
                            "doc_type",
                            "",
                        ),
                        "fiscal_year": chunk.get(
                            "fiscal_year",
                            "",
                        ),
                        "section": chunk.get(
                            "section",
                            "",
                        ),
                        "page": str(
                            chunk.get(
                                "page",
                                "",
                            )
                        ),
                    },
                )
            )

        retriever = (
            LCBM25.from_documents(
                docs,
                k=self.top_k,
            )
        )

        self._lc_retriever = (
            retriever
        )

        return retriever

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_chunk_count(
        self,
    ) -> int:

        return len(
            self._chunks
        )

    def is_loaded(
        self,
    ) -> bool:

        return (
            self._retriever
            is not None
            and len(self._chunks) > 0
        )

    def search_direct(
        self,
        query: str,
        index_path: str,
        top_k: int = 10,
    ) -> List[Dict]:

        self._load_index(
            index_path
        )

        self.top_k = top_k

        return self._search(
            query
        )

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_bm25(
    state,
):

    retriever = BM25Retriever(
        top_k=TOP_K
    )

    return retriever.run(
        state
    )