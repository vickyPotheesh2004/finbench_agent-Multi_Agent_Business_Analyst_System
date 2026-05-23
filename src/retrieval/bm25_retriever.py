"""
src/retrieval/bm25_retriever.py

Production-Grade BM25 Retriever
Optimized for:

- Windows
- Colab
- FinanceBench
- Large SEC filings
- Tiny corpus safety
- Stable ranking
- Fast repeated retrieval
- bm25s compatibility
"""

from __future__ import annotations

import json
import logging
import os
import re

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

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
    "guidance",
    "margin",
    "earnings",
    "liabilities",
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared Cache
# ─────────────────────────────────────────────────────────────────────────────

_INDEX_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def normalize_query(
    query: str,
) -> str:

    query = (
        query or ""
    ).strip().lower()

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query


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

    text = text.lower()

    boost = 0.0

    for term in IMPORTANT_TERMS:

        if term in text:

            boost += 0.03

    return boost

# ─────────────────────────────────────────────────────────────────────────────
# BM25 Retriever
# ─────────────────────────────────────────────────────────────────────────────


class BM25Retriever:

    def __init__(
        self,
        top_k: int = TOP_K,
    ):

        self.top_k = max(
            1,
            int(top_k),
        )

        self._retriever = None

        self._chunks: List[
            Dict[str, Any]
        ] = []

        self._index_path = ""

    # ─────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        state,
    ):

        index_path = getattr(
            state,
            "bm25_index_path",
            "",
        ) or ""

        query = getattr(
            state,
            "query",
            "",
        ) or ""

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
                "[BM25] Empty chunk store"
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
                results[0].get(
                    "bm25_score_norm",
                    0.0,
                )
            )
            if results
            else 0.0
        )

        logger.info(
            "[BM25] query='%s' results=%d",
            query[:80],
            len(results),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────
    # Load Index
    # ─────────────────────────────────────────────────────────────────────

    def _load_index(
        self,
        index_path: str,
    ) -> None:

        if (
            self._index_path
            == index_path
            and self._retriever
            is not None
            and self._chunks
        ):

            return

        if index_path in _INDEX_CACHE:

            cached = _INDEX_CACHE[
                index_path
            ]

            self._retriever = cached[
                "retriever"
            ]

            self._chunks = cached[
                "chunks"
            ]

            self._index_path = (
                index_path
            )

            return

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

            self._retriever = None

            self._chunks = []

            return

        if not meta_path.exists():

            logger.warning(
                "[BM25] Metadata missing"
            )

            self._retriever = None

            self._chunks = []

            return

        # Metadata

        try:

            with open(
                meta_path,
                "r",
                encoding="utf-8",
            ) as f:

                self._chunks = (
                    json.load(f)
                )

        except Exception:

            logger.exception(
                "[BM25] Failed loading metadata"
            )

            self._retriever = None

            self._chunks = []

            return

        # BM25

        try:

            self._retriever = (
                bm25s.BM25.load(
                    str(index_dir),
                    load_corpus=True,
                )
            )

        except Exception:

            logger.exception(
                "[BM25] Failed loading BM25"
            )

            self._retriever = None

            return

        self._index_path = (
            index_path
        )

        _INDEX_CACHE[
            index_path
        ] = {
            "retriever": self._retriever,
            "chunks": self._chunks,
        }

        logger.info(
            "[BM25] Loaded index | chunks=%d",
            len(self._chunks),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────

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

        except Exception:

            logger.exception(
                "[BM25] Tokenization failed"
            )

            return self._fallback_score_all(
                query
            )

        safe_k = min(
            max(2, self.top_k),
            n_chunks,
        )

        # Main Retrieval

        try:

            results, scores = (
                self._retriever.retrieve(
                    query_tokens,
                    k=safe_k,
                )
            )

        except Exception:

            logger.exception(
                "[BM25] retrieve failed"
            )

            return self._fallback_score_all(
                query
            )

        # Normalize weird bm25s formats

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

            text = chunk.get(
                "text",
                "",
            )

            score_f = float(score)

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

        for idx, item in enumerate(
            output,
            start=1,
        ):

            item["rank"] = idx

        return output[
            : self.top_k
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Fallback Retrieval
    # ─────────────────────────────────────────────────────────────────────

    def _fallback_score_all(
        self,
        query: str,
    ) -> List[Dict]:

        if not self._chunks:
            return []

        query = normalize_query(
            query
        )

        q_terms = set(
            query.split()
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

            text_terms = set(
                text.split()
            )

            overlap = len(
                q_terms
                & text_terms
            )

            fuzzy = (
                fuzz.partial_ratio(
                    query,
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

        max_score = max(
            float(s[0])
            for s in scored
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
                        float(score)
                        / max_score,
                        6,
                    ),
                    "rank": rank,
                    "retriever": "bm25_fallback",
                }
            )

        return output

    # ─────────────────────────────────────────────────────────────────────
    # Chunk Resolution
    # ─────────────────────────────────────────────────────────────────────

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

            if chunk_id:

                chunk = (
                    self._find_chunk(
                        chunk_id
                    )
                )

                if chunk:
                    return chunk

            if (
                0 <= rank < n_chunks
            ):

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

        chunk = self._find_chunk(
            str(item)
        )

        if chunk:
            return chunk

        if (
            0 <= rank < n_chunks
        ):

            return self._chunks[
                rank
            ]

        return None

    def _find_chunk(
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

    # ─────────────────────────────────────────────────────────────────────
    # LangChain
    # ─────────────────────────────────────────────────────────────────────

    def as_langchain_retriever(
        self,
        index_path: str,
    ):

        try:

            from langchain_core.documents import (
                Document,
            )

            from langchain_community.retrievers import (
                BM25Retriever as LCBM25,
            )

        except Exception:

            logger.exception(
                "[BM25] LangChain unavailable"
            )

            return None

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

        return (
            LCBM25.from_documents(
                docs,
                k=self.top_k,
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

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
            and bool(self._chunks)
        )

    def search_direct(
        self,
        query: str,
        index_path: str,
        top_k: int = 10,
    ) -> List[Dict]:

        self.top_k = max(
            1,
            int(top_k),
        )

        self._load_index(
            index_path
        )

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