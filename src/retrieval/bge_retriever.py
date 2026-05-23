"""
src/retrieval/bge_retriever.py

Production-Grade BGE-M3 Semantic Retriever
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. BGE-M3 semantic retrieval
2. ChromaDB persistent vector search
3. Batch embeddings
4. Singleton model loading
5. GPU auto-detection
6. FP16 optimization
7. CPU fallback
8. Collection existence validation
9. Memory-safe embedding
10. LangChain compatibility
11. Financial metadata preservation
12. Graceful degradation
13. Retrieval deduplication
14. Dynamic top-k handling
15. Environment-based disable support
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Globals
# ─────────────────────────────────────────────────────────────────────────────

_ST_MODEL = None
_CHROMADB = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_TOP_K = 10

MIN_SIMILARITY = 0.0

RETRIEVER_LABEL = "bge_m3"

_COLLECTION_PREFIX = "finbench_"

_BATCH_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Imports
# ─────────────────────────────────────────────────────────────────────────────


def get_sentence_transformers():

    global _ST_MODEL

    if _ST_MODEL is None:

        import sentence_transformers

        _ST_MODEL = (
            sentence_transformers
        )

    return _ST_MODEL


def get_chromadb():

    global _CHROMADB

    if _CHROMADB is None:

        import chromadb

        _CHROMADB = chromadb

    return _CHROMADB

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

# ─────────────────────────────────────────────────────────────────────────────
# BGERetriever
# ─────────────────────────────────────────────────────────────────────────────


class BGERetriever:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        data_dir: str = "data",
    ):

        self.model_name = model_name

        self.top_k = top_k

        self.data_dir = data_dir

        self._model = None

        self._client = None

        self._disabled = bool(
            os.environ.get(
                "DISABLE_BGE"
            )
        ) or bool(
            os.environ.get(
                "DISABLE_CHROMADB"
            )
        )

        if self._disabled:

            logger.warning(
                "[BGE] Disabled via environment"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state):

        query = getattr(
            state,
            "query",
            "",
        ) or ""

        collection = getattr(
            state,
            "chromadb_collection",
            "",
        ) or ""

        if not query:

            logger.warning(
                "[BGE] Empty query"
            )

            state.retrieval_stage_1 = []

            return state

        if not collection:

            logger.warning(
                "[BGE] Missing collection"
            )

            state.retrieval_stage_1 = []

            return state

        results = self.retrieve(
            query=query,
            collection_name=collection,
            data_dir=self.data_dir,
            top_k=self.top_k,
        )

        state.retrieval_stage_1 = (
            results
        )

        state.bge_results = (
            results
        )

        logger.info(
            "[BGE] query='%s' results=%d",
            query[:50],
            len(results),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        collection_name: str,
        data_dir: str = "data",
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        if not query:
            return []

        if not collection_name:
            return []

        if self._disabled:

            logger.warning(
                "[BGE] Disabled"
            )

            return []

        try:

            collection = (
                self._load_collection(
                    collection_name,
                    data_dir,
                )
            )

            if collection is None:

                logger.warning(
                    "[BGE] Collection missing: %s",
                    collection_name,
                )

                return []

            model = self._load_model()

            query = normalize_query(
                query
            )

            query_instruction = (
                "Represent this sentence "
                "for searching relevant passages: "
                f"{query}"
            )

            embedding = model.encode(
                query_instruction,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            results = collection.query(
                query_embeddings=[
                    embedding
                ],
                n_results=min(
                    top_k,
                    collection.count(),
                ),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            formatted = (
                self._format_results(
                    results,
                    top_k,
                )
            )

            formatted = (
                deduplicate_results(
                    formatted
                )
            )

            return formatted

        except Exception as exc:

            logger.error(
                "[BGE] Retrieve failed: %s",
                exc,
                exc_info=True,
            )

            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Build Collection
    # ─────────────────────────────────────────────────────────────────────────

    def build_collection(
        self,
        chunks: List[Dict],
        collection_name: str,
        data_dir: str = "data",
    ) -> bool:

        if not chunks:

            logger.warning(
                "[BGE] No chunks"
            )

            return False

        if self._disabled:

            logger.warning(
                "[BGE] Disabled"
            )

            return False

        try:

            model = self._load_model()

            client = self._get_client(
                data_dir
            )

            try:

                client.delete_collection(
                    name=collection_name
                )

            except Exception:
                pass

            collection = (
                client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine"
                    },
                )
            )

            texts = [
                c.get("text", "")
                for c in chunks
            ]

            ids = [
                c.get(
                    "chunk_id",
                    f"chunk_{i}",
                )
                for i, c in enumerate(
                    chunks
                )
            ]

            metadata = [
                self._build_metadata(c)
                for c in chunks
            ]

            for i in range(
                0,
                len(texts),
                _BATCH_SIZE,
            ):

                batch_texts = texts[
                    i:i + _BATCH_SIZE
                ]

                batch_ids = ids[
                    i:i + _BATCH_SIZE
                ]

                batch_meta = metadata[
                    i:i + _BATCH_SIZE
                ]

                instructed = [
                    f"passage: {t}"
                    for t in batch_texts
                ]

                embeddings = (
                    model.encode(
                        instructed,
                        batch_size=_BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist()
                )

                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                )

                logger.info(
                    "[BGE] Embedded %d/%d",
                    min(
                        i + _BATCH_SIZE,
                        len(texts),
                    ),
                    len(texts),
                )

            logger.info(
                "[BGE] Collection built: %s",
                collection_name,
            )

            return True

        except Exception as exc:

            logger.error(
                "[BGE] Build failed: %s",
                exc,
                exc_info=True,
            )

            return False

    # ─────────────────────────────────────────────────────────────────────────
    # LangChain
    # ─────────────────────────────────────────────────────────────────────────

    def as_langchain_retriever(
        self,
        collection_name: str,
        data_dir: str = "data",
        top_k: int = DEFAULT_TOP_K,
    ):

        return BGELangChainRetriever(
            bge_retriever=self,
            collection_name=collection_name,
            data_dir=data_dir,
            top_k=top_k,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Collection Checks
    # ─────────────────────────────────────────────────────────────────────────

    def collection_exists(
        self,
        collection_name: str,
        data_dir: str = "data",
    ) -> bool:

        try:

            client = self._get_client(
                data_dir
            )

            names = [
                c.name
                for c in client.list_collections()
            ]

            return (
                collection_name
                in names
            )

        except Exception:

            return False

    def get_collection_count(
        self,
        collection_name: str,
        data_dir: str = "data",
    ) -> int:

        try:

            collection = (
                self._load_collection(
                    collection_name,
                    data_dir,
                )
            )

            return (
                collection.count()
                if collection
                else 0
            )

        except Exception:

            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self):

        if self._model is not None:
            return self._model

        st = (
            get_sentence_transformers()
        )

        forced = os.environ.get(
            "BGE_DEVICE",
            "",
        ).strip().lower()

        allow_fp16 = (
            os.environ.get(
                "BGE_FP16",
                "1",
            ) != "0"
        )

        device = "cpu"

        if forced in (
            "cpu",
            "cuda",
            "mps",
        ):

            device = forced

        else:

            try:

                import torch

                if (
                    torch.cuda.is_available()
                ):
                    device = "cuda"

            except Exception:
                pass

        use_fp16 = False

        if device == "cuda":

            try:

                import torch

                free_bytes, total_bytes = (
                    torch.cuda.mem_get_info()
                )

                free_gb = (
                    free_bytes / 1e9
                )

                if free_gb < 1.0:

                    logger.warning(
                        "[BGE] GPU low memory → CPU"
                    )

                    device = "cpu"

                elif (
                    free_gb < 2.5
                    and allow_fp16
                ):

                    use_fp16 = True

            except Exception:
                pass

        logger.info(
            "[BGE] Loading model "
            "device=%s fp16=%s",
            device,
            use_fp16,
        )

        try:

            kwargs = {
                "device": device
            }

            if (
                use_fp16
                and device == "cuda"
            ):

                import torch

                kwargs[
                    "model_kwargs"
                ] = {
                    "torch_dtype": torch.float16
                }

            self._model = (
                st.SentenceTransformer(
                    self.model_name,
                    **kwargs,
                )
            )

        except Exception as exc:

            logger.warning(
                "[BGE] GPU load failed: %s",
                exc,
            )

            self._model = (
                st.SentenceTransformer(
                    self.model_name,
                    device="cpu",
                )
            )

        logger.info(
            "[BGE] Model loaded"
        )

        return self._model

    def _get_client(
        self,
        data_dir: str,
    ):

        chromadb_path = os.path.join(
            data_dir,
            "chromadb",
        )

        os.makedirs(
            chromadb_path,
            exist_ok=True,
        )

        chromadb = get_chromadb()

        return chromadb.PersistentClient(
            path=chromadb_path
        )

    def _load_collection(
        self,
        collection_name: str,
        data_dir: str,
    ):

        try:

            client = self._get_client(
                data_dir
            )

            return client.get_collection(
                name=collection_name
            )

        except Exception:

            return None

    @staticmethod
    def _build_metadata(
        chunk: Dict,
    ) -> Dict:

        return {
            "chunk_id": str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            ),
            "company": str(
                chunk.get(
                    "company",
                    "UNKNOWN",
                )
            ),
            "doc_type": str(
                chunk.get(
                    "doc_type",
                    "UNKNOWN",
                )
            ),
            "fiscal_year": str(
                chunk.get(
                    "fiscal_year",
                    "UNKNOWN",
                )
            ),
            "section": str(
                chunk.get(
                    "section",
                    "UNKNOWN",
                )
            ),
            "page": int(
                chunk.get(
                    "page",
                    0,
                )
            ),
            "prefix": str(
                chunk.get(
                    "prefix",
                    "",
                )
            ),
        }

    @staticmethod
    def _format_results(
        chroma_results: Dict,
        top_k: int,
    ) -> List[Dict]:

        formatted = []

        documents = (
            chroma_results.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            chroma_results.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            chroma_results.get(
                "distances",
                [[]],
            )[0]
        )

        for rank, (
            doc,
            meta,
            dist,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            similarity = max(
                0.0,
                1.0 - float(dist),
            )

            if (
                similarity
                < MIN_SIMILARITY
            ):
                continue

            formatted.append(
                {
                    "text": doc,
                    "chunk_id": meta.get(
                        "chunk_id",
                        "",
                    ),
                    "rank": rank,
                    "bge_score": round(
                        similarity,
                        6,
                    ),
                    "retriever": RETRIEVER_LABEL,
                    "company": meta.get(
                        "company",
                        "UNKNOWN",
                    ),
                    "doc_type": meta.get(
                        "doc_type",
                        "UNKNOWN",
                    ),
                    "fiscal_year": meta.get(
                        "fiscal_year",
                        "UNKNOWN",
                    ),
                    "section": meta.get(
                        "section",
                        "UNKNOWN",
                    ),
                    "page": meta.get(
                        "page",
                        0,
                    ),
                    "prefix": meta.get(
                        "prefix",
                        "",
                    ),
                }
            )

            if (
                len(formatted)
                >= top_k
            ):
                break

        return formatted

# ─────────────────────────────────────────────────────────────────────────────
# LangChain Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class BGELangChainRetriever:

    def __init__(
        self,
        bge_retriever: BGERetriever,
        collection_name: str,
        data_dir: str = "data",
        top_k: int = DEFAULT_TOP_K,
    ):

        self._retriever = (
            bge_retriever
        )

        self._collection = (
            collection_name
        )

        self._data_dir = data_dir

        self._top_k = top_k

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
                "[BGE] LangChain unavailable"
            )

            return []

        results = (
            self._retriever.retrieve(
                query=query,
                collection_name=self._collection,
                data_dir=self._data_dir,
                top_k=self._top_k,
            )
        )

        return [
            Document(
                page_content=r["text"],
                metadata={
                    "chunk_id": r[
                        "chunk_id"
                    ],
                    "bge_score": r[
                        "bge_score"
                    ],
                    "rank": r["rank"],
                    "retriever": r[
                        "retriever"
                    ],
                    "company": r[
                        "company"
                    ],
                    "doc_type": r[
                        "doc_type"
                    ],
                    "fiscal_year": r[
                        "fiscal_year"
                    ],
                    "section": r[
                        "section"
                    ],
                    "page": r["page"],
                },
            )
            for r in results
        ]

    def get_relevant_documents(
        self,
        query: str,
    ):

        return self.invoke(query)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_bge(
    state,
    data_dir: str = "data",
):

    state_data_dir = getattr(
        state,
        "chromadb_data_dir",
        "",
    ) or ""

    effective_dir = (
        state_data_dir
        if state_data_dir
        else data_dir
    )

    retriever = BGERetriever(
        model_name=DEFAULT_MODEL,
        top_k=DEFAULT_TOP_K,
        data_dir=effective_dir,
    )

    return retriever.run(state)