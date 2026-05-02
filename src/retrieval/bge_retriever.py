"""
N08 BGE-M3 Semantic Retriever — Tier 3 Dense Embedding Search
PDR-BAAAI-001 · Rev 1.0 · Node N08

CHANGELOG:
  2026-04-30 S13  Bug #3: run_bge() reads state.chromadb_data_dir
  2026-04-30 S15  Bug #8: collection check before model load + DISABLE_BGE
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports
_sentence_transformers = None
_chromadb = None


def _get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        import sentence_transformers
        _sentence_transformers = sentence_transformers
    return _sentence_transformers


def _get_chromadb():
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL      = "BAAI/bge-m3"
DEFAULT_TOP_K      = 10
MIN_SIMILARITY     = 0.0
RETRIEVER_LABEL    = "bge_m3"

_COLLECTION_PREFIX = "finbench_"


# ── BGERetriever ──────────────────────────────────────────────────────────────

class BGERetriever:
    """N08 BGE-M3 Semantic Retriever."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k:      int = DEFAULT_TOP_K,
        data_dir:   str = "data",
    ) -> None:
        self.model_name = model_name
        self.top_k      = top_k
        self.data_dir   = data_dir
        self._model     = None
        self._client    = None
        # Bug #8: respect DISABLE_BGE / DISABLE_CHROMADB env vars
        self._disabled  = bool(os.environ.get("DISABLE_BGE")) or \
                          bool(os.environ.get("DISABLE_CHROMADB"))
        if self._disabled:
            logger.info(
                "N08: BGE-M3 disabled (DISABLE_BGE or DISABLE_CHROMADB set)"
            )

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """LangGraph N08 node entry point."""
        query      = getattr(state, "query",               "") or ""
        collection = getattr(state, "chromadb_collection", "") or ""

        if not query:
            logger.warning("N08 BGE-M3: empty query — returning empty results")
            state.retrieval_stage_1 = []
            return state

        if not collection:
            logger.warning("N08 BGE-M3: no chromadb_collection in state")
            state.retrieval_stage_1 = []
            return state

        results = self.retrieve(
            query           = query,
            collection_name = collection,
            data_dir        = self.data_dir,
            top_k           = self.top_k,
        )

        state.retrieval_stage_1 = results
        logger.info(
            "N08 BGE-M3: query='%s...' | collection=%s | results=%d",
            query[:50], collection, len(results),
        )
        return state

    # ── Core retrieval method ─────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        collection_name: str,
        data_dir:        str  = "data",
        top_k:           int  = DEFAULT_TOP_K,
    ) -> List[Dict]:
        """Embed query and search ChromaDB collection.

        Bug #8: check collection EXISTS before loading the BGE model.
        """
        if not query or not collection_name:
            return []

        if self._disabled:
            logger.warning("N08: BGE-M3 disabled — returning empty")
            return []

        try:
            # Check collection exists FIRST (cheap) before loading model (expensive)
            collection = self._load_collection(collection_name, data_dir)
            if collection is None:
                logger.warning(
                    "N08: collection '%s' not found — skipping model load",
                    collection_name,
                )
                return []

            # Only NOW do we load the 1.5GB model
            model = self._load_model()

            # Embed the query — BGE-M3 instruction prefix improves retrieval
            query_with_instruction = (
                f"Represent this sentence for searching relevant passages: {query}"
            )
            query_embedding = model.encode(
                query_with_instruction,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            # Query ChromaDB
            chroma_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(chroma_results, top_k)

        except Exception as exc:
            logger.error("N08 BGE-M3 retrieve error: %s", exc, exc_info=True)
            return []

    # ── Index building ────────────────────────────────────────────────────────

    def build_collection(
        self,
        chunks:          List[Dict],
        collection_name: str,
        data_dir:        str = "data",
    ) -> bool:
        """Embed all chunks and store in ChromaDB collection.

        Bug #8: respect DISABLE_BGE env var.
        """
        if not chunks:
            logger.warning("N08: No chunks provided — skipping collection build")
            return False

        if self._disabled:
            logger.warning("N08: BGE-M3 disabled — skipping collection build")
            return False

        try:
            model      = self._load_model()
            client     = self._get_client(data_dir)

            # Delete existing collection if rebuilding
            try:
                client.delete_collection(name=collection_name)
                logger.info("N08: Deleted existing collection '%s'", collection_name)
            except Exception:
                pass

            collection = client.create_collection(
                name     = collection_name,
                metadata = {"hnsw:space": "cosine"},
            )

            # Batch embed for efficiency
            batch_size = 32
            texts      = [c.get("text", "") for c in chunks]
            ids        = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
            metadatas  = [self._build_metadata(c) for c in chunks]

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids   = ids[i:i + batch_size]
                batch_meta  = metadatas[i:i + batch_size]

                instructed = [f"passage: {t}" for t in batch_texts]
                embeddings = model.encode(
                    instructed,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

                collection.add(
                    ids        = batch_ids,
                    documents  = batch_texts,
                    embeddings = embeddings,
                    metadatas  = batch_meta,
                )
                logger.info(
                    "N08: Embedded batch %d-%d / %d",
                    i, min(i + batch_size, len(texts)), len(texts),
                )

            logger.info(
                "N08: Collection '%s' built — %d chunks embedded",
                collection_name, len(chunks),
            )
            return True

        except Exception as exc:
            logger.error("N08 build_collection error: %s", exc, exc_info=True)
            return False

    # ── LangChain compatibility ───────────────────────────────────────────────

    def as_langchain_retriever(
        self,
        collection_name: str,
        data_dir:        str = "data",
        top_k:           int = DEFAULT_TOP_K,
    ):
        """Return a LangChain-compatible retriever wrapping this BGERetriever."""
        return BGELangChainRetriever(
            bge_retriever   = self,
            collection_name = collection_name,
            data_dir        = data_dir,
            top_k           = top_k,
        )

    def collection_exists(
        self,
        collection_name: str,
        data_dir:        str = "data",
    ) -> bool:
        """Check if a ChromaDB collection exists."""
        try:
            client = self._get_client(data_dir)
            names  = [c.name for c in client.list_collections()]
            return collection_name in names
        except Exception:
            return False

    def get_collection_count(
        self,
        collection_name: str,
        data_dir:        str = "data",
    ) -> int:
        """Return number of documents in a collection."""
        try:
            col = self._load_collection(collection_name, data_dir)
            return col.count() if col is not None else 0
        except Exception:
            return 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            st = _get_sentence_transformers()
            logger.info("N08: Loading model '%s' ...", self.model_name)
            self._model = st.SentenceTransformer(
                self.model_name,
                device="cpu",
            )
            logger.info("N08: Model loaded — %s", self.model_name)
        return self._model

    def _get_client(self, data_dir: str):
        """Get or create ChromaDB persistent client."""
        chromadb_path = os.path.join(data_dir, "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)
        chroma = _get_chromadb()
        return chroma.PersistentClient(path=chromadb_path)

    def _load_collection(
        self,
        collection_name: str,
        data_dir:        str,
    ):
        """Load existing ChromaDB collection. Returns None if not found."""
        try:
            client = self._get_client(data_dir)
            return client.get_collection(name=collection_name)
        except Exception:
            return None

    @staticmethod
    def _build_metadata(chunk: Dict) -> Dict:
        """Build ChromaDB metadata dict from a chunk."""
        return {
            "chunk_id":    str(chunk.get("chunk_id",    "")),
            "company":     str(chunk.get("company",     "UNKNOWN")),
            "doc_type":    str(chunk.get("doc_type",    "UNKNOWN")),
            "fiscal_year": str(chunk.get("fiscal_year", "UNKNOWN")),
            "section":     str(chunk.get("section",     "UNKNOWN")),
            "page":        int(chunk.get("page",        0)),
            "prefix":      str(chunk.get("prefix",      "")),
        }

    @staticmethod
    def _format_results(chroma_results: Dict, top_k: int) -> List[Dict]:
        """Convert raw ChromaDB query results to pipeline-standard format."""
        formatted = []

        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            similarity = max(0.0, 1.0 - float(dist))

            result = {
                "text":        doc,
                "chunk_id":    meta.get("chunk_id",    ""),
                "rank":        rank,
                "bge_score":   round(similarity, 6),
                "retriever":   RETRIEVER_LABEL,
                "company":     meta.get("company",     "UNKNOWN"),
                "doc_type":    meta.get("doc_type",    "UNKNOWN"),
                "fiscal_year": meta.get("fiscal_year", "UNKNOWN"),
                "section":     meta.get("section",     "UNKNOWN"),
                "page":        meta.get("page",         0),
                "prefix":      meta.get("prefix",      ""),
            }
            formatted.append(result)

            if rank >= top_k:
                break

        return formatted


# ── LangChain compatibility wrapper ──────────────────────────────────────────

class BGELangChainRetriever:
    """Thin wrapper making BGERetriever compatible with LangChain's
    retriever interface (used by N09 EnsembleRetriever)."""

    def __init__(
        self,
        bge_retriever:   BGERetriever,
        collection_name: str,
        data_dir:        str = "data",
        top_k:           int = DEFAULT_TOP_K,
    ) -> None:
        self._retriever      = bge_retriever
        self._collection     = collection_name
        self._data_dir       = data_dir
        self._top_k          = top_k

    def invoke(self, query: str) -> List[Any]:
        """LangChain-compatible invoke method."""
        from langchain_core.documents import Document

        results = self._retriever.retrieve(
            query           = query,
            collection_name = self._collection,
            data_dir        = self._data_dir,
            top_k           = self._top_k,
        )

        return [
            Document(
                page_content = r["text"],
                metadata     = {
                    "chunk_id":    r["chunk_id"],
                    "bge_score":   r["bge_score"],
                    "rank":        r["rank"],
                    "retriever":   r["retriever"],
                    "company":     r["company"],
                    "doc_type":    r["doc_type"],
                    "fiscal_year": r["fiscal_year"],
                    "section":     r["section"],
                    "page":        r["page"],
                },
            )
            for r in results
        ]

    def get_relevant_documents(self, query: str) -> List[Any]:
        """Alias for older LangChain versions."""
        return self.invoke(query)


# ── Convenience wrapper for LangGraph N08 node ───────────────────────────────

def run_bge(state, data_dir: str = "data") -> object:
    """Convenience wrapper used by the LangGraph pipeline node N08.

    Bug #3 fix: if state has chromadb_data_dir, use that.
    """
    state_data_dir = getattr(state, "chromadb_data_dir", "") or ""
    effective_dir = state_data_dir if state_data_dir else data_dir

    retriever = BGERetriever(
        model_name = DEFAULT_MODEL,
        top_k      = DEFAULT_TOP_K,
        data_dir   = effective_dir,
    )
    return retriever.run(state)