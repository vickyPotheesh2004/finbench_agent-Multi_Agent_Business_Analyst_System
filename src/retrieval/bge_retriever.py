"""
N08 BGE-M3 Semantic Retriever — Tier 3 Dense Embedding Search
PDR-BAAAI-001 · Rev 1.0 · Node N08

Purpose:
    Dense semantic retrieval using BAAI/bge-m3 embeddings stored in ChromaDB.
    Runs in parallel with N07 BM25. Returns top-10 chunks by cosine similarity.
    Results feed into N09 RRF+Reranker for merging with BM25 results.

Constraints satisfied:
    C1  $0 cost — sentence-transformers is free, ChromaDB is free
    C2  100% local — zero network calls during inference
    C5  seed=42 via SeedManager
    C7  N/A at this node (no LLM prompt)
    C8  All results carry 5-field metadata prefix
    C9  No _rlef_ fields touched

Gate M3:
    MRR@10 >= 0.85 on FinanceBench query eval set required before deploy.
    Evaluated by: python eval/run_eval.py --gate m3 --seed 42

CHANGELOG:
  2026-04-30 S13  Bug #3: run_bge() now reads state.chromadb_data_dir to ensure
                  retriever path matches what chunker wrote to.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — only loaded when BGERetriever is actually used
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
MIN_SIMILARITY     = 0.0   # include all results — RRF handles filtering
RETRIEVER_LABEL    = "bge_m3"

# ChromaDB collection name prefix
_COLLECTION_PREFIX = "finbench_"


# ── BGERetriever ──────────────────────────────────────────────────────────────

class BGERetriever:
    """
    N08 BGE-M3 Semantic Retriever.

    Two usage modes:
        1. retriever.retrieve(query, collection_name, data_dir) → List[Dict]
        2. retriever.run(ba_state)                              → BAState
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k:      int = DEFAULT_TOP_K,
        data_dir:   str = "data",
    ) -> None:
        self.model_name = model_name
        self.top_k      = top_k
        self.data_dir   = data_dir
        self._model     = None   # lazy loaded
        self._client    = None   # lazy loaded ChromaDB client

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        Reads:  state.query, state.chromadb_collection
        Writes: state.retrieval_stage_1 (top-10 semantic chunks)
        """
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
        """
        Embed query and search ChromaDB collection.
        """
        if not query or not collection_name:
            return []

        try:
            model      = self._load_model()
            collection = self._load_collection(collection_name, data_dir)

            if collection is None:
                logger.warning("N08: collection '%s' not found", collection_name)
                return []

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
        """
        Embed all chunks and store in ChromaDB collection.

        Called at N03 ingestion time (or on first query if collection missing).
        """
        if not chunks:
            logger.warning("N08: No chunks provided — skipping collection build")
            return False

        try:
            model      = self._load_model()
            client     = self._get_client(data_dir)

            # Delete existing collection if rebuilding
            try:
                client.delete_collection(name=collection_name)
                logger.info("N08: Deleted existing collection '%s'", collection_name)
            except Exception:
                pass  # Collection did not exist — that is fine

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

                # Instruction prefix for document chunks
                instructed = [
                    f"passage: {t}" for t in batch_texts
                ]
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
        """
        Return a LangChain-compatible retriever wrapping this BGERetriever.
        """
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
                device="cpu",   # C2: local only — no GPU required
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
        """
        Build ChromaDB metadata dict from a chunk.
        All values must be str, int, float, or bool for ChromaDB.
        """
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
        """
        Convert raw ChromaDB query results to pipeline-standard format.

        ChromaDB returns distances (lower = more similar for cosine).
        We convert to similarity score: score = 1 - distance.
        """
        formatted = []

        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            # Convert cosine distance to similarity score
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
    """
    Thin wrapper making BGERetriever compatible with LangChain's
    retriever interface (used by N09 EnsembleRetriever).
    Implements invoke(query) → List[Document]
    """

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
        """
        LangChain-compatible invoke method.
        Returns List[Document] with page_content and metadata.
        """
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
    """
    Convenience wrapper used by the LangGraph pipeline node N08.

    Bug #3 fix (S13): if state has chromadb_data_dir (set by chunker),
    use that to ensure BGE reads the same path the chunker wrote to.

    Args:
        state    : BAState object
        data_dir : Default ChromaDB parent dir (overridden by state if present)

    Returns:
        BAState with retrieval_stage_1 populated
    """
    state_data_dir = getattr(state, "chromadb_data_dir", "") or ""
    effective_dir = state_data_dir if state_data_dir else data_dir

    retriever = BGERetriever(
        model_name = DEFAULT_MODEL,
        top_k      = DEFAULT_TOP_K,
        data_dir   = effective_dir,
    )
    return retriever.run(state)