"""
src/retrieval/bge_retriever.py

Production-Grade BGE Semantic Retriever
Optimized for:

- Windows
- Colab
- T4 GPUs
- Ollama coexistence
- Low VRAM
- ChromaDB stability
- FinanceBench scale
"""

from __future__ import annotations

import gc
import logging
import re
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENV SAFETY
# ─────────────────────────────────────────────────────────────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMADB_TELEMETRY"] = "False"

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Globals
# ─────────────────────────────────────────────────────────────────────────────

_ST = None
_CHROMADB = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 10

MIN_SIMILARITY = 0.05

RETRIEVER_LABEL = "bge"

_BATCH_SIZE = 16

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Imports
# ─────────────────────────────────────────────────────────────────────────────


def get_sentence_transformers():

    global _ST

    if _ST is None:

        import sentence_transformers

        _ST = sentence_transformers

    return _ST


def get_chromadb():

    global _CHROMADB

    if _CHROMADB is None:

        import chromadb

        _CHROMADB = chromadb

    return _CHROMADB


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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

    final = []

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

        final.append(item)

    return final


def cleanup_gpu():

    try:

        import torch

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

    except Exception:
        pass

    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
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

        self._disabled = bool(
            os.environ.get(
                "DISABLE_BGE",
                "",
            )
        ) or bool(
            os.environ.get(
                "DISABLE_CHROMADB",
                "",
            )
        )

    # ─────────────────────────────────────────────────────────────────────

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

        if not query or not collection:

            state.bge_results = []

            return state

        results = self.retrieve(
            query=query,
            collection_name=collection,
            data_dir=self.data_dir,
            top_k=self.top_k,
        )

        state.bge_results = results

        return state

    # ─────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        collection_name: str,
        data_dir: str = "data",
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        if not query:
            return []

        if self._disabled:
            return []

        try:

            collection = self._load_collection(
                collection_name,
                data_dir,
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

            embedding = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()

            results = collection.query(
                query_embeddings=[
                    embedding
                ],
                n_results=min(
                    top_k,
                    max(
                        1,
                        collection.count(),
                    ),
                ),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            formatted = self._format_results(
                results,
                top_k,
            )

            return deduplicate_results(
                formatted
            )

        except Exception as exc:

            logger.exception(
                "[BGE] Retrieval failed"
            )

            return []

    # ─────────────────────────────────────────────────────────────────────

    def build_collection(
        self,
        chunks: List[Dict],
        collection_name: str,
        data_dir: str = "data",
    ) -> bool:

        if not chunks:
            return False

        if self._disabled:
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

            metadatas = [
                self._build_metadata(c)
                for c in chunks
            ]

            total = len(texts)

            for start in range(
                0,
                total,
                _BATCH_SIZE,
            ):

                end = start + _BATCH_SIZE

                batch_texts = texts[
                    start:end
                ]

                batch_ids = ids[
                    start:end
                ]

                batch_meta = metadatas[
                    start:end
                ]

                embeddings = model.encode(
                    batch_texts,
                    batch_size=_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                )

            cleanup_gpu()

            logger.info(
                "[BGE] Collection built: %s | chunks=%d",
                collection_name,
                len(chunks),
            )

            return True

        except Exception:

            logger.exception(
                "[BGE] Build failed"
            )

            cleanup_gpu()

            return False

    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self):

        if self._model is not None:
            return self._model

        st = (
            get_sentence_transformers()
        )

        # IMPORTANT:
        # Keep CPU only.
        # Ollama already owns GPU VRAM.

        device = "cpu"

        logger.info(
            "[BGE] Loading model on %s",
            device,
        )

        self._model = (
            st.SentenceTransformer(
                self.model_name,
                device=device,
            )
        )

        return self._model

    # ─────────────────────────────────────────────────────────────────────

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
            path=chromadb_path,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    # ─────────────────────────────────────────────────────────────────────

    def _load_collection(
        self,
        collection_name: str,
        data_dir: str,
    ) -> Optional[object]:

        try:

            client = self._get_client(
                data_dir
            )

            return client.get_collection(
                name=collection_name
            )

        except Exception:

            logger.exception(
                "[BGE] Failed loading collection"
            )

            return None

    # ─────────────────────────────────────────────────────────────────────

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
        }

    # ─────────────────────────────────────────────────────────────────────

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

            if similarity < MIN_SIMILARITY:
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
                }
            )

            if len(formatted) >= top_k:
                break

        return formatted


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