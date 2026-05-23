"""
src/ingestion/chunker.py

Production-Grade Chunker + Index Builder

Optimized for:
- Windows
- Colab
- FinanceBench
- Low RAM
- T4 GPUs
- Large SEC filings
- Stable ChromaDB integration
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

MAX_CHUNK_TOKENS = 350

MIN_CHUNK_CHARS = 50

MAX_CHUNKS_CAP = 5000

CHUNK_OVERLAP = 100

MAX_SECTION_CHARS = 50_000

MAX_SECTION_NAME = 200

MAX_PREFIX_CHARS = 500

BM25_BATCH_SIZE = 500

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def cleanup_memory():

    gc.collect()

    try:

        import torch

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# DocumentChunk
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DocumentChunk:

    chunk_id: str

    text: str

    company: str

    doc_type: str

    fiscal_year: str

    section: str

    page: int

    char_count: int = 0

    token_estimate: int = 0

    prefix: str = ""

    def __post_init__(
        self,
    ):

        self.text = (
            self.text or ""
        ).strip()

        self.company = (
            self.company
            or "UNKNOWN"
        )

        self.doc_type = (
            self.doc_type
            or "UNKNOWN"
        )

        self.fiscal_year = (
            self.fiscal_year
            or "UNKNOWN"
        )

        self.section = (
            self.section
            or "DOCUMENT"
        )[
            :MAX_SECTION_NAME
        ]

        self.page = int(
            self.page or 0
        )

        self.char_count = len(
            self.text
        )

        self.token_estimate = max(
            1,
            self.char_count // 4,
        )

        self.prefix = (
            f"{self.company} / "
            f"{self.doc_type} / "
            f"{self.fiscal_year} / "
            f"{self.section} / "
            f"{self.page}"
        )[
            :MAX_PREFIX_CHARS
        ]

    @property
    def prefixed_text(
        self,
    ) -> str:

        return (
            f"{self.prefix}\n"
            f"{self.text}"
        )

    def to_dict(
        self,
    ) -> Dict:

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "prefix": self.prefix,
            "company": self.company,
            "doc_type": self.doc_type,
            "fiscal_year": self.fiscal_year,
            "section": self.section,
            "page": self.page,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────


class Chunker:

    def __init__(
        self,
        bm25_dir: str = "data/bm25_index",
        chromadb_dir: str = "data/chromadb",
        max_tokens: int = MAX_CHUNK_TOKENS,
        seed: int = SEED,
    ):

        self.bm25_dir = bm25_dir

        self.chromadb_dir = chromadb_dir

        self.max_tokens = int(
            max_tokens
        )

        self.seed = int(seed)

    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        state,
    ):

        raw_text = getattr(
            state,
            "raw_text",
            "",
        ) or ""

        section_tree = getattr(
            state,
            "section_tree",
            {},
        ) or {}

        company = getattr(
            state,
            "company_name",
            "UNKNOWN",
        )

        doc_type = getattr(
            state,
            "doc_type",
            "UNKNOWN",
        )

        fiscal_year = getattr(
            state,
            "fiscal_year",
            "UNKNOWN",
        )

        session_id = getattr(
            state,
            "session_id",
            "default",
        )

        if not raw_text.strip():

            logger.warning(
                "[N03] Empty document"
            )

            state.chunk_count = 0

            return state

        chunks = self.chunk(
            raw_text=raw_text,
            section_tree=section_tree,
            company=company,
            doc_type=doc_type,
            fiscal_year=fiscal_year,
        )

        safe_session = (
            self._sanitize_for_chroma(
                session_id
            )
        )

        collection_name = (
            f"doc-{safe_session}"
        )

        bm25_path = os.path.join(
            self.bm25_dir,
            safe_session,
        )

        # BM25

        try:

            self._build_bm25_index(
                chunks,
                bm25_path,
            )

        except Exception:

            logger.exception(
                "[N03] BM25 failed"
            )

        # Chroma

        try:

            self._build_chromadb_index(
                chunks,
                collection_name,
            )

        except Exception:

            logger.exception(
                "[N03] Chroma failed"
            )

        state.chunk_count = len(
            chunks
        )

        state.bm25_index_path = (
            bm25_path
        )

        state.chromadb_collection = (
            collection_name
        )

        state.chromadb_data_dir = (
            os.path.dirname(
                os.path.normpath(
                    self.chromadb_dir
                )
            )
            or "."
        )

        cleanup_memory()

        logger.info(
            "[N03] Chunking complete | chunks=%d",
            len(chunks),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────

    def chunk(
        self,
        raw_text: str,
        section_tree: Dict,
        company: str = "UNKNOWN",
        doc_type: str = "UNKNOWN",
        fiscal_year: str = "UNKNOWN",
    ) -> List[DocumentChunk]:

        sections = (
            section_tree.get(
                "children",
                [],
            )
            if section_tree
            else []
        )

        chunks = []

        if sections:

            chunks = (
                self._chunk_by_sections(
                    raw_text,
                    sections,
                    company,
                    doc_type,
                    fiscal_year,
                )
            )

        if not chunks:

            chunks = (
                self._chunk_by_paragraphs(
                    raw_text,
                    company,
                    doc_type,
                    fiscal_year,
                )
            )

        if (
            not chunks
            and raw_text.strip()
        ):

            chunks = [
                DocumentChunk(
                    chunk_id="chunk_0000",
                    text=raw_text.strip(),
                    company=company,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section="DOCUMENT",
                    page=0,
                )
            ]

        chunks = chunks[
            :MAX_CHUNKS_CAP
        ]

        self._assert_metadata(
            chunks
        )

        return chunks

    # ─────────────────────────────────────────────────────────────────────

    def _chunk_by_sections(
        self,
        raw_text: str,
        sections: List[Dict],
        company: str,
        doc_type: str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:

        chunks = []

        chunk_id = 0

        for idx, section in enumerate(
            sections
        ):

            section_name = (
                section.get(
                    "name",
                    "UNKNOWN",
                )
            )

            start_page = int(
                section.get(
                    "start_page",
                    0,
                )
            )

            next_section = ""

            if idx + 1 < len(
                sections
            ):

                next_section = sections[
                    idx + 1
                ].get(
                    "name",
                    "",
                )

            section_text = (
                self._extract_section_text(
                    raw_text,
                    section_name,
                    next_section,
                )
            )

            if (
                not section_text
                or len(section_text)
                < MIN_CHUNK_CHARS
            ):

                continue

            sub_chunks = (
                self._split_large_text(
                    text=section_text,
                    company=company,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section=section_name,
                    page=start_page,
                    base_id=chunk_id,
                )
            )

            chunks.extend(
                sub_chunks
            )

            chunk_id += len(
                sub_chunks
            )

        return chunks

    # ─────────────────────────────────────────────────────────────────────

    def _chunk_by_paragraphs(
        self,
        raw_text: str,
        company: str,
        doc_type: str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:

        paragraphs = re.split(
            r"\n\s*\n",
            raw_text,
        )

        chunks = []

        chunk_id = 0

        current_section = (
            "DOCUMENT"
        )

        for para in paragraphs:

            para = para.strip()

            if not para:
                continue

            if (
                len(para)
                < MIN_CHUNK_CHARS
            ):

                continue

            sub_chunks = (
                self._split_large_text(
                    text=para,
                    company=company,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section=current_section,
                    page=0,
                    base_id=chunk_id,
                )
            )

            chunks.extend(
                sub_chunks
            )

            chunk_id += len(
                sub_chunks
            )

        return chunks

    # ─────────────────────────────────────────────────────────────────────

    def _split_large_text(
        self,
        text: str,
        company: str,
        doc_type: str,
        fiscal_year: str,
        section: str,
        page: int,
        base_id: int = 0,
    ) -> List[DocumentChunk]:

        max_chars = (
            self.max_tokens * 4
        )

        text = text.strip()

        if len(text) <= max_chars:

            return [
                DocumentChunk(
                    chunk_id=f"chunk_{base_id:05d}",
                    text=text,
                    company=company,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section=section,
                    page=page,
                )
            ]

        chunks = []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        current = ""

        idx = 0

        for sent in sentences:

            sent = sent.strip()

            if not sent:
                continue

            projected = (
                len(current)
                + len(sent)
            )

            if (
                projected
                > max_chars
                and current
            ):

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{base_id + idx:05d}",
                        text=current.strip(),
                        company=company,
                        doc_type=doc_type,
                        fiscal_year=fiscal_year,
                        section=section,
                        page=page,
                    )
                )

                idx += 1

                overlap = current[
                    -CHUNK_OVERLAP:
                ]

                current = (
                    overlap
                    + " "
                    + sent
                )

            else:

                current = (
                    current
                    + " "
                    + sent
                ).strip()

        if (
            current
            and len(current)
            >= MIN_CHUNK_CHARS
        ):

            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{base_id + idx:05d}",
                    text=current.strip(),
                    company=company,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section=section,
                    page=page,
                )
            )

        return chunks

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_section_text(
        raw_text: str,
        section_name: str,
        end_section_name: str = "",
    ) -> str:

        if (
            not raw_text
            or not section_name
        ):

            return ""

        pattern = re.compile(
            re.escape(
                section_name[:50]
            ),
            re.IGNORECASE,
        )

        match = pattern.search(
            raw_text
        )

        if not match:
            return ""

        start = match.start()

        end = -1

        if end_section_name:

            next_pattern = re.compile(
                re.escape(
                    end_section_name[
                        :50
                    ]
                ),
                re.IGNORECASE,
            )

            next_match = (
                next_pattern.search(
                    raw_text,
                    pos=start
                    + len(
                        section_name
                    ),
                )
            )

            if next_match:

                end = next_match.start()

        if end < 0:

            end = min(
                start
                + MAX_SECTION_CHARS,
                len(raw_text),
            )

        return raw_text[
            start:end
        ].strip()

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assert_metadata(
        chunks: List[DocumentChunk],
    ):

        for chunk in chunks:

            if not all(
                [
                    chunk.company,
                    chunk.doc_type,
                    chunk.fiscal_year,
                    chunk.section,
                    chunk.page
                    is not None,
                ]
            ):

                raise ValueError(
                    f"C8 violation: {chunk.chunk_id}"
                )

    # ─────────────────────────────────────────────────────────────────────

    def _build_bm25_index(
        self,
        chunks: List[DocumentChunk],
        bm25_path: str,
    ):

        if not chunks:
            return

        import bm25s

        os.makedirs(
            bm25_path,
            exist_ok=True,
        )

        corpus = [
            c.prefixed_text
            for c in chunks
        ]

        retriever = bm25s.BM25()

        tokenized = bm25s.tokenize(
            corpus,
            stopwords="en",
        )

        retriever.index(
            tokenized
        )

        retriever.save(
            bm25_path,
            corpus=corpus,
        )

        meta_path = os.path.join(
            bm25_path,
            "chunks_meta.json",
        )

        with open(
            meta_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                [
                    c.to_dict()
                    for c in chunks
                ],
                f,
                ensure_ascii=False,
            )

        logger.info(
            "[N03] BM25 built | chunks=%d",
            len(chunks),
        )

    # ─────────────────────────────────────────────────────────────────────

    def _build_chromadb_index(
        self,
        chunks: List[DocumentChunk],
        collection_name: str,
    ):

        if not chunks:
            return

        disabled = os.environ.get(
            "DISABLE_CHROMADB",
            "",
        ).lower()

        if disabled in (
            "1",
            "true",
            "yes",
        ):

            logger.info(
                "[N03] Chroma disabled"
            )

            return

        try:

            from src.retrieval.bge_retriever import (
                BGERetriever,
            )

            retriever = (
                BGERetriever(
                    data_dir=os.path.dirname(
                        os.path.normpath(
                            self.chromadb_dir
                        )
                    )
                )
            )

            ok = (
                retriever.build_collection(
                    chunks=[
                        c.to_dict()
                        for c in chunks
                    ],
                    collection_name=collection_name,
                    data_dir=os.path.dirname(
                        os.path.normpath(
                            self.chromadb_dir
                        )
                    ),
                )
            )

            if ok:

                logger.info(
                    "[N03] Chroma built=%s",
                    collection_name,
                )

            else:

                logger.warning(
                    "[N03] Chroma build failed"
                )

        except Exception:

            logger.exception(
                "[N03] Chroma exception"
            )

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_for_chroma(
        value: str,
        max_len: int = 16,
    ) -> str:

        value = re.sub(
            r"[^a-zA-Z0-9]+",
            "-",
            value,
        )

        value = value.strip("-")

        value = value[:max_len]

        if len(value) < 3:

            value = (
                value + "xxx"
            )[:3]

        return value.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_chunker(
    state,
    bm25_dir: str = "data/bm25_index",
    chromadb_dir: str = "data/chromadb",
):

    chunker = Chunker(
        bm25_dir=bm25_dir,
        chromadb_dir=chromadb_dir,
    )

    return chunker.run(state)