"""
N03 Chunker + Index Builder — Section-Boundary Chunking
PDR-BAAAI-001 · Rev 1.0 · Node N03

Purpose:
    Splits document text at section boundaries (NEVER at arbitrary
    512-token word boundaries). Adds mandatory 5-field metadata prefix
    to every chunk (C8). Builds bm25s sparse index + ChromaDB collection.

CHANGELOG:
  2026-04-30 S8  Bug #1.5: _chunk_by_paragraphs() no longer consolidates;
                 emit one chunk per paragraph.
  2026-05-10 S27 Bug Fix 4: TWO critical bugs fixed
                 1. _extract_section_text() was capping at 3000 chars,
                    dropping most of the document. Now uses next-section
                    boundary or 50,000 char cap (huge increase).
                 2. MAX_CHUNK_TOKENS 800 -> 400 (chunk_size 3200 -> 1600
                    chars). More chunks = better BGE precision.
                 Result: Apple 10-K 30 chunks -> ~150 chunks expected.
                 Narrative recall improves significantly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEED             = 42
# Bug Fix 4: chunk_size reduced from 800 tokens (3200 chars) to 400 tokens
# (1600 chars). Smaller chunks = better BGE precision for finding numbers.
MAX_CHUNK_TOKENS = 400      # was 800
MIN_CHUNK_CHARS  = 50
MAX_CHUNKS_CAP   = 5000
CHUNK_OVERLAP    = 100

# Bug Fix 4: section text extraction limit raised from 3000 to 50000.
# Apple 10-K MD&A section alone is 40K+ chars. 3000 was dropping 90%+
# of every long section.
MAX_SECTION_CHARS = 50_000   # was implicit 3000


class DocumentChunk:
    """A single document chunk with mandatory 5-field metadata (C8)."""

    __slots__ = (
        "chunk_id", "text", "prefix",
        "company", "doc_type", "fiscal_year",
        "section", "page",
        "char_count", "token_estimate",
    )

    def __init__(
        self,
        chunk_id:    str,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
    ) -> None:
        self.chunk_id      = chunk_id
        self.text          = text
        self.company       = company
        self.doc_type      = doc_type
        self.fiscal_year   = fiscal_year
        self.section       = section
        self.page          = page
        self.char_count    = len(text)
        self.token_estimate= len(text) // 4

        # C8: mandatory 5-field prefix
        self.prefix = f"{company} / {doc_type} / {fiscal_year} / {section} / {page}"

    @property
    def prefixed_text(self) -> str:
        return f"{self.prefix}\n{self.text}"

    def to_dict(self) -> Dict:
        return {
            "chunk_id":       self.chunk_id,
            "text":           self.text,
            "prefix":         self.prefix,
            "company":        self.company,
            "doc_type":       self.doc_type,
            "fiscal_year":    self.fiscal_year,
            "section":        self.section,
            "page":           self.page,
            "char_count":     self.char_count,
            "token_estimate": self.token_estimate,
        }


class Chunker:
    """N03 Chunker + Index Builder."""

    def __init__(
        self,
        bm25_dir:     str = "data/bm25_index",
        chromadb_dir: str = "data/chromadb",
        max_tokens:   int = MAX_CHUNK_TOKENS,
        seed:         int = SEED,
    ) -> None:
        self.bm25_dir     = bm25_dir
        self.chromadb_dir = chromadb_dir
        self.max_tokens   = max_tokens
        self.seed         = seed

    # ── LangGraph pipeline node ───────────────────────────────────────────────

    def run(self, state) -> object:
        raw_text     = getattr(state, "raw_text",      "") or ""
        section_tree = getattr(state, "section_tree",  {}) or {}
        company      = getattr(state, "company_name",  "UNKNOWN") or "UNKNOWN"
        doc_type     = getattr(state, "doc_type",      "UNKNOWN") or "UNKNOWN"
        fiscal_year  = getattr(state, "fiscal_year",   "UNKNOWN") or "UNKNOWN"
        session_id   = getattr(state, "session_id",    "default") or "default"

        if not raw_text:
            logger.warning("N03: empty raw_text — skipping chunking")
            return state

        chunks = self.chunk(
            raw_text     = raw_text,
            section_tree = section_tree,
            company      = company,
            doc_type     = doc_type,
            fiscal_year  = fiscal_year,
        )

        # Bug B1 fix (2026-05-20): ChromaDB requires collection names to
        # start AND end with alphanumeric chars. Truncating session_id can
        # leave trailing dashes/underscores. Sanitize the truncated id.
        def _sanitize_for_chroma(s: str, max_len: int = 16) -> str:
            """Truncate to max_len and trim trailing non-alphanumeric chars."""
            truncated = s[:max_len]
            # Strip trailing _ - . chars
            truncated = re.sub(r"[^A-Za-z0-9]+$", "", truncated)
            # Strip leading non-alphanumeric chars
            truncated = re.sub(r"^[^A-Za-z0-9]+", "", truncated)
            # Must be at least 3 chars for ChromaDB
            return truncated if len(truncated) >= 3 else (truncated + "xxx")[:max_len]

        safe_session = _sanitize_for_chroma(session_id)
        collection_name = f"doc-{safe_session}"
        bm25_path       = os.path.join(self.bm25_dir, safe_session)

        self._build_bm25_index(chunks, bm25_path)
        self._build_chromadb_index(chunks, collection_name)

        state.chunk_count          = len(chunks)
        state.bm25_index_path      = bm25_path
        state.chromadb_collection  = collection_name

        chromadb_path_norm = os.path.normpath(self.chromadb_dir)
        if os.path.basename(chromadb_path_norm) == "chromadb":
            data_dir = os.path.dirname(chromadb_path_norm) or "."
        else:
            data_dir = os.path.dirname(chromadb_path_norm) or "."
        if hasattr(state, "chromadb_data_dir"):
            state.chromadb_data_dir = data_dir
        else:
            try:
                state.chromadb_data_dir = data_dir
            except Exception:
                pass

        logger.info(
            "N03 Chunker: %d chunks | bm25=%s | chromadb=%s | data_dir=%s",
            len(chunks), bm25_path, collection_name, data_dir,
        )
        return state

    # ── Core chunk method ─────────────────────────────────────────────────────

    def chunk(
        self,
        raw_text:     str,
        section_tree: Dict,
        company:      str = "UNKNOWN",
        doc_type:     str = "UNKNOWN",
        fiscal_year:  str = "UNKNOWN",
    ) -> List[DocumentChunk]:
        sections = section_tree.get("children", []) if section_tree else []
        chunks: List[DocumentChunk] = []

        if sections:
            chunks = self._chunk_by_sections(
                raw_text, sections, company, doc_type, fiscal_year
            )

        if not chunks:
            chunks = self._chunk_by_paragraphs(
                raw_text, company, doc_type, fiscal_year
            )

        if not chunks and raw_text.strip():
            chunks = [DocumentChunk(
                chunk_id    = "chunk_0000",
                text        = raw_text.strip(),
                company     = company,
                doc_type    = doc_type,
                fiscal_year = fiscal_year,
                section     = "DOCUMENT",
                page        = 0,
            )]

        chunks = chunks[:MAX_CHUNKS_CAP]
        self._assert_metadata_prefixes(chunks)
        return chunks

    # ── Private chunking strategies ───────────────────────────────────────────

    def _chunk_by_sections(
        self,
        raw_text:    str,
        sections:    List[Dict],
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:
        """Split text at section boundaries.

        Bug Fix 4: Pass next-section name so _extract_section_text knows
        where the current section ENDS. Previously capped at 3000 chars.
        """
        chunks   = []
        chunk_id = 0

        # Build list of section names + their start positions for boundary detection
        section_starts: List[str] = []
        for section in sections:
            name = section.get("name", "")
            if name:
                section_starts.append(name)
            for child in section.get("children", []):
                cname = child.get("name", "")
                if cname:
                    section_starts.append(cname)

        for i, section in enumerate(sections):
            section_name = section.get("name",       "UNKNOWN")
            start_page   = section.get("start_page", 0)

            # Bug Fix 4: find next section name to use as end boundary
            next_section_name = ""
            if i + 1 < len(sections):
                next_section_name = sections[i + 1].get("name", "")

            section_text = self._extract_section_text(
                raw_text, section_name, start_page,
                end_section_name=next_section_name,
            )

            if not section_text or len(section_text) < MIN_CHUNK_CHARS:
                # Recurse into children
                children = section.get("children", [])
                for j, child in enumerate(children):
                    child_next = ""
                    if j + 1 < len(children):
                        child_next = children[j + 1].get("name", "")
                    elif i + 1 < len(sections):
                        child_next = sections[i + 1].get("name", "")

                    child_text = self._extract_section_text(
                        raw_text,
                        child.get("name", ""),
                        child.get("start_page", 0),
                        end_section_name=child_next,
                    )
                    if child_text and len(child_text) >= MIN_CHUNK_CHARS:
                        sub_chunks = self._split_large_text(
                            child_text,
                            company, doc_type, fiscal_year,
                            child.get("name", section_name),
                            child.get("start_page", start_page),
                            chunk_id,
                        )
                        chunks.extend(sub_chunks)
                        chunk_id += len(sub_chunks)
                continue

            sub_chunks = self._split_large_text(
                section_text,
                company, doc_type, fiscal_year,
                section_name, start_page, chunk_id,
            )
            chunks.extend(sub_chunks)
            chunk_id += len(sub_chunks)

            # Also chunk children (often have detailed content)
            children = section.get("children", [])
            for j, child in enumerate(children):
                child_next = ""
                if j + 1 < len(children):
                    child_next = children[j + 1].get("name", "")
                elif i + 1 < len(sections):
                    child_next = sections[i + 1].get("name", "")

                child_text = self._extract_section_text(
                    raw_text,
                    child.get("name", ""),
                    child.get("start_page", 0),
                    end_section_name=child_next,
                )
                if child_text and len(child_text) >= MIN_CHUNK_CHARS:
                    child_chunks = self._split_large_text(
                        child_text,
                        company, doc_type, fiscal_year,
                        child.get("name", section_name),
                        child.get("start_page", start_page),
                        chunk_id,
                    )
                    chunks.extend(child_chunks)
                    chunk_id += len(child_chunks)

        return chunks

    def _chunk_by_paragraphs(
        self,
        raw_text:    str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
    ) -> List[DocumentChunk]:
        """Emit one chunk per paragraph (never consolidate).

        Only split a single paragraph further if it exceeds max_chars.
        """
        max_chars  = self.max_tokens * 4
        paragraphs = re.split(r'\n\s*\n', raw_text)
        chunks: List[DocumentChunk] = []
        chunk_id = 0

        current_section = "DOCUMENT"

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < MIN_CHUNK_CHARS:
                wc = len(para.split())
                if 1 <= wc <= 5 and len(para) <= 60:
                    current_section = para[:50]
                continue

            if len(para) > max_chars:
                sub_chunks = self._split_large_text(
                    para,
                    company, doc_type, fiscal_year,
                    current_section, 0, chunk_id,
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunks.append(DocumentChunk(
                    chunk_id    = f"chunk_{chunk_id:04d}",
                    text        = para,
                    company     = company,
                    doc_type    = doc_type,
                    fiscal_year = fiscal_year,
                    section     = current_section,
                    page        = 0,
                ))
                chunk_id += 1

        return chunks

    def _split_large_text(
        self,
        text:        str,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
        base_id:     int = 0,
    ) -> List[DocumentChunk]:
        """Split a large text block into smaller chunks with overlap."""
        max_chars = self.max_tokens * 4
        chunks    = []

        if len(text) <= max_chars:
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append(DocumentChunk(
                    chunk_id    = f"chunk_{base_id:04d}",
                    text        = text.strip(),
                    company     = company,
                    doc_type    = doc_type,
                    fiscal_year = fiscal_year,
                    section     = section,
                    page        = page,
                ))
            return chunks

        sentences = re.split(r'(?<=[.!?])\s+', text)
        current   = ""
        idx       = 0

        for sent in sentences:
            if len(current) + len(sent) > max_chars and current:
                if len(current) >= MIN_CHUNK_CHARS:
                    chunks.append(DocumentChunk(
                        chunk_id    = f"chunk_{base_id + idx:04d}",
                        text        = current.strip(),
                        company     = company,
                        doc_type    = doc_type,
                        fiscal_year = fiscal_year,
                        section     = section,
                        page        = page,
                    ))
                    idx += 1
                current = current[-CHUNK_OVERLAP:] + " " + sent
            else:
                current = (current + " " + sent).strip()

        if current and len(current) >= MIN_CHUNK_CHARS:
            chunks.append(DocumentChunk(
                chunk_id    = f"chunk_{base_id + idx:04d}",
                text        = current.strip(),
                company     = company,
                doc_type    = doc_type,
                fiscal_year = fiscal_year,
                section     = section,
                page        = page,
            ))

        return chunks

    @staticmethod
    def _extract_section_text(
        raw_text: str,
        section_name: str,
        page: int,
        end_section_name: str = "",
    ) -> str:
        """Extract text for a section, bounded by the next section.

        Bug Fix 4: Previously hardcoded 3000-char cap. Now:
        1. Find section_name in raw_text
        2. If end_section_name given AND found later in text, use that
           as the end boundary
        3. Otherwise cap at MAX_SECTION_CHARS (50,000)
        """
        if not section_name or not raw_text:
            return ""

        pattern = re.compile(
            re.escape(section_name[:30]),
            re.IGNORECASE,
        )
        m = pattern.search(raw_text)
        if not m:
            return ""

        start = m.start()

        # Bug Fix 4: try to find next section as end boundary
        end = -1
        if end_section_name:
            end_pattern = re.compile(
                re.escape(end_section_name[:30]),
                re.IGNORECASE,
            )
            end_match = end_pattern.search(raw_text, pos=start + len(section_name))
            if end_match:
                end = end_match.start()

        if end < 0:
            # No next section found — cap at MAX_SECTION_CHARS
            end = min(start + MAX_SECTION_CHARS, len(raw_text))

        return raw_text[start:end].strip()

    @staticmethod
    def _assert_metadata_prefixes(chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            if not all([
                chunk.company,
                chunk.doc_type,
                chunk.fiscal_year,
                chunk.section,
                chunk.page is not None,
            ]):
                raise ValueError(
                    f"C8 violation: chunk {chunk.chunk_id} missing metadata. "
                    f"company={chunk.company!r} doc_type={chunk.doc_type!r} "
                    f"fiscal_year={chunk.fiscal_year!r} section={chunk.section!r} "
                    f"page={chunk.page!r}"
                )

    def _build_bm25_index(self, chunks: List[DocumentChunk], bm25_path: str) -> None:
        """Build BM25 sparse index from chunks.

        Bug Y2 fix (2026-05-12): Also persist chunks_meta.json so BM25Retriever
        can load chunk metadata at query time. Without this file, BM25Retriever
        silently returns 0 results and the entire FinanceBench eval scored 0%.
        """
        if not chunks:
            return

        try:
            os.makedirs(bm25_path, exist_ok=True)

            import bm25s
            retriever = bm25s.BM25()
            corpus = [c.prefixed_text for c in chunks]
            retriever.index(bm25s.tokenize(corpus, stopwords="en"))
            retriever.save(bm25_path, corpus=corpus)

            # Bug Y2: persist chunks metadata for BM25Retriever to load at query time
            import json
            chunks_meta_path = os.path.join(bm25_path, "chunks_meta.json")
            chunks_dict = [c.to_dict() for c in chunks]
            with open(chunks_meta_path, "w", encoding="utf-8") as f:
                json.dump(chunks_dict, f, ensure_ascii=False)
            logger.info(
                "BM25 index + meta saved: %d chunks -> %s",
                len(chunks), bm25_path,
            )
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)

    def _build_chromadb_index(
        self,
        chunks: List[DocumentChunk],
        collection_name: str,
    ) -> None:
        """Build ChromaDB collection from chunks (delegated to BGERetriever)."""
        if not chunks:
            return

        if os.environ.get("DISABLE_CHROMADB", "").lower() in ("1", "true", "yes"):
            logger.debug("DISABLE_CHROMADB set — skipping ChromaDB index")
            return

        chromadb_path = os.path.normpath(self.chromadb_dir)

        if os.path.basename(chromadb_path) == "chromadb":
            data_dir = os.path.dirname(chromadb_path) or "."
        else:
            data_dir = os.path.dirname(chromadb_path) or "."

        try:
            from src.retrieval.bge_retriever import BGERetriever
        except ImportError:
            logger.warning("BGERetriever import failed — ChromaDB index skipped")
            return

        try:
            chunk_dicts = [c.to_dict() for c in chunks]
            retriever   = BGERetriever()
            ok = retriever.build_collection(
                chunks          = chunk_dicts,
                collection_name = collection_name,
                data_dir        = data_dir,
            )
            if ok:
                logger.debug(
                    "ChromaDB collection '%s' built via BGE: %d documents",
                    collection_name, len(chunks),
                )
            else:
                logger.warning(
                    "BGE.build_collection returned False for '%s'",
                    collection_name,
                )
        except Exception as exc:
            logger.warning("ChromaDB index build failed: %s", exc)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_chunker(
    state,
    bm25_dir:     str = "data/bm25_index",
    chromadb_dir: str = "data/chromadb",
) -> object:
    """Convenience wrapper for LangGraph N03 node."""
    return Chunker(bm25_dir=bm25_dir, chromadb_dir=chromadb_dir).run(state)