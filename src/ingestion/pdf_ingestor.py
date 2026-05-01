"""
N01 PDF Ingestor - Multi-Format Document Ingestion
PDR-BAAAI-001 Rev 1.0 Node N01

CHANGELOG:
  2026-04-30 S9   Bug #2:   rewrote _ingest_html() for full HTML extraction
                            (h1-h6, b/strong, <table>).
  2026-04-30 S10  Bug #2.1: real SEC 10-Ks are iXBRL — added auto-detection
                            and ix:nonFraction/ix:nonNumeric fact extraction
                            plus visual styled headings via inline CSS.
  2026-04-30 S11  Bug #2.2: dual-pass parsing. Real iXBRL files need:
                              (a) lxml-xml parser for ix: facts (namespaces)
                              (b) lxml HTML parser for <span>/<div>/<p>
                                  styled-heading layout (XHTML in body)
                            Also fixed dei:DocumentFiscalYearFocus matching
                            empty-text tags first → skip empties.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".csv",
    ".pptx", ".html", ".htm", ".txt", ".eml",
    ".json", ".xml",
}

HEADING_FONT_SIZE_MIN = 13.0
HTML_CHARS_PER_PAGE   = 5000

HTML_HEADING_FONT_PT  = 11.0
HTML_BOLD_WEIGHT_MIN  = 600

MAX_IXBRL_FACTS  = 4000
MAX_VISUAL_HEADS = 1500

SEC_SECTIONS = [
    "business", "risk factors", "properties",
    "legal proceedings", "mine safety",
    "market", "selected financial data",
    "management", "financial statements",
    "quantitative", "controls", "other information",
    "directors", "executive compensation",
    "security ownership", "certain relationships",
    "principal accountant",
]


class TableCell:
    """Structured representation of a single table cell."""
    __slots__ = ("row_header", "col_header", "value",
                 "page", "table_number", "section")

    def __init__(
        self,
        row_header:   str = "",
        col_header:   str = "",
        value:        str = "",
        page:         int = 0,
        table_number: int = 0,
        section:      str = "",
    ) -> None:
        self.row_header   = row_header
        self.col_header   = col_header
        self.value        = value
        self.page         = page
        self.table_number = table_number
        self.section      = section

    def to_dict(self) -> Dict:
        return {
            "row_header":   self.row_header,
            "col_header":   self.col_header,
            "value":        self.value,
            "page":         self.page,
            "table_number": self.table_number,
            "section":      self.section,
        }


class PDFIngestor:
    """N01 Multi-Format Document Ingestor."""

    def __init__(
        self,
        enable_images: bool = False,
        llm_client          = None,
    ) -> None:
        self.enable_images = enable_images
        self._llm          = llm_client

    def run(self, state) -> object:
        doc_path = getattr(state, "document_path", "") or ""

        if not doc_path:
            logger.warning("N01: no document_path in state")
            return state

        if not os.path.exists(doc_path):
            logger.error("N01: file not found - %s", doc_path)
            return state

        result = self.ingest(doc_path)

        state.raw_text          = result.get("raw_text",          "")
        state.table_cells       = result.get("table_cells",       [])
        state.heading_positions = result.get("heading_positions", [])

        if not state.company_name:
            state.company_name = result.get("company_name", "")
        if not state.doc_type:
            state.doc_type     = result.get("doc_type",     "")
        if not state.fiscal_year:
            state.fiscal_year  = result.get("fiscal_year",  "")

        logger.info(
            "N01 Ingestor: %d chars | %d table_cells | %d headings | %s",
            len(state.raw_text),
            len(state.table_cells),
            len(state.heading_positions),
            os.path.basename(doc_path),
        )

        if self.enable_images and doc_path.lower().endswith(".pdf"):
            try:
                from src.ingestion.image_processor import ImageProcessor
                image_proc = ImageProcessor(
                    enable_ocr    = True,
                    enable_vision = self._llm is not None,
                    llm_client    = self._llm,
                )
                state = image_proc.run(state)
                logger.info("N01b: image processing complete")
            except Exception as exc:
                logger.warning("N01b image processor failed: %s", exc)

        return state

    def ingest(self, file_path: str) -> Dict:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._ingest_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self._ingest_docx(file_path)
        elif ext in (".xlsx", ".xls"):
            return self._ingest_xlsx(file_path)
        elif ext == ".csv":
            return self._ingest_csv(file_path)
        elif ext == ".pptx":
            return self._ingest_pptx(file_path)
        elif ext in (".html", ".htm"):
            return self._ingest_html(file_path)
        elif ext == ".json":
            return self._ingest_json(file_path)
        else:
            return self._ingest_txt(file_path)

    def _ingest_pdf(self, file_path: str) -> Dict:
        raw_text          = ""
        table_cells       = []
        heading_positions = []

        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    raw_text += page_text + "\n"

                    tables = page.extract_tables() or []
                    for table_num, table in enumerate(tables, start=1):
                        if not table or len(table) < 2:
                            continue
                        col_headers = [
                            str(c).strip() if c else ""
                            for c in table[0]
                        ]
                        for row in table[1:]:
                            if not row:
                                continue
                            row_header = str(row[0]).strip() if row[0] else ""
                            for col_idx, cell_val in enumerate(row[1:], start=1):
                                col_header = (
                                    col_headers[col_idx]
                                    if col_idx < len(col_headers) else ""
                                )
                                value = str(cell_val).strip() if cell_val else ""
                                if value:
                                    table_cells.append(TableCell(
                                        row_header   = row_header,
                                        col_header   = col_header,
                                        value        = value,
                                        page         = page_num,
                                        table_number = table_num,
                                    ).to_dict())
        except ImportError:
            logger.warning("pdfplumber not installed - PDF tables skipped")
        except Exception as exc:
            logger.warning("pdfplumber error on %s: %s", file_path, exc)

        try:
            import fitz
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict").get("blocks", [])
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_size = span.get("size", 0)
                            flags     = span.get("flags", 0)
                            text      = span.get("text", "").strip()
                            is_bold   = bool(flags & 2 ** 4)
                            if font_size >= HEADING_FONT_SIZE_MIN and text and len(text) > 3:
                                heading_positions.append({
                                    "text":      text,
                                    "font_size": round(font_size, 1),
                                    "is_bold":   is_bold,
                                    "page":      page_num,
                                })
            doc.close()
        except ImportError:
            logger.warning("PyMuPDF not installed - headings skipped")
        except Exception as exc:
            logger.warning("PyMuPDF error on %s: %s", file_path, exc)

        company_name, doc_type, fiscal_year = self._extract_metadata(
            raw_text, heading_positions
        )
        return {
            "raw_text":          raw_text,
            "table_cells":       table_cells,
            "heading_positions": heading_positions,
            "company_name":      company_name,
            "doc_type":          doc_type,
            "fiscal_year":       fiscal_year,
        }

    def _ingest_docx(self, file_path: str) -> Dict:
        raw_text          = ""
        table_cells       = []
        heading_positions = []

        try:
            from docx import Document
            doc = Document(file_path)

            for para in doc.paragraphs:
                raw_text += para.text + "\n"
                if para.style and "Heading" in para.style.name:
                    level = para.style.name.replace("Heading ", "").strip()
                    heading_positions.append({
                        "text":      para.text.strip(),
                        "font_size": 16.0 if level == "1" else 13.0,
                        "is_bold":   True,
                        "page":      0,
                    })

            for t_num, table in enumerate(doc.tables, start=1):
                rows = list(table.rows)
                if len(rows) < 2:
                    continue
                col_headers = [c.text.strip() for c in rows[0].cells]
                for row in rows[1:]:
                    cells      = row.cells
                    row_header = cells[0].text.strip() if cells else ""
                    for col_idx, cell in enumerate(cells[1:], start=1):
                        col_header = (
                            col_headers[col_idx]
                            if col_idx < len(col_headers) else ""
                        )
                        value = cell.text.strip()
                        if value:
                            table_cells.append(TableCell(
                                row_header   = row_header,
                                col_header   = col_header,
                                value        = value,
                                page         = 0,
                                table_number = t_num,
                            ).to_dict())
        except ImportError:
            logger.warning("python-docx not installed")
        except Exception as exc:
            logger.warning("DOCX error: %s", exc)

        company_name, doc_type, fiscal_year = self._extract_metadata(
            raw_text, heading_positions
        )
        return {
            "raw_text": raw_text, "table_cells": table_cells,
            "heading_positions": heading_positions,
            "company_name": company_name, "doc_type": doc_type,
            "fiscal_year": fiscal_year,
        }

    def _ingest_xlsx(self, file_path: str) -> Dict:
        raw_text    = ""
        table_cells = []

        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            for sheet_name in wb.sheetnames:
                ws   = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))
                if not rows:
                    continue
                col_headers = [
                    str(c).strip() if c is not None else ""
                    for c in rows[0]
                ]
                raw_text += f"\n[Sheet: {sheet_name}]\n"
                for row in rows[1:]:
                    row_header = str(row[0]).strip() if row[0] is not None else ""
                    for col_idx, val in enumerate(row[1:], start=1):
                        if val is not None:
                            col_header = (
                                col_headers[col_idx]
                                if col_idx < len(col_headers) else ""
                            )
                            value = str(val).strip()
                            raw_text += f"{row_header} | {col_header} | {value}\n"
                            table_cells.append(TableCell(
                                row_header   = row_header,
                                col_header   = col_header,
                                value        = value,
                                page         = 0,
                                table_number = 0,
                                section      = sheet_name,
                            ).to_dict())
        except ImportError:
            logger.warning("openpyxl not installed")
        except Exception as exc:
            logger.warning("XLSX error: %s", exc)

        return {
            "raw_text": raw_text, "table_cells": table_cells,
            "heading_positions": [],
            "company_name": "", "doc_type": "", "fiscal_year": "",
        }

    def _ingest_csv(self, file_path: str) -> Dict:
        raw_text    = ""
        table_cells = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows   = list(reader)
            if rows:
                col_headers = rows[0]
                for row_idx, row in enumerate(rows[1:], start=1):
                    row_header = row[0].strip() if row else ""
                    for col_idx, val in enumerate(row[1:], start=1):
                        col_header = (
                            col_headers[col_idx].strip()
                            if col_idx < len(col_headers) else ""
                        )
                        value = val.strip()
                        raw_text += f"{row_header} | {col_header} | {value}\n"
                        if value:
                            table_cells.append(TableCell(
                                row_header   = row_header,
                                col_header   = col_header,
                                value        = value,
                                page         = 0,
                                table_number = row_idx,
                            ).to_dict())
        except Exception as exc:
            logger.warning("CSV error: %s", exc)

        return {
            "raw_text": raw_text, "table_cells": table_cells,
            "heading_positions": [],
            "company_name": "", "doc_type": "", "fiscal_year": "",
        }

    def _ingest_pptx(self, file_path: str) -> Dict:
        raw_text          = ""
        heading_positions = []

        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            for slide_num, slide in enumerate(prs.slides, start=1):
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            text = para.text.strip()
                            if text:
                                raw_text += text + "\n"
                                if para.runs and para.runs[0].font.size:
                                    font_pt = para.runs[0].font.size.pt
                                    if font_pt >= HEADING_FONT_SIZE_MIN:
                                        heading_positions.append({
                                            "text":      text,
                                            "font_size": round(font_pt, 1),
                                            "is_bold":   bool(para.runs[0].font.bold),
                                            "page":      slide_num,
                                        })
        except ImportError:
            logger.warning("python-pptx not installed")
        except Exception as exc:
            logger.warning("PPTX error: %s", exc)

        return {
            "raw_text": raw_text, "table_cells": [],
            "heading_positions": heading_positions,
            "company_name": "", "doc_type": "", "fiscal_year": "",
        }

    # ════════════════════════════════════════════════════════════════════════
    # HTML / iXBRL INGESTION  ── Bugs #2, #2.1, #2.2 ──
    # ════════════════════════════════════════════════════════════════════════

    def _ingest_html(self, file_path: str) -> Dict:
        """Unified HTML / iXBRL ingestor — DUAL-PASS PARSING.

        Pass 1 (lxml-xml): preserve namespaces for ix:nonFraction etc.
        Pass 2 (lxml HTML): proper layout/CSS for <span>/<div> styled headings

        This is the only reliable way to handle SEC iXBRL files where the
        document is XHTML+namespaced-XBRL, and visual headings live inside
        XHTML <span> tags wrapped by <ix:*> tags.
        """
        raw_text          = ""
        table_cells       = []
        heading_positions = []

        try:
            import warnings
            try:
                from bs4 import XMLParsedAsHTMLWarning
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            except ImportError:
                pass

            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            is_ixbrl = (
                "ix:nonFraction" in content
                or "ix:nonNumeric" in content
                or "xmlns:ix=" in content
            )

            # ── PASS 1: XML-aware parser for iXBRL facts + clean text ────
            try:
                soup_xml = BeautifulSoup(content, "lxml-xml")
            except Exception:
                soup_xml = None

            # ── PASS 2: HTML parser for visual layout (styled headings) ─
            try:
                soup_html = BeautifulSoup(content, "lxml")
            except Exception:
                try:
                    soup_html = BeautifulSoup(content, "html.parser")
                except Exception:
                    soup_html = None

            logger.debug(
                "HTML ingest: iXBRL=%s | xml=%s | html=%s",
                is_ixbrl, soup_xml is not None, soup_html is not None,
            )

            # Use HTML soup for raw_text (better text concatenation in HTML mode)
            text_soup = soup_html if soup_html is not None else soup_xml
            if text_soup is not None:
                # Strip noise
                for tag in text_soup(["script", "style", "noscript",
                                      "meta", "link"]):
                    tag.decompose()
                raw_text = text_soup.get_text(separator="\n")
                raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
                raw_text = re.sub(r"[ \t]+", " ", raw_text)

            # ── Step 1: iXBRL facts (XML soup, namespace-aware) ──────────
            if is_ixbrl and soup_xml is not None:
                ixbrl_cells = self._html_extract_ixbrl_facts(soup_xml, raw_text)
                table_cells.extend(ixbrl_cells)
                logger.debug("Extracted %d iXBRL facts", len(ixbrl_cells))

            # ── Step 2: <h1>-<h6> headings (HTML soup) ───────────────────
            if soup_html is not None:
                heading_positions.extend(
                    self._html_extract_headings(soup_html, raw_text)
                )

            # ── Step 3: <b>/<strong> bold-as-heading (HTML soup) ─────────
            if soup_html is not None:
                heading_positions.extend(
                    self._html_extract_bold_headings(soup_html, raw_text)
                )

             # ── Step 4: visual styled headings (HTML soup) ───────────────
            if soup_html is not None:
                heading_positions.extend(
                    self._html_extract_styled_headings(soup_html, raw_text)
                )

            # ── Step 4b: SEC-conventional section markers by TEXT PATTERN
            #          (PART I, Item 1, Item 1A. etc. — these may be in
            #           tiny 9pt spans in real 10-Ks, font-size won't find them)
            if soup_html is not None:
                heading_positions.extend(
                    self._html_extract_sec_section_markers(soup_html, raw_text)
                )

            # ── Step 5: HTML table cells (HTML soup) ─────────────────────
            if soup_html is not None:
                table_cells.extend(
                    self._html_extract_table_cells(soup_html, raw_text)
                )

            # ── Step 6: dedupe headings ───────────────────────────────────
            heading_positions = self._dedupe_headings(heading_positions)
            heading_positions = heading_positions[:MAX_VISUAL_HEADS]

            logger.debug(
                "HTML/iXBRL extracted: %d chars | %d headings | %d cells",
                len(raw_text), len(heading_positions), len(table_cells),
            )

        except ImportError:
            logger.warning("BeautifulSoup not installed — using regex fallback")
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
                raw_text = re.sub(r"<script[^>]*>.*?</script>", " ",
                                  raw_text, flags=re.DOTALL | re.IGNORECASE)
                raw_text = re.sub(r"<style[^>]*>.*?</style>", " ",
                                  raw_text, flags=re.DOTALL | re.IGNORECASE)
                raw_text = re.sub(r"<[^>]+>", " ", raw_text)
                raw_text = re.sub(r"\s+", " ", raw_text)
            except Exception as exc:
                logger.warning("HTML regex fallback error: %s", exc)
        except Exception as exc:
            logger.warning("HTML error: %s", exc)

        company_name, doc_type, fiscal_year = self._extract_metadata(
            raw_text, heading_positions
        )

        # iXBRL files have AUTHORITATIVE metadata — always override regex
        # when iXBRL values are present (regex may match noise like CIK
        # numbers and produce garbage like 'FY0000')
        ixbrl_meta = self._html_extract_ixbrl_metadata(file_path)
        if ixbrl_meta.get("company_name"):
            company_name = ixbrl_meta["company_name"]
        if ixbrl_meta.get("fiscal_year"):
            fiscal_year = ixbrl_meta["fiscal_year"]
        if ixbrl_meta.get("doc_type"):
            doc_type = ixbrl_meta["doc_type"]

        return {
            "raw_text":          raw_text,
            "table_cells":       table_cells,
            "heading_positions": heading_positions,
            "company_name":      company_name,
            "doc_type":          doc_type,
            "fiscal_year":       fiscal_year,
        }

    @staticmethod
    def _estimate_page_from_offset(offset: int) -> int:
        if offset <= 0:
            return 1
        return max(1, (offset // HTML_CHARS_PER_PAGE) + 1)

    def _html_extract_ixbrl_facts(self, soup, raw_text: str) -> List[Dict]:
        """Extract iXBRL <ix:nonFraction> and <ix:nonNumeric> as TableCells.
        Must be called with the XML-mode soup (preserves namespaces)."""
        cells: List[Dict] = []

        nonfractions = (
            soup.find_all("ix:nonFraction") or soup.find_all("nonFraction")
        )
        nonnumerics = (
            soup.find_all("ix:nonNumeric") or soup.find_all("nonNumeric")
        )

        # Numeric facts
        for tag in nonfractions[:MAX_IXBRL_FACTS]:
            name  = (tag.get("name", "") or "").strip()
            value = tag.get_text(strip=True)
            if not name or not value:
                continue

            ctx   = (tag.get("contextRef", "") or "").strip()
            scale = tag.get("scale", "") or ""

            display_value = value
            if scale and scale.lstrip("-").isdigit():
                try:
                    s = int(scale)
                    if s != 0:
                        display_value = f"{value} ×10^{s}"
                except (ValueError, TypeError):
                    pass

            snippet = value[:20]
            page = 1
            if snippet:
                idx = raw_text.find(snippet)
                if idx >= 0:
                    page = self._estimate_page_from_offset(idx)

            cells.append(TableCell(
                row_header   = name,
                col_header   = ctx,
                value        = display_value,
                page         = page,
                table_number = 0,
                section      = "iXBRL_NUMERIC",
            ).to_dict())

        # Text facts
        for tag in nonnumerics[:MAX_IXBRL_FACTS]:
            name  = (tag.get("name", "") or "").strip()
            value = tag.get_text(strip=True)[:200]
            if not name or not value:
                continue
            ctx = (tag.get("contextRef", "") or "").strip()

            cells.append(TableCell(
                row_header   = name,
                col_header   = ctx,
                value        = value,
                page         = 1,
                table_number = 0,
                section      = "iXBRL_TEXT",
            ).to_dict())

        return cells

    def _html_extract_ixbrl_metadata(self, file_path: str) -> Dict[str, str]:
        """Extract authoritative metadata from iXBRL dei: tags.

        Bug #2.2: previously returned empty FY because some files have
        MULTIPLE dei:DocumentFiscalYearFocus tags, the first being empty.
        Now we skip empty-text matches.
        """
        out = {"company_name": "", "doc_type": "", "fiscal_year": ""}
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            soup = BeautifulSoup(content, "lxml-xml")
        except Exception:
            return out

        def fact(name: str) -> str:
            """Return text of FIRST tag with this name AND non-empty content."""
            for tag in soup.find_all(["ix:nonNumeric", "nonNumeric"]):
                if (tag.get("name", "") or "").strip() == name:
                    text = tag.get_text(strip=True)
                    if text:                 # SKIP empty values
                        return text
            return ""

        company = fact("dei:EntityRegistrantName")
        doctype = fact("dei:DocumentType")
        fy_year = fact("dei:DocumentFiscalYearFocus")

        if company:
            out["company_name"] = company
        if doctype:
            out["doc_type"] = doctype
        if fy_year and fy_year.isdigit() and len(fy_year) == 4 and fy_year != "0000":
            out["fiscal_year"] = f"FY{fy_year}"

        return out

    def _html_extract_headings(self, soup, raw_text: str) -> List[Dict]:
        """Extract <h1>-<h6> elements as heading_positions."""
        out = []
        size_by_level = {1: 20.0, 2: 18.0, 3: 16.0, 4: 14.0, 5: 13.0, 6: 13.0}

        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                text = tag.get_text(separator=" ", strip=True)
                if not text or len(text) < 3 or len(text) > 200:
                    continue
                page = 1
                snippet = text[:30]
                idx = raw_text.find(snippet)
                if idx >= 0:
                    page = self._estimate_page_from_offset(idx)
                out.append({
                    "text":      text,
                    "font_size": size_by_level[level],
                    "is_bold":   True,
                    "page":      page,
                })
        return out

    def _html_extract_bold_headings(self, soup, raw_text: str) -> List[Dict]:
        """Extract <b>/<strong> short text as candidate headings."""
        out = []
        for tag in soup.find_all(["b", "strong"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue
            wc = len(text.split())
            if len(text) < 3 or len(text) > 120 or wc > 12:
                continue
            if re.match(r"^[\d\s,.\-$()]+$", text):
                continue
            page = 1
            snippet = text[:30]
            idx = raw_text.find(snippet)
            if idx >= 0:
                page = self._estimate_page_from_offset(idx)
            out.append({
                "text":      text,
                "font_size": 13.5,
                "is_bold":   True,
                "page":      page,
            })
        return out

    def _html_extract_styled_headings(
        self, soup, raw_text: str
    ) -> List[Dict]:
        """Extract elements with inline font-size + font-weight CSS as headings.

        Bug #2.2 fix: must be called with HTML-mode soup (lxml), not XML soup,
        because real iXBRL files wrap visual <span>/<div> tags in <ix:*> XML
        namespaces that hide them from XML mode tag-name searches.
        """
        out = []
        candidates = soup.find_all(["span", "div", "p", "td"])

        for elem in candidates:
            style = (elem.get("style", "") or "").lower()
            if not style:
                continue

            size_match = re.search(r"font-size\s*:\s*([\d.]+)\s*pt", style)
            if not size_match:
                continue
            try:
                font_pt = float(size_match.group(1))
            except (ValueError, TypeError):
                continue

            if font_pt < HTML_HEADING_FONT_PT:
                continue

            weight_match = re.search(r"font-weight\s*:\s*([\w\d]+)", style)
            is_bold = False
            if weight_match:
                w = weight_match.group(1).lower()
                if w in ("bold", "bolder"):
                    is_bold = True
                else:
                    try:
                        if int(w) >= HTML_BOLD_WEIGHT_MIN:
                            is_bold = True
                    except ValueError:
                        pass

            # Heading must be either: bold, OR very large font (>=14pt)
            if not is_bold and font_pt < 14.0:
                continue

            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue

            wc = len(text.split())
            if len(text) < 3 or len(text) > 200 or wc > 25:
                continue
            if re.match(r"^[\d\s,.\-$()%\\]+$", text):
                continue
            if text.islower() and wc > 3:
                continue

            page = 1
            snippet = text[:30]
            idx = raw_text.find(snippet)
            if idx >= 0:
                page = self._estimate_page_from_offset(idx)

            out.append({
                "text":      text,
                "font_size": font_pt,
                "is_bold":   is_bold,
                "page":      page,
            })

        return out
    # SEC section pattern: PART I/II/III/IV, Item 1/1A/2/.../15/16
    _SEC_SECTION_RX = re.compile(
        r"^(?:"
        r"PART\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X)"  # PART I-X
        r"|Item\s+\d{1,2}[A-Z]?\."                     # Item 1., Item 1A., etc.
        r")\s*$",
        re.IGNORECASE,
    )

    def _html_extract_sec_section_markers(
        self, soup, raw_text: str
    ) -> List[Dict]:
        """Find SEC-conventional section markers by TEXT PATTERN.

        Real SEC 10-Ks often put 'PART I' and 'Item 1.' in tiny 9pt spans
        that font-size detection misses. We catch these by matching the
        text pattern: PART [roman] or Item N[letter].
        """
        out = []
        seen = set()
        # Search both <span> (typical body) and <div> (typical wrapper)
        for tag in soup.find_all(["span", "div", "p", "a", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text or len(text) > 30:
                continue
            if not self._SEC_SECTION_RX.match(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            page = 1
            idx = raw_text.find(text)
            if idx >= 0:
                page = self._estimate_page_from_offset(idx)

            out.append({
                "text":      text,
                "font_size": 14.0,   # treat as heading-equivalent for chunker
                "is_bold":   True,
                "page":      page,
            })
        return out

    def _html_extract_table_cells(self, soup, raw_text: str) -> List[Dict]:
        """Extract every <td>/<th> cell with row+column headers from <table>."""
        cells: List[Dict] = []

        for t_num, table in enumerate(soup.find_all("table"), start=1):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            page = 1
            first_row_text = (
                rows[0].get_text(" ", strip=True) if rows else ""
            )
            if first_row_text:
                idx = raw_text.find(first_row_text[:50])
                if idx >= 0:
                    page = self._estimate_page_from_offset(idx)

            header_row = None
            for r in rows:
                ths = r.find_all("th")
                if ths:
                    header_row = r
                    break
            if header_row is None:
                header_row = rows[0]

            col_headers = [
                c.get_text(" ", strip=True)
                for c in header_row.find_all(["th", "td"])
            ]

            try:
                start_idx = rows.index(header_row) + 1
            except ValueError:
                start_idx = 1
            data_rows = rows[start_idx:]

            for row in data_rows:
                row_cells = row.find_all(["th", "td"])
                if not row_cells:
                    continue

                row_header = row_cells[0].get_text(" ", strip=True)
                for col_idx, c in enumerate(row_cells[1:], start=1):
                    value = c.get_text(" ", strip=True)
                    if not value or value in ("—", "-", "$", "(", ")"):
                        continue
                    col_header = (
                        col_headers[col_idx]
                        if col_idx < len(col_headers) else ""
                    )
                    cells.append(TableCell(
                        row_header   = row_header,
                        col_header   = col_header,
                        value        = value,
                        page         = page,
                        table_number = t_num,
                    ).to_dict())

        return cells

    @staticmethod
    def _dedupe_headings(headings: List[Dict]) -> List[Dict]:
        """Deduplicate by text (case-insensitive). Keep first occurrence."""
        seen = set()
        out  = []
        for h in headings:
            key = (h.get("text", "") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out

    def _ingest_txt(self, file_path: str) -> Dict:
        raw_text = ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except Exception as exc:
            logger.warning("TXT error: %s", exc)
        return {
            "raw_text": raw_text, "table_cells": [],
            "heading_positions": [],
            "company_name": "", "doc_type": "", "fiscal_year": "",
        }

    def _ingest_json(self, file_path: str) -> Dict:
        raw_text = ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data     = json.load(f)
            raw_text = json.dumps(data, indent=2)
        except Exception as exc:
            logger.warning("JSON error: %s", exc)
        return {
            "raw_text": raw_text, "table_cells": [],
            "heading_positions": [],
            "company_name": "", "doc_type": "", "fiscal_year": "",
        }

    def _extract_metadata(
        self,
        raw_text:          str,
        heading_positions: List[Dict],
    ) -> Tuple[str, str, str]:
        company_name = self._extract_company(raw_text)
        doc_type     = self._extract_doc_type(raw_text)
        fiscal_year  = self._extract_fiscal_year(raw_text)
        return company_name, doc_type, fiscal_year

    @staticmethod
    def _extract_company(text: str) -> str:
        snippet  = text[:2000]
        patterns = [
            r"((?:[A-Z][a-z]+\s*){1,4}(?:Inc|Corp|Ltd|LLC|Co|Company|Group|Holdings)\.?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, snippet)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_doc_type(text: str) -> str:
        snippet = text[:3000].upper()
        for dtype in ["10-K", "10-Q", "8-K", "S-1", "20-F", "6-K", "DEF 14A"]:
            if dtype.replace("-", "") in snippet.replace("-", "").replace(" ", ""):
                return dtype
        return "UNKNOWN"

    @staticmethod
    def _extract_fiscal_year(text: str) -> str:
        patterns = [
            r"[Ff]iscal [Yy]ear\s*(?:ended|ending)?\s*(?:(?:September|December|March|June)\s+\d{1,2},?\s*)?(\d{4})",
            r"[Ff][Yy]\s*(\d{4})",
            r"[Yy]ear [Ee]nded\s+\w+\s+\d{1,2},?\s*(\d{4})",
            r"[Aa]nnual [Rr]eport\s+(\d{4})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text[:5000])
            if m:
                return f"FY{m.group(1)}"
        return ""


def run_pdf_ingestor(state, enable_images: bool = False, llm_client=None) -> object:
    """Convenience wrapper for LangGraph N01 node."""
    return PDFIngestor(
        enable_images = enable_images,
        llm_client    = llm_client,
    ).run(state)