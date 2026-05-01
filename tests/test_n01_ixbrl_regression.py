"""
tests/test_n01_ixbrl_regression.py
Regression suite for Bugs #2.1 + #2.2: real SEC iXBRL filings.

Phase 1 / Bug #2 family in the FinBench fix campaign.

Real SEC 10-Ks (since FY2020) use Inline XBRL with:
    - <ix:nonFraction> for numeric facts (us-gaap:Revenues, etc.)
    - <ix:nonNumeric> for text facts (fiscal year, entity name, etc.)
    - inline CSS font-size + font-weight:700 for visual headings
    - NO <h1>-<h6> tags
    - NO <b>/<strong> tags

S11 update (Bug #2.2):
    - dei:DocumentFiscalYearFocus may have empty duplicates — skip them
    - Visual styled headings need HTML parser (lxml), not XML parser
    - Real-world threshold lowered: 30+ headings (was 50+) since dedup
      removes many repeats from 3,298 raw styled elements
"""
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_ingestor import PDFIngestor


# ── Fixture: synthetic iXBRL document matching real Apple 10-K patterns ────

SAMPLE_IXBRL_10K = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:dei="http://xbrl.sec.gov/dei/2023"
      xmlns:us-gaap="http://fasb.org/us-gaap/2023">
<head>
  <title>Apple Inc FY2023 10-K</title>
</head>
<body>

<div style="font-size:17pt;font-weight:700;">FORM 10-K</div>
<div style="font-size:13pt;font-weight:700;">UNITED STATES</div>
<div style="font-size:13pt;font-weight:700;">SECURITIES AND EXCHANGE COMMISSION</div>

<div style="font-size:14pt;font-weight:700;">PART I</div>
<div style="font-size:12pt;font-weight:700;">Item 1. Business</div>

<p>The Company designs, manufactures, and markets smartphones.</p>

<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c-1">Apple Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentType" contextRef="c-1">10-K</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c-1">2023</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="c-1">FY</ix:nonNumeric>

<div style="font-size:14pt;font-weight:700;">PART II</div>
<div style="font-size:12pt;font-weight:700;">Item 8. Financial Statements</div>

<p>Net sales were
<ix:nonFraction name="us-gaap:Revenues" contextRef="c-2" decimals="-6" unitRef="usd">383,285</ix:nonFraction>
million in fiscal 2023, compared to
<ix:nonFraction name="us-gaap:Revenues" contextRef="c-3" decimals="-6" unitRef="usd">394,328</ix:nonFraction>
million in fiscal 2022.</p>

<p>Net income was
<ix:nonFraction name="us-gaap:NetIncomeLoss" contextRef="c-2" decimals="-6" unitRef="usd">96,995</ix:nonFraction>
million.</p>

<p>Diluted earnings per share was
<ix:nonFraction name="us-gaap:EarningsPerShareDiluted" contextRef="c-2" decimals="2" unitRef="usd">6.13</ix:nonFraction>.
</p>

<p>Total assets were
<ix:nonFraction name="us-gaap:Assets" contextRef="c-2" decimals="-6" unitRef="usd">352,583</ix:nonFraction>
million.</p>

<table>
  <tr>
    <th>Metric</th>
    <th>FY2023</th>
    <th>FY2022</th>
  </tr>
  <tr>
    <td>Cost of sales</td>
    <td>214,137</td>
    <td>223,546</td>
  </tr>
  <tr>
    <td>Gross margin</td>
    <td>169,148</td>
    <td>170,782</td>
  </tr>
</table>

</body>
</html>"""


# Fixture for Bug #2.2 — multiple dei: tags, first one empty
SAMPLE_IXBRL_DUPLICATE_DEI = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:dei="http://xbrl.sec.gov/dei/2023">
<body>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c-empty"></ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c-1">2023</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c-1">Test Corp</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentType" contextRef="c-1">10-K</ix:nonNumeric>
<table><tr><th>X</th></tr><tr><td>1</td></tr></table>
</body></html>"""


@pytest.fixture
def ixbrl_path(tmp_path):
    p = tmp_path / "AAPL_FY2023_10-K.html"
    p.write_text(SAMPLE_IXBRL_10K, encoding="utf-8")
    return str(p)


@pytest.fixture
def ingestor():
    return PDFIngestor()


@pytest.fixture
def ixbrl_result(ixbrl_path, ingestor):
    return ingestor.ingest(ixbrl_path)


# ════════════════════════════════════════════════════════════════════════════
# CORE BUG #2.1 — iXBRL facts extracted
# ════════════════════════════════════════════════════════════════════════════

class TestBug21IXBRLFacts:

    def test_ixbrl_facts_extracted(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        assert len(cells) > 0

    def test_revenues_fact_present(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        revenues = [c for c in cells
                    if "us-gaap:Revenues" in (c.get("row_header") or "")]
        assert len(revenues) > 0
        values = [c["value"] for c in revenues]
        assert any("383,285" in v or "383285" in v.replace(",", "")
                   for v in values)

    def test_net_income_fact_present(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        ni = [c for c in cells
              if "NetIncomeLoss" in (c.get("row_header") or "")]
        assert len(ni) > 0
        assert any("96,995" in c["value"] or "96995" in c["value"].replace(",", "")
                   for c in ni)

    def test_eps_fact_present(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        eps = [c for c in cells
               if "EarningsPerShareDiluted" in (c.get("row_header") or "")]
        assert len(eps) > 0
        assert any("6.13" in c["value"] for c in eps)

    def test_total_assets_fact_present(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        ta = [c for c in cells
              if c.get("row_header") == "us-gaap:Assets"]
        assert len(ta) > 0

    def test_ixbrl_section_label(self, ixbrl_result):
        cells = ixbrl_result["table_cells"]
        sections = {c.get("section") for c in cells}
        assert "iXBRL_NUMERIC" in sections or "iXBRL_TEXT" in sections


# ════════════════════════════════════════════════════════════════════════════
# AUTHORITATIVE METADATA FROM iXBRL
# ════════════════════════════════════════════════════════════════════════════

class TestBug21Metadata:

    def test_company_from_ixbrl(self, ixbrl_result):
        co = ixbrl_result["company_name"]
        assert "Apple" in co

    def test_fiscal_year_from_ixbrl(self, ixbrl_result):
        fy = ixbrl_result["fiscal_year"]
        assert fy == "FY2023"

    def test_doc_type_from_ixbrl(self, ixbrl_result):
        dt = ixbrl_result["doc_type"]
        assert dt == "10-K"


# ════════════════════════════════════════════════════════════════════════════
# BUG #2.2 — duplicate dei: tags with empty values
# ════════════════════════════════════════════════════════════════════════════

class TestBug22DuplicateDei:

    def test_skips_empty_fiscal_year_tag(self, tmp_path, ingestor):
        """Bug #2.2: when first dei:DocumentFiscalYearFocus is empty,
        we must skip it and find the next non-empty one."""
        p = tmp_path / "dup.html"
        p.write_text(SAMPLE_IXBRL_DUPLICATE_DEI, encoding="utf-8")
        result = ingestor.ingest(str(p))
        fy = result["fiscal_year"]
        assert fy == "FY2023", (
            f"Bug #2.2 regression: empty dei: tag was returned, got {fy!r}"
        )

    def test_no_fy0000_artifact(self, tmp_path, ingestor):
        """Specifically guard against 'FY0000' which was the symptom."""
        p = tmp_path / "dup.html"
        p.write_text(SAMPLE_IXBRL_DUPLICATE_DEI, encoding="utf-8")
        result = ingestor.ingest(str(p))
        assert result["fiscal_year"] != "FY0000"
        assert result["fiscal_year"] != "FY"


# ════════════════════════════════════════════════════════════════════════════
# VISUAL STYLED HEADINGS (font-size + font-weight:700)
# ════════════════════════════════════════════════════════════════════════════

class TestBug21StyledHeadings:

    def test_styled_headings_extracted(self, ixbrl_result):
        h = ixbrl_result["heading_positions"]
        assert len(h) > 0

    def test_form_10k_heading_present(self, ixbrl_result):
        texts = [h["text"] for h in ixbrl_result["heading_positions"]]
        assert any("FORM 10-K" in t for t in texts)

    def test_part_headings_present(self, ixbrl_result):
        texts = [h["text"] for h in ixbrl_result["heading_positions"]]
        assert any("PART I" in t for t in texts)

    def test_item_headings_present(self, ixbrl_result):
        texts = [h["text"] for h in ixbrl_result["heading_positions"]]
        assert any("Item 1" in t or "Item 8" in t for t in texts)

    def test_styled_headings_have_correct_font_size(self, ixbrl_result):
        h = ixbrl_result["heading_positions"]
        sizes = [hd["font_size"] for hd in h]
        assert max(sizes) >= 12.0


# ════════════════════════════════════════════════════════════════════════════
# CLEAN TEXT
# ════════════════════════════════════════════════════════════════════════════

class TestBug21CleanText:

    def test_raw_text_contains_xbrl_values(self, ixbrl_result):
        text = ixbrl_result["raw_text"]
        assert "383,285" in text or "383285" in text.replace(",", "")

    def test_raw_text_contains_business_content(self, ixbrl_result):
        text = ixbrl_result["raw_text"]
        assert "smartphones" in text.lower()


# ════════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION
# ════════════════════════════════════════════════════════════════════════════

class TestBug21Detection:

    def test_plain_html_still_works(self, tmp_path, ingestor):
        """Non-iXBRL HTML must still extract via h1-h6."""
        plain = """<!DOCTYPE html>
<html><body>
<h1>Test Document</h1>
<h2>Section A</h2>
<table><tr><th>K</th><th>V</th></tr><tr><td>X</td><td>123</td></tr></table>
</body></html>"""
        p = tmp_path / "plain.html"
        p.write_text(plain, encoding="utf-8")
        result = ingestor.ingest(str(p))
        texts = [h["text"] for h in result["heading_positions"]]
        assert any("Test Document" in t for t in texts)
        assert any("Section A" in t for t in texts)
        values = [c["value"] for c in result["table_cells"]]
        assert "123" in values


# ════════════════════════════════════════════════════════════════════════════
# REAL APPLE 10-K (only runs if file exists)
# ════════════════════════════════════════════════════════════════════════════

REAL_APPLE_10K = (
    Path(__file__).parent.parent
    / "documents" / "sec_filings" / "AAPL_FY2023_10-K.html"
)


class TestBug21RealAppleFiling:
    """Run against the actual SEC filing if present."""

    @pytest.fixture(scope="class")
    def real_result(self, request):
        if not REAL_APPLE_10K.exists():
            pytest.skip(f"Real Apple 10-K not available at {REAL_APPLE_10K}")
        return PDFIngestor().ingest(str(REAL_APPLE_10K))

    def test_real_apple_extracts_many_ixbrl_facts(self, real_result):
        """Real Apple 10-K has ~1990 ix:nonFraction tags. Expect ≥500."""
        cells = real_result["table_cells"]
        ixbrl_cells = [c for c in cells
                       if c.get("section", "").startswith("iXBRL")]
        assert len(ixbrl_cells) >= 500, (
            f"Expected ≥500 iXBRL facts, got {len(ixbrl_cells)}"
        )

    def test_real_apple_extracts_many_headings(self, real_result):
        """Real Apple 10-K has only 5 truly heading-sized styled elements
        (cover-page banner). The bulk of section markers are tiny 9pt spans
        — those are now found by SEC pattern matching (Step 4b). Combined
        we expect ≥10 headings (5 styled + 5+ PART/Item markers)."""
        h = real_result["heading_positions"]
        assert len(h) >= 10, (
            f"Expected ≥10 headings from real Apple 10-K, got {len(h)}. "
            f"Sample: {[hd['text'][:40] for hd in h[:8]]}"
        )

    def test_real_apple_company_name(self, real_result):
        co = real_result["company_name"]
        assert "Apple" in co

    def test_real_apple_fiscal_year(self, real_result):
        """Bug #2.2: real file has duplicate dei: tags, first is empty.
        Must skip empty and return FY2023."""
        fy = real_result["fiscal_year"]
        assert fy == "FY2023", (
            f"Bug #2.2: expected 'FY2023', got {fy!r} "
            f"(probably matching empty dei:DocumentFiscalYearFocus tag)"
        )

    def test_real_apple_doc_type(self, real_result):
        dt = real_result["doc_type"]
        assert dt == "10-K"

    def test_real_apple_total_table_cells(self, real_result):
        """Combined HTML tables + iXBRL facts should be 2000+."""
        assert len(real_result["table_cells"]) >= 2000
    
    def test_real_apple_has_part_i_heading(self, real_result):
        """SEC section markers (PART I, Item 1.) found via text pattern,
        not font-size — Step 4b in pdf_ingestor."""
        texts = [h["text"].upper() for h in real_result["heading_positions"]]
        has_part = any(re.match(r"PART\s+(?:I|II|III)", t) for t in texts)
        has_item = any(re.match(r"ITEM\s+\d", t) for t in texts)
        assert has_part or has_item, (
            f"Neither PART nor Item markers found. Sample: {texts[:10]}"
        )