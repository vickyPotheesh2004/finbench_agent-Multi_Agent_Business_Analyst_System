"""
src/utils/query_classifier.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Regex-based query classifier. Returns one of the CANONICAL BAState query
types directly — no translation layer required.

Canonical types (must match BAState.normalise_query_type VALID set):
  - "numerical"  — extract a specific number / metric
  - "ratio"      — compute or look up a ratio / margin / percentage
  - "multi_doc"  — compare across documents or companies
  - "text"       — narrative / qualitative explanation
  - "forensic"   — fraud / anomaly / Benford / suspicious patterns

2026-06-05: renamed constants from "numeric"/"narrative" to canonical
"numerical"/"text" so pipeline.py no longer needs QUERY_TYPE_MAP.
"""
from __future__ import annotations


class QueryType:
    """Canonical query type constants. Match BAState validator."""

    NUMERICAL = "numerical"   # was NUMERIC
    RATIO     = "ratio"
    MULTI_DOC = "multi_doc"   # new
    TEXT      = "text"        # was NARRATIVE
    FORENSIC  = "forensic"


def classify_query(query: str) -> str:
    """
    Classify a query into one of the canonical BAState query types.

    Order of checks matters — forensic and multi-doc patterns are
    checked first because they are more specific than the generic
    ratio / numerical / text catch-alls.
    """
    if not query:
        return QueryType.TEXT

    q = query.lower()

    # ── forensic (most specific) ──────────────────────────────────────────
    forensic_terms = (
        "fraud",
        "anomaly",
        "anomalous",
        "suspicious",
        "benford",
        "irregular",
        "manipulation",
        "earnings management",
        "restatement",
    )
    if any(t in q for t in forensic_terms):
        return QueryType.FORENSIC

    # ── multi-doc / comparison ────────────────────────────────────────────
    multi_doc_terms = (
        "compare",
        "comparison",
        "versus",
        " vs ",
        " vs. ",
        "across companies",
        "between companies",
        "peers",
        "competitor",
    )
    if any(t in q for t in multi_doc_terms):
        return QueryType.MULTI_DOC

    # ── ratio / percentage ────────────────────────────────────────────────
    ratio_terms = (
        "margin",
        "ratio",
        "percentage",
        "percent ",
        " % ",
        "growth",
        "yoy",
        "year-over-year",
        "return on",
        "turnover",
        "leverage",
    )
    if any(t in q for t in ratio_terms):
        return QueryType.RATIO

    # ── numerical extraction ──────────────────────────────────────────────
    numeric_terms = (
        "how much",
        "what was",
        "what is",
        "amount",
        "value of",
        "revenue",
        "income",
        "cash",
        "eps",
        "earnings per share",
        "capex",
        "capital expenditure",
        "ebitda",
        "ebit",
        "gross profit",
        "total assets",
        "total liabilities",
        "long-term debt",
        "shareholders equity",
        "stockholders equity",
        "dividends",
    )
    if any(t in q for t in numeric_terms):
        return QueryType.NUMERICAL

    # ── fallback: narrative ───────────────────────────────────────────────
    return QueryType.TEXT
