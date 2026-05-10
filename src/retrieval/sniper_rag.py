"""
N06 SniperRAG — Tier 1 Direct Table Cell Extraction
PDR-BAAAI-001 · Rev 1.0 · Node N06

CHANGELOG:
  2026-05-03 S17  Bug A3: index iXBRL row_headers by humanized name + alias.
  2026-05-10 S22  Bug A/B/C: us-gaap tag preference scoring.
  2026-05-10 S23  Bug A/B/C v2: pattern reorder, softened penalty.
  2026-05-10 S24  Bug A/B/C v3: tie-breaker by largest value.
  2026-05-10 S25  Bug A/B/C v4: FY filtering before tie-breaker.
  2026-05-10 S26  Bug A/B/C v5: iXBRL CONTEXT-AWARE retrieval.
  2026-05-10 S27  Bug A/B/C v6:
                  - STRICT-MATCH-ONLY mode for ambiguous tags (e.g.
                    us-gaap:liabilities must NOT match
                    us-gaap:liabilitiesandstockholdersequity).
                  - SYNTHETIC METRIC computation for non-GAAP metrics
                    like free_cash_flow (= OCF − CapEx).
                  - Graceful fallback when context detection fails on
                    non-iXBRL data (preserves test compatibility).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    row_header:    str
    col_header:    str
    value:         str
    unit:          str
    page:          int
    section:       str
    company:       str
    doc_type:      str
    fiscal_year:   str
    numeric_value: Optional[float] = None

    @property
    def metadata_key(self) -> str:
        """C8: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE"""
        return (
            f"{self.company}/{self.doc_type}/{self.fiscal_year}"
            f"/{self.section}/{self.page}"
        )


@dataclass
class SniperResult:
    sniper_hit:      bool
    answer:          str
    value:           str
    unit:            str
    confidence:      float
    matched_pattern: str
    cell:            Optional[TableCell]
    citation:        str
    reason:          str


# ── Compiled regex patterns (ORDER MATTERS — most specific first) ────────────
RAW_PATTERNS: Dict[str, str] = {
    "operating_cash_flow": r"(?:net\s+)?cash\s+(?:provided\s+by\s+|from\s+)?operating\s+activities|operating\s+cash\s+flow",
    "free_cash_flow":      r"free\s+cash\s+flow|fcf",
    "long_term_debt":      r"long.?term\s+(?:debt|notes?\s+payable|borrowings?|obligations?)",
    "deferred_revenue":    r"deferred\s+revenue(?:s)?|unearned\s+revenue",
    "share_repurchase":    r"(?:repurchases?|buybacks?)\s+(?:of\s+)?(?:common\s+)?(?:stock|shares?)|treasury\s+stock\s+purchases?",
    "dividends_paid":      r"dividends?\s+paid|cash\s+dividends?",
    "capex":               r"capital\s+expenditures?|capex|purchases?\s+of\s+property",
    "interest_expense":    r"interest\s+(?:expense|cost|charges?)|finance\s+(?:cost|charge)",
    "income_tax":          r"(?:provision\s+for\s+)?income\s+tax(?:es)?|tax\s+(?:expense|provision)",
    "operating_income":    r"(?:(?:total\s+)?operating\s+(?:income|loss|profit))|(?:income\s+(?:loss\s+)?from\s+operations)",
    "gross_profit":        r"gross\s+(?:profit|margin|income)",
    "ebitda":              r"(?:adjusted\s+)?ebitda|earnings\s+before\s+interest[,\s]+(?:taxes[,\s]+)?depreciation",
    "ebit":                r"\bebit\b|earnings\s+before\s+interest\s+and\s+taxes",
    "eps_diluted":         r"(?:diluted\s+)?(?:earnings|loss)\s+per\s+(?:diluted\s+)?(?:common\s+)?share|diluted\s+eps|eps\s+diluted",
    "eps_basic":           r"basic\s+(?:earnings|loss)\s+per\s+(?:common\s+)?share|basic\s+eps",
    "r_and_d":             r"research\s+(?:and|&)\s+development(?:\s+(?:expense|cost))?",
    "sg_and_a":            r"(?:selling[,\s]+)?general\s+(?:and\s+)?administrative(?:\s+(?:expense|cost))?|sg(?:\s*[&and]+\s*)?a",
    "cogs":                r"cost\s+of\s+(?:goods?\s+)?(?:revenue|sales?|products?|services?)|cost\s+of\s+revenues?",
    "current_assets":      r"(?:total\s+)?current\s+assets",
    "current_liabilities": r"(?:total\s+)?current\s+liabilities",
    "total_assets":        r"total\s+assets",
    "total_liabilities":   r"total\s+(?:liabilities(?:\s+and)?|debt)",
    "shareholders_equity": r"(?:total\s+)?(?:stockholders?|shareholders?)\s+(?:equity|deficit)",
    "accounts_receivable": r"(?:net\s+)?accounts?\s+receivable|trade\s+receivables?",
    "inventory":           r"(?:total\s+)?inventor(?:y|ies)",
    "goodwill":            r"goodwill(?:\s+impairment)?",
    "net_income":          r"net\s+(?:income|earnings|loss)(?:\s+attributable)?(?:\s+to\s+(?:common\s+)?shareholders?)?",
    "revenue":             r"(?:total\s+)?(?:net\s+)?(?:revenues?|net\s+sales?|total\s+sales?)",
    "cash":                r"cash\s+(?:and\s+(?:cash\s+)?equivalents?)?|cash\s+and\s+short.?term\s+investments?",
}

COMPILED_PATTERNS: Dict[str, re.Pattern] = {
    name: re.compile(pat, re.IGNORECASE)
    for name, pat in RAW_PATTERNS.items()
}


# ── Pattern → canonical iXBRL tags ────────────────────────────────────────────
_PATTERN_TO_IXBRL_TAGS: Dict[str, List[str]] = {
    "revenue":             ["us-gaap:revenues", "us-gaap:salesrevenuenet",
                            "us-gaap:revenuefromcontractwithcustomerexcludingassessedtax"],
    "cogs":                ["us-gaap:costofrevenue", "us-gaap:costofgoodssold",
                            "us-gaap:costofgoodsandservicessold"],
    "gross_profit":        ["us-gaap:grossprofit"],
    "operating_income":    ["us-gaap:operatingincomeloss"],
    "net_income":          ["us-gaap:netincomeloss"],
    "eps_diluted":         ["us-gaap:earningspersharediluted"],
    "eps_basic":           ["us-gaap:earningspersharebasic"],
    "operating_cash_flow": ["us-gaap:netcashprovidedbyusedinoperatingactivities",
                            "us-gaap:netcashprovidedbyoperatingactivities"],
    "long_term_debt":      ["us-gaap:longtermdebtnoncurrent"],
    "deferred_revenue":    ["us-gaap:deferredrevenue", "us-gaap:contractwithcustomerliability"],
    "total_assets":        ["us-gaap:assets"],
    "total_liabilities":   ["us-gaap:liabilities"],
    "shareholders_equity": ["us-gaap:stockholdersequity"],
    "current_assets":      ["us-gaap:assetscurrent"],
    "current_liabilities": ["us-gaap:liabilitiescurrent"],
    "cash":                ["us-gaap:cashandcashequivalentsatcarryingvalue",
                            "us-gaap:cashandcashequivalents"],
    "interest_expense":    ["us-gaap:interestexpense"],
    "income_tax":          ["us-gaap:incometaxexpensebenefit"],
    "r_and_d":             ["us-gaap:researchanddevelopmentexpense"],
    "sg_and_a":            ["us-gaap:sellinggeneralandadministrativeexpense"],
    "capex":               ["us-gaap:paymentstoacquireproductiveassets",
                            "us-gaap:paymentstoacquirepropertyplantandequipment"],
    "dividends_paid":      ["us-gaap:paymentsofdividends",
                            "us-gaap:paymentsofdividendscommonstock"],
    "share_repurchase":    ["us-gaap:paymentsforrepurchaseofcommonstock"],
    "accounts_receivable": ["us-gaap:accountsreceivablenetcurrent"],
    "inventory":           ["us-gaap:inventorynet"],
    "goodwill":            ["us-gaap:goodwill"],
}


# ── STRICT-MATCH patterns (Bug A/B/C v6) ─────────────────────────────────────
# For these patterns, the row_header MUST EQUAL the canonical tag (after
# normalization). Substring match would pull in wrong cells like
# us-gaap:liabilitiesandstockholdersequity matching us-gaap:liabilities.
_STRICT_MATCH_PATTERNS: Set[str] = {
    "total_assets",        # Assets ≠ AssetsCurrent ≠ AssetsNoncurrent
    "total_liabilities",   # Liabilities ≠ LiabilitiesAndStockholdersEquity
    "shareholders_equity", # StockholdersEquity ≠ StockholdersEquityIncluding...
    "current_assets",
    "current_liabilities",
    "long_term_debt",      # LongTermDebtNoncurrent only, not LongTermDebt
    "income_tax",          # IncomeTaxExpenseBenefit only
    "cash",                # CashAndCashEquivalentsAtCarryingValue
}


# ── Categorize patterns by statement type ────────────────────────────────────
_INCOME_FLOW_PATTERNS = {
    "revenue", "cogs", "gross_profit", "operating_income", "net_income",
    "eps_diluted", "eps_basic", "operating_cash_flow", "free_cash_flow",
    "interest_expense", "income_tax", "r_and_d", "sg_and_a",
    "capex", "dividends_paid", "share_repurchase", "ebitda", "ebit",
}
_BALANCE_SHEET_PATTERNS = {
    "total_assets", "total_liabilities", "shareholders_equity",
    "current_assets", "current_liabilities", "cash",
    "long_term_debt", "deferred_revenue", "accounts_receivable",
    "inventory", "goodwill",
}


# ── Synthetic metrics (computed from primitives) ─────────────────────────────
# Some metrics are non-GAAP (no iXBRL tag) — derive them at query time.
# Each entry: (formula_description, list of required pattern dependencies, fn)
# fn takes a dict of {pattern_name: SniperResult} and returns float|None
def _compute_free_cash_flow(deps: Dict[str, "SniperResult"]) -> Optional[float]:
    """FCF = OCF - CapEx"""
    ocf = deps.get("operating_cash_flow")
    capex = deps.get("capex")
    if ocf and ocf.cell and ocf.cell.numeric_value is not None and \
       capex and capex.cell and capex.cell.numeric_value is not None:
        # CapEx is usually reported as positive payment (a use of cash)
        # FCF = OCF - |CapEx|
        return float(ocf.cell.numeric_value) - abs(float(capex.cell.numeric_value))
    return None


_SYNTHETIC_METRICS: Dict[str, Tuple[str, List[str], Callable]] = {
    "free_cash_flow": (
        "OCF - CapEx",
        ["operating_cash_flow", "capex"],
        _compute_free_cash_flow,
    ),
}


_FY_PATTERNS = [
    re.compile(r"\bfy\s*(\d{4})\b",                 re.IGNORECASE),
    re.compile(r"\bfiscal\s+year\s+(\d{4})\b",      re.IGNORECASE),
    re.compile(r"\b(?:in|for|at\s+end\s+of)\s+(?:fy\s*)?(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\b"),
    re.compile(r"\bfy\s*'?(\d{2,4})\b",             re.IGNORECASE),
]

_UNIT_PATTERNS: Dict[str, re.Pattern] = {
    "billions":  re.compile(r"\bbillion[s]?\b|\bbn\b", re.IGNORECASE),
    "millions":  re.compile(r"\bmillion[s]?\b|\bmn\b|\bm\b(?!\w)", re.IGNORECASE),
    "thousands": re.compile(r"\bthousand[s]?\b|\bk\b(?!\w)", re.IGNORECASE),
    "%":         re.compile(r"\bpercent(?:age)?\b|%"),
}

_CONF_EXACT      = 0.98
_CONF_PREFIX     = 0.92
_CONF_CONTAINS   = 0.85
_CONF_UNIT_BONUS = 0.02
_CONF_CONTEXT_BONUS = 0.10
_HIT_THRESHOLD   = 0.95


# ── Helper functions ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u2019\u2018\u201c\u201d]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_tag(tag: str) -> str:
    """Normalize tag for comparison: lowercase, no punctuation/spaces."""
    return tag.lower().replace(":", "").replace("_", "").replace(" ", "")


def _parse_numeric(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    s = value_str.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True
    s = re.sub(r"[$,\s]", "", s)
    s = s.rstrip("%")
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _extract_fy_from_query(query: str) -> Optional[str]:
    norm = _normalise(query)
    for pat in _FY_PATTERNS:
        m = pat.search(norm)
        if m:
            year_str = m.group(1) if m.lastindex >= 1 else m.group(0)
            if len(year_str) == 2:
                year_str = f"20{year_str}"
            return f"FY{year_str}"
    return None


def _detect_unit_from_context(text: str) -> str:
    for unit_name, pat in _UNIT_PATTERNS.items():
        if pat.search(text):
            return unit_name
    return "units"


def _humanize_ixbrl(name: str) -> str:
    if not name:
        return ""
    if ":" in name:
        name = name.split(":", 1)[1]
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return spaced.lower().strip()


# ── Pattern preferences ──────────────────────────────────────────────────────
_PATTERN_PREFERENCES: Dict[str, Dict[str, List[str]]] = {
    "revenue": {
        "prefer": ["us-gaap:revenues", "us-gaap:salesrevenuenet",
                   "us-gaap:revenuefromcontractwithcustomerexcludingassessedtax"],
        "avoid":  ["deferredrevenue", "unearnedrevenue",
                   "servicerevenue", "productrevenue", "revenueremaining"],
    },
    "cogs": {
        "prefer": ["costofrevenue", "costofgoodssold", "costofgoodsandservicessold"],
        "avoid":  ["operatingexpenses"],
    },
    "operating_cash_flow": {
        "prefer": ["netcashprovidedbyusedinoperatingactivities",
                   "netcashprovidedbyoperatingactivities"],
        "avoid":  ["cashandcashequivalents", "investingactivities",
                   "financingactivities", "carryingvalue"],
    },
    "free_cash_flow": {
        "prefer": ["freecashflow"],
        "avoid":  ["cashandcashequivalents", "investingactivities",
                   "financingactivities", "operatingactivities"],
    },
    "long_term_debt": {
        "prefer": ["longtermdebtnoncurrent"],
        "avoid":  ["currentportion", "longtermdebtcurrent", "shortterm"],
    },
    "deferred_revenue": {
        "prefer": ["deferredrevenue", "contractwithcustomerliability", "unearnedrevenue"],
        "avoid":  ["revenuefromcontractwithcustomerexcluding", "us-gaap:revenues"],
    },
    "net_income": {
        "prefer": ["netincomeloss"],
        "avoid":  ["netincomelossattributabletononcontrollinginterest", "comprehensiveincome"],
    },
    "gross_profit": {
        "prefer": ["grossprofit"],
        "avoid":  ["operatingexpenses"],
    },
    "operating_income": {
        "prefer": ["operatingincomeloss"],
        "avoid":  ["operatingexpenses", "nonoperating"],
    },
    "total_assets": {
        "prefer": ["us-gaap:assets"],
        "avoid":  ["currentassets", "noncurrentassets",
                   "intangibleassets", "rightofuseasset"],
    },
    "total_liabilities": {
        "prefer": ["us-gaap:liabilities"],
        "avoid":  ["currentliabilities", "longtermdebtnoncurrent",
                   "deferredrevenue", "rightofuselease",
                   "liabilitiesandstockholdersequity"],
    },
    "shareholders_equity": {
        "prefer": ["stockholdersequity"],
        "avoid":  ["stockholdersequityincludingportionattributabletononcontrolling",
                   "minorityinterest", "liabilitiesandstockholdersequity"],
    },
    "current_assets": {
        "prefer": ["assetscurrent"],
        "avoid":  ["assetsnoncurrent", "us-gaap:assets"],
    },
    "current_liabilities": {
        "prefer": ["liabilitiescurrent"],
        "avoid":  ["liabilitiesnoncurrent", "us-gaap:liabilities"],
    },
    "cash": {
        "prefer": ["cashandcashequivalentsatcarryingvalue", "cashandcashequivalents"],
        "avoid":  ["restrictedcash", "operatingcashflow",
                   "investingactivities", "financingactivities"],
    },
    "eps_diluted": {
        "prefer": ["earningspersharediluted"],
        "avoid":  ["earningspersharebasic"],
    },
    "eps_basic": {
        "prefer": ["earningspersharebasic"],
        "avoid":  ["earningspersharediluted"],
    },
    "capex": {
        "prefer": ["paymentstoacquireproductiveassets",
                   "paymentstoacquirepropertyplantandequipment", "capitalexpenditures"],
        "avoid":  ["paymentstoacquirebusinesses"],
    },
    "dividends_paid": {
        "prefer": ["paymentsofdividends", "paymentsofdividendscommonstock"],
        "avoid":  ["dividendsdeclared"],
    },
    "share_repurchase": {
        "prefer": ["paymentsforrepurchaseofcommonstock", "treasurystock"],
        "avoid":  ["issuanceofcommonstock"],
    },
    "r_and_d": {
        "prefer": ["researchanddevelopmentexpense"],
        "avoid":  ["sellinggeneralandadministrative"],
    },
    "sg_and_a": {
        "prefer": ["sellinggeneralandadministrativeexpense"],
        "avoid":  ["researchanddevelopment", "costofrevenue"],
    },
    "interest_expense": {
        "prefer": ["interestexpense"],
        "avoid":  ["interestincome"],
    },
    "income_tax": {
        "prefer": ["incometaxexpensebenefit", "provisionforincometaxes"],
        "avoid":  ["deferredincometax", "incometaxreconciliation"],
    },
    "inventory": {
        "prefer": ["inventorynet"],
        "avoid":  [],
    },
    "accounts_receivable": {
        "prefer": ["accountsreceivablenetcurrent", "receivablesnetcurrent"],
        "avoid":  ["allowancefordoubtfulaccounts"],
    },
    "goodwill": {
        "prefer": ["goodwill"],
        "avoid":  ["impairment"],
    },
}


def _apply_pattern_preference(
    cell: "TableCell", pattern_name: str, base_conf: float,
) -> float:
    if pattern_name not in _PATTERN_PREFERENCES:
        return base_conf

    prefs = _PATTERN_PREFERENCES[pattern_name]
    row_normalized = _normalize_tag(cell.row_header or "")

    conf = base_conf
    matched_prefer = False
    for prefer_tag in prefs.get("prefer", []):
        prefer_normalized = _normalize_tag(prefer_tag)
        if prefer_normalized in row_normalized:
            conf = min(conf + 0.05, 1.0)
            matched_prefer = True
            break

    if not matched_prefer:
        for avoid_tag in prefs.get("avoid", []):
            avoid_normalized = _normalize_tag(avoid_tag)
            if avoid_normalized in row_normalized:
                conf -= 0.10
                break

    return max(0.0, min(1.0, conf))


_IXBRL_ALIAS_MAP: Dict[str, List[str]] = {
    "net income loss": ["net income", "net loss", "net earnings"],
    "revenues": ["revenue", "net sales", "total revenue"],
    "revenue from contract with customer excluding assessed tax":
        ["revenue", "net sales", "total revenue"],
    "earnings per share diluted": ["diluted eps", "diluted earnings per share"],
    "earnings per share basic": ["basic eps", "basic earnings per share"],
    "gross profit": ["gross margin", "gross profit"],
    "operating income loss": ["operating income", "operating loss"],
    "assets": ["total assets"],
    "liabilities": ["total liabilities"],
    "stockholders equity": ["shareholders equity", "total equity"],
    "cash and cash equivalents at carrying value": ["cash", "cash and cash equivalents"],
    "long term debt noncurrent": ["long-term debt", "long term debt", "noncurrent debt"],
    "research and development expense": ["research and development", "r and d"],
    "selling general and administrative expense":
        ["sg and a", "selling general administrative"],
    "cost of revenue": ["cost of sales", "cost of goods sold"],
    "cost of goods and services sold": ["cost of sales", "cost of goods sold"],
    "income tax expense benefit": ["income tax", "tax expense"],
    "interest expense": ["interest expense"],
}


def _ixbrl_aliases(humanized: str) -> List[str]:
    return _IXBRL_ALIAS_MAP.get(humanized, [])


# ── TableIndex with iXBRL CONTEXT-AWARE indexing ─────────────────────────────

class TableIndex:
    def __init__(self) -> None:
        self._cells:   List[TableCell] = []
        self._row_map: Dict[str, List[TableCell]] = {}
        self._primary_context:   Optional[str] = None
        self._balance_context:   Optional[str] = None
        self._all_contexts:      Set[str]      = set()
        # Bug v6: only enable context filtering for real iXBRL data
        self._has_ixbrl_contexts: bool = False

    @classmethod
    def from_raw_cells(cls, raw_cells: List[Dict]) -> "TableIndex":
        idx = cls()
        for raw in raw_cells:
            row_header_raw = raw.get("row_header", "") or ""
            row_header_norm = _normalise(row_header_raw)

            value_raw = raw.get("value", "") or ""
            cell = TableCell(
                row_header=row_header_norm,
                col_header=_normalise(raw.get("col_header", "")),
                value=value_raw.strip() if isinstance(value_raw, str) else str(value_raw),
                unit=raw.get("unit", "units"),
                page=int(raw.get("page", 0)),
                section=raw.get("section", "UNKNOWN"),
                company=raw.get("company", "UNKNOWN"),
                doc_type=raw.get("doc_type", "UNKNOWN"),
                fiscal_year=raw.get("fiscal_year", "UNKNOWN"),
                numeric_value=_parse_numeric(value_raw if isinstance(value_raw, str) else str(value_raw)),
            )
            idx._cells.append(cell)
            idx._row_map.setdefault(cell.row_header, []).append(cell)
            if cell.col_header:
                idx._all_contexts.add(cell.col_header)

            humanized = _humanize_ixbrl(row_header_raw)
            if humanized and humanized != cell.row_header:
                idx._row_map.setdefault(humanized, []).append(cell)
                for alias in _ixbrl_aliases(humanized):
                    idx._row_map.setdefault(alias, []).append(cell)

        idx._detect_canonical_contexts()
        logger.info(
            "TableIndex built: %d cells, %d unique rows, ixbrl=%s, primary_ctx=%s, balance_ctx=%s",
            len(idx._cells), len(idx._row_map), idx._has_ixbrl_contexts,
            idx._primary_context, idx._balance_context,
        )
        return idx

    def _detect_canonical_contexts(self) -> None:
        """Auto-detect canonical iXBRL contexts.

        Bug v6: Only enable context filtering when col_headers look like
        real iXBRL context refs (start with 'c-'). Otherwise tests using
        synthetic data with col_headers like 'fy2022' would be broken.
        """
        if not self._cells:
            return

        # Detect: do most col_headers match iXBRL context pattern (c-X)?
        ixbrl_pattern_count = 0
        total_with_ctx = 0
        for cell in self._cells:
            if cell.col_header:
                total_with_ctx += 1
                if re.match(r"^c-\d+$", cell.col_header):
                    ixbrl_pattern_count += 1

        # Need at least 50% to be iXBRL contexts to enable feature
        if total_with_ctx == 0 or ixbrl_pattern_count / total_with_ctx < 0.5:
            self._has_ixbrl_contexts = False
            return

        self._has_ixbrl_contexts = True

        # Count distinct row_headers per col_header
        ctx_to_rows: Dict[str, Set[str]] = defaultdict(set)
        for cell in self._cells:
            if cell.col_header and cell.row_header:
                ctx_to_rows[cell.col_header].add(cell.row_header)

        if not ctx_to_rows:
            return

        sorted_contexts = sorted(
            ctx_to_rows.items(),
            key=lambda x: (-len(x[1]), x[0]),
        )

        if sorted_contexts:
            self._primary_context = sorted_contexts[0][0]

        balance_anchor_tags = {
            "us-gaap:assets", "us-gaap:liabilities", "us-gaap:stockholdersequity",
            "us-gaap:assetscurrent", "us-gaap:liabilitiescurrent",
            "us-gaap:cashandcashequivalentsatcarryingvalue",
        }
        for ctx, rows in sorted_contexts:
            if ctx == self._primary_context:
                continue
            for row in rows:
                if row in balance_anchor_tags:
                    self._balance_context = ctx
                    break
            if self._balance_context:
                break

    def get_canonical_context(self, pattern_name: str) -> Optional[str]:
        if not self._has_ixbrl_contexts:
            return None
        if pattern_name in _BALANCE_SHEET_PATTERNS:
            return self._balance_context or self._primary_context
        return self._primary_context

    def search_by_row(self, normalised_row: str) -> List[TableCell]:
        return self._row_map.get(normalised_row, [])

    def search_prefix(self, prefix: str) -> List[TableCell]:
        return [
            cell for key, cells in self._row_map.items()
            if key.startswith(prefix)
            for cell in cells
        ]

    def search_contains(self, substring: str) -> List[TableCell]:
        return [
            cell for key, cells in self._row_map.items()
            if substring in key
            for cell in cells
        ]

    def search_by_ixbrl_tags(
        self, tags: List[str], strict: bool = False,
    ) -> List[TableCell]:
        """Search cells matching iXBRL tags.

        Bug v6: strict mode requires EXACT equality after normalization
        (avoids us-gaap:liabilities matching us-gaap:liabilitiesandstockholdersequity).
        """
        matches: List[TableCell] = []
        seen_ids = set()
        normalized_tags = [_normalize_tag(t) for t in tags]

        for cell in self._cells:
            if id(cell) in seen_ids:
                continue
            row_norm = _normalize_tag(cell.row_header or "")
            for tag_norm in normalized_tags:
                if strict:
                    if row_norm == tag_norm:
                        matches.append(cell)
                        seen_ids.add(id(cell))
                        break
                else:
                    if tag_norm in row_norm:
                        matches.append(cell)
                        seen_ids.add(id(cell))
                        break
        return matches

    def __len__(self) -> int:
        return len(self._cells)

    def is_empty(self) -> bool:
        return len(self._cells) == 0


# ── SniperRAG ─────────────────────────────────────────────────────────────────

class SniperRAG:
    """N06 SniperRAG — Tier 1 Direct Table Cell Extraction."""

    def __init__(self, table_index: TableIndex) -> None:
        self.index = table_index

    def run(self, state) -> object:
        if self.index.is_empty() and hasattr(state, "table_cells") and state.table_cells:
            self.index = TableIndex.from_raw_cells(state.table_cells)

        query  = getattr(state, "query", "") or ""
        result = self.hit(query)

        state.sniper_hit        = result.sniper_hit
        state.sniper_result     = result.answer if result.sniper_hit else None
        state.sniper_confidence = result.confidence

        logger.info(
            "N06 SniperRAG: hit=%s | confidence=%.3f | pattern=%s",
            result.sniper_hit, result.confidence, result.matched_pattern,
        )
        return state

    def hit(self, query: str) -> SniperResult:
        if self.index.is_empty():
            return self._miss("Table index is empty — no cells to search")

        norm_query = _normalise(query)
        matched_metric, matched_pattern_name = self._identify_metric(norm_query)

        if matched_metric is None:
            return self._miss(
                f"No financial metric pattern matched query: '{query[:80]}'"
            )

        # ── Bug v6: SYNTHETIC METRIC computation (e.g. free_cash_flow) ──
        if matched_pattern_name in _SYNTHETIC_METRICS:
            synthetic_result = self._compute_synthetic(matched_pattern_name, query)
            if synthetic_result.sniper_hit:
                return synthetic_result
            # Fall through to regular search if synthetic computation fails

        fy_hint    = _extract_fy_from_query(query)
        query_unit = _detect_unit_from_context(norm_query)
        canonical_ctx = self.index.get_canonical_context(matched_pattern_name)
        use_strict = matched_pattern_name in _STRICT_MATCH_PATTERNS

        # ── Strategy 0: Direct iXBRL tag lookup (with strict mode) ──
        candidates: List[Tuple[TableCell, float]] = []
        ixbrl_tags = _PATTERN_TO_IXBRL_TAGS.get(matched_pattern_name, [])
        if ixbrl_tags:
            tag_cells = self.index.search_by_ixbrl_tags(ixbrl_tags, strict=use_strict)

            if canonical_ctx and tag_cells:
                ctx_filtered = [c for c in tag_cells if c.col_header == canonical_ctx]
                if ctx_filtered:
                    tag_cells = ctx_filtered

            if tag_cells:
                candidates = self._score_candidates(
                    tag_cells, fy_hint, query_unit, canonical_ctx,
                    base_confidence=_CONF_EXACT,
                    pattern_name=matched_pattern_name,
                )

        # Strategy 1: Exact match
        if not candidates:
            cells_s1 = self.index.search_by_row(matched_metric)
            if canonical_ctx and cells_s1:
                ctx_filtered = [c for c in cells_s1 if c.col_header == canonical_ctx]
                if ctx_filtered:
                    cells_s1 = ctx_filtered
            candidates = self._score_candidates(
                cells_s1, fy_hint, query_unit, canonical_ctx,
                base_confidence=_CONF_EXACT, pattern_name=matched_pattern_name,
            )

        # Strategy 2: Prefix match
        if not candidates:
            cells_s2 = self.index.search_prefix(matched_metric)
            if canonical_ctx and cells_s2:
                ctx_filtered = [c for c in cells_s2 if c.col_header == canonical_ctx]
                if ctx_filtered:
                    cells_s2 = ctx_filtered
            candidates = self._score_candidates(
                cells_s2, fy_hint, query_unit, canonical_ctx,
                base_confidence=_CONF_PREFIX, pattern_name=matched_pattern_name,
            )

        # Strategy 3: Contains match
        if not candidates:
            first_word = matched_metric.split()[0] if matched_metric else ""
            if len(first_word) >= 4:
                cells_s3 = self.index.search_contains(first_word)
                if canonical_ctx and cells_s3:
                    ctx_filtered = [c for c in cells_s3 if c.col_header == canonical_ctx]
                    if ctx_filtered:
                        cells_s3 = ctx_filtered
                candidates = self._score_candidates(
                    cells_s3, fy_hint, query_unit, canonical_ctx,
                    base_confidence=_CONF_CONTAINS, pattern_name=matched_pattern_name,
                )

        if not candidates:
            return self._miss(
                f"No table cells matched metric '{matched_metric}'"
            )

        best_cell, best_conf = max(
            candidates,
            key=lambda x: (x[1], abs(x[0].numeric_value or 0.0)),
        )

        if best_conf >= _HIT_THRESHOLD:
            return self._build_hit(best_cell, best_conf, matched_pattern_name)

        return SniperResult(
            sniper_hit=False, answer="", value=best_cell.value, unit=best_cell.unit,
            confidence=best_conf, matched_pattern=matched_pattern_name,
            cell=best_cell, citation=best_cell.metadata_key,
            reason=f"Best confidence {best_conf:.3f} < threshold {_HIT_THRESHOLD}",
        )

    def _compute_synthetic(self, pattern_name: str, query: str) -> SniperResult:
        """Compute a synthetic (non-GAAP) metric from primitive cells.

        Example: free_cash_flow = operating_cash_flow - capex.
        Recursively calls hit() on each dependency, then applies the formula.
        """
        if pattern_name not in _SYNTHETIC_METRICS:
            return self._miss(f"Not a synthetic metric: {pattern_name}")

        formula_desc, deps_required, compute_fn = _SYNTHETIC_METRICS[pattern_name]

        deps: Dict[str, SniperResult] = {}
        for dep_name in deps_required:
            # Build a synthetic query for the dependency to leverage hit()
            # We re-run hit() but skip the synthetic check to avoid recursion
            dep_result = self._hit_primitive(dep_name, query)
            if not dep_result.sniper_hit:
                return self._miss(
                    f"Synthetic '{pattern_name}' missing dep '{dep_name}'"
                )
            deps[dep_name] = dep_result

        computed_value = compute_fn(deps)
        if computed_value is None:
            return self._miss(f"Synthetic '{pattern_name}' computation failed")

        # Build a synthetic cell from the first dependency for citation
        anchor_cell = deps[deps_required[0]].cell
        if anchor_cell is None:
            return self._miss("No anchor cell for synthetic")

        formatted_value = f"{computed_value:,.0f}"
        synthetic_cell = TableCell(
            row_header=f"computed:{pattern_name}",
            col_header=anchor_cell.col_header,
            value=formatted_value,
            unit=anchor_cell.unit,
            page=anchor_cell.page,
            section=anchor_cell.section,
            company=anchor_cell.company,
            doc_type=anchor_cell.doc_type,
            fiscal_year=anchor_cell.fiscal_year,
            numeric_value=computed_value,
        )

        unit_str = f" {anchor_cell.unit}" if anchor_cell.unit and anchor_cell.unit != "units" else ""
        answer = f"{formatted_value}{unit_str} [{synthetic_cell.metadata_key}] (computed: {formula_desc})"

        return SniperResult(
            sniper_hit=True, answer=answer, value=formatted_value,
            unit=anchor_cell.unit, confidence=0.95,
            matched_pattern=pattern_name, cell=synthetic_cell,
            citation=synthetic_cell.metadata_key,
            reason=f"Computed: {formula_desc}",
        )

    def _hit_primitive(self, pattern_name: str, query: str) -> SniperResult:
        """Internal: run hit() forcing a specific pattern, no synthetic recursion."""
        if self.index.is_empty():
            return self._miss("Table index empty")

        fy_hint    = _extract_fy_from_query(query)
        query_unit = _detect_unit_from_context(_normalise(query))
        canonical_ctx = self.index.get_canonical_context(pattern_name)
        use_strict = pattern_name in _STRICT_MATCH_PATTERNS

        candidates: List[Tuple[TableCell, float]] = []
        ixbrl_tags = _PATTERN_TO_IXBRL_TAGS.get(pattern_name, [])
        if ixbrl_tags:
            tag_cells = self.index.search_by_ixbrl_tags(ixbrl_tags, strict=use_strict)
            if canonical_ctx and tag_cells:
                ctx_filtered = [c for c in tag_cells if c.col_header == canonical_ctx]
                if ctx_filtered:
                    tag_cells = ctx_filtered
            if tag_cells:
                candidates = self._score_candidates(
                    tag_cells, fy_hint, query_unit, canonical_ctx,
                    base_confidence=_CONF_EXACT, pattern_name=pattern_name,
                )

        if not candidates:
            return self._miss(f"No primitive cells for '{pattern_name}'")

        best_cell, best_conf = max(
            candidates,
            key=lambda x: (x[1], abs(x[0].numeric_value or 0.0)),
        )

        if best_conf >= _HIT_THRESHOLD:
            return self._build_hit(best_cell, best_conf, pattern_name)
        return self._miss(f"Primitive '{pattern_name}' below threshold")

    def _identify_metric(
        self, norm_query: str
    ) -> Tuple[Optional[str], Optional[str]]:
        for name, pattern in COMPILED_PATTERNS.items():
            m = pattern.search(norm_query)
            if m:
                matched_text = _normalise(m.group(0)).strip()
                return matched_text, name
        return None, None

    def _score_candidates(
        self,
        cells:           List[TableCell],
        fy_hint:         Optional[str],
        query_unit:      str,
        canonical_ctx:   Optional[str],
        base_confidence: float,
        pattern_name:    str = "",
    ) -> List[Tuple[TableCell, float]]:
        scored: List[Tuple[TableCell, float]] = []
        for cell in cells:
            conf = base_confidence

            if canonical_ctx and cell.col_header == canonical_ctx:
                conf = min(conf + _CONF_CONTEXT_BONUS, 1.0)

            if pattern_name:
                conf = _apply_pattern_preference(cell, pattern_name, conf)

            if query_unit != "units" and query_unit == cell.unit:
                conf = min(conf + _CONF_UNIT_BONUS, 1.0)

            # Fallback FY scoring (when context detection didn't help)
            if fy_hint is not None and not canonical_ctx:
                cell_fy  = _normalise(cell.fiscal_year)
                query_fy = _normalise(fy_hint)
                is_real_fy = (
                    cell_fy and cell_fy != "unknown"
                    and cell_fy not in ("", "n/a", "none")
                )
                if is_real_fy and query_fy not in cell_fy and cell_fy not in query_fy:
                    conf -= 0.05

            if not cell.value or cell.value in ("—", "-", "N/A", ""):
                conf -= 0.10
            scored.append((cell, round(conf, 4)))
        return scored

    @staticmethod
    def _build_hit(
        cell: TableCell, confidence: float, pattern_name: str
    ) -> SniperResult:
        display_value = cell.value
        if cell.numeric_value is not None and cell.numeric_value < 0:
            if not display_value.startswith("-"):
                display_value = f"-{display_value.strip('()')}"
        unit_str = f" {cell.unit}" if cell.unit and cell.unit != "units" else ""
        answer   = f"{display_value}{unit_str} [{cell.metadata_key}]"
        return SniperResult(
            sniper_hit=True, answer=answer, value=cell.value, unit=cell.unit,
            confidence=confidence, matched_pattern=pattern_name, cell=cell,
            citation=cell.metadata_key,
            reason=f"Pattern '{pattern_name}' matched with confidence {confidence:.3f}",
        )

    @staticmethod
    def _miss(reason: str) -> SniperResult:
        logger.debug("SniperRAG MISS: %s", reason)
        return SniperResult(
            sniper_hit=False, answer="", value="", unit="", confidence=0.0,
            matched_pattern="", cell=None, citation="", reason=reason,
        )


# ── Convenience wrapper for LangGraph N06 node ───────────────────────────────

def run_sniper(query: str, table_cells: List[Dict]) -> SniperResult:
    index  = TableIndex.from_raw_cells(table_cells)
    sniper = SniperRAG(index)
    result = sniper.hit(query)
    logger.info(
        "SniperRAG: hit=%s | confidence=%.3f | pattern=%s",
        result.sniper_hit, result.confidence, result.matched_pattern,
    )
    return result