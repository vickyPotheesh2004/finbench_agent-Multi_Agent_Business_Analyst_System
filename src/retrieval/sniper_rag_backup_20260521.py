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
                  - STRICT-MATCH-ONLY mode for ambiguous tags.
                  - SYNTHETIC METRIC computation for non-GAAP metrics.
                  - Graceful fallback when context detection fails.
  2026-05-21 S28  Synthetic ratio engine (8 new metrics):
                  gross_margin, operating_margin, net_margin,
                  effective_tax_rate, ROE, ROA, D/E, current_ratio.
                  EPS basic vs diluted disambiguation (pattern order).
                  Segment-aware revenue/income extraction.
                  accounts_payable pattern added.
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


# ---- Compiled regex patterns (ORDER MATTERS) ----
RAW_PATTERNS: Dict[str, str] = {
    "operating_cash_flow": r"(?:net\s+)?cash\s+(?:provided\s+by\s+|from\s+)?operating\s+activities|operating\s+cash\s+flow",
    "free_cash_flow":      r"free\s+cash\s+flow|fcf",
    "long_term_debt":      r"long.?term\s+(?:debt|notes?\s+payable|borrowings?|obligations?)",
    "deferred_revenue":    r"deferred\s+revenue(?:s)?|unearned\s+revenue",
    "share_repurchase":    r"(?:repurchases?|buybacks?)\s+(?:of\s+)?(?:common\s+)?(?:stock|shares?)|treasury\s+stock\s+purchases?",
    "dividends_paid":      r"dividends?\s+paid|cash\s+dividends?",
    "capex":               r"capital\s+expenditures?|capex|purchases?\s+of\s+property",
    "interest_expense":    r"interest\s+(?:expense|cost|charges?)|finance\s+(?:cost|charge)",
    "income_before_tax":   r"income\s+(?:loss\s+)?(?:from\s+continuing\s+operations\s+)?before\s+(?:income\s+)?tax(?:es)?|pre[\s-]?tax\s+income",
    "income_tax":          r"(?:provision\s+for\s+)?income\s+tax(?:es)?|tax\s+(?:expense|provision)",
    "operating_income":    r"(?:(?:total\s+)?operating\s+(?:income|loss|profit))|(?:income\s+(?:loss\s+)?from\s+operations)",
    "gross_margin":        r"gross\s+(?:profit\s+)?margin(?:\s+percentage)?|gross\s+margin\s+%|gross\s+profit\s+as\s+(?:a\s+)?percent",
    "operating_margin":    r"operating\s+(?:income\s+)?margin(?:\s+percentage)?|operating\s+margin\s+%",
    "net_margin":          r"net\s+(?:income\s+)?margin|net\s+profit\s+margin|profit\s+margin(?:\s+percentage)?",
    "gross_profit":        r"gross\s+(?:profit|income)(?!\s+margin)(?!\s+as\s+percent)",
    "effective_tax_rate":  r"effective\s+(?:income\s+)?tax\s+rate|tax\s+rate",
    "return_on_equity":    r"return\s+on\s+equity|(?<!current\s)\broe\b",
    "return_on_assets":    r"return\s+on\s+(?:total\s+)?assets|\broa\b",
    "debt_to_equity":      r"debt[\s-]+to[\s-]+equity(?:\s+ratio)?|d/?e\s+ratio|leverage\s+ratio",
    "current_ratio":       r"current\s+ratio",
    "ebitda":              r"(?:adjusted\s+)?ebitda|earnings\s+before\s+interest[,\s]+(?:taxes[,\s]+)?depreciation",
    "ebit":                r"\bebit\b|earnings\s+before\s+interest\s+and\s+taxes",
    "eps_basic":           r"basic\s+(?:earnings|loss)\s+per\s+(?:common\s+)?share|basic\s+eps|eps\s+basic",
    "eps_diluted":         r"diluted\s+(?:earnings|loss)\s+per\s+(?:diluted\s+)?(?:common\s+)?share|(?:earnings|loss)\s+per\s+(?:diluted\s+)?(?:common\s+)?share|diluted\s+eps|eps\s+diluted|(?<!basic\s)(?:earnings|loss)\s+per\s+share",
    "r_and_d":             r"research\s+(?:and|&)\s+development(?:\s+(?:expense|cost))?",
    "sg_and_a":            r"(?:selling[,\s]+)?general\s+(?:and\s+)?administrative(?:\s+(?:expense|cost))?|sg(?:\s*[&and]+\s*)?a",
    "cogs":                r"cost\s+of\s+(?:goods?\s+)?(?:revenue|sales?|products?|services?)|cost\s+of\s+revenues?",
    "current_assets":      r"(?:total\s+)?current\s+assets",
    "current_liabilities": r"(?:total\s+)?current\s+liabilities",
    "total_assets":        r"total\s+assets",
    "total_liabilities":   r"total\s+(?:liabilities(?:\s+and)?|debt)",
    "shareholders_equity": r"(?:total\s+)?(?:stockholders?|shareholders?)\s+(?:equity|deficit)",
    "accounts_receivable": r"(?:net\s+)?accounts?\s+receivable|trade\s+receivables?",
    "accounts_payable":    r"(?:total\s+)?accounts?\s+payable",
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


_PATTERN_TO_IXBRL_TAGS: Dict[str, List[str]] = {
    "income_before_tax":   ["us-gaap:incomelossfromcontinuingoperationsbeforeincometaxes",
                            "us-gaap:incomelossfromcontinuingoperationsbeforeincometaxesextraordinaryitemsnoncontrollinginterest"],
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
    "accounts_payable":    ["us-gaap:accountspayablecurrent"],
    "inventory":           ["us-gaap:inventorynet"],
    "goodwill":            ["us-gaap:goodwill"],
}

_STRICT_MATCH_PATTERNS: Set[str] = {
    "total_assets", "total_liabilities", "shareholders_equity",
    "current_assets", "current_liabilities", "long_term_debt",
    "income_tax", "cash",
}

_INCOME_FLOW_PATTERNS = {
    "revenue", "cogs", "gross_profit", "operating_income", "net_income",
    "eps_diluted", "eps_basic", "operating_cash_flow", "free_cash_flow",
    "interest_expense", "income_tax", "income_before_tax", "r_and_d",
    "sg_and_a", "capex", "dividends_paid", "share_repurchase", "ebitda", "ebit",
}
_BALANCE_SHEET_PATTERNS = {
    "total_assets", "total_liabilities", "shareholders_equity",
    "current_assets", "current_liabilities", "cash",
    "long_term_debt", "deferred_revenue", "accounts_receivable",
    "accounts_payable", "inventory", "goodwill",
}

# ---- Segment keywords ----
_SEGMENT_KEYWORDS: Dict[str, str] = {
    "aws": "aws", "amazon web services": "aws",
    "google services": "google_services", "google search": "google_services",
    "youtube": "google_services", "google cloud": "google_cloud",
    "family of apps": "family_of_apps", "reality labs": "reality_labs",
    "automotive": "automotive",
    "energy generation": "energy_storage", "energy storage": "energy_storage",
    "azure": "azure", "intelligent cloud": "intelligent_cloud",
}
_SEGMENT_NAME_VARIANTS: Dict[str, List[str]] = {
    "aws":              ["amazon web services", "aws"],
    "google_services":  ["google services", "google search", "youtube ads"],
    "google_cloud":     ["google cloud"],
    "family_of_apps":   ["family of apps", "foa"],
    "reality_labs":     ["reality labs"],
    "automotive":       ["automotive"],
    "energy_storage":   ["energy generation and storage", "energy storage"],
    "azure":            ["azure"],
    "intelligent_cloud":["intelligent cloud"],
}

# ---- Synthetic metrics ----
def _compute_free_cash_flow(deps):
    ocf = deps.get("operating_cash_flow")
    capex = deps.get("capex")
    if (ocf and ocf.cell and ocf.cell.numeric_value is not None and
        capex and capex.cell and capex.cell.numeric_value is not None):
        return float(ocf.cell.numeric_value) - abs(float(capex.cell.numeric_value))
    return None

def _compute_ratio(deps, num_key, den_key):
    num, den = deps.get(num_key), deps.get(den_key)
    if (num and num.cell and num.cell.numeric_value is not None and
        den and den.cell and den.cell.numeric_value is not None):
        d = float(den.cell.numeric_value)
        if abs(d) < 1e-9: return None
        return round(float(num.cell.numeric_value) / d * 100, 1)
    return None

def _compute_plain_ratio(deps, num_key, den_key):
    num, den = deps.get(num_key), deps.get(den_key)
    if (num and num.cell and num.cell.numeric_value is not None and
        den and den.cell and den.cell.numeric_value is not None):
        d = float(den.cell.numeric_value)
        if abs(d) < 1e-9: return None
        return round(float(num.cell.numeric_value) / d, 2)
    return None

_SYNTHETIC_METRICS: Dict[str, Tuple[str, List[str], Callable]] = {
    "free_cash_flow":     ("OCF - CapEx", ["operating_cash_flow", "capex"], _compute_free_cash_flow),
    "gross_margin":       ("Gross Profit / Revenue x 100", ["gross_profit", "revenue"], lambda d: _compute_ratio(d, "gross_profit", "revenue")),
    "operating_margin":   ("Operating Income / Revenue x 100", ["operating_income", "revenue"], lambda d: _compute_ratio(d, "operating_income", "revenue")),
    "net_margin":         ("Net Income / Revenue x 100", ["net_income", "revenue"], lambda d: _compute_ratio(d, "net_income", "revenue")),
    "effective_tax_rate": ("Income Tax / Pre-Tax Income x 100", ["income_tax", "income_before_tax"], lambda d: _compute_ratio(d, "income_tax", "income_before_tax")),
    "return_on_equity":   ("Net Income / Equity x 100", ["net_income", "shareholders_equity"], lambda d: _compute_ratio(d, "net_income", "shareholders_equity")),
    "return_on_assets":   ("Net Income / Total Assets x 100", ["net_income", "total_assets"], lambda d: _compute_ratio(d, "net_income", "total_assets")),
    "debt_to_equity":     ("Liabilities / Equity", ["total_liabilities", "shareholders_equity"], lambda d: _compute_plain_ratio(d, "total_liabilities", "shareholders_equity")),
    "current_ratio":      ("Current Assets / Current Liabilities", ["current_assets", "current_liabilities"], lambda d: _compute_plain_ratio(d, "current_assets", "current_liabilities")),
}

# ---- FY/unit detection ----
_FY_PATTERNS = [
    re.compile(r"\bfy\s*(\d{4})\b", re.I),
    re.compile(r"\bfiscal\s+year\s+(\d{4})\b", re.I),
    re.compile(r"\b(?:in|for|at\s+end\s+of)\s+(?:fy\s*)?(\d{4})\b", re.I),
    re.compile(r"\b(20\d{2})\b"),
    re.compile(r"\bfy\s*'?(\d{2,4})\b", re.I),
]
_UNIT_PATTERNS: Dict[str, re.Pattern] = {
    "billions":  re.compile(r"\bbillion[s]?\b|\bbn\b", re.I),
    "millions":  re.compile(r"\bmillion[s]?\b|\bmn\b|\bm\b(?!\w)", re.I),
    "thousands": re.compile(r"\bthousand[s]?\b|\bk\b(?!\w)", re.I),
    "%":         re.compile(r"\bpercent(?:age)?\b|%"),
}
_CONF_EXACT      = 0.98
_CONF_PREFIX     = 0.92
_CONF_CONTAINS   = 0.85
_CONF_UNIT_BONUS = 0.02
_CONF_CONTEXT_BONUS = 0.10
_HIT_THRESHOLD   = 0.95
_UNIT_DISPLAY = {"x10^6": "million", "x10^9": "billion", "x10^3": "thousand",
                 "USDperShare": "per share", "shares": "shares"}

# ---- Helpers ----
def _normalise(text):
    text = text.lower().strip()
    text = re.sub(r"[\u2019\u2018\u201c\u201d]", "'", text)
    return re.sub(r"\s+", " ", text)

def _normalize_tag(tag):
    return tag.lower().replace(":", "").replace("_", "").replace(" ", "")

def _parse_numeric(value_str):
    if not value_str: return None
    s = value_str.strip()
    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1]
    s = re.sub(r"[$,\s]", "", s).rstrip("%")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None

def _extract_fy_from_query(query):
    norm = _normalise(query)
    for pat in _FY_PATTERNS:
        m = pat.search(norm)
        if m:
            y = m.group(1) if m.lastindex >= 1 else m.group(0)
            if len(y) == 2: y = f"20{y}"
            return f"FY{y}"
    return None

def _detect_unit_from_context(text):
    for name, pat in _UNIT_PATTERNS.items():
        if pat.search(text): return name
    return "units"

def _humanize_ixbrl(name):
    if not name: return ""
    if ":" in name: name = name.split(":", 1)[1]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower().strip()

# ---- Pattern preferences ----
_PATTERN_PREFERENCES: Dict[str, Dict[str, List[str]]] = {
    "income_before_tax": {"prefer": ["incomelossfromcontinuingoperationsbeforeincometaxes"], "avoid": ["incometaxexpensebenefit"]},
    "revenue": {"prefer": ["us-gaap:revenues", "us-gaap:salesrevenuenet", "us-gaap:revenuefromcontractwithcustomerexcludingassessedtax"], "avoid": ["deferredrevenue", "unearnedrevenue", "servicerevenue", "productrevenue", "revenueremaining"]},
    "cogs": {"prefer": ["costofrevenue", "costofgoodssold", "costofgoodsandservicessold"], "avoid": ["operatingexpenses"]},
    "operating_cash_flow": {"prefer": ["netcashprovidedbyusedinoperatingactivities", "netcashprovidedbyoperatingactivities"], "avoid": ["cashandcashequivalents", "investingactivities", "financingactivities", "carryingvalue"]},
    "free_cash_flow": {"prefer": ["freecashflow"], "avoid": ["cashandcashequivalents", "investingactivities", "financingactivities", "operatingactivities"]},
    "long_term_debt": {"prefer": ["longtermdebtnoncurrent"], "avoid": ["currentportion", "longtermdebtcurrent", "shortterm"]},
    "deferred_revenue": {"prefer": ["deferredrevenue", "contractwithcustomerliability", "unearnedrevenue"], "avoid": ["revenuefromcontractwithcustomerexcluding", "us-gaap:revenues"]},
    "net_income": {"prefer": ["netincomeloss"], "avoid": ["netincomelossattributabletononcontrollinginterest", "comprehensiveincome"]},
    "gross_profit": {"prefer": ["grossprofit"], "avoid": ["operatingexpenses"]},
    "operating_income": {"prefer": ["operatingincomeloss"], "avoid": ["operatingexpenses", "nonoperating"]},
    "total_assets": {"prefer": ["us-gaap:assets"], "avoid": ["currentassets", "noncurrentassets", "intangibleassets", "rightofuseasset"]},
    "total_liabilities": {"prefer": ["us-gaap:liabilities"], "avoid": ["currentliabilities", "longtermdebtnoncurrent", "deferredrevenue", "rightofuselease", "liabilitiesandstockholdersequity"]},
    "shareholders_equity": {"prefer": ["stockholdersequity"], "avoid": ["stockholdersequityincludingportionattributabletononcontrolling", "minorityinterest", "liabilitiesandstockholdersequity"]},
    "current_assets": {"prefer": ["assetscurrent"], "avoid": ["assetsnoncurrent", "us-gaap:assets"]},
    "current_liabilities": {"prefer": ["liabilitiescurrent"], "avoid": ["liabilitiesnoncurrent", "us-gaap:liabilities"]},
    "cash": {"prefer": ["cashandcashequivalentsatcarryingvalue", "cashandcashequivalents"], "avoid": ["restrictedcash", "operatingcashflow", "investingactivities", "financingactivities"]},
    "eps_diluted": {"prefer": ["earningspersharediluted"], "avoid": ["earningspersharebasic"]},
    "eps_basic": {"prefer": ["earningspersharebasic"], "avoid": ["earningspersharediluted"]},
    "capex": {"prefer": ["paymentstoacquireproductiveassets", "paymentstoacquirepropertyplantandequipment", "capitalexpenditures"], "avoid": ["paymentstoacquirebusinesses"]},
    "dividends_paid": {"prefer": ["paymentsofdividends", "paymentsofdividendscommonstock"], "avoid": ["dividendsdeclared"]},
    "share_repurchase": {"prefer": ["paymentsforrepurchaseofcommonstock", "treasurystock"], "avoid": ["issuanceofcommonstock"]},
    "r_and_d": {"prefer": ["researchanddevelopmentexpense"], "avoid": ["sellinggeneralandadministrative"]},
    "sg_and_a": {"prefer": ["sellinggeneralandadministrativeexpense"], "avoid": ["researchanddevelopment", "costofrevenue"]},
    "interest_expense": {"prefer": ["interestexpense"], "avoid": ["interestincome"]},
    "income_tax": {"prefer": ["incometaxexpensebenefit", "provisionforincometaxes"], "avoid": ["deferredincometax", "incometaxreconciliation"]},
    "inventory": {"prefer": ["inventorynet"], "avoid": []},
    "accounts_receivable": {"prefer": ["accountsreceivablenetcurrent", "receivablesnetcurrent"], "avoid": ["allowancefordoubtfulaccounts"]},
    "accounts_payable": {"prefer": ["accountspayablecurrent", "accountspayable"], "avoid": []},
    "goodwill": {"prefer": ["goodwill"], "avoid": ["impairment"]},
}

def _apply_pattern_preference(cell, pattern_name, base_conf):
    if pattern_name not in _PATTERN_PREFERENCES: return base_conf
    prefs = _PATTERN_PREFERENCES[pattern_name]
    row_normalized = _normalize_tag(cell.row_header or "")
    conf = base_conf
    matched = False
    for pt in prefs.get("prefer", []):
        if _normalize_tag(pt) in row_normalized:
            conf = min(conf + 0.05, 1.0); matched = True; break
    if not matched:
        for at in prefs.get("avoid", []):
            if _normalize_tag(at) in row_normalized:
                conf -= 0.10; break
    return max(0.0, min(1.0, conf))

_IXBRL_ALIAS_MAP: Dict[str, List[str]] = {
    "net income loss": ["net income", "net loss", "net earnings"],
    "revenues": ["revenue", "net sales", "total revenue"],
    "revenue from contract with customer excluding assessed tax": ["revenue", "net sales", "total revenue"],
    "earnings per share diluted": ["diluted eps", "diluted earnings per share"],
    "earnings per share basic": ["basic eps", "basic earnings per share"],
    "gross profit": ["gross margin", "gross profit"],
    "operating income loss": ["operating income", "operating loss"],
    "assets": ["total assets"], "liabilities": ["total liabilities"],
    "stockholders equity": ["shareholders equity", "total equity"],
    "cash and cash equivalents at carrying value": ["cash", "cash and cash equivalents"],
    "long term debt noncurrent": ["long-term debt", "long term debt", "noncurrent debt"],
    "research and development expense": ["research and development", "r and d"],
    "selling general and administrative expense": ["sg and a", "selling general administrative"],
    "cost of revenue": ["cost of sales", "cost of goods sold"],
    "cost of goods and services sold": ["cost of sales", "cost of goods sold"],
    "income tax expense benefit": ["income tax", "tax expense"],
    "interest expense": ["interest expense"],
    "accounts payable current": ["accounts payable"],
}
def _ixbrl_aliases(humanized): return _IXBRL_ALIAS_MAP.get(humanized, [])

# ---- TableIndex ----
class TableIndex:
    def __init__(self):
        self._cells = []; self._row_map = {}
        self._primary_context = None; self._balance_context = None
        self._all_contexts = set(); self._has_ixbrl_contexts = False

    @classmethod
    def from_raw_cells(cls, raw_cells):
        idx = cls()
        for raw in raw_cells:
            rh_raw = raw.get("row_header", "") or ""
            rh_norm = _normalise(rh_raw)
            vr = raw.get("value", "") or ""
            cell = TableCell(
                row_header=rh_norm, col_header=_normalise(raw.get("col_header", "")),
                value=vr.strip() if isinstance(vr, str) else str(vr),
                unit=raw.get("unit", "units"), page=int(raw.get("page", 0)),
                section=raw.get("section", "UNKNOWN"), company=raw.get("company", "UNKNOWN"),
                doc_type=raw.get("doc_type", "UNKNOWN"), fiscal_year=raw.get("fiscal_year", "UNKNOWN"),
                numeric_value=_parse_numeric(vr if isinstance(vr, str) else str(vr)))
            idx._cells.append(cell)
            idx._row_map.setdefault(cell.row_header, []).append(cell)
            if cell.col_header: idx._all_contexts.add(cell.col_header)
            h = _humanize_ixbrl(rh_raw)
            if h and h != cell.row_header:
                idx._row_map.setdefault(h, []).append(cell)
                for a in _ixbrl_aliases(h): idx._row_map.setdefault(a, []).append(cell)
        idx._detect_canonical_contexts()
        logger.info("TableIndex: %d cells, %d rows, ixbrl=%s, pri=%s, bal=%s",
                     len(idx._cells), len(idx._row_map), idx._has_ixbrl_contexts,
                     idx._primary_context, idx._balance_context)
        return idx

    def _detect_canonical_contexts(self):
        if not self._cells: return
        ic = sum(1 for c in self._cells if c.col_header and re.match(r"^c-\d+$", c.col_header))
        tc = sum(1 for c in self._cells if c.col_header)
        if tc == 0 or ic / tc < 0.5: self._has_ixbrl_contexts = False; return
        self._has_ixbrl_contexts = True
        ctx_rows = defaultdict(set)
        for c in self._cells:
            if c.col_header and c.row_header: ctx_rows[c.col_header].add(c.row_header)
        if not ctx_rows: return
        sc = sorted(ctx_rows.items(), key=lambda x: (-len(x[1]), x[0]))
        if sc: self._primary_context = sc[0][0]
        ba = {"us-gaap:assets","us-gaap:liabilities","us-gaap:stockholdersequity",
              "us-gaap:assetscurrent","us-gaap:liabilitiescurrent",
              "us-gaap:cashandcashequivalentsatcarryingvalue"}
        for ctx, rows in sc:
            if ctx == self._primary_context: continue
            if any(r in ba for r in rows): self._balance_context = ctx; break

    def get_canonical_context(self, pn):
        if not self._has_ixbrl_contexts: return None
        return (self._balance_context or self._primary_context) if pn in _BALANCE_SHEET_PATTERNS else self._primary_context

    def search_by_row(self, r): return self._row_map.get(r, [])
    def search_prefix(self, p): return [c for k,cs in self._row_map.items() if k.startswith(p) for c in cs]
    def search_contains(self, s): return [c for k,cs in self._row_map.items() if s in k for c in cs]

    def search_by_ixbrl_tags(self, tags, strict=False):
        matches, seen = [], set()
        nt = [_normalize_tag(t) for t in tags]
        for cell in self._cells:
            if id(cell) in seen: continue
            rn = _normalize_tag(cell.row_header or "")
            for tn in nt:
                if (strict and rn == tn) or (not strict and tn in rn):
                    matches.append(cell); seen.add(id(cell)); break
        return matches

    def __len__(self): return len(self._cells)
    def is_empty(self): return len(self._cells) == 0


# ---- SniperRAG ----
class SniperRAG:
    def __init__(self, table_index):
        self.index = table_index
        self._current_segment = None

    def run(self, state):
        if self.index.is_empty() and hasattr(state, "table_cells") and state.table_cells:
            self.index = TableIndex.from_raw_cells(state.table_cells)
        query = getattr(state, "query", "") or ""
        result = self.hit(query)
        state.sniper_hit = result.sniper_hit
        state.sniper_result = result.answer if result.sniper_hit else None
        state.sniper_confidence = result.confidence
        logger.info("N06 SniperRAG: hit=%s | conf=%.3f | pat=%s", result.sniper_hit, result.confidence, result.matched_pattern)
        return state

    def hit(self, query):
        if self.index.is_empty(): return self._miss("Table index empty")
        nq = _normalise(query)
        mm, mpn = self._identify_metric(nq)
        if mm is None: return self._miss(f"No pattern matched: '{query[:80]}'")

        # Segment search
        seg = self._current_segment
        if seg and mpn in ("revenue", "operating_income", "gross_profit", "net_income"):
            sr = self._search_segment(seg, mpn, query)
            if sr.sniper_hit: return sr

        # Synthetic
        if mpn in _SYNTHETIC_METRICS:
            sr = self._compute_synthetic(mpn, query)
            if sr.sniper_hit: return sr

        fy = _extract_fy_from_query(query)
        qu = _detect_unit_from_context(nq)
        cc = self.index.get_canonical_context(mpn)
        strict = mpn in _STRICT_MATCH_PATTERNS

        # Strategy 0: iXBRL tag
        cands = []
        tags = _PATTERN_TO_IXBRL_TAGS.get(mpn, [])
        if tags:
            tc = self.index.search_by_ixbrl_tags(tags, strict=strict)
            if cc and tc:
                cf = [c for c in tc if c.col_header == cc]
                if cf: tc = cf
            if tc: cands = self._score(tc, fy, qu, cc, _CONF_EXACT, mpn)

        # Strategy 1: exact
        if not cands:
            s1 = self.index.search_by_row(mm)
            if cc and s1:
                cf = [c for c in s1 if c.col_header == cc]
                if cf: s1 = cf
            cands = self._score(s1, fy, qu, cc, _CONF_EXACT, mpn)

        # Strategy 2: prefix
        if not cands:
            s2 = self.index.search_prefix(mm)
            if cc and s2:
                cf = [c for c in s2 if c.col_header == cc]
                if cf: s2 = cf
            cands = self._score(s2, fy, qu, cc, _CONF_PREFIX, mpn)

        # Strategy 3: contains
        if not cands:
            fw = mm.split()[0] if mm else ""
            if len(fw) >= 4:
                s3 = self.index.search_contains(fw)
                if cc and s3:
                    cf = [c for c in s3 if c.col_header == cc]
                    if cf: s3 = cf
                cands = self._score(s3, fy, qu, cc, _CONF_CONTAINS, mpn)

        if not cands: return self._miss(f"No cells matched '{mm}'")
        bc, bf = max(cands, key=lambda x: (x[1], abs(x[0].numeric_value or 0.0)))
        if bf >= _HIT_THRESHOLD: return self._build_hit(bc, bf, mpn)
        return SniperResult(sniper_hit=False, answer="", value=bc.value, unit=bc.unit,
                            confidence=bf, matched_pattern=mpn, cell=bc,
                            citation=bc.metadata_key, reason=f"Conf {bf:.3f} < {_HIT_THRESHOLD}")

    def _search_segment(self, segment, pattern_name, query):
        variants = _SEGMENT_NAME_VARIANTS.get(segment, [segment])
        matching = []
        for cell in self.index._cells:
            if cell.numeric_value is None: continue
            rl = (cell.row_header or "").lower()
            if any(v in rl for v in variants): matching.append(cell)
        if not matching: return self._miss(f"No cells for segment '{segment}'")
        fy = _extract_fy_from_query(query)
        cc = self.index.get_canonical_context(pattern_name)
        cands = self._score(matching, fy, "units", cc, 0.90, pattern_name)
        if not cands: return self._miss(f"No scored for segment '{segment}'")
        bc, bf = max(cands, key=lambda x: (x[1], abs(x[0].numeric_value or 0.0)))
        if bf >= 0.85: return self._build_hit(bc, bf, f"segment_{segment}")
        return self._miss(f"Segment '{segment}' conf {bf:.3f} < 0.85")

    def _compute_synthetic(self, pn, query):
        if pn not in _SYNTHETIC_METRICS: return self._miss(f"Not synthetic: {pn}")
        fd, dr, fn = _SYNTHETIC_METRICS[pn]
        deps = {}
        for dn in dr:
            r = self._hit_primitive(dn, query)
            if not r.sniper_hit: return self._miss(f"Synthetic '{pn}' missing '{dn}'")
            deps[dn] = r
        cv = fn(deps)
        if cv is None: return self._miss(f"Synthetic '{pn}' compute failed")
        ac = deps[dr[0]].cell
        if ac is None: return self._miss("No anchor cell")
        ir = pn in ("gross_margin","operating_margin","net_margin","effective_tax_rate","return_on_equity","return_on_assets")
        ip = pn in ("debt_to_equity","current_ratio")
        fv = f"{cv:.1f}%" if ir else (f"{cv:.2f}x" if ip else f"{cv:,.0f}")
        sc = TableCell(row_header=f"computed:{pn}", col_header=ac.col_header, value=fv,
                       unit=ac.unit, page=ac.page, section=ac.section, company=ac.company,
                       doc_type=ac.doc_type, fiscal_year=ac.fiscal_year, numeric_value=cv)
        if ir or ip:
            ans = f"{fv} [{sc.metadata_key}] (computed: {fd})"
        else:
            uf = _UNIT_DISPLAY.get(ac.unit or "", ac.unit or "")
            us = f" {uf}" if uf and uf != "units" else ""
            ans = f"{fv}{us} [{sc.metadata_key}] (computed: {fd})"
        return SniperResult(sniper_hit=True, answer=ans, value=fv, unit=ac.unit,
                            confidence=0.95, matched_pattern=pn, cell=sc,
                            citation=sc.metadata_key, reason=f"Computed: {fd}")

    def _hit_primitive(self, pn, query):
        if self.index.is_empty(): return self._miss("Empty")
        fy = _extract_fy_from_query(query)
        qu = _detect_unit_from_context(_normalise(query))
        cc = self.index.get_canonical_context(pn)
        strict = pn in _STRICT_MATCH_PATTERNS
        cands = []
        tags = _PATTERN_TO_IXBRL_TAGS.get(pn, [])
        if tags:
            tc = self.index.search_by_ixbrl_tags(tags, strict=strict)
            if cc and tc:
                cf = [c for c in tc if c.col_header == cc]
                if cf: tc = cf
            if tc: cands = self._score(tc, fy, qu, cc, _CONF_EXACT, pn)
        if not cands: return self._miss(f"No primitive for '{pn}'")
        bc, bf = max(cands, key=lambda x: (x[1], abs(x[0].numeric_value or 0.0)))
        if bf >= _HIT_THRESHOLD: return self._build_hit(bc, bf, pn)
        return self._miss(f"Primitive '{pn}' below threshold")

    def _identify_metric(self, nq):
        self._current_segment = None
        for kw, sid in _SEGMENT_KEYWORDS.items():
            if kw in nq: self._current_segment = sid; break
        for name, pat in COMPILED_PATTERNS.items():
            m = pat.search(nq)
            if m:
                mt = _normalise(m.group(0)).strip()
                if self._current_segment and name not in ("revenue","operating_income","gross_profit","net_income"):
                    self._current_segment = None
                return mt, name
        self._current_segment = None
        return None, None

    def _score(self, cells, fy, qu, cc, base, pn=""):
        scored = []
        for cell in cells:
            conf = base
            if cc and cell.col_header == cc: conf = min(conf + _CONF_CONTEXT_BONUS, 1.0)
            if pn: conf = _apply_pattern_preference(cell, pn, conf)
            if qu != "units" and qu == cell.unit: conf = min(conf + _CONF_UNIT_BONUS, 1.0)
            if fy and not cc:
                cf = _normalise(cell.fiscal_year); qf = _normalise(fy)
                if cf and cf != "unknown" and cf not in ("","n/a","none"):
                    if qf not in cf and cf not in qf: conf -= 0.05
            if not cell.value or cell.value in ("\u2014","-","N/A",""): conf -= 0.10
            scored.append((cell, round(conf, 4)))
        return scored

    @staticmethod
    def _build_hit(cell, confidence, pattern_name):
        dv = cell.value
        if cell.numeric_value is not None and cell.numeric_value < 0:
            if not dv.startswith("-"): dv = f"-{dv.strip('()')}"
        uf = _UNIT_DISPLAY.get(cell.unit or "", cell.unit or "")
        us = f" {uf}" if uf and uf != "units" else ""
        ans = f"{dv}{us} [{cell.metadata_key}]"
        return SniperResult(sniper_hit=True, answer=ans, value=cell.value, unit=cell.unit,
                            confidence=confidence, matched_pattern=pattern_name, cell=cell,
                            citation=cell.metadata_key,
                            reason=f"Pattern '{pattern_name}' conf {confidence:.3f}")

    @staticmethod
    def _miss(reason):
        logger.debug("SniperRAG MISS: %s", reason)
        return SniperResult(sniper_hit=False, answer="", value="", unit="", confidence=0.0,
                            matched_pattern="", cell=None, citation="", reason=reason)


def run_sniper(query, table_cells):
    idx = TableIndex.from_raw_cells(table_cells)
    sniper = SniperRAG(idx)
    result = sniper.hit(query)
    logger.info("SniperRAG: hit=%s | conf=%.3f | pat=%s", result.sniper_hit, result.confidence, result.matched_pattern)
    return result