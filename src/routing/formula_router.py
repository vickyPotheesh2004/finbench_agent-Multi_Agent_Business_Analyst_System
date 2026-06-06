"""
N06b Formula Router
═══════════════════════════════════════════════════════════════════
Sits right after N06 Sniper. If the question is a known financial
formula (ratio / margin / growth / turnover), it:
  1. Detects which formula the question asks for
  2. Extracts the required inputs from Sniper's table_cells
  3. Calls maths_lib deterministically
  4. Returns the answer, bypassing the LLM entirely

Pure deterministic. No LLM. No network. Auditable.

2026-06-04 FIX-A: Use extract_lib.resolver.resolve_metric for proper
                  scored cell extraction. Fixes 'revenue:53.8' bug where
                  naive substring matching picked a margin row (53.8%)
                  instead of the actual revenue total (34,229).
"""
from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import maths_lib as ml
    _HAS_MATHS = True
except Exception:
    ml = None
    _HAS_MATHS = False

try:
    from extract_lib.resolver import resolve_metric as _resolve_metric
    _HAS_EXTRACT = True
except Exception:
    _resolve_metric = None
    _HAS_EXTRACT = False


# ── Formula detection: question text -> formula_id ──────────────────
# Maps natural-language phrasing to maths_lib formula IDs.
FORMULA_KEYWORDS: Dict[str, List[str]] = {
    "gross_margin":        ["gross margin", "gross profit margin"],
    "net_margin":          ["net margin", "net profit margin", "net income margin"],
    "operating_margin":    ["operating margin", "operating profit margin"],
    "ebitda_margin":       ["ebitda margin"],
    "current_ratio":       ["current ratio"],
    "quick_ratio":         ["quick ratio", "acid test", "acid-test"],
    "cash_ratio":          ["cash ratio"],
    "debt_to_equity":      ["debt to equity", "debt-to-equity", "d/e ratio"],
    "debt_to_assets":      ["debt to assets", "debt-to-assets"],
    "return_on_equity":    ["return on equity", "roe"],
    "return_on_assets":    ["return on assets", "roa"],
    "inventory_turnover":  ["inventory turnover"],
    "asset_turnover":      ["asset turnover"],
    "interest_coverage":   ["interest coverage", "times interest earned"],
    "effective_tax_rate":  ["effective tax rate"],
    "dividend_payout_ratio": ["payout ratio", "dividend payout"],
    "working_capital":     ["working capital"],
}


# ── Input extraction: formula input name -> row_header synonyms ─────
INPUT_SYNONYMS: Dict[str, List[str]] = {
    "revenue":             ["net sales", "total revenue", "net revenue",
                            "total net sales", "revenue", "sales", "total sales"],
    "cogs":                ["cost of sales", "cost of goods sold",
                            "cost of revenue", "cost of products sold", "cogs"],
    "net_income":          ["net income", "net earnings",
                            "profit for the year", "net income attributable"],
    "operating_income":    ["operating income", "income from operations",
                            "operating profit"],
    "ebitda":              ["ebitda"],
    "current_assets":      ["total current assets"],
    "current_liabilities": ["total current liabilities"],
    "cash":                ["cash and cash equivalents", "cash equivalents", "cash"],
    "inventory":           ["inventory", "inventories", "total inventory"],
    "total_assets":        ["total assets"],
    "total_debt":          ["total debt", "long-term debt", "total borrowings"],
    "shareholders_equity": ["total stockholders equity", "shareholders equity",
                            "total equity", "stockholders equity"],
    "interest_expense":    ["interest expense", "interest paid"],
    "ebit":                ["operating income", "ebit", "income from operations"],
    "tax_expense":         ["income tax expense", "provision for income taxes",
                            "tax expense"],
    "pretax_income":       ["income before taxes", "pretax income",
                            "income before income taxes"],
    "dividends":           ["dividends paid", "dividends declared", "total dividends"],
    "average_inventory":   ["inventory", "inventories"],
}


def _parse_number(s: Any) -> Optional[float]:
    """Parse '34,229' / '(1,234)' / '$5.2' into float. Parens = negative."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _detect_formula(question: str) -> Optional[str]:
    q = (question or "").lower()

    # FIX-v4 (2026-06-05): Skip formula router entirely when the question
    # is asking for EXPLANATION / DRIVERS / NARRATIVE rather than a
    # numeric ratio value. The router has no LLM; it can only return a
    # number. Returning a number to a "why" question scores zero AND
    # poisons the answer with garbage like "384647.06".
    narrative_markers = (
        "why",
        "what drove",
        "what caused",
        "explain",
        "reason for",
        "reasons for",
        "drivers of",
        "factors",
        "contributed to",
        "due to",
        "is it",
        "is the",
        "is this",
        "discuss",
        "describe",
        "is x a y",   # placeholder — specific phrasing checked below
    )
    if any(m in q for m in narrative_markers):
        return None
    # "Is 3M a capital-intensive business" — detect with regex
    if re.search(r"\bis\s+\w+\s+(a|an)\s+\w+(-?\w+)?\s+(business|company)", q):
        return None

    # Prefer longer keyword matches first (more specific)
    matches = []
    for fid, kws in FORMULA_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                matches.append((len(kw), fid))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


# Map formula input names to extract_lib metric IDs.
# extract_lib uses metric IDs like "revenue", "net_income" already, so this is mostly identity.
_INPUT_TO_METRIC: Dict[str, str] = {
    "revenue":             "revenue",
    "cogs":                "cogs",
    "net_income":          "net_income",
    "operating_income":    "operating_income",
    "ebitda":              "ebitda",
    "current_assets":      "current_assets",
    "current_liabilities": "current_liabilities",
    "cash":                "cash",
    "inventory":           "inventory",
    "total_assets":        "total_assets",
    "total_debt":          "long_term_debt",
    "shareholders_equity": "shareholders_equity",
    "interest_expense":    "interest_expense",
    "ebit":                "operating_income",
    "tax_expense":         "income_tax",
    "pretax_income":       "income_before_tax",
    "dividends":           "dividends_paid",
    "average_inventory":   "inventory",
}


def _find_input(cells: List[Dict], input_name: str, period: str = "") -> Optional[float]:
    """
    Find a formula input value from table cells.

    FIX-A (2026-06-04): Primary path now uses extract_lib's scored
    resolve_metric (rejects decoys, scores by anti-pattern match).
    Falls back to legacy substring matching only if extract_lib unavailable
    or returns no result.
    """
    # ── Path 1: extract_lib (scored, anti-pattern aware) ────────────────
    if _HAS_EXTRACT and _resolve_metric is not None:
        metric_id = _INPUT_TO_METRIC.get(input_name, input_name)
        try:
            result = _resolve_metric(metric_id, cells, period or "")
            if result and result.valid and result.value is not None:
                logger.debug(
                    "formula_router: extract_lib resolved '%s' = %s (conf=%.2f)",
                    input_name, result.value, getattr(result, "confidence", 0.0),
                )
                return float(result.value)
        except Exception:
            logger.exception("formula_router: extract_lib.resolve_metric failed")

    # ── Path 2: legacy substring matching (fallback only) ───────────────
    keywords = INPUT_SYNONYMS.get(input_name, [input_name.replace("_", " ")])
    # exact row_header match first, then substring
    for exact in (True, False):
        for cell in cells:
            rh = str(cell.get("row_header", "")).lower().strip()
            if not rh:
                continue
            for kw in keywords:
                if (exact and kw == rh) or (not exact and kw in rh):
                    v = _parse_number(cell.get("value"))
                    if v is not None:
                        return v
    return None


def run_formula_router(state):
    """N06b: deterministic formula answering. Sets state.formula_hit."""
    state.formula_hit = False
    state.formula_answer = ""
    state.formula_result = None

    if not _HAS_MATHS:
        return state

    question = getattr(state, "query", "") or ""
    fid = _detect_formula(question)
    if not fid or fid not in ml.FORMULA_REGISTRY:
        return state

    cells = getattr(state, "table_cells", None) or []
    if not cells:
        return state

    meta = ml.FORMULA_METADATA_REGISTRY.get(fid, {})
    needed = meta.get("inputs", [])
    if not needed:
        return state

    # FIX-A: Pass fiscal_year as period hint to extract_lib for scoring boost
    period_hint = getattr(state, "fiscal_year", "") or ""

    inputs: Dict[str, float] = {}
    missing: List[str] = []
    for inp in needed:
        v = _find_input(cells, inp, period=period_hint)
        if v is None:
            missing.append(inp)
        else:
            inputs[inp] = v

    if missing:
        # Can't satisfy this formula from extracted cells — let normal flow run
        return state

    try:
        result = ml.FORMULA_REGISTRY[fid](**inputs)
    except Exception:
        return state

    if not result.valid or result.value is None:
        return state

    # FIX-v3 (2026-06-05): Sanity bounds — reject obviously-wrong formula
    # results that come from picking the wrong cells. Without this guard
    # the formula router can return things like "12154.28%" gross margin
    # which scores 0 against the gold answer.
    val_for_check = float(result.value)
    unit_for_check = (result.unit or "").strip()
    if unit_for_check == "%":
        # Margins/ratios as % are typically 0-100, occasionally up to 200.
        # Anything above 500% is almost certainly wrong inputs.
        if abs(val_for_check) > 500.0:
            logger.warning(
                "formula_router: rejected garbage result %.2f%% for %s (likely wrong inputs)",
                val_for_check, fid,
            )
            return state
    elif unit_for_check in ("x", ""):
        # Pure ratios should rarely exceed 100x; reject if exceeds 10000
        if abs(val_for_check) > 10000.0 and fid in (
            "current_ratio", "quick_ratio", "cash_ratio",
            "debt_to_equity", "debt_to_assets",
            "interest_coverage", "asset_turnover", "inventory_turnover",
        ):
            logger.warning(
                "formula_router: rejected garbage ratio %.2f for %s (likely wrong inputs)",
                val_for_check, fid,
            )
            return state

    # Format the answer
    val = result.value
    unit = result.unit or ""
    if unit == "%":
        answer = f"{val:.2f}%"
    elif unit in ("x", ""):
        answer = f"{val:.2f}"
    else:
        answer = f"{val:.2f} {unit}".strip()

    state.formula_hit = True
    state.formula_answer = answer
    state.formula_result = {
        "formula_id":   result.formula_id,
        "formula_name": result.formula_name,
        "expression":   result.expression,
        "inputs_used":  result.inputs_used,
        "value":        result.value,
        "unit":         result.unit,
    }
    # Set as the final answer (deterministic path)
    state.final_answer = answer
    state.final_answer_pre_xgb = answer
    state.confidence_score = 0.95
    state.winning_pod = "FORMULA_ROUTER"
    return state