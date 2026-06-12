"""
src/analysis/decision_engine.py
N20a — Decision Engine (Yes/No + Classification Questions, OFFLINE)

Handles questions like:
  - "Is 3M capital-intensive based on FY2022?"
  - "Is Apple's liquidity healthy?"
  - "Is Tesla highly leveraged?"
  - "Is Amazon profitable?"
  - "Is Boeing solvent?"

Pipeline:
  1. Pattern-match the question against known decision patterns
  2. Extract required input metrics via extract_lib + raw_text fallback
  3. Compute the deciding ratio via maths_lib
  4. Fire the appropriate logic_lib L02 rule
  5. Format a natural-language answer with the ratio + classification

100% deterministic. ZERO LLM calls. Full audit trail.

Constraints satisfied:
  C1, C2 ($0, local)  |  C5 (no randomness)  |  C7 (n/a, no LLM)
  C8 (citations via raw_text positions)  |  C9 (no _rlef_ writes)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional lib imports — all soft, gracefully fall back if missing
try:
    from extract_lib.synonyms import METRIC_SYNONYMS as _EXT_SYNS
    _HAS_EXTRACT = True
except Exception:
    _EXT_SYNS = {}
    _HAS_EXTRACT = False

try:
    from logic_lib import fire as _logic_fire
    _HAS_LOGIC = True
except Exception:
    _logic_fire = None
    _HAS_LOGIC = False


# ─────────────────────────────────────────────────────────────────────────────
# Decision patterns — question phrasings mapped to (metric_inputs, rule_id)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionPattern:
    """One decision-question pattern."""
    pattern: str                   # regex on the lowercase question
    rule_id: str                   # logic_lib rule to fire
    rule_input_name: str           # the kwarg name the rule expects
    metric_a: str                  # extract_lib metric ID for numerator
    metric_b: Optional[str]        # extract_lib metric ID for denominator (None = use a directly)
    operation: str = "ratio_pct"   # "ratio_pct", "ratio", "value", "diff"
    label: str = ""                # human-readable name


DECISION_PATTERNS: List[DecisionPattern] = [
    # ── Capital intensity ────────────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|are)\s+\w+(\s+\w+)?\s+(a\s+)?capital[\s\-]intensive",
        rule_id="capital_intensity_class",
        rule_input_name="capex_to_revenue",
        metric_a="capex",
        metric_b="revenue",
        operation="ratio_pct",
        label="Capital intensity (CapEx / Revenue)",
    ),
    DecisionPattern(
        pattern=r"\bcapital[\s\-]intensive\b",
        rule_id="capital_intensity_class",
        rule_input_name="capex_to_revenue",
        metric_a="capex",
        metric_b="revenue",
        operation="ratio_pct",
        label="Capital intensity (CapEx / Revenue)",
    ),
    # ── Liquidity (current ratio) ────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|are)\s+\w+(\s+\w+)?\s+liquid(ity)?\b",
        rule_id="liquidity_current_ratio",
        rule_input_name="current_ratio",
        metric_a="current_assets",
        metric_b="current_liabilities",
        operation="ratio",
        label="Liquidity (Current Ratio)",
    ),
    DecisionPattern(
        pattern=r"\bcurrent\s+ratio\s+(healthy|good|strong|weak|adequate)\b",
        rule_id="liquidity_current_ratio",
        rule_input_name="current_ratio",
        metric_a="current_assets",
        metric_b="current_liabilities",
        operation="ratio",
        label="Liquidity (Current Ratio)",
    ),
    # ── Leverage (Debt / Equity) ─────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|are)\s+\w+(\s+\w+)?\s+(highly\s+)?leveraged\b",
        rule_id="leverage_debt_equity",
        rule_input_name="debt_to_equity",
        metric_a="long_term_debt",
        metric_b="shareholders_equity",
        operation="ratio",
        label="Leverage (Debt / Equity)",
    ),
    # ── Profitability (Net Margin) ───────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|are)\s+\w+(\s+\w+)?\s+profitabl[ye]\b",
        rule_id="profitability_net_margin",
        rule_input_name="net_margin",
        metric_a="net_income",
        metric_b="revenue",
        operation="ratio_pct",
        label="Profitability (Net Margin)",
    ),
    DecisionPattern(
        pattern=r"\bnet\s+margin\s+(healthy|good|strong|weak|excellent)\b",
        rule_id="profitability_net_margin",
        rule_input_name="net_margin",
        metric_a="net_income",
        metric_b="revenue",
        operation="ratio_pct",
        label="Profitability (Net Margin)",
    ),
    # ── ROE quality ──────────────────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(roe|return\s+on\s+equity)\s+(healthy|good|strong|weak|excellent)\b",
        rule_id="profitability_roe",
        rule_input_name="roe",
        metric_a="net_income",
        metric_b="shareholders_equity",
        operation="ratio_pct",
        label="Return on Equity (ROE)",
    ),
    # ── FCF positivity ───────────────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|does)\s+\w+(\s+\w+)?\s+generat\w*\s+(positive\s+)?free\s+cash\s+flow",
        rule_id="fcf_positive",
        rule_input_name="fcf",
        metric_a="free_cash_flow",
        metric_b=None,
        operation="value",
        label="Free Cash Flow",
    ),
    # ── Dividend sustainability ──────────────────────────────────────────
    DecisionPattern(
        pattern=r"\b(is|are)\s+\w+(\s+\w+)?\s+dividend(s)?\s+sustainable\b",
        rule_id="dividend_sustainability",
        rule_input_name="payout_ratio",
        metric_a="dividends_paid",
        metric_b="net_income",
        operation="ratio_pct",
        label="Dividend Sustainability (Payout Ratio)",
    ),
    # ── Working capital health ───────────────────────────────────────────
    DecisionPattern(
        pattern=r"\bworking\s+capital\s+(healthy|positive|negative|strong|weak)\b",
        rule_id="working_capital_health",
        rule_input_name="working_capital",
        metric_a="current_assets",
        metric_b="current_liabilities",
        operation="diff",
        label="Working Capital",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Number extraction (re-used from pipeline raw_text scan style)
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    # FIX-v5: \d+ instead of [\d]{1,3} so "1577" matches fully
    r"\(?\$?\s*\-?\s*\d+(?:,\d{3})*(?:\.\d+)?\)?"
    r"(?:\s*(?:million|billion|thousand|bn|mn|m|b))?",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[,;:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_number_to_float(s: str) -> Optional[float]:
    """Convert '$1,234.5' or '(1,234)' to 1234.5 / -1234. Returns None on fail."""
    if not s:
        return None
    raw = str(s).strip()
    neg = ("(" in raw and ")" in raw)
    raw = re.sub(r"[\$\s\(\)%,]", "", raw)
    # Strip trailing scale suffix
    m_scale = re.match(r"^([-\d\.]+)\s*(million|billion|thousand|bn|mn|m|b)?$",
                       raw, re.IGNORECASE)
    if not m_scale:
        return None
    try:
        v = float(m_scale.group(1))
    except (ValueError, TypeError):
        return None
    if neg:
        v = -abs(v)
    # Apply scale (we keep things in MILLIONS for consistency with extract_lib)
    suf = (m_scale.group(2) or "").lower()
    if suf in ("billion", "bn", "b"):
        v *= 1000.0  # billion → million
    elif suf in ("thousand",):
        v /= 1000.0  # thousand → million
    return v


def _extract_metric_from_raw_text(
    raw_text: str,
    metric_id: str,
) -> Optional[float]:
    """Reuse the same scan logic as pipeline._scan_raw_text_for_metric."""
    if not _HAS_EXTRACT or not raw_text:
        return None
    syn = _EXT_SYNS.get(metric_id, {})
    positives = syn.get("positive", [])
    if not positives:
        return None

    raw_norm = _norm(raw_text)
    floor = _MEGA_FLOORS.get(metric_id, 0.0)

    for synonym in sorted(positives, key=len, reverse=True):
        s_norm = _norm(synonym)
        idx = raw_norm.find(s_norm)
        if idx < 0:
            continue
        window_start = idx + len(s_norm)
        window_end = min(window_start + 300, len(raw_norm))
        window = raw_norm[window_start:window_end]
        # FIX-v13: collect ALL candidates in the window, skip year-like and
        # below-floor values, then for mega-metrics prefer the LARGEST.
        candidates = []
        for m in _NUMBER_RE.finditer(window):
            number_str = m.group(0).strip()
            digits_only = re.sub(r"[^\d]", "", number_str)
            if len(digits_only) < 2:
                continue
            val = _parse_number_to_float(number_str)
            if val is None:
                continue
            av = abs(val)
            # skip year-like
            if 1990 <= av <= 2030 and av == int(av):
                continue
            # skip below floor for mega metrics
            if av < floor:
                continue
            candidates.append(val)
        if not candidates:
            continue
        if metric_id in _LARGEST_WINS:
            # Balance-sheet dominance: the LARGEST plausible value wins
            # (e.g. $8,700M PPE beats a stray "63" note reference).
            return max(candidates, key=lambda x: abs(x))
        # MOVE-5 fix (2026-06-12): flow/income metrics (capex, net income,
        # operating income) take the FIRST sane number after the synonym —
        # PROXIMITY wins. "Capex totaled $1,749M. Net sales were $34,229M"
        # sits in one window; largest-wins returned 34,229 for CAPEX and
        # produced the absurd "capital intensity 100.0%" self-test answer.
        return candidates[0]
    return None


def _extract_metric_from_cells(
    cells: List[Dict],
    metric_id: str,
    period: str = "",
) -> Optional[float]:
    """Use extract_lib resolver on table_cells if available."""
    if not _HAS_EXTRACT:
        return None
    try:
        from extract_lib.resolver import resolve_metric
        result = resolve_metric(metric_id, cells, period or "")
        if result and result.valid and result.value is not None:
            return float(result.value)
    except Exception:
        logger.debug("[decision_engine] extract_lib.resolve_metric failed", exc_info=True)
    return None


# FIX-v13 (2026-06-07): magnitude floors so the raw_text fallback can't
# return a tiny wrong value (e.g. revenue=1.0 from "net sales rose 1.0%").
_MEGA_FLOORS = {
    "revenue": 50.0,
    "total_assets": 50.0,
    "total_liabilities": 50.0,
    "shareholders_equity": 20.0,
    "ppe": 20.0,
    "cogs": 20.0,
    "current_assets": 20.0,
    "current_liabilities": 20.0,
    "long_term_debt": 5.0,
    "net_income": 5.0,
    "operating_income": 5.0,
    "capex": 5.0,
}

# MOVE-5 (2026-06-12): only BALANCE-SHEET items keep "largest wins" — their
# true value dominates stray note references. Flow/income items (capex,
# net_income, operating_income) must use PROXIMITY instead, because their
# window often also contains the much larger revenue figure.
_LARGEST_WINS = {
    "revenue", "total_assets", "total_liabilities", "shareholders_equity",
    "ppe", "cogs", "current_assets", "current_liabilities",
}


def _get_metric_value(
    metric_id: str,
    cells: List[Dict],
    raw_text: str,
    period: str,
) -> Optional[float]:
    """Try cells first (magnitude-aware via FIX-v13 resolver), then raw_text.

    FIX-v13: enforce a magnitude floor on BOTH paths so a tiny wrong value
    never wins for a mega-metric.
    """
    floor = _MEGA_FLOORS.get(metric_id, 0.0)

    v = _extract_metric_from_cells(cells, metric_id, period)
    if v is not None and abs(v) >= floor:
        return v

    v2 = _extract_metric_from_raw_text(raw_text, metric_id)
    if v2 is not None and abs(v2) >= floor:
        return v2

    # Neither cleared the floor. Return the cells value if it exists (better
    # provenance), else the raw_text value, else None. Downstream sanity
    # checks will abstain if it's still absurd.
    if v is not None:
        return v
    return v2


# ─────────────────────────────────────────────────────────────────────────────
# Decision pattern detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_decision_pattern(question: str) -> Optional[DecisionPattern]:
    """Return the first matching DecisionPattern, or None."""
    if not question:
        return None
    q = question.lower()
    for pat in DECISION_PATTERNS:
        if re.search(pat.pattern, q):
            return pat
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionAnswer:
    answered: bool
    output: str = ""               # natural-language answer
    classification: str = ""       # the L02 branch label (e.g. "moderately capital-intensive")
    metric_value: Optional[float] = None
    rule_id: str = ""
    pattern_label: str = ""
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


def run_decision_engine(
    question: str,
    cells: List[Dict],
    raw_text: str,
    company: str = "",
    fiscal_year: str = "",
    doc_type: str = "",
) -> DecisionAnswer:
    """
    Main entry point. Returns DecisionAnswer with answered=True if any
    decision pattern matched AND inputs were found AND the rule fired.

    Never raises. Returns DecisionAnswer(answered=False) on any failure.
    """
    if not _HAS_LOGIC:
        logger.debug("[decision_engine] logic_lib not installed")
        return DecisionAnswer(answered=False)

    pat = detect_decision_pattern(question)
    if pat is None:
        return DecisionAnswer(answered=False)

    logger.info(
        "[decision_engine] Matched pattern '%s' → rule=%s",
        pat.label, pat.rule_id,
    )

    # ── Extract inputs ──────────────────────────────────────────────────
    val_a = _get_metric_value(pat.metric_a, cells, raw_text, fiscal_year)
    if val_a is None:
        logger.info("[decision_engine] Failed to extract %s", pat.metric_a)
        return DecisionAnswer(
            answered=False,
            audit_trail={"reason": f"missing input {pat.metric_a}"},
        )

    val_b: Optional[float] = None
    if pat.metric_b:
        val_b = _get_metric_value(pat.metric_b, cells, raw_text, fiscal_year)
        if val_b is None or val_b == 0:
            logger.info(
                "[decision_engine] Failed to extract %s (or zero)",
                pat.metric_b,
            )
            return DecisionAnswer(
                answered=False,
                audit_trail={"reason": f"missing input {pat.metric_b}"},
            )

    # ── Compute deciding metric ─────────────────────────────────────────
    metric_value: float
    if pat.operation == "ratio_pct":
        metric_value = abs(val_a) / abs(val_b) * 100.0   # type: ignore[arg-type]
    elif pat.operation == "ratio":
        metric_value = abs(val_a) / abs(val_b)            # type: ignore[arg-type]
    elif pat.operation == "diff":
        metric_value = val_a - val_b                       # type: ignore[arg-type]
    else:   # "value"
        metric_value = val_a

    # FIX-v5: SANITY BOUNDS — reject absurd ratios (e.g. 102882% capital intensity)
    sanity_bounds = {
        "ratio_pct": (-1000.0, 1000.0),
        "ratio":     (-1000.0, 1000.0),
        "diff":      (-1e9, 1e9),
        "value":     (-1e12, 1e12),
    }
    lo, hi = sanity_bounds.get(pat.operation, (-1e12, 1e12))
    if not (lo <= metric_value <= hi):
        logger.warning(
            "[decision_engine] ABSTAIN: %s = %.2f outside [%s,%s] | val_a=%s val_b=%s",
            pat.label, metric_value, lo, hi, val_a, val_b,
        )
        return DecisionAnswer(
            answered=False,
            audit_trail={"reason": "absurd_ratio", "val_a": val_a, "val_b": val_b,
                          "computed": metric_value},
        )

    # FIX-v5: revenue cell must be >= $50M (else it's a wrong cell pick)
    if pat.metric_a == "revenue" and val_a is not None and abs(val_a) < 50.0:
        logger.warning("[decision_engine] ABSTAIN: revenue extracted as %s (too small)", val_a)
        return DecisionAnswer(answered=False, audit_trail={"reason": "revenue_too_small", "val_a": val_a})
    if pat.metric_b == "revenue" and val_b is not None and abs(val_b) < 50.0:
        logger.warning("[decision_engine] ABSTAIN: revenue extracted as %s (too small)", val_b)
        return DecisionAnswer(answered=False, audit_trail={"reason": "revenue_too_small", "val_b": val_b})

    logger.info(
        "[decision_engine] computed | %s | val_a=%s val_b=%s → %.4f (op=%s)",
        pat.label, val_a, val_b, metric_value, pat.operation,
    )

    # ── Fire the logic_lib rule ─────────────────────────────────────────
    try:
        rule_result = _logic_fire(
            pat.rule_id,
            **{pat.rule_input_name: metric_value},
        )
    except Exception as exc:
        logger.exception(
            "[decision_engine] logic_lib.fire('%s') failed: %s",
            pat.rule_id, exc,
        )
        return DecisionAnswer(
            answered=False,
            audit_trail={"reason": f"rule_fire_failed: {exc}"},
        )

    if not rule_result or not getattr(rule_result, "fired", False):
        return DecisionAnswer(
            answered=False,
            audit_trail={"reason": "rule did not fire"},
        )

    classification = str(getattr(rule_result, "output", ""))
    branch         = str(getattr(rule_result, "branch", ""))

    # ── Format natural-language answer ──────────────────────────────────
    unit_suffix = "%" if pat.operation == "ratio_pct" else ""
    if pat.operation == "ratio":
        value_str = f"{metric_value:.2f}"
    elif pat.operation == "diff":
        value_str = f"${abs(metric_value):,.0f}M"
        if metric_value < 0:
            value_str = "-" + value_str
    else:
        value_str = f"{metric_value:.1f}{unit_suffix}"

    # First letter capitalised, end with period.
    classification_cap = (
        classification[:1].upper() + classification[1:]
        if classification else "Determined"
    )
    output = f"{classification_cap}. {pat.label}: {value_str}."

    citation = f"{company}/{doc_type}/{fiscal_year}/{pat.label.replace(' ', '_')}/{branch}"
    output = f"{output} [{citation}]"

    return DecisionAnswer(
        answered=True,
        output=output,
        classification=classification,
        metric_value=metric_value,
        rule_id=pat.rule_id,
        pattern_label=pat.label,
        audit_trail={
            "rule_id":           pat.rule_id,
            "rule_input_name":   pat.rule_input_name,
            "metric_a":          pat.metric_a,
            "metric_b":          pat.metric_b,
            "val_a":             val_a,
            "val_b":             val_b,
            "computed_value":    metric_value,
            "branch":            branch,
            "operation":         pat.operation,
        },
        confidence=0.85,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with: python -m src.analysis.decision_engine)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_question = "Is 3M a capital-intensive business based on FY2022 data?"
    test_raw_text = (
        "Capital expenditures for FY2022 totaled $1,749 million for property, "
        "plant and equipment. Net sales for the year were $34,229 million."
    )

    print("\n[decision_engine] self-test")
    print(f"  Q: {test_question}")
    print(f"  logic_lib available: {_HAS_LOGIC}")
    print(f"  extract_lib available: {_HAS_EXTRACT}")

    ans = run_decision_engine(
        question=test_question,
        cells=[],
        raw_text=test_raw_text,
        company="3M",
        fiscal_year="FY2022",
        doc_type="10-K",
    )

    print(f"\n  answered:        {ans.answered}")
    print(f"  classification:  {ans.classification}")
    print(f"  output:          {ans.output}")
    print(f"  metric_value:    {ans.metric_value}")
    print(f"  rule_id:         {ans.rule_id}")
    print(f"  confidence:      {ans.confidence}")
    print(f"  audit_trail:     {ans.audit_trail}")
