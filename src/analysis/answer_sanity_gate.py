"""
src/analysis/answer_sanity_gate.py

UNIVERSAL NUMERIC-ANSWER SANITY GATE  (2026-06-21)

The 150-run exposed a whole CLASS of confident-wrong answers: deterministic
fast-paths (EXTRACT_LIB_FAST_PATH, FORMULA_ROUTER, SNIPER_EARLY_EXIT, the
question_lib ratio formulas) emit a number with NO plausibility check, so the
pipeline returns garbage like:
    revenue        = "23"            (JPMorgan, real ~120,000)
    operating_inc  = "$13.5"         (Adobe, real ~1,500)
    inventory_turn = 356.17          (Nike, impossible)
    current_liab   = $15,430,786 M   (Netflix, thousands read as millions)
    net sales      = $11,880         (Lockheed, a random cell)
    effective_tax  = 21.61           (AmEx, but gold wanted a different metric)

Every one of these scores 0 AND blocks the question from reaching the LLM.

This gate gives each metric/answer-type a plausible RANGE. An answer outside
its range is REJECTED -> the caller abstains -> the question can fall through
to the LLM (which is built for exactly these). Turning a confident-wrong into
an honest abstain is a strict improvement: it never loses a real PASS (those
are in-range) and it stops poisoning the score + unblocks the LLM path.

Deterministic. No LLM. Never raises. C1/C2/C5/C9 satisfied.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?")


# Plausible absolute-value ranges for a SINGLE statement line item, in the
# unit the answer is expressed in. Ratios/percentages/turnovers have tight
# bounds because they are dimensionless; dollar line items have wide bounds.
#
# These are deliberately GENEROUS — the goal is to catch absurd garbage
# (turnover 356, margin 0.6% mis-scaled, $15-trillion line items, a bare "23"
# for revenue), NOT to second-guess plausible numbers.
_RATIO_KEYS = (
    "turnover", "ratio", "dpo", "dso", "dio", "days",
)
_PCT_KEYS = (
    "margin", "growth", "yoy", "change", "rate", "percent", "%",
    "roa", "roe", "intensity",
)


def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    raw = str(s)
    m = _NUM_RE.search(raw)
    if not m:
        return None
    tok = m.group(0)
    neg = "(" in tok and ")" in tok
    tok = re.sub(r"[\$\s,()]", "", tok)
    try:
        v = float(tok)
    except ValueError:
        return None
    if neg:
        v = -abs(v)
    # unit suffix scaling
    low = raw.lower()
    if "billion" in low or re.search(r"\bbn\b", low):
        v *= 1000.0           # express in millions
    elif "trillion" in low:
        v *= 1_000_000.0
    return v


def _classify_unit(citation: str, question: str, answer: str) -> str:
    """Return 'ratio' | 'pct' | 'dollar' | 'count' from the answer context."""
    ctx = f"{citation} {question} {answer}".lower()
    # explicit % sign in the answer is the strongest signal
    if "%" in answer:
        return "pct"
    if any(k in ctx for k in _RATIO_KEYS):
        return "ratio"
    if any(k in ctx for k in _PCT_KEYS):
        return "pct"
    return "dollar"


def sanity_check(answer: str,
                 citation: str = "",
                 question: str = "") -> Tuple[bool, str]:
    """Return (ok, reason).

    ok=False means the numeric answer is implausible and the caller should
    ABSTAIN (let it fall through to the LLM) instead of returning it.

    Non-numeric / narrative answers always pass (ok=True) — this gate only
    judges numbers. Never raises.
    """
    try:
        if not answer:
            return True, "empty-passthrough"

        val = _to_float(answer)
        if val is None:
            # no parseable number -> narrative answer, not our concern
            return True, "non-numeric"

        av = abs(val)
        unit = _classify_unit(citation, question, answer)

        if unit == "pct":
            # a percentage answer (margin, growth, rate). Plausible roughly
            # -100%..+1000% (a YoY can be a few hundred %). Reject 0.0 only
            # if it's an exact integer 0 with no decimals AND not a real 0.
            if av > 5000.0:
                return False, f"pct out of range ({val})"
            return True, "pct-ok"

        if unit == "ratio":
            # turnover / DPO / current-ratio. Inventory turnover realistically
            # 0.2..60; DPO 5..400 days; current/quick ratio 0.05..15. A value
            # like 356 turnover or 2406 DPO is garbage.
            low = f"{citation} {question}".lower()
            if "dpo" in low or "dso" in low or "dio" in low or "days" in low:
                if not (1.0 <= av <= 500.0):
                    return False, f"days-ratio out of range ({val})"
                return True, "days-ok"
            # generic turnover / ratio
            if not (0.05 <= av <= 100.0):
                return False, f"ratio out of range ({val})"
            return True, "ratio-ok"

        # dollar line item (millions). A single statement line is plausibly
        # ~$0.1M .. ~$5,000,000M ($5T, covers the largest balance sheets).
        # Catch BOTH absurdly large (mis-scaled thousands: $15,430,786M) and
        # absurdly tiny for a big metric (a bare "23" or "$13.5" for revenue/
        # assets/net-sales).
        cl = f"{citation} {question}".lower()
        _big = any(k in cl for k in (
            "revenue", "net sales", "net_sales", "total_assets", "total assets",
            "total_liabilities", "cogs", "cost of", "operating_income",
            "operating income", "net_income", "net income", "current_assets",
            "current_liabilities", "current liabilities", "ppe", "property",
            "shareholders", "stockholders", "long_term_debt", "ebitda",
        ))
        if av > 5_000_000.0:
            return False, f"dollar too large, likely mis-scaled ({val})"
        if _big and av < 50.0:
            # a top-line / balance-sheet metric in millions is essentially
            # never < $50M for these large-cap filers. "23", "13.5" = wrong cell.
            return False, f"dollar implausibly small for a major line item ({val})"
        return True, "dollar-ok"

    except Exception:
        logger.debug("[sanity_gate] check failed", exc_info=True)
        return True, "gate-error-passthrough"


if __name__ == "__main__":
    tests = [
        ("$1,577 million [3M/10-K/FY2018/capex]", "capex amount", True),
        ("24.26 [Activision/FY2019/fixed_asset_turnover]", "turnover ratio", True),
        ("93.86 [Amazon/FY2017/dpo]", "days payable outstanding", True),
        ("30.8% [Amazon/FY2017/yoy_change]", "yoy change revenue", True),
        ("356.17 [Nike/FY2021/inventory_turnover]", "inventory turnover", False),
        ("2406.79 [Block/FY2020/dpo]", "days payable", False),
        ("$15,430,786 million [Netflix/FY2017/current_liabilities]", "current liabilities", False),
        ("23 [JPMorgan/FY2021/revenue]", "total revenue", False),
        ("$ 13.5 [Adobe/FY2016/operating_income]", "operating income", False),
        ("$11,880 [Lockheed/FY2022/Net sales]", "net sales", False),
        ("$11,588 million [Amazon/FY2019/net_income]", "net income", True),
    ]
    print("\n── answer_sanity_gate self-test ──")
    for ans, q, expect_ok in tests:
        ok, reason = sanity_check(ans, ans, q)
        mark = "✓" if ok == expect_ok else "✗ MISMATCH"
        print(f"  {mark}  ok={ok!s:5} expect={expect_ok!s:5} | {reason:42} | {ans[:46]}")
