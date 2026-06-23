"""
src/analysis/answer_sanity_gate.py

UNIVERSAL NUMERIC-ANSWER SANITY GATE  (2026-06-21, recalibrated 2026-06-22)

Rejects IMPOSSIBLE numeric answers (turnover 356, DPO 2406, $15-trillion line
items, a bare "23" for revenue) so the question abstains / falls through to the
LLM instead of returning confident-wrong garbage.

2026-06-22 FIX — the first version was TOO aggressive and classified correct
DOLLAR answers as "ratios" (because the question mentioned 'ratio'/'turnover'),
then rejected them for being "too large for a ratio". That zeroed the score:
Costco "$59,268" and General Mills "$3,215.4 million" were GOLD answers wrongly
killed. The classifier now lets the UNIT IN THE ANSWER ($ / million / billion)
win over question keywords, and only judges a value as a ratio/pct when the
answer carries no money marker.

Deterministic. No LLM. Never raises. C1/C2/C5/C9 satisfied.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?")

_RATIO_KEYS = ("turnover", "ratio", "dpo", "dso", "dio", "days")
_PCT_KEYS = (
    "margin", "growth", "yoy", "change", "rate", "percent", "%",
    "roa", "roe", "intensity",
)
_DOLLAR_METRIC_KEYS = (
    "revenue", "net sales", "net_sales", "net income", "net_income",
    "cogs", "cost of", "operating income", "operating_income",
    "total assets", "total_assets", "total liabilities", "total_liabilities",
    "current assets", "current_assets", "current liabilities",
    "current_liabilities", "inventory", "accounts payable", "accounts_payable",
    "ppe", "property", "cash", "ebitda", "dividends_paid", "dividends paid",
    "capex", "capital expenditure", "shareholders", "stockholders",
    "long_term_debt", "long-term debt",
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
    low = raw.lower()
    if "billion" in low or re.search(r"\bbn\b", low):
        v *= 1000.0
    elif "trillion" in low:
        v *= 1_000_000.0
    return v


def _classify_unit(citation: str, question: str, answer: str) -> str:
    """Return 'ratio' | 'pct' | 'dollar'.

    The UNIT IN THE ANSWER wins over question keywords. A value with a '$' sign
    or 'million'/'billion'/'thousand' is a DOLLAR amount even if the question
    says 'ratio'/'turnover'. Only a marker-less answer is judged by keywords.
    """
    ans = (answer or "").lower()
    ctx = f"{citation} {question} {answer}".lower()

    # 1) explicit % sign in the answer => percentage
    if "%" in answer:
        return "pct"

    # 2) money marker in the ANSWER => dollar, regardless of question wording
    if "$" in answer or any(u in ans for u in (
        "million", "billion", "thousand", " mn", " bn",
    )):
        return "dollar"

    # 3) citation metric is a known dollar line item AND no ratio/pct word
    has_dollar_metric = any(k in ctx for k in _DOLLAR_METRIC_KEYS)
    has_ratio_pct_word = any(k in ctx for k in (
        "turnover", "dpo", "dso", "dio", "days", "margin", "ratio",
        "per share", "growth", "yoy", "rate",
    ))
    if has_dollar_metric and not has_ratio_pct_word:
        return "dollar"

    # 4) marker-less answer => use context keywords
    if any(k in ctx for k in _RATIO_KEYS):
        return "ratio"
    if any(k in ctx for k in _PCT_KEYS):
        return "pct"
    return "dollar"


def sanity_check(answer: str,
                 citation: str = "",
                 question: str = "") -> Tuple[bool, str]:
    """Return (ok, reason). ok=False => caller should ABSTAIN.

    Only judges numbers; narrative answers always pass. Never raises.
    """
    try:
        if not answer:
            return True, "empty-passthrough"

        val = _to_float(answer)
        if val is None:
            return True, "non-numeric"

        av = abs(val)
        unit = _classify_unit(citation, question, answer)

        if unit == "pct":
            # percentage answer (margin/growth/rate). Allow wide band; only
            # reject the truly impossible (a raw dollar count grabbed as a %).
            if av > 100_000.0:
                return False, f"pct out of range ({val})"
            return True, "pct-ok"

        if unit == "ratio":
            low = f"{citation} {question}".lower()
            if any(k in low for k in ("dpo", "dso", "dio", "days")):
                if not (1.0 <= av <= 1000.0):
                    return False, f"days-ratio out of range ({val})"
                return True, "days-ok"
            # generic turnover / current ratio: a real turnover is < ~150 and
            # a real current/quick ratio is < ~15. 356 (Nike) is impossible.
            if not (0.01 <= av <= 150.0):
                return False, f"ratio out of range ({val})"
            return True, "ratio-ok"

        # dollar line item (millions). Plausibly up to ~$5,000,000M ($5T) for
        # the largest filers. Above that is a mis-scaled thousands/whole-dollar
        # value (Netflix 15,430,786) -> reject.
        if av > 5_000_000.0:
            return False, f"dollar too large, likely mis-scaled ({val})"
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
        ("$59,268 [Costco/FY2021]", "merchandise costs", True),
        ("$3,215.4 million [General Mills/FY2020]", "net income", True),
        ("$11,588 million [Amazon/FY2019/net_income]", "net income", True),
        ("356.17 [Nike/FY2021/inventory_turnover]", "inventory turnover", False),
        ("2406.79 [Block/FY2020/dpo]", "days payable", False),
        ("$15,430,786 million [Netflix/FY2017/current_liabilities]", "current liabilities", False),
    ]
    print("\n-- answer_sanity_gate self-test --")
    for ans, q, expect_ok in tests:
        ok, reason = sanity_check(ans, ans, q)
        mark = "OK " if ok == expect_ok else "XX MISMATCH"
        print(f"  {mark} ok={ok!s:5} expect={expect_ok!s:5} | {reason:38} | {ans[:46]}")
