"""
src/utils/lib_bridge.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Lib Bridge — single source of truth for the 6 FinBench support libraries.

Each lib is imported with a try/except so the system runs gracefully
even when a lib is not installed. Use the *_AVAILABLE flags to gate
optional functionality.

Libraries
─────────
  1. maths_lib       — 200+ deterministic financial formulas
  2. extract_lib     — scored cell extraction with anti-pattern rejection
  3. pattern_lib     — question classification + period detection
  4. format_lib      — deterministic answer formatting
  5. logic_lib       — 195 rules across 6 layers (forensic + sanity)
  6. algo_lib        — advanced finance algorithms registry
  7. verify_lib      — final-answer verification + abstention

Constraints satisfied
─────────────────────
  C1  $0 cost — all libs are open-source, no paid APIs
  C2  100% local — pure Python, no network
  C5  no randomness — deterministic libs
  C9  no _rlef_ field handling — libs operate on public state only

2026-06-04 v2: FIXED verify_lib and logic_lib API signatures after
               testing showed they differ from initial assumptions:
                 - verify_lib.verify_answer takes (metric, value), not (answer)
                 - VerifyVerdict has .ok/.abstain/.answer/.reasons (not .passed)
                 - logic_lib.fire takes (rule_id), not (layer)
                 - Added list_by_layer enumeration for layer-based firing
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. maths_lib
# ─────────────────────────────────────────────────────────────────────────────
try:
    import maths_lib as _maths_lib
    MATHS_LIB_AVAILABLE = True
except Exception:
    _maths_lib = None
    MATHS_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 2. extract_lib
# ─────────────────────────────────────────────────────────────────────────────
try:
    import extract_lib as _extract_lib
    from extract_lib.resolver import resolve_metric as _resolve_metric  # noqa: F401
    EXTRACT_LIB_AVAILABLE = True
except Exception:
    _extract_lib = None
    _resolve_metric = None
    EXTRACT_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 3. pattern_lib
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pattern_lib as _pattern_lib
    from pattern_lib import classify as _pattern_classify  # noqa: F401
    PATTERN_LIB_AVAILABLE = True
except Exception:
    _pattern_lib = None
    _pattern_classify = None
    PATTERN_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 4. format_lib
# ─────────────────────────────────────────────────────────────────────────────
try:
    import format_lib as _format_lib
    from format_lib import render as _format_render  # noqa: F401
    FORMAT_LIB_AVAILABLE = True
except Exception:
    _format_lib = None
    _format_render = None
    FORMAT_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 5. logic_lib  —  fire() takes rule_id, NOT layer
# ─────────────────────────────────────────────────────────────────────────────
try:
    import logic_lib as _logic_lib
    from logic_lib import fire as _logic_fire                # noqa: F401
    from logic_lib import list_by_layer as _logic_list_by_layer  # noqa: F401
    LOGIC_LIB_AVAILABLE = True
except Exception:
    _logic_lib = None
    _logic_fire = None
    _logic_list_by_layer = None
    LOGIC_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 6. algo_lib (package name: fina_algo_lib)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import fina_algo_lib as _algo_lib
    from fina_algo_lib import get_algorithm as _get_algorithm  # noqa: F401
    ALGO_LIB_AVAILABLE = True
except Exception:
    _algo_lib = None
    _get_algorithm = None
    ALGO_LIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 7. verify_lib  —  verify_answer takes (metric, value), returns VerifyVerdict
# ─────────────────────────────────────────────────────────────────────────────
try:
    import verify_lib as _verify_lib
    from verify_lib import verify_answer as _verify_answer  # noqa: F401
    VERIFY_LIB_AVAILABLE = True
except Exception:
    _verify_lib = None
    _verify_answer = None
    VERIFY_LIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers — safe wrappers that never raise
# ─────────────────────────────────────────────────────────────────────────────


def lib_status() -> Dict[str, bool]:
    """Return install status of every supported library."""
    return {
        "maths_lib":   MATHS_LIB_AVAILABLE,
        "extract_lib": EXTRACT_LIB_AVAILABLE,
        "pattern_lib": PATTERN_LIB_AVAILABLE,
        "format_lib":  FORMAT_LIB_AVAILABLE,
        "logic_lib":   LOGIC_LIB_AVAILABLE,
        "algo_lib":    ALGO_LIB_AVAILABLE,
        "verify_lib":  VERIFY_LIB_AVAILABLE,
    }


def lib_summary() -> str:
    """Human-readable summary of which libs are installed."""
    s = lib_status()
    ok    = [k for k, v in s.items() if v]
    miss  = [k for k, v in s.items() if not v]
    parts = [f"libs installed ({len(ok)}/7): {', '.join(ok) or 'none'}"]
    if miss:
        parts.append(f"missing: {', '.join(miss)}")
    return "  |  ".join(parts)


# ── extract_lib ──────────────────────────────────────────────────────────────

def resolve_metric_safe(
    metric_id: str,
    cells: List[Dict],
    period: str = "",
) -> Optional[Any]:
    """
    Safe wrapper around extract_lib.resolver.resolve_metric.
    Returns the ExtractResult or None if extract_lib unavailable / errored.
    """
    if not EXTRACT_LIB_AVAILABLE or _resolve_metric is None:
        return None
    try:
        return _resolve_metric(metric_id, cells, period or "")
    except Exception:
        logger.exception("[lib_bridge] extract_lib.resolve_metric failed")
        return None


# ── pattern_lib ──────────────────────────────────────────────────────────────

def classify_question_safe(question: str) -> Optional[Any]:
    """
    Safe wrapper around pattern_lib.classify.
    Returns PatternResult dict or None if pattern_lib unavailable / errored.
    """
    if not PATTERN_LIB_AVAILABLE or _pattern_classify is None:
        return None
    try:
        return _pattern_classify(question)
    except Exception:
        logger.exception("[lib_bridge] pattern_lib.classify failed")
        return None


# ── format_lib ───────────────────────────────────────────────────────────────

def format_value_safe(
    value: Any,
    formatter_id: str = "default_number",
    **kwargs,
) -> Optional[str]:
    """
    Safe wrapper around format_lib.render.
    Returns formatted string or None if format_lib unavailable / errored.
    """
    if not FORMAT_LIB_AVAILABLE or _format_render is None:
        return None
    try:
        result = _format_render(formatter_id, value, **kwargs)
        if result is None:
            return None
        # FormatResult typically has a .text or .value attribute
        if hasattr(result, "text"):
            return str(result.text)
        if hasattr(result, "value"):
            return str(result.value)
        return str(result)
    except Exception:
        logger.exception("[lib_bridge] format_lib.render failed")
        return None


# ── logic_lib ────────────────────────────────────────────────────────────────

def fire_rule_safe(rule_id: str, **kwargs) -> Optional[Any]:
    """
    Safe wrapper around logic_lib.fire(rule_id, **kwargs).
    Returns RuleResult or None if logic_lib unavailable / errored / rule missing.
    """
    if not LOGIC_LIB_AVAILABLE or _logic_fire is None:
        return None
    try:
        return _logic_fire(rule_id, **kwargs)
    except KeyError:
        logger.debug("[lib_bridge] unknown logic rule: %s", rule_id)
        return None
    except Exception:
        logger.exception("[lib_bridge] logic_lib.fire failed for %s", rule_id)
        return None


def fire_rules_in_layer_safe(layer: str, **kwargs) -> List[Any]:
    """
    Fire every rule in a layer. Returns list of RuleResult.
    Empty list if logic_lib unavailable, layer empty, or all rules failed.

    Layers: 'extraction_resolver', 'decision_rules', 'composite_resolver',
            'reconciliation', 'narrative_reasoner', 'verifier'
    """
    if not LOGIC_LIB_AVAILABLE or _logic_list_by_layer is None:
        return []
    try:
        rule_ids = _logic_list_by_layer(layer)
    except Exception:
        logger.exception("[lib_bridge] logic_lib.list_by_layer failed")
        return []

    results = []
    for rid in rule_ids:
        r = fire_rule_safe(rid, **kwargs)
        if r is not None:
            results.append(r)
    return results


# ── algo_lib ─────────────────────────────────────────────────────────────────

def get_algorithm_safe(algo_id: str) -> Optional[Any]:
    """
    Safe wrapper around fina_algo_lib.get_algorithm.
    Returns callable or None if algo_lib unavailable / errored.
    """
    if not ALGO_LIB_AVAILABLE or _get_algorithm is None:
        return None
    try:
        return _get_algorithm(algo_id)
    except Exception:
        logger.exception("[lib_bridge] algo_lib.get_algorithm failed")
        return None


# ── verify_lib ───────────────────────────────────────────────────────────────

def verify_value_safe(
    metric: str,
    value: Any,
    confidence: float = 1.0,
    **context,
) -> Tuple[bool, bool, List[str]]:
    """
    Safe wrapper around verify_lib.verify_answer.

    Signature matches the real lib:
        verify_answer(metric: str, value: Any, confidence: float, **ctx)
        returns VerifyVerdict(ok, abstain, answer, reasons, checks_run)

    Returns:
        (ok, abstain, reasons)
        - ok:      True if checks passed OR verify_lib unavailable
        - abstain: True if verifier recommends abstaining (low conf / hard fail)
        - reasons: list of human-readable check failures (empty on success)
    """
    if not VERIFY_LIB_AVAILABLE or _verify_answer is None:
        # Fail-open: if lib not installed, don't block answers
        return True, False, []
    try:
        verdict = _verify_answer(
            metric,
            value,
            confidence=confidence,
            **context,
        )
        ok      = bool(getattr(verdict, "ok", True))
        abstain = bool(getattr(verdict, "abstain", False))
        reasons = list(getattr(verdict, "reasons", []) or [])
        return ok, abstain, reasons
    except Exception:
        logger.exception("[lib_bridge] verify_lib.verify_answer failed")
        return True, False, ["verify_lib internal error"]


# ── Convenience: parse + verify a final_answer string ────────────────────────

_NUMBER_RE = re.compile(r"-?\$?\s*\(?-?\s*[\d,]+(?:\.\d+)?\)?")


def _parse_number(token: str) -> Optional[float]:
    """Extract a float from '$1,577', '(1,234)', '34.5%' etc."""
    if not token:
        return None
    s = str(token).strip()
    neg = "(" in s and ")" in s
    s = re.sub(r"[\$\s\(\)%,]", "", s)
    if not s or s == "-":
        return None
    try:
        v = float(s)
        return -abs(v) if neg else v
    except ValueError:
        return None


def verify_final_answer_safe(
    answer:     str,
    metric:     str   = "answer",
    confidence: float = 1.0,
    **context,
) -> Tuple[bool, bool, List[str]]:
    """
    Smart adapter: extract the first number from an answer string and
    verify it against verify_lib.

    Pipeline.py calls this with the rendered final_answer; we pull out
    the headline number and run verification.

    Returns (ok, abstain, reasons) — see verify_value_safe.
    """
    if not VERIFY_LIB_AVAILABLE or _verify_answer is None:
        return True, False, []

    if not answer:
        return True, False, []   # nothing to verify

    # Find the first number-looking token
    m = _NUMBER_RE.search(answer)
    if not m:
        # No number found — skip verification, don't block string answers
        return True, False, []

    value = _parse_number(m.group(0))
    if value is None:
        return True, False, []

    return verify_value_safe(metric, value, confidence=confidence, **context)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- lib_bridge sanity check --[/bold cyan]")
    rprint(f"\n[bold]Library status:[/bold] {lib_summary()}\n")

    for name, available in lib_status().items():
        symbol = "✓" if available else "✗"
        colour = "green" if available else "yellow"
        rprint(f"  [[{colour}]{symbol}[/{colour}]] {name:<14} "
               f"= {'installed' if available else 'not installed'}")

    rprint("\n[bold cyan]-- Safe wrappers test --[/bold cyan]")

    # 1. extract_lib
    r1 = resolve_metric_safe("revenue", [], "FY2023")
    rprint(f"  resolve_metric_safe        → {r1}")

    # 2. pattern_lib
    r2 = classify_question_safe("What was Apple's net income in FY2023?")
    rprint(f"  classify_question_safe     → {r2}")

    # 3. format_lib
    r3 = format_value_safe(123456.78)
    rprint(f"  format_value_safe          → {r3}")

    # 4. logic_lib — single rule
    r4a = fire_rule_safe("nonexistent_rule_id")
    rprint(f"  fire_rule_safe (unknown)   → {r4a}")

    # 5. logic_lib — layer-based firing
    r4b = fire_rules_in_layer_safe("verifier", value=100.0)
    rprint(f"  fire_rules_in_layer_safe   → {len(r4b)} verifier rules fired")

    # 6. algo_lib
    r5 = get_algorithm_safe("dcf")
    rprint(f"  get_algorithm_safe         → {r5}")

    # 7. verify_lib — direct value verification
    ok, abstain, reasons = verify_value_safe(
        "revenue", 1577.0, confidence=0.95,
    )
    rprint(f"  verify_value_safe          → ok={ok} abstain={abstain} "
           f"reasons={len(reasons)}")

    # 8. verify_lib — string answer verification
    ok2, abs2, r2 = verify_final_answer_safe(
        "$1,577 million [3M / 10-K / FY2018 / CashFlow / 42]",
        metric="capex",
        confidence=0.95,
    )
    rprint(f"  verify_final_answer_safe   → ok={ok2} abstain={abs2} "
           f"reasons={len(r2)}")

    rprint("\n[bold green]lib_bridge v2 ready.[/bold green]\n")
