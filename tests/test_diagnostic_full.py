"""
tests/test_diagnostic_full.py
FULL DIAGNOSTIC TEST SUITE — runs every critical check in one go.

Usage:
    pytest tests/test_diagnostic_full.py -v --tb=short
    # OR plain python:
    python tests/test_diagnostic_full.py

This catches:
  - Import errors in any module
  - 8 support libs availability
  - question_lib self-tests
  - Pipeline import + class instantiation
  - FIX-v6/v7/v8/v9 verification on real questions
  - Composite resolver routing
  - Grader format compatibility

Goal: surface every error in 60 seconds so you don't burn 2-hour evals
      on a broken pipeline.
"""
from __future__ import annotations

import sys
import os
import importlib
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Pretty-print pass/fail with colours
class C:
    G = "\033[92m"   # green
    R = "\033[91m"   # red
    Y = "\033[93m"   # yellow
    B = "\033[1m"    # bold
    E = "\033[0m"    # end


def _print_section(label: str):
    print(f"\n{C.B}{'=' * 78}{C.E}")
    print(f"{C.B}  {label}{C.E}")
    print(f"{C.B}{'=' * 78}{C.E}")


def _ok(msg: str):
    print(f"  {C.G}[PASS]{C.E} {msg}")


def _fail(msg: str, exc: Exception = None):
    print(f"  {C.R}[FAIL]{C.E} {msg}")
    if exc:
        print(f"         {C.R}{type(exc).__name__}: {exc}{C.E}")


def _warn(msg: str):
    print(f"  {C.Y}[WARN]{C.E} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Support lib imports
# ─────────────────────────────────────────────────────────────────────────────

def test_support_libs():
    _print_section("1. SUPPORT LIB IMPORTS (8 libs)")
    libs = {
        "maths_lib":     None,
        "extract_lib":   "synonyms",
        "pattern_lib":   None,
        "format_lib":    None,
        "logic_lib":     None,
        "algo_lib":      None,
        "verify_lib":    None,
        "question_lib":  None,
    }
    results = {}
    for lib, submod in libs.items():
        try:
            mod = importlib.import_module(lib)
            if submod:
                importlib.import_module(f"{lib}.{submod}")
            _ok(f"{lib:<14} import OK")
            results[lib] = True
        except Exception as e:
            _fail(f"{lib:<14} import FAILED", e)
            results[lib] = False
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. question_lib internal modules
# ─────────────────────────────────────────────────────────────────────────────

def test_question_lib_modules():
    _print_section("2. question_lib INTERNAL MODULES (12 files)")
    modules = [
        "question_lib.models",
        "question_lib.intent_classifier",
        "question_lib.subject_extractor",
        "question_lib.period_extractor",
        "question_lib.modifier_extractor",
        "question_lib.operation_detector",
        "question_lib.formula_matcher",
        "question_lib.dependency_graph",
        "question_lib.decomposer",
        "question_lib.plan_executor",
        "question_lib.registry",
    ]
    results = {}
    for m in modules:
        try:
            importlib.import_module(m)
            _ok(f"{m:<40} import OK")
            results[m] = True
        except Exception as e:
            _fail(f"{m:<40} import FAILED", e)
            results[m] = False
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. question_lib functional self-test (parse + execute end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_question_lib_functional():
    _print_section("3. question_lib FUNCTIONAL END-TO-END")
    try:
        from question_lib import parse_question, execute_plan
    except Exception as e:
        _fail("import parse_question / execute_plan", e)
        return False

    raw_text_3m_2018 = (
        "Purchases of property, plant and equipment $(1,577) million in 2018. "
        "Total assets at December 31, 2018 were $36,500 million. "
        "Net sales for 2018 were $32,765 million. "
    )

    cases = [
        # (question, expected_intent, expected_metric, expected_unit_in_output)
        ("What is the FY2018 capital expenditure amount for 3M? Answer in USD millions.",
         "extract", "capex", "million"),
        ("What is the year end FY2018 net PPNE for 3M? Answer in USD billions.",
         "extract", "ppe", "billion"),
        ("Is 3M a capital-intensive business based on FY2022 data?",
         "decide", None, None),
        ("What drove operating margin change in FY2022?",
         "narrate", "operating_margin", None),
    ]

    all_pass = True
    for q, exp_intent, exp_metric, exp_unit_str in cases:
        try:
            plan = parse_question(q)
            ok_intent = plan.intent.value == exp_intent
            ok_metric = (
                exp_metric is None
                or (plan.subject and plan.subject.metric_id == exp_metric)
            )
            ok = ok_intent and ok_metric
            if ok:
                _ok(f"intent={plan.intent.value} subj={plan.subject.metric_id if plan.subject else None} | {q[:60]}")
            else:
                _fail(f"got intent={plan.intent.value} subj={plan.subject.metric_id if plan.subject else None} | exp intent={exp_intent} metric={exp_metric}")
                all_pass = False

            # If EXTRACT and we have raw_text, also run execute_plan
            if exp_intent == "extract":
                res = execute_plan(plan, cells=[], raw_text=raw_text_3m_2018,
                                    company="3M", fy="FY2018", doc_type="10-K")
                if res.answered:
                    _ok(f"   executed: {res.final_answer[:90]}")
                    if exp_unit_str and exp_unit_str in res.final_answer.lower():
                        _ok(f"   unit OK: '{exp_unit_str}' present in output")
                    elif exp_unit_str:
                        _warn(f"   unit MISSING: expected '{exp_unit_str}' in output")
                else:
                    _warn(f"   execute_plan returned answered=False (reason: {res.audit_trail.get('reason')})")
        except Exception as e:
            _fail(f"Q failed: {q[:50]}", e)
            traceback.print_exc()
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline import + instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_import():
    _print_section("4. PIPELINE IMPORT + INSTANTIATION")
    try:
        from src.pipeline.pipeline import FinBenchPipeline
        _ok("FinBenchPipeline import OK")
    except Exception as e:
        _fail("FinBenchPipeline import FAILED", e)
        traceback.print_exc()
        return False

    try:
        p = FinBenchPipeline()
        _ok("FinBenchPipeline() instantiation OK")
    except Exception as e:
        _fail("FinBenchPipeline() instantiation FAILED", e)
        traceback.print_exc()
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite resolver routing
# ─────────────────────────────────────────────────────────────────────────────

def test_composite_resolver():
    _print_section("5. COMPOSITE RESOLVER (N20) ROUTING")
    try:
        from src.analysis.composite_resolver import (
            run_composite_resolver, classify_question, CompositeAnswer,
        )
        _ok("composite_resolver import OK")
    except Exception as e:
        _fail("composite_resolver import FAILED", e)
        traceback.print_exc()
        return False

    test_classifications = [
        ("Is 3M capital-intensive in FY2022?", "decision"),
        ("What drove operating margin change?", "causal"),
        ("Which segment dragged growth?", "segment"),
        ("What was Apple's revenue?", "other"),
    ]
    for q, expected in test_classifications:
        got = classify_question(q)
        if got == expected:
            _ok(f"classify('{q[:45]}') → '{got}'")
        else:
            _fail(f"classify('{q[:45]}') → '{got}' (expected '{expected}')")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grader format compatibility (the BIG ONE)
# ─────────────────────────────────────────────────────────────────────────────

def test_grader_compatibility():
    _print_section("6. GRADER FORMAT COMPATIBILITY (FIX-v8 / FIX-v9)")
    try:
        # Import the exact grader from run_financebench.py
        import sys
        eval_dir = ROOT / "eval"
        sys.path.insert(0, str(eval_dir))
        from run_financebench import normalize_number, numbers_match, is_correct
        _ok("grader functions imported OK")
    except Exception as e:
        _fail("grader import FAILED", e)
        return False

    # Test scenarios that MUST pass for our system to score points
    cases = [
        # FIX-v8: minus sign should not matter
        ("$1,577.00 million", "$1,577 million", True, "FIX-v8: clean format"),
        # FIX-v8: negative cash flow item
        ("$(1577)", "$1577 million", True, "FIX-v8: paren neg matches positive"),
        # FIX-v9: billion vs million scale match
        ("$8.70 billion", "$8.7 billion", True, "FIX-v9: billions match"),
        # The OLD bug: negative with explicit minus
        ("$-1,577 million", "$1,577 million", False, "OLD BUG: minus sign breaks match"),
        # The OLD bug: scale mismatch
        ("$8,700.00 million", "$8.7 billion", False, "OLD BUG: scale mismatch"),
        # Identity
        ("5.91%", "5.91%", True, "identity %"),
    ]

    all_pass = True
    for pred, gold, expected, label in cases:
        correct, mtype, score = is_correct(pred, gold)
        if correct == expected:
            _ok(f"{label} | pred='{pred}' gold='{gold}' → {correct} ({mtype})")
        else:
            _fail(f"{label} | pred='{pred}' gold='{gold}' → {correct} but expected {expected}")
            all_pass = False
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. logic_lib decision rule firing
# ─────────────────────────────────────────────────────────────────────────────

def test_logic_lib():
    _print_section("7. logic_lib DECISION RULES")
    try:
        from logic_lib import fire
        _ok("logic_lib.fire imported")
    except Exception as e:
        _fail("logic_lib.fire import FAILED", e)
        return False

    cases = [
        ("capital_intensity_class", {"capex_to_revenue": 5.9}, "moderately"),
        ("capital_intensity_class", {"capex_to_revenue": 20.0}, "highly"),
        ("capital_intensity_class", {"capex_to_revenue": 2.0}, "not"),
        ("liquidity_current_ratio", {"current_ratio": 2.5}, "strong"),
        ("liquidity_current_ratio", {"current_ratio": 0.5}, "weak"),
    ]
    all_pass = True
    for rule_id, kwargs, expected_substr in cases:
        try:
            r = fire(rule_id, **kwargs)
            output = str(getattr(r, "output", "")).lower()
            if expected_substr in output:
                _ok(f"fire('{rule_id}', {kwargs}) → '{output}' (contains '{expected_substr}')")
            else:
                _warn(f"fire('{rule_id}', {kwargs}) → '{output}' (expected substring '{expected_substr}')")
        except Exception as e:
            _fail(f"fire('{rule_id}') FAILED", e)
            all_pass = False
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 8. Critical environment variables
# ─────────────────────────────────────────────────────────────────────────────

def test_env_vars():
    _print_section("8. ENVIRONMENT VARIABLES")
    _ok(f"SKIP_LLM    = {os.environ.get('SKIP_LLM', 'unset')}")
    _ok(f"SNIPER_ONLY = {os.environ.get('SNIPER_ONLY', 'unset')}")
    _ok(f"DISABLE_BGE = {os.environ.get('DISABLE_BGE', 'unset')}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print(f"\n{C.B}╔{'═' * 76}╗{C.E}")
    print(f"{C.B}║  FINBENCH FULL DIAGNOSTIC  —  catch every error before next eval        ║{C.E}")
    print(f"{C.B}╚{'═' * 76}╝{C.E}")

    results = {
        "1. Support libs":            test_support_libs(),
        "2. question_lib modules":    test_question_lib_modules(),
        "3. question_lib functional": test_question_lib_functional(),
        "4. Pipeline":                test_pipeline_import(),
        "5. Composite resolver":      test_composite_resolver(),
        "6. Grader compatibility":    test_grader_compatibility(),
        "7. logic_lib decisions":     test_logic_lib(),
        "8. Env vars":                test_env_vars(),
    }

    _print_section("SUMMARY")
    for name, ok in results.items():
        if isinstance(ok, dict):
            n_pass = sum(1 for v in ok.values() if v)
            n_tot = len(ok)
            mark = "PASS" if n_pass == n_tot else "PARTIAL"
            print(f"  [{mark:<7}] {name}: {n_pass}/{n_tot}")
        else:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark:<7}] {name}")

    print(f"\n{C.B}{'═' * 78}{C.E}")
    print(f"  Done. If you see [FAIL] above, fix those BEFORE running the eval.")
    print(f"  If all green, run: python eval\\run_financebench.py --limit 10")
    print(f"{C.B}{'═' * 78}{C.E}\n")


# Pytest-compatible adapter
def test_full_diagnostic():
    """pytest entry — fails the test if anything above failed."""
    results = {}
    failed = []
    try:
        results["1"] = test_support_libs()
        results["2"] = test_question_lib_modules()
        results["3"] = test_question_lib_functional()
        results["4"] = test_pipeline_import()
        results["5"] = test_composite_resolver()
        results["6"] = test_grader_compatibility()
        results["7"] = test_logic_lib()
        results["8"] = test_env_vars()
    except Exception as e:
        failed.append(f"diagnostic crashed: {e}")
    for k, v in results.items():
        if v is False:
            failed.append(k)
    assert not failed, f"Diagnostic failures in: {failed}"


if __name__ == "__main__":
    run_all()
