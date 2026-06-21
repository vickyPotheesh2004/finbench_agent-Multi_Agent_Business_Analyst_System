"""
diag_silver_regression.py - Check Fix C against multiple ratio cases at once,
WITHOUT running the full slow pipeline. Uses synthetic operand pairs that
mirror the real bug plus cases that must NOT be touched, so we catch any
Fix-C misfire (over-correction) in seconds.

Run:  python diag_silver_regression.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]

from question_lib.models import SubFormula, Operation
from question_lib.plan_executor import _execute_sub

def ratio(a, b, op=Operation.RATIO):
    sub = SubFormula(name="t", formula_id="t", inputs=["x", "y"], operation=op)
    return _execute_sub(sub, {"x": a, "y": b}, {})

print("="*70)
print(" FIX-C REGRESSION CHECK")
print("="*70)
print(" Each line: inputs -> result | EXPECT")
print("-"*70)

cases = [
    # (label, a, b, op, expected_approx, should_reconcile)
    ("Adobe OCF ratio (THE BUG): 1469.5 / 2,213,556",
     1469.5, 2213556.0, Operation.RATIO, 0.66, True),
    ("Same-scale OCF ratio: 1469.5 / 2213.556 (already clean)",
     1469.5, 2213.556, Operation.RATIO, 0.66, False),
    ("Current ratio same scale: 5000 / 3000",
     5000.0, 3000.0, Operation.RATIO, 1.667, False),
    ("Operating margin %: 1494 / 5854 (Adobe FY16)",
     1494.0, 5854.0, Operation.RATIO_PCT, 25.52, False),
    ("Net margin %: 630 / 4795",
     630.0, 4795.0, Operation.RATIO_PCT, 13.14, False),
    ("LEGIT tiny ratio - interest coverage style: 5000 / 12 (NO reconcile)",
     5000.0, 12.0, Operation.RATIO, 416.67, False),
    ("LEGIT 1000x that is REAL: 2000 / 2.0 (gap=1000, but both plausible $M)",
     2000.0, 2.0, Operation.RATIO, 1000.0, False),
    ("Debt/equity normal: 8000 / 12000",
     8000.0, 12000.0, Operation.RATIO, 0.667, False),
    ("Zero operand -> abstain (Fix B): 0 / 5000",
     0.0, 5000.0, Operation.RATIO, None, None),
]

passed = 0
for label, a, b, op, exp, reconcile in cases:
    r = ratio(a, b, op)
    if exp is None:
        ok = r is None
        got = "None (abstain)" if r is None else f"{r:.4f}"
        exp_s = "None (abstain)"
    else:
        ok = (r is not None) and abs(r - exp) < max(0.05, abs(exp)*0.01)
        got = f"{r:.4f}" if r is not None else "None"
        exp_s = f"{exp:.4f}"
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"        got={got}  expect={exp_s}")

print("-"*70)
print(f" {passed}/{len(cases)} passed")
print("="*70)
print("\n READ CAREFULLY:")
print(" - The two LEGIT cases (interest coverage, 2000/2.0) test for")
print("   OVER-correction. If Fix C wrongly fires on them, they FAIL and")
print("   we must narrow the 300-3000x band.")
print(" - 2000/2.0 has gap=1000 and SITS INSIDE the band -> this is the")
print("   known risk. If it fails, Fix C is too aggressive.")
